#include <torch/extension.h>
#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <c10/cuda/CUDAGuard.h>

#include <pybind11/numpy.h>

#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <algorithm>
#include <vector>

using std::vector;
namespace py = pybind11;

#define min_value (1e-9)

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_NUM(x, y) TORCH_CHECK((x)>=(y), #x "should be >=" #y)
#define access_G(nonlinear, batch_num, x, row, y, col) (nonlinear? G[batch_num][x][y][row][col] : G[0][x][row][y][col])

template<typename scalar_t>
__global__ void ir_drop_accurate_GS_kernel_vdown(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> G,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_up,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_down,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_up_prev,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_down_prev,
    const int row_size,
    const scalar_t g_wire,
    const scalar_t g_load_down,
    const scalar_t beta,
    bool nonlinear_cell
)
{
    const int batch_num = blockIdx.x;
    const int x = blockIdx.y;
    const int y = blockIdx.z;
    const int col = threadIdx.x;
    int i = row_size-1;

    scalar_t up, down, other_side, sumg;

    up = v_down_prev[batch_num][x][y][i-1][col];
    down = 0;
    other_side = v_up[batch_num][x][y][i][col];
    sumg = g_wire+g_load_down + access_G(nonlinear_cell, batch_num, x, i, y, col);
    v_down[batch_num][x][y][i][col] = beta * (g_wire * up+ g_load_down*down + access_G(nonlinear_cell, batch_num, x, i, y, col) * other_side)/sumg \
                                    + (1-beta) * (v_down_prev[batch_num][x][y][i][col]);


    for (i = row_size-2; i>0; --i)
    {
        up = v_down_prev[batch_num][x][y][i-1][col];
        down = v_down[batch_num][x][y][i+1][col];
        other_side = v_up[batch_num][x][y][i][col];
        sumg = g_wire*2 + access_G(nonlinear_cell, batch_num, x, i, y, col);
        v_down[batch_num][x][y][i][col] = beta * (g_wire * (up+down) + access_G(nonlinear_cell, batch_num, x, i, y, col) * other_side)/sumg \
                                    + (1-beta) * (v_down_prev[batch_num][x][y][i][col]);
    }

    down = v_down[batch_num][x][y][i+1][col];
    other_side = v_up[batch_num][x][y][i][col];
    sumg = g_wire + access_G(nonlinear_cell, batch_num, x, i, y, col);
    v_down[batch_num][x][y][i][col] = beta * (g_wire * down + access_G(nonlinear_cell, batch_num, x, i, y, col) * other_side)/sumg \
                                    + (1-beta) * (v_down_prev[batch_num][x][y][i][col]);
}

template<typename scalar_t>
__global__ void ir_drop_accurate_GS_kernel_vup(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> G,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_up,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_down,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_up_prev,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_down_prev,
    const int col_size,
    const scalar_t g_wire,
    const scalar_t g_load_left,
    const scalar_t beta,
    bool nonlinear_cell
)
{
    const int batch_num = blockIdx.x;
    const int x = blockIdx.y;
    const int y = blockIdx.z;
    const int row = threadIdx.x;
    int j = 0;

    scalar_t left, right, other_side, sumg;
    left = input[batch_num][x][row];
    right = v_up_prev[batch_num][x][y][row][j+1];
    other_side = v_down_prev[batch_num][x][y][row][j];
    sumg = g_wire + g_load_left + access_G(nonlinear_cell, batch_num, x, row, y, j);

    v_up[batch_num][x][y][row][j] = beta * (g_load_left * left+ g_wire*right + access_G(nonlinear_cell, batch_num, x, row, y, j) * other_side)/sumg \
                                        + (1-beta) * v_up_prev[batch_num][x][y][row][j];
    for (j = 1; j < col_size-1; ++j)
    {
        left = v_up[batch_num][x][y][row][j-1];
        right = v_up_prev[batch_num][x][y][row][j+1];
        other_side = v_down_prev[batch_num][x][y][row][j];
        sumg = g_wire*2 + access_G(nonlinear_cell, batch_num, x, row, y, j);
        v_up[batch_num][x][y][row][j] = beta * (g_wire * (left+right) + access_G(nonlinear_cell, batch_num, x, row, y, j) * other_side)/sumg \
                                            + (1-beta) * v_up_prev[batch_num][x][y][row][j];
    }

    sumg = g_wire + access_G(nonlinear_cell, batch_num, x, row, y, j);
    left = v_up[batch_num][x][y][row][j-1];
    other_side = v_down_prev[batch_num][x][y][row][j];
    v_up[batch_num][x][y][row][j] = beta * (g_wire * left + access_G(nonlinear_cell, batch_num, x, row, y, j) * other_side)/sumg \
                                            + (1-beta) * v_up_prev[batch_num][x][y][row][j];
}

template<typename scalar_t>
__global__ void get_current(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_up,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_down,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> G,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> current,
    const int row_size,
    bool nonlinear_cell
)
{
    const int batch_num = blockIdx.x;
    const int x = blockIdx.y;
    const int y = blockIdx.z;
    const int col = threadIdx.x;

    for (int i=0; i<row_size; ++i)
    {
        current[batch_num][x][y][col] += (v_up[batch_num][x][y][i][col]-v_down[batch_num][x][y][i][col])*access_G(nonlinear_cell, batch_num, x, i, y, col);
    }
}

template<typename scalar_t>
__global__ void deal_v(
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> V,
    scalar_t th,
    scalar_t v,
    int col_size
)
{
    const int batch_num = blockIdx.x;
    const int x = blockIdx.y;
    const int y = blockIdx.z;
    const int row = threadIdx.x;

    for (int j = 0; j<col_size; ++j)
    {
        scalar_t temp = V[batch_num][x][y][row][j];
        if (temp<th && temp>-th)
        {
            if (temp>0)
                V[batch_num][x][y][row][j] = v;
            else
                V[batch_num][x][y][row][j] = -v;
        }
    }
}


// # R = z*R_off + (1-z)*Ron, (Roff-Ron)*z + Ron = R
// def get_nonlinear_ratio_z(r, r_min, r_max):
//     return (r-r_min)/(r_max-r_min)

// def nonlinear_R_tensor(a_on, b_on, a_off, b_off, z, V):
//     return (1-z)*nonlinear_R_tensor(a_on, b_on, V)+z*nonlinear_R_tensor(a_off, b_off, V)

template<typename scalar_t>
torch::Tensor get_nonlinear_ratio_z(torch::Tensor R, scalar_t r_min, scalar_t r_max)
{
    auto z = (R-r_min)/(r_max-r_min);
    z.index_put_({z<0}, 0);
    return z;
}

template<typename scalar_t>
torch::Tensor nonlinear_R_tensor(torch::Tensor V, torch::Tensor z, scalar_t a_on, scalar_t b_on, scalar_t a_off, scalar_t b_off)
{
    auto nonlinear_I = [](scalar_t a, scalar_t b, torch::Tensor V) -> torch::Tensor
    {
        return b*(V*a).sinh();
    };

    V = V.abs();
    auto I_on = nonlinear_I(a_on, b_on, V);
    auto I_off = nonlinear_I(a_off, b_off, V);
    return (1-z)*V/I_on+z*V/I_off;
}



// ir_drop_accurate_GSMethod(in_b, g_b_neg, bsize, true_insize//phyArrParams.arrRowSize, true_outsize//phyArrParams.arrColSize)
// #inv [bsize, numX, phyArrParams.arrRowSize]
// G [numX, phyArrParams.arrRowSize, true_outsize]
// return [bsize, numx, true_outsize]
torch::Tensor ir_drop_gs(
    torch::Tensor input,
    torch::Tensor G,
    int batch_size,
    int row_size,
    int col_size,
    int numX,
    int numY,
    int iter_times, 
    double g_wire,
    double beta,
    bool enable_break,
    double break_threshould,
    bool nonlinear_cell_enable,
    vector<double> nonlinear_params,
    bool ret_V = false
)
{
    CHECK_INPUT(input);
    CHECK_INPUT(G);
    CHECK_NUM(row_size, 2);
    CHECK_NUM(col_size, 2);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    // input is [batch_size, numX, row_size]
    // voltage is [batch_size, numX, numY, row_size, col_size]
    auto voltage_up = input.unsqueeze(2).unsqueeze(4).repeat({1, 1, numY, 1, col_size});
    auto voltage_down = G.new_zeros({batch_size, numX, numY, row_size, col_size});

    auto voltage_up_prev = voltage_up.clone();
    auto voltage_down_prev = voltage_down.clone();
    const dim3 blocks(batch_size, numX, numY);

    G = G.view({1, numX, row_size, numY, col_size});
    torch::Tensor z;

    if (nonlinear_cell_enable)
    {
        z = get_nonlinear_ratio_z<double>(1/G, nonlinear_params[0], nonlinear_params[1]).transpose(2, 3).contiguous();
    }

    for (int i = 0; i<iter_times; ++i)
    {
        if (nonlinear_cell_enable)
        {
            auto V = voltage_up - voltage_down;
            
            AT_DISPATCH_FLOATING_TYPES(V.type(), "deal_v", ([&]
            {
                deal_v<scalar_t><<<blocks, row_size>>>(
                    V.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                    1e-5,
                    1e-5,
                    col_size 
                );
            }));
            G = 1/nonlinear_R_tensor<double>(V, z, nonlinear_params[2], nonlinear_params[3], nonlinear_params[4], nonlinear_params[5]);
        }

        AT_DISPATCH_FLOATING_TYPES(G.type(), "ir_drop_accurate_GS_kernel_vup", ([&]
        {
            ir_drop_accurate_GS_kernel_vup<scalar_t><<<blocks, row_size>>>(
                input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(), 
                G.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_up.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_up_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                col_size,
                g_wire,
                g_wire,
                beta,
                nonlinear_cell_enable);
        }));
        AT_DISPATCH_FLOATING_TYPES(G.type(), "ir_drop_accurate_GS_kernel_vdown", ([&]
        {
            ir_drop_accurate_GS_kernel_vdown<scalar_t><<<blocks, col_size>>>(
                input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(), 
                G.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_up.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_up_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                row_size,
                g_wire,
                g_wire,
                beta,
                nonlinear_cell_enable);
        }));

        std::swap(voltage_up, voltage_up_prev);
        std::swap(voltage_down, voltage_down_prev);
        if (enable_break)
        {
            auto diff_up = ((voltage_up - voltage_up_prev).abs()>break_threshould).sum().item<int>();
            auto diff_down = ((voltage_down - voltage_down_prev).abs()>break_threshould).sum().item<int>();
            if (diff_up == 0 && diff_down == 0)
            {
                // std::cout << "break in " << i << ' ' << std::endl;
                break;
            }
        }
    }


    if (ret_V)
    {
        return (voltage_up_prev-voltage_down_prev);
    }

    // in fact, voltage_down_prev and voltage_up_prev are newest.
    // current = (last_vup-last_vdown).mul(x) # [bsize, numx, rowsize, numy, colsize] * [numx, rowsize, numy, colsize]
        // return current.sum(2).reshape(bsize, numX, -1) # [bsize, numx, true_outsize]

    auto current = G.new_zeros({batch_size, numX, numY, col_size});
    AT_DISPATCH_FLOATING_TYPES(G.type(), "get_current", ([&]
    {
        get_current<scalar_t><<<blocks, col_size>>>(
                voltage_up_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                G.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                current.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                row_size,
                nonlinear_cell_enable
            );
    }));

    return current.view({batch_size, numX, -1});
}


template<typename scalar_t>
__global__ void ir2s_vup(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> G,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_up,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_down,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_up_prev,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_down_prev,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> I_row_sum,
    const int col_size,
    const scalar_t r_wire,
    const scalar_t r_load_left,
    bool nonlinear_cell
)
{
    const int batch_num = blockIdx.x;
    const int x = blockIdx.y;
    const int y = blockIdx.z;
    const int row = threadIdx.x;
    int j;

    for (j=0; j<col_size; ++j)
    {
        I_row_sum[batch_num][x][y][row] += (v_up_prev[batch_num][x][y][row][j] - v_down_prev[batch_num][x][y][row][j])*access_G(nonlinear_cell, batch_num, x, row, y, j);
    }

    j = 0;
    v_up[batch_num][x][y][row][j] = input[batch_num][x][row] - I_row_sum[batch_num][x][y][row]*r_load_left;
    scalar_t pass_current = (v_up_prev[batch_num][x][y][row][j] - v_down_prev[batch_num][x][y][row][j])*access_G(nonlinear_cell, batch_num, x, row, y, j);
    I_row_sum[batch_num][x][y][row] -= pass_current;

    for (j=1; j<col_size; ++j)
    {
        v_up[batch_num][x][y][row][j] = v_up[batch_num][x][y][row][j-1] - I_row_sum[batch_num][x][y][row]*r_wire;
        pass_current = (v_up_prev[batch_num][x][y][row][j] - v_down_prev[batch_num][x][y][row][j])*access_G(nonlinear_cell, batch_num, x, row, y, j);
        I_row_sum[batch_num][x][y][row] -= pass_current;
    }
}

template<typename scalar_t>
__global__ void ir2s_vdown(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> G,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_up,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_down,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_up_prev,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_down_prev,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> I_col_sum,
    const int row_size,
    const scalar_t r_wire,
    const scalar_t r_load_down,
    bool nonlinear_cell
)
{
    const int batch_num = blockIdx.x;
    const int x = blockIdx.y;
    const int y = blockIdx.z;
    const int col = threadIdx.x;
    int i;

    for (i=0; i<row_size; ++i)
    {
        I_col_sum[batch_num][x][y][col] += (v_up_prev[batch_num][x][y][i][col] - v_down_prev[batch_num][x][y][i][col])*access_G(nonlinear_cell, batch_num, x, i, y, col);
    }

    i = row_size-1;
    v_down[batch_num][x][y][i][col] = I_col_sum[batch_num][x][y][col]*r_load_down;
    scalar_t pass_current = (v_up_prev[batch_num][x][y][i][col] - v_down_prev[batch_num][x][y][i][col])*access_G(nonlinear_cell, batch_num, x, i, y, col);
    I_col_sum[batch_num][x][y][col] -= pass_current;

    for (i=row_size-2; i>=0; --i)
    {
        v_down[batch_num][x][y][i][col] = v_down[batch_num][x][y][i+1][col] + I_col_sum[batch_num][x][y][col]*r_wire;
        pass_current = (v_up_prev[batch_num][x][y][i][col] - v_down_prev[batch_num][x][y][i][col])*access_G(nonlinear_cell, batch_num, x, i, y, col);
        I_col_sum[batch_num][x][y][col] -= pass_current;
    }
}



//inv [bsize, numX, phyArrParams.arrRowSize]
//x [numX, phyArrParams.arrRowSize, true_outsize]
//return [bsize, numX, true_outsize]
torch::Tensor ir2s(
    torch::Tensor input,
    torch::Tensor G,
    int batch_size,
    int row_size,
    int col_size,
    int numX,
    int numY,
    int iter_times, 
    double r_wire,
    bool enable_break,
    double break_threshould,
    bool nonlinear_cell_enable,
    vector<double> nonlinear_params,
    bool ret_V = false
)
{
    CHECK_INPUT(input);
    CHECK_INPUT(G);
    CHECK_NUM(row_size, 2);
    CHECK_NUM(col_size, 2);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    auto voltage_up = input.unsqueeze(2).unsqueeze(4).repeat({1, 1, numY, 1, col_size});
    auto voltage_down = G.new_zeros({batch_size, numX, numY, row_size, col_size});

    auto voltage_up_prev = voltage_up.clone();
    auto voltage_down_prev = voltage_down.clone();
    auto I_row_sum = G.new_zeros({batch_size, numX, numY, row_size}); 
    auto I_col_sum = G.new_zeros({batch_size, numX, numY, col_size}); 
    const dim3 blocks(batch_size, numX, numY);


    G = G.view({1, numX, row_size, numY, col_size});
    torch::Tensor z;

    if (nonlinear_cell_enable)
    {
        z = get_nonlinear_ratio_z<double>(1/G, nonlinear_params[0], nonlinear_params[1]).transpose(2, 3).contiguous();
    }
    
    for (int i=0; i<iter_times; ++i)
    {
        if (nonlinear_cell_enable)
        {
            auto V = voltage_up - voltage_down;
            
            AT_DISPATCH_FLOATING_TYPES(V.type(), "deal_v", ([&]
            {
                deal_v<scalar_t><<<blocks, row_size>>>(
                    V.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                    1e-5,
                    1e-5,
                    col_size 
                );
            }));
            G = 1/nonlinear_R_tensor<double>(V, z, nonlinear_params[2], nonlinear_params[3], nonlinear_params[4], nonlinear_params[5]);
            // std::cout << G << std::endl;
        }

        AT_DISPATCH_FLOATING_TYPES(G.type(), "ir2s_vup", ([&]
        {
            ir2s_vup<scalar_t><<<blocks, row_size>>>(
                input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(), 
                G.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_up.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_up_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                I_row_sum.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), 
                col_size,
                r_wire,
                r_wire, 
                nonlinear_cell_enable);
        }));
        AT_DISPATCH_FLOATING_TYPES(G.type(), "ir2s_vdown", ([&]
        {
            ir2s_vdown<scalar_t><<<blocks, col_size>>>(
                input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(), 
                G.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_up.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_up_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                I_col_sum.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), 
                row_size,
                r_wire,
                r_wire, 
                nonlinear_cell_enable);
        }));

        std::swap(voltage_up, voltage_up_prev);
        std::swap(voltage_down, voltage_down_prev);
        if (enable_break)
        {
            auto diff_up = ((voltage_up - voltage_up_prev).abs()>break_threshould).sum().item<int>();
            auto diff_down = ((voltage_down - voltage_down_prev).abs()>break_threshould).sum().item<int>();
            if (diff_up == 0 && diff_down == 0)
            {
                // std::cout << "break in " << i << ' ' << std::endl;
                break;
            }
        }

    }

    if (ret_V)
    {
        return (voltage_up_prev-voltage_down_prev);
    }

    auto current = G.new_zeros({batch_size, numX, numY, col_size});
    AT_DISPATCH_FLOATING_TYPES(G.type(), "get_current", ([&]
    {
        get_current<scalar_t><<<blocks, col_size>>>(
                voltage_up_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                G.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                current.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                row_size, 
                nonlinear_cell_enable
            );
    }));

    return current.view({batch_size, numX, -1});

}

/**
 * @cal ir drop in OU size,  
 * 
 * @param input [bsize, numX, ouRowSize]
 * @param G [numX, ouRowSize, ouColSize]
 * @param batch_size 
 * @param row_size 
 * @param col_size 
 * @param numX 
 * @param numY 
 * @param iter_times 
 * @param r_wire 
 * @param r_load_left 
 * @param r_load_down 
 * @param enable_break 
 * @param break_threshould 
 * @return torch::Tensor [bsize, numX, ouColSize]
 */
torch::Tensor ir2s_ou_local(
    torch::Tensor input,
    torch::Tensor G,
    int batch_size,
    int row_size,
    int col_size,
    int numX,
    int numY,
    int iter_times, 
    double r_wire,
    double r_load_left,
    double r_load_down,
    bool enable_break,
    double break_threshould,
    bool nonlinear_cell_enable,
    vector<double> nonlinear_params
)
{
    CHECK_INPUT(input);
    CHECK_INPUT(G);
    CHECK_NUM(row_size, 2);
    CHECK_NUM(col_size, 2);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    auto voltage_up = input.unsqueeze(2).unsqueeze(4).repeat({1, 1, numY, 1, col_size});
    auto voltage_down = G.new_zeros({batch_size, numX, numY, row_size, col_size});

    auto voltage_up_prev = voltage_up.clone();
    auto voltage_down_prev = voltage_down.clone();
    auto I_row_sum = G.new_zeros({batch_size, numX, numY, row_size}); 
    auto I_col_sum = G.new_zeros({batch_size, numX, numY, col_size}); 
    const dim3 blocks(batch_size, numX, numY);


    G = G.view({1, numX, row_size, numY, col_size});
    torch::Tensor z;

    if (nonlinear_cell_enable)
    {
        z = get_nonlinear_ratio_z<double>(1/G, nonlinear_params[0], nonlinear_params[1]).transpose(2, 3).contiguous();
    }
    
    for (int i=0; i<iter_times; ++i)
    {   
        if (nonlinear_cell_enable)
        {
            auto V = voltage_up - voltage_down;
            
            AT_DISPATCH_FLOATING_TYPES(V.type(), "deal_v", ([&]
            {
                deal_v<scalar_t><<<blocks, row_size>>>(
                    V.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                    1e-5,
                    1e-5,
                    col_size 
                );
            }));
            G = 1/nonlinear_R_tensor<double>(V, z, nonlinear_params[2], nonlinear_params[3], nonlinear_params[4], nonlinear_params[5]);
        }

        AT_DISPATCH_FLOATING_TYPES(G.type(), "ir2s_vup", ([&]
        {
            ir2s_vup<scalar_t><<<blocks, row_size>>>(
                input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(), 
                G.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_up.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_up_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                I_row_sum.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), 
                col_size,
                r_wire,
                r_load_left, 
                nonlinear_cell_enable);
        }));
        AT_DISPATCH_FLOATING_TYPES(G.type(), "ir2s_vdown", ([&]
        {
            ir2s_vdown<scalar_t><<<blocks, col_size>>>(
                input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(), 
                G.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_up.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_up_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                I_col_sum.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), 
                row_size,
                r_wire,
                r_load_down,
                nonlinear_cell_enable);
        }));

        std::swap(voltage_up, voltage_up_prev);
        std::swap(voltage_down, voltage_down_prev);
        if (enable_break)
        {
            auto diff_up = ((voltage_up - voltage_up_prev).abs()>break_threshould).sum().item<int>();
            auto diff_down = ((voltage_down - voltage_down_prev).abs()>break_threshould).sum().item<int>();
            if (diff_up == 0 && diff_down == 0)
            {
                // std::cout << "break in " << i << ' ' << std::endl;
                break;
            }
        }

    }

    auto current = G.new_zeros({batch_size, numX, numY, col_size});
    AT_DISPATCH_FLOATING_TYPES(G.type(), "get_current", ([&]
    {
        get_current<scalar_t><<<blocks, col_size>>>(
                voltage_up_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                G.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                current.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                row_size, 
                nonlinear_cell_enable
            );
    }));

    return current.view({batch_size, numX, -1});

}


torch::Tensor ir_gs_ou_local(
    torch::Tensor input,
    torch::Tensor G,
    int batch_size,
    int row_size,
    int col_size,
    int numX,
    int numY,
    int iter_times, 
    double g_wire,
    double g_load_left,
    double g_load_down,
    double beta,
    bool enable_break,
    double break_threshould,
    bool nonlinear_cell_enable,
    vector<double> nonlinear_params
)
{
    CHECK_INPUT(input);
    CHECK_INPUT(G);
    CHECK_NUM(row_size, 2);
    CHECK_NUM(col_size, 2);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    // input is [batch_size, numX, row_size]
    // voltage is [batch_size, numX, numY, row_size, col_size]
    auto voltage_up = input.unsqueeze(2).unsqueeze(4).repeat({1, 1, numY, 1, col_size});
    auto voltage_down = G.new_zeros({batch_size, numX, numY, row_size, col_size});

    auto voltage_up_prev = voltage_up.clone();
    auto voltage_down_prev = voltage_down.clone();
    const dim3 blocks(batch_size, numX, numY);

    G = G.view({1, numX, row_size, numY, col_size});
    torch::Tensor z;

    if (nonlinear_cell_enable)
    {
        z = get_nonlinear_ratio_z<double>(1/G, nonlinear_params[0], nonlinear_params[1]).transpose(2, 3).contiguous();
    }

    for (int i = 0; i<iter_times; ++i)
    {
        if (nonlinear_cell_enable)
        {
            auto V = voltage_up - voltage_down;
            
            AT_DISPATCH_FLOATING_TYPES(V.type(), "deal_v", ([&]
            {
                deal_v<scalar_t><<<blocks, row_size>>>(
                    V.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                    1e-5,
                    1e-5,
                    col_size 
                );
            }));
            G = 1/nonlinear_R_tensor<double>(V, z, nonlinear_params[2], nonlinear_params[3], nonlinear_params[4], nonlinear_params[5]);
        }

        AT_DISPATCH_FLOATING_TYPES(G.type(), "ir_drop_accurate_GS_kernel_vup", ([&]
        {
            ir_drop_accurate_GS_kernel_vup<scalar_t><<<blocks, row_size>>>(
                input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(), 
                G.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_up.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_up_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                col_size,
                g_wire,
                g_load_left,
                beta,
                nonlinear_cell_enable);
        }));
        AT_DISPATCH_FLOATING_TYPES(G.type(), "ir_drop_accurate_GS_kernel_vdown", ([&]
        {
            ir_drop_accurate_GS_kernel_vdown<scalar_t><<<blocks, col_size>>>(
                input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(), 
                G.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_up.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_up_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                row_size,
                g_wire,
                g_load_down,
                beta,
                nonlinear_cell_enable);
        }));

        std::swap(voltage_up, voltage_up_prev);
        std::swap(voltage_down, voltage_down_prev);
        if (enable_break)
        {
            auto diff_up = ((voltage_up - voltage_up_prev).abs()>break_threshould).sum().item<int>();
            auto diff_down = ((voltage_down - voltage_down_prev).abs()>break_threshould).sum().item<int>();
            if (diff_up == 0 && diff_down == 0)
            {
                // std::cout << "break in " << i << ' ' << std::endl;
                break;
            }
        }
    }

    // in fact, voltage_down_prev and voltage_up_prev are newest.
    // current = (last_vup-last_vdown).mul(x) # [bsize, numx, rowsize, numy, colsize] * [numx, rowsize, numy, colsize]
        // return current.sum(2).reshape(bsize, numX, -1) # [bsize, numx, true_outsize]

    auto current = G.new_zeros({batch_size, numX, numY, col_size});
    AT_DISPATCH_FLOATING_TYPES(G.type(), "get_current", ([&]
    {
        get_current<scalar_t><<<blocks, col_size>>>(
                voltage_up_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                G.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                current.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                row_size,
                nonlinear_cell_enable
            );
    }));

    return current.view({batch_size, numX, -1});
}


template<typename scalar_t>
__global__ void ir2s_vup_entire(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> G,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_up,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_down,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_up_prev,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_down_prev,
    torch::PackedTensorAccessor32<scalar_t, 6, torch::RestrictPtrTraits> I_row_sum,
    const int ou_row_size,
    const int ou_col_size,
    const scalar_t r_wire,
    bool nonlinear_cell
)
{
    const int batch_num = blockIdx.x/ou_row_size;
    const int x = blockIdx.y;
    const int y = blockIdx.z;
    const int ou_row = blockIdx.x%ou_row_size;
    const int ou_x = threadIdx.x;
    const int ou_y = threadIdx.y;
    const int s = ou_y*ou_col_size;

    int row = ou_x * ou_row_size + ou_row;
    int j;

    //auto I_row_sum = G.new_zeros({batch_size, numX, numY, row_size, num_ou_Y}); 
    for (j=s; j<s+ou_col_size; ++j)
    {
        I_row_sum[batch_num][x][y][ou_x][ou_y][ou_row] += (v_up_prev[batch_num][x][y][row][j] - v_down_prev[batch_num][x][y][row][j])*access_G(nonlinear_cell, batch_num, x, row, y, j);
    }

    j = s;
    v_up[batch_num][x][y][row][j] = input[batch_num][x][row] - I_row_sum[batch_num][x][y][ou_x][ou_y][ou_row]*(r_wire*(j+1));
    scalar_t pass_current = (v_up_prev[batch_num][x][y][row][j] - v_down_prev[batch_num][x][y][row][j])*access_G(nonlinear_cell, batch_num, x, row, y, j);
    I_row_sum[batch_num][x][y][ou_x][ou_y][ou_row] -= pass_current;

    for (j=s+1; j<s+ou_col_size; ++j)
    {
        v_up[batch_num][x][y][row][j] = v_up[batch_num][x][y][row][j-1] - I_row_sum[batch_num][x][y][ou_x][ou_y][ou_row]*r_wire;
        pass_current = (v_up_prev[batch_num][x][y][row][j] - v_down_prev[batch_num][x][y][row][j])*access_G(nonlinear_cell, batch_num, x, row, y, j);
        I_row_sum[batch_num][x][y][ou_x][ou_y][ou_row] -= pass_current;
    }
}

template<typename scalar_t>
__global__ void ir2s_vdown_entire(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> G,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_up,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_down,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_up_prev,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_down_prev,
    torch::PackedTensorAccessor32<scalar_t, 6, torch::RestrictPtrTraits> I_col_sum,
    const int ou_row_size,
    const int ou_col_size,
    const scalar_t r_wire,
    bool nonlinear_cell
)
{
    const int batch_num = blockIdx.x/ou_col_size;
    const int x = blockIdx.y;
    const int y = blockIdx.z;
    const int ou_col = blockIdx.x%ou_col_size;
    const int ou_x = threadIdx.x;
    const int ou_y = threadIdx.y;
    const int row_size = blockDim.x*ou_row_size;
    const int s = ou_x*ou_row_size;

    int col = ou_y*ou_col_size + ou_col;
    int i;

    for (i=s; i<s+ou_row_size; ++i)
    {
        I_col_sum[batch_num][x][y][ou_x][ou_y][ou_col] += (v_up_prev[batch_num][x][y][i][col] - v_down_prev[batch_num][x][y][i][col])*access_G(nonlinear_cell, batch_num, x, i, y, col);
    }

    i = s+ou_row_size-1;
    v_down[batch_num][x][y][i][col] = I_col_sum[batch_num][x][y][ou_x][ou_y][ou_col]*(r_wire*(row_size-i));
    scalar_t pass_current = (v_up_prev[batch_num][x][y][i][col] - v_down_prev[batch_num][x][y][i][col])*access_G(nonlinear_cell, batch_num, x, i, y, col);
    I_col_sum[batch_num][x][y][ou_x][ou_y][ou_col] -= pass_current;

    for (i=s+ou_row_size-2; i>=s; --i)
    {
        v_down[batch_num][x][y][i][col] = v_down[batch_num][x][y][i+1][col] + I_col_sum[batch_num][x][y][ou_x][ou_y][ou_col]*r_wire;
        pass_current = (v_up_prev[batch_num][x][y][i][col] - v_down_prev[batch_num][x][y][i][col])*access_G(nonlinear_cell, batch_num, x, i, y, col);
        I_col_sum[batch_num][x][y][ou_x][ou_y][ou_col] -= pass_current;
    }
}

template<typename scalar_t>
__global__ void get_current_entire(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_up,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_down,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> G,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> current,
    const int ou_row_size,
    const int ou_col_size,
    bool nonlinear_cell
)
{
    const int batch_num = blockIdx.x/ou_col_size;
    const int x = blockIdx.y;
    const int y = blockIdx.z;
    const int ou_col = blockIdx.x%ou_col_size;
    const int ou_x = threadIdx.x;
    const int ou_y = threadIdx.y;

    const int s = ou_x*ou_row_size;
    int col = ou_y*ou_col_size + ou_col;

    //auto current = G.new_zeros({batch_size, numX, num_ou_X, numY, col_size});
    for (int i=s; i<s+ou_row_size; ++i)
    {
        current[batch_num][x][ou_x][y][col] += (v_up[batch_num][x][y][i][col]-v_down[batch_num][x][y][i][col])*access_G(nonlinear_cell, batch_num, x, i, y, col);
    }
}


/**
 * @brief 
 * 
 * @param input 
 * @param G 
 * @param batch_size 
 * @param row_size 
 * @param col_size 
 * @param ou_row_size 
 * @param ou_col_size 
 * @param numX 
 * @param numY 
 * @param iter_times 
 * @param r_wire 
 * @param enable_break 
 * @param break_threshould 
 * @return torch::Tensor 
 */
torch::Tensor ir2s_ou(
    torch::Tensor input,
    torch::Tensor G,
    int batch_size,
    int row_size,
    int col_size,
    int ou_row_size,
    int ou_col_size,
    int numX,
    int numY,
    int iter_times, 
    double r_wire,
    bool enable_break,
    double break_threshould,
    bool nonlinear_cell_enable,
    vector<double> nonlinear_params
)
{
    CHECK_INPUT(input);
    CHECK_INPUT(G);
    CHECK_NUM(row_size, 2);
    CHECK_NUM(col_size, 2);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    auto voltage_up = input.unsqueeze(2).unsqueeze(4).repeat({1, 1, numY, 1, col_size});
    auto voltage_down = G.new_zeros({batch_size, numX, numY, row_size, col_size});

    int num_ou_X = row_size/ou_row_size;
    int num_ou_Y = col_size/ou_col_size;

    auto voltage_up_prev = voltage_up.clone();
    auto voltage_down_prev = voltage_down.clone();
    auto I_row_sum = G.new_zeros({batch_size, numX, numY, num_ou_X, num_ou_Y, ou_row_size}); 
    auto I_col_sum = G.new_zeros({batch_size, numX, numY, num_ou_X, num_ou_Y, ou_col_size}); 
    const dim3 blocks_up(batch_size*ou_row_size, numX, numY);
    const dim3 blocks_down(batch_size*ou_col_size, numX, numY);
    const dim3 threads_up(num_ou_X, num_ou_Y);
    const dim3 threads_down(num_ou_X, num_ou_Y);
    const dim3 blocks(batch_size, numX, numY);

    G = G.view({1, numX, row_size, numY, col_size});
    torch::Tensor z;

    if (nonlinear_cell_enable)
    {
        z = get_nonlinear_ratio_z<double>(1/G, nonlinear_params[0], nonlinear_params[1]).transpose(2, 3).contiguous();
    }

    
    for (int i=0; i<iter_times; ++i)
    {
        if (nonlinear_cell_enable)
        {
            auto V = voltage_up - voltage_down;
            
            AT_DISPATCH_FLOATING_TYPES(V.type(), "deal_v", ([&]
            {
                deal_v<scalar_t><<<blocks, row_size>>>(
                    V.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                    1e-5,
                    1e-5,
                    col_size 
                );
            }));
            G = 1/nonlinear_R_tensor<double>(V, z, nonlinear_params[2], nonlinear_params[3], nonlinear_params[4], nonlinear_params[5]);
        }

        AT_DISPATCH_FLOATING_TYPES(G.type(), "ir2s_vup", ([&]
        {
            ir2s_vup_entire<scalar_t><<<blocks_up, threads_up>>>(
                input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(), 
                G.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_up.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_up_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                I_row_sum.packed_accessor32<scalar_t, 6, torch::RestrictPtrTraits>(), 
                ou_row_size,
                ou_col_size,
                r_wire,
                nonlinear_cell_enable);
        }));
        AT_DISPATCH_FLOATING_TYPES(G.type(), "ir2s_vdown", ([&]
        {
            ir2s_vdown_entire<scalar_t><<<blocks_down, threads_down>>>(
                input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(), 
                G.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_up.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_up_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                I_col_sum.packed_accessor32<scalar_t, 6, torch::RestrictPtrTraits>(), 
                ou_row_size,
                ou_col_size,
                r_wire,
                nonlinear_cell_enable);
        }));

        std::swap(voltage_up, voltage_up_prev);
        std::swap(voltage_down, voltage_down_prev);
        if (enable_break)
        {
            auto diff_up = ((voltage_up - voltage_up_prev).abs()>break_threshould).sum().item<int>();
            auto diff_down = ((voltage_down - voltage_down_prev).abs()>break_threshould).sum().item<int>();
            // std::cout << "iters " << i << ' ' << std::endl;
            // std::cout << diff_up << ' ' << diff_down << std::endl;
            if (diff_up == 0 && diff_down == 0)
            {
                // std::cout << "break in " << i << ' ' << std::endl;
                break;
            }
        }

    }

    auto current = G.new_zeros({batch_size, numX, num_ou_X, numY, col_size});
    AT_DISPATCH_FLOATING_TYPES(G.type(), "get_current", ([&]
    {
        get_current_entire<scalar_t><<<blocks_down, threads_down>>>(
                voltage_up_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                G.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                current.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                ou_row_size,
                ou_col_size,
                nonlinear_cell_enable
            );
    }));

    return current.view({batch_size, numX*num_ou_X, -1});

}

template<typename scalar_t>
__global__ void ir_gs_vup_ou_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> G,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_up,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_down,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_up_prev,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_down_prev,
    const int ou_row_size,
    const int ou_col_size,
    const scalar_t g_wire,
    const scalar_t beta,
    bool nonlinear_cell
)
{

    const int batch_num = blockIdx.x/ou_row_size;
    const int x = blockIdx.y;
    const int y = blockIdx.z;
    const int ou_row = blockIdx.x%ou_row_size;
    const int ou_x = threadIdx.x;
    const int ou_y = threadIdx.y;

    const int s = ou_y*ou_col_size;
    int row = ou_x*ou_row_size + ou_row;

    int j = s;

    scalar_t left, right, other_side, sumg;
    scalar_t g_load_left = g_wire/(j+1);

    left = input[batch_num][x][row];
    right = v_up_prev[batch_num][x][y][row][j+1];
    other_side = v_down_prev[batch_num][x][y][row][j];
    sumg = g_wire+g_load_left+access_G(nonlinear_cell, batch_num, x, row, y, j);

    v_up[batch_num][x][y][row][j] = beta * (g_load_left * left+ g_wire*right + access_G(nonlinear_cell, batch_num, x, row, y, j) * other_side)/sumg \
                                        + (1-beta) * v_up_prev[batch_num][x][y][row][j];
    for (j = s+1; j < s+ou_col_size-1; ++j)
    {
        left = v_up[batch_num][x][y][row][j-1];
        right = v_up_prev[batch_num][x][y][row][j+1];
        other_side = v_down_prev[batch_num][x][y][row][j];
        sumg = g_wire*2 + access_G(nonlinear_cell, batch_num, x, row, y, j);
        v_up[batch_num][x][y][row][j] = beta * (g_wire * (left+right) + access_G(nonlinear_cell, batch_num, x, row, y, j) * other_side)/sumg \
                                            + (1-beta) * v_up_prev[batch_num][x][y][row][j];
    }

    sumg = g_wire + access_G(nonlinear_cell, batch_num, x, row, y, j);
    left = v_up[batch_num][x][y][row][j-1];
    other_side = v_down_prev[batch_num][x][y][row][j];
    v_up[batch_num][x][y][row][j] = beta * (g_wire * left + access_G(nonlinear_cell, batch_num, x, row, y, j) * other_side)/sumg \
                                            + (1-beta) * v_up_prev[batch_num][x][y][row][j];
}

template<typename scalar_t>
__global__ void ir_gs_vdown_ou_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> G,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_up,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_down,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_up_prev,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> v_down_prev,
    const int ou_row_size,
    const int ou_col_size,
    const scalar_t g_wire,
    const scalar_t beta,
    bool nonlinear_cell
)
{
    const int batch_num = blockIdx.x/ou_col_size;
    const int x = blockIdx.y;
    const int y = blockIdx.z;
    const int ou_col = blockIdx.x%ou_col_size;
    const int ou_x = threadIdx.x;
    const int ou_y = threadIdx.y;
    const int row_size = blockDim.x*ou_row_size;
    const int s = ou_x*ou_row_size;

    int col = ou_y*ou_col_size + ou_col;
    int i = s+ou_row_size-1;


    scalar_t up, down, other_side, sumg;
    scalar_t g_load_down = g_wire/(row_size-i);

    up = v_down_prev[batch_num][x][y][i-1][col];
    down = 0;
    other_side = v_up[batch_num][x][y][i][col];
    sumg = g_wire+g_load_down + access_G(nonlinear_cell, batch_num, x, i, y, col);
    v_down[batch_num][x][y][i][col] = beta * (g_wire * up+ g_load_down*down + access_G(nonlinear_cell, batch_num, x, i, y, col) * other_side)/sumg \
                                    + (1-beta) * (v_down_prev[batch_num][x][y][i][col]);


    for (i = s+ou_row_size-2; i>s; --i)
    {
        up = v_down_prev[batch_num][x][y][i-1][col];
        down = v_down[batch_num][x][y][i+1][col];
        other_side = v_up[batch_num][x][y][i][col];
        sumg = g_wire*2 + access_G(nonlinear_cell, batch_num, x, i, y, col);
        v_down[batch_num][x][y][i][col] = beta * (g_wire * (up+down) + access_G(nonlinear_cell, batch_num, x, i, y, col) * other_side)/sumg \
                                    + (1-beta) * (v_down_prev[batch_num][x][y][i][col]);
    }

    down = v_down[batch_num][x][y][i+1][col];
    other_side = v_up[batch_num][x][y][i][col];
    sumg = g_wire + access_G(nonlinear_cell, batch_num, x, i, y, col);
    v_down[batch_num][x][y][i][col] = beta * (g_wire * down + access_G(nonlinear_cell, batch_num, x, i, y, col) * other_side)/sumg \
                                    + (1-beta) * (v_down_prev[batch_num][x][y][i][col]);
}

torch::Tensor ir_gs_ou(
    torch::Tensor input,
    torch::Tensor G,
    int batch_size,
    int row_size,
    int col_size,
    int ou_row_size,
    int ou_col_size,
    int numX,
    int numY,
    int iter_times, 
    double g_wire,
    double beta,
    bool enable_break,
    double break_threshould,
    bool nonlinear_cell_enable,
    vector<double> nonlinear_params
)
{
    CHECK_INPUT(input);
    CHECK_INPUT(G);
    CHECK_NUM(row_size, 2);
    CHECK_NUM(col_size, 2);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    // input is [batch_size, numX, row_size]
    // voltage is [batch_size, numX, numY, row_size, col_size]
    auto voltage_up = input.unsqueeze(2).unsqueeze(4).repeat({1, 1, numY, 1, col_size});
    auto voltage_down = G.new_zeros({batch_size, numX, numY, row_size, col_size});

    int num_ou_X = row_size/ou_row_size;
    int num_ou_Y = col_size/ou_col_size;

    auto voltage_up_prev = voltage_up.clone();
    auto voltage_down_prev = voltage_down.clone();
    const dim3 blocks_up(batch_size*ou_row_size, numX, numY);
    const dim3 blocks_down(batch_size*ou_col_size, numX, numY);
    const dim3 threads_up(num_ou_X, num_ou_Y);
    const dim3 threads_down(num_ou_X, num_ou_Y);
    const dim3 blocks(batch_size, numX, numY);

    G = G.view({1, numX, row_size, numY, col_size});
    torch::Tensor z;

    if (nonlinear_cell_enable)
    {
        z = get_nonlinear_ratio_z<double>(1/G, nonlinear_params[0], nonlinear_params[1]).transpose(2, 3).contiguous();
    }

    for (int i = 0; i<iter_times; ++i)
    {
        if (nonlinear_cell_enable)
        {
            auto V = voltage_up - voltage_down;
            
            AT_DISPATCH_FLOATING_TYPES(V.type(), "deal_v", ([&]
            {
                deal_v<scalar_t><<<blocks, row_size>>>(
                    V.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                    1e-5,
                    1e-5,
                    col_size 
                );
            }));
            G = 1/nonlinear_R_tensor<double>(V, z, nonlinear_params[2], nonlinear_params[3], nonlinear_params[4], nonlinear_params[5]);
        }

        AT_DISPATCH_FLOATING_TYPES(G.type(), "ir_drop_accurate_GS_kernel_vup", ([&]
        {
            ir_gs_vup_ou_kernel<scalar_t><<<blocks_up, threads_up>>>(
                input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(), 
                G.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_up.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_up_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                ou_row_size,
                ou_col_size,
                g_wire,
                beta, 
                nonlinear_cell_enable);
        }));
        AT_DISPATCH_FLOATING_TYPES(G.type(), "ir_drop_accurate_GS_kernel_vdown", ([&]
        {
            ir_gs_vdown_ou_kernel<scalar_t><<<blocks_down, threads_down>>>(
                input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(), 
                G.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_up.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_up_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                ou_row_size,
                ou_col_size,
                g_wire,
                beta,
                nonlinear_cell_enable);
        }));

        std::swap(voltage_up, voltage_up_prev);
        std::swap(voltage_down, voltage_down_prev);
        if (enable_break)
        {
            
            auto diff_up = ((voltage_up - voltage_up_prev).abs()>break_threshould).sum().item<int>();
            auto diff_down = ((voltage_down - voltage_down_prev).abs()>break_threshould).sum().item<int>();
            // std::cout << "iters " << i << ' ' << std::endl;
            // std::cout << diff_up << ' ' << diff_down << std::endl;
            if (diff_up == 0 && diff_down == 0)
            {
                // std::cout << "break in " << i << ' ' << std::endl;
                break;
            }
        }
    }

    auto current = G.new_zeros({batch_size, numX, num_ou_X, numY, col_size});
    AT_DISPATCH_FLOATING_TYPES(G.type(), "get_current", ([&]
    {
        get_current_entire<scalar_t><<<blocks_down, threads_down>>>(
                voltage_up_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                voltage_down_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                G.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                current.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                ou_row_size,
                ou_col_size,
                nonlinear_cell_enable
            );
    }));

    return current.view({batch_size, numX*num_ou_X, -1});
}

// torch::Tensor ir_gs_ou_memCMOS(
//     torch::Tensor input, // input is [batch_size, numX, numY, row_size]
//     torch::Tensor G,
//     int batch_size,
//     int row_size,
//     int col_size,
//     int ou_row_size,
//     int ou_col_size,
//     int numX,
//     int numY,
//     int iter_times, 
//     double g_wire,
//     double beta,
//     bool enable_break,
//     double break_threshould
// )
// {
//     CHECK_INPUT(input);
//     CHECK_INPUT(G);
//     CHECK_NUM(row_size, 2);
//     CHECK_NUM(col_size, 2);

//     // input is [batch_size, numX, numY, row_size]
//     // voltage is [batch_size, numX, numY, row_size, col_size]
//     auto voltage_up = input.unsqueeze(2).unsqueeze(4).repeat({1, 1, numY, 1, col_size});
//     auto voltage_down = G.new_zeros({batch_size, numX, numY, row_size, col_size});

//     int num_ou_X = row_size/ou_row_size;
//     int num_ou_Y = col_size/ou_col_size;

//     auto voltage_up_prev = voltage_up.clone();
//     auto voltage_down_prev = voltage_down.clone();
//     const dim3 blocks_up(batch_size*ou_row_size, numX, numY);
//     const dim3 blocks_down(batch_size*ou_col_size, numX, numY);
//     const dim3 threads_up(num_ou_X, num_ou_Y);
//     const dim3 threads_down(num_ou_X, num_ou_Y);

//     G = G.view({numX, row_size, numY, col_size});

//     for (int i = 0; i<iter_times; ++i)
//     {
//         AT_DISPATCH_FLOATING_TYPES(G.type(), "ir_drop_accurate_GS_kernel_vup", ([&]
//         {
//             ir_gs_vup_ou_kernel<scalar_t><<<blocks_up, threads_up>>>(
//                 input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(), 
//                 G.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), 
//                 voltage_up.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
//                 voltage_down.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
//                 voltage_up_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
//                 voltage_down_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
//                 ou_row_size,
//                 ou_col_size,
//                 g_wire,
//                 beta);
//         }));
//         AT_DISPATCH_FLOATING_TYPES(G.type(), "ir_drop_accurate_GS_kernel_vdown", ([&]
//         {
//             ir_gs_vdown_ou_kernel<scalar_t><<<blocks_down, threads_down>>>(
//                 input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(), 
//                 G.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), 
//                 voltage_up.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
//                 voltage_down.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
//                 voltage_up_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
//                 voltage_down_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
//                 ou_row_size,
//                 ou_col_size,
//                 g_wire,
//                 beta);
//         }));

//         std::swap(voltage_up, voltage_up_prev);
//         std::swap(voltage_down, voltage_down_prev);
//         if (enable_break)
//         {
//              auto diff_up = ((voltage_up - voltage_up_prev).abs()>break_threshould).sum().item<int>();
//              auto diff_down = ((voltage_down - voltage_down_prev).abs()>break_threshould).sum().item<int>();
//              if (diff_up == 0 && diff_down == 0)
//              {
//                 //  std::cout << "break in " << i << ' ' << std::endl;
//                  break;
//              }
//         }
//     }

//     auto current = G.new_zeros({batch_size, numX, num_ou_X, numY, col_size});
//     AT_DISPATCH_FLOATING_TYPES(G.type(), "get_current", ([&]
//     {
//         get_current_entire<scalar_t><<<blocks_down, threads_down>>>(
//                 voltage_up_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
//                 voltage_down_prev.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
//                 G.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), 
//                 current.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
//                 ou_row_size,
//                 ou_col_size
//             );
//     }));

//     return current.view({batch_size, numX*num_ou_X, -1});
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("ir_drop_gs", &ir_drop_gs, "ir drop solve with GS Method");
    m.def("ir2s", &ir2s, "ir drop ir2s solve");
    m.def("ir2s_ou_local", &ir2s_ou_local, "ir drop ir2s solve single ou size");
    m.def("ir2s_ou", &ir2s_ou, "ir drop ir2s solve the entire crx with ou size");
    m.def("ir_gs_ou_local", &ir_gs_ou_local, "ir drop gs method solve single ou size");
    m.def("ir_gs_ou", &ir_gs_ou, "ir drop gs method solve entire crx with ou size");
}
