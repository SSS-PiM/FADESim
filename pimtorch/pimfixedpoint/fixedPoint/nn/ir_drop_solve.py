from ctypes import sizeof
import torch
from torch import Tensor
import math
from .commonConst import ir_drop_mode, phyArrParams, GS_method, ir_drop_compensate_scheme, phyArrMode

from .commonConst import debug

#inv [bsize, numX, phyArrParams.arrRowSize]
#x [numX, phyArrParams.arrRowSize, true_outsize]
#return [numX, bsize, true_outsize]
def no_ir_drop_cal(inv: Tensor, x: Tensor) -> Tensor:
    inv = torch.transpose(inv, 0, 1)
    #now inv is [numX, bsize, rowsize]
    return torch.matmul(inv, x)

#inv [bsize, numX, arrRowSize]  
# x is [numX, arrRowSize, true_outsize]
# ou_row_num = arrRowSize//ou_row_size
#return is [numX, bsize, ou_row_num, true_outsize]
def no_ir_drop_cal_ou(inv: Tensor, x: Tensor, ou_size: tuple, bsize: int, true_outsize: int, numX: int) -> Tensor:

    #inv changes to [numX*ou_row_num, bsize, ou_row_size]
    inv = inv.view(bsize, -1, ou_size[0]).transpose(0, 1)
    x = x.reshape(numX, -1, ou_size[0], true_outsize)
    x = x.reshape(-1, ou_size[1], true_outsize)

    #[numX*ou_row_nunum, bsize, ou_row_size] * [numX*ou_row_num, ou_row_size, true_outsize]
    #return is [numX*ou_row_num, bsize, true_outsize]
    return torch.matmul(inv, x)

#inv [bsize, numX, phyArrParams.arrRowSize]
#x [numX, phyArrParams.arrRowSize, true_outsize]
#return [bsize, numX, true_outsize]
def ir_drop_ir2s_fastsolve(inv: Tensor, x: Tensor, bsize: int, numX: int, numY: int, arrColSize: int = phyArrParams.arrColSize) -> Tensor:
    #inv = torch.transpose(inv, 0, 1) # inv is now [numX, bsize, rowsize]
    upv = inv.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, numY, arrColSize)
    downv = x.new_zeros((bsize, numX, phyArrParams.arrRowSize, numY, arrColSize))
    x = x.reshape(1, numX, phyArrParams.arrRowSize, numY, arrColSize)

    for iter in range(phyArrParams.ir_drop_iter_time):
        if phyArrParams.ir_drop_less_than_thresholdBreak_enable:
            upv_last = upv.clone()
            downv_last = downv.clone()

        current = upv.sub(downv).mul(x) # current is [bsize, numx, rowsize, numy, colsize]
        i_bitline_sum = current.sum(2)
        i_wordline_sum = current.sum(4)

        upv[..., 0] = inv.unsqueeze(-1) - i_wordline_sum*phyArrParams.r_wire
        i_wordline_sum.sub_(current[..., 0])

        for col in range(1, arrColSize):
            upv[..., col] = upv[..., col-1] - i_wordline_sum*phyArrParams.r_wire
            i_wordline_sum.sub_(current[..., col])
        
        downv[:, :, phyArrParams.arrRowSize-1, ...] = i_bitline_sum*phyArrParams.r_wire
        i_bitline_sum.sub_(current[:, :, phyArrParams.arrRowSize-1, ...])

        for row in range(phyArrParams.arrRowSize-2, -1, -1):
            downv[:, :, row, ...] = downv[:, :, row+1, ...] + i_bitline_sum*phyArrParams.r_wire
            i_bitline_sum.sub_(current[:, :, row, ...])
        
        if phyArrParams.ir_drop_less_than_thresholdBreak_enable:
            diff_up = ((upv - upv_last).abs()>phyArrParams.ir_drop_less_than_threshold).sum().item()
            diff_down = ((downv - downv_last).abs()>phyArrParams.ir_drop_less_than_threshold).sum().item()
            # print("iter = {}, larger than threshould number = {}".format(iter, diff_up+diff_down))
            if diff_up==0 and diff_down==0:
                print(iter)
                break
    
    current = upv.sub(downv).mul(x)
    return current.sum(2).reshape(bsize, numX, -1) # [bsize, numx, true_outsize]

class ir_drop_accurate_math_model:
    crx_size = phyArrParams.crx_size
    row_size = phyArrParams.arrRowSize
    col_size = phyArrParams.arrColSize
    r_wire = phyArrParams.r_wire
    nodes_list = []
    values_list = []

    # build wordline & bitline node
    for i in range(crx_size*2):
        nodes_list.append((i, i))
        values_list.append(-2.0/r_wire) # remains to sub cell G
    
    for i in range(crx_size*2, crx_size*2+row_size+col_size):
        nodes_list.append((i, i))
        values_list.append(1.0)
    
    for i in range(crx_size*2+row_size+col_size, (crx_size+row_size+col_size)*2):
        nodes_list.append((i, i))
        values_list.append(-1.0/r_wire)
    
    # start = 2*crx_size + 2*rowsize +2 *colsize
    # [start, start+2*crx_size) should set to g_ij
    for i in range(crx_size):
        nodes_list.append((i, i+crx_size))
        values_list.append(0)  # currently, i don't know the Gij, so set to 0
    
    for i in range(crx_size):
        nodes_list.append((i+crx_size, i))
        values_list.append(0)  # currently, i don't know the Gij, so set to 0
    
    # build wordline wire conductance
    for i in range(row_size):
        for j in range(col_size-1):
            num = i*col_size + j
            nodes_list.append((num, num+1))
            values_list.append(1.0/r_wire)
    
            nodes_list.append((num+1, num))
            values_list.append(1.0/r_wire)

        num = (i+1)*col_size -1
        right = 2*crx_size + row_size + col_size + i
        nodes_list.append((num, right)) 
        nodes_list.append((right, num)) 

        values_list.append(1.0/r_wire)
        values_list.append(1.0/r_wire)

    # build bitline wire 
    for i in range(row_size-1):
        for j in range(col_size):
            num = i*col_size + j + crx_size
            nodes_list.append((num, num+col_size))
            values_list.append(1.0/r_wire)
    
            nodes_list.append((num+col_size, num))
            values_list.append(1.0/r_wire)
    for j in range(col_size):
        num = j+crx_size
        up = crx_size*2+row_size*2+col_size+j
        nodes_list.append((num, up)) 
        nodes_list.append((up, num)) 

        values_list.append(1.0/r_wire)
        values_list.append(1.0/r_wire)

    # build wordline left connected to wordline
    for i in range(row_size):
        num = i*col_size
        left = crx_size*2+i
        nodes_list.append((num, left)) 
        values_list.append(1.0/r_wire)
    
    #build bitline down connected to bitline
    for j in range(col_size):
        num = crx_size*2-col_size+j
        down = crx_size*2 + row_size + j
        nodes_list.append((num, down))
        values_list.append(1.0/r_wire)
    
    nodes_tensor = torch.tensor(nodes_list).transpose(0, 1)
    values_tensor = torch.tensor(values_list)

#inv [bsize, numX, phyArrParams.arrRowSize]
#x [numX, phyArrParams.arrRowSize, true_outsize]
#return [bsize, numX, true_outsize]
def ir_drop_accurate_math(inv: Tensor, x: Tensor, bsize: int, numX: int, numY: int) -> Tensor:

    x = x.reshape(numX, phyArrParams.arrRowSize, numY, phyArrParams.arrColSize)
    node = ir_drop_accurate_math_model.nodes_tensor.to(inv.device)
    value = ir_drop_accurate_math_model.values_tensor.to(inv.device)

    output = x.new_empty((bsize, numX, numY, phyArrParams.arrColSize))
    for i in range(numX):
        for j in range(numY):
            temp_node = node.clone()
            temp_value = value.clone()
            g_mat_flat = x[i, :, j, :].flatten()
            g_mat_flat2 = torch.cat((g_mat_flat, g_mat_flat))

            number_cnt = 2*(phyArrParams.crx_size+phyArrParams.arrRowSize+phyArrParams.arrColSize)
            temp_value[0:2*phyArrParams.crx_size] -= g_mat_flat2 
            temp_value[number_cnt:(number_cnt+2*phyArrParams.crx_size)] = g_mat_flat2

            B = inv.new_zeros(bsize, number_cnt)
            B[:, 2*phyArrParams.crx_size:2*phyArrParams.crx_size+phyArrParams.arrRowSize] = inv[:, i, :]
            A = torch.sparse_coo_tensor(temp_node, temp_value, [number_cnt, number_cnt], device=inv.device)
            ans = torch.linalg.solve(A.to_dense(), B.t()).t() # ans is [bsize, number_cnt]
            output[:, i, j, :] = ans[:, phyArrParams.crx_size*2-phyArrParams.arrColSize:phyArrParams.crx_size*2].div(phyArrParams.r_wire)   
    
    return output.reshape(bsize, numX, -1)

#inv [bsize, numX, phyArrParams.arrRowSize]
#x [numX, phyArrParams.arrRowSize, true_outsize]
# return [bsize, numx, true_outsize]
def ir_drop_accurate_GSMethod(inv: Tensor, x: Tensor, bsize: int, numX: int, numY: int, arrColSize: int = phyArrParams.arrColSize) -> Tensor:
    beta_scale = phyArrParams.ir_drop_GS_method_beta_scale
    last_vup = inv.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, numY, arrColSize)
    last_vdown = x.new_zeros((bsize, numX, phyArrParams.arrRowSize, numY, arrColSize))
    now_vup = last_vup.clone()
    now_vdown = last_vdown.clone()
    
    x = x.reshape(numX, phyArrParams.arrRowSize, numY, arrColSize)
    g_wire = 1.0/phyArrParams.r_wire

    if phyArrParams.ir_drop_GS_method_mode==GS_method.nodeBynode:
        for _ in range(phyArrParams.ir_drop_iter_time):
            for i in range(phyArrParams.arrRowSize-1, -1, -1):
                for j in range(arrColSize):
                    left = inv[..., i].unsqueeze(-1).repeat(1, 1, numY) if j==0 else now_vup[:, :, i, :, j-1] # left is [bsize, numx, numY]
                    last_vup_ij = last_vup[:, :, i, :, j]
                    last_vdown_ij = last_vdown[:, :, i, :, j]

                    other_point = last_vdown_ij # is [bsize, numX, numY]
                    x_ij = x[:, i, :, j]
                    x_o = x_ij.mul(other_point)


                    last_vup_ij_x1b = last_vup_ij.mul(1.0-beta_scale)
                    last_vdown_ij_x1b = last_vdown_ij.mul(1.0-beta_scale)    

                    if j==arrColSize-1:
                        sumg = x_ij.add(g_wire)  # sumg is [numX, numY]
                        now_vup[:, :, i, :, j] = (left.mul(g_wire) + x_o).mul(beta_scale).div(sumg) + last_vup_ij_x1b
                    else:
                        right = last_vup[:, :, i, :, j+1] 
                        sumg = x_ij.add(2*g_wire)  # sumg is [numX, numY]
                        now_vup[:, :, i, :, j] = ((left+right).mul(g_wire) + x_o).mul(beta_scale).div(sumg) + last_vup_ij_x1b

                    right = inv.new_zeros((bsize, numX, numY)) if i==phyArrParams.arrRowSize-1 else now_vdown[:, :, i+1, :, j]
                    other_point = now_vup[:, :, i, :, j] ## 必须要注意nodeGSMethod的计算方式中，不是每次都是取上一次的计算值，是有一种特别的顺序要求
                                                        #例如这个位置就是取得now_vup

                    x_o = x_ij.mul(other_point)
                    if i==0:
                        sumg = x_ij.add(g_wire)  # sumg is [numX, numY]
                        now_vdown[:, :, i, :, j] = (right.mul(g_wire) + x_o).mul(beta_scale).div(sumg) + last_vdown_ij_x1b
                    else:
                        left = last_vdown[:, :, i-1, :, j]
                        sumg = x_ij.add(2*g_wire)  # sumg is [numX, numY]
                        now_vdown[:, :, i, :, j] = ((left+right).mul(g_wire) + x_o).mul(beta_scale).div(sumg) + last_vdown_ij_x1b
            last_vup, now_vup = now_vup, last_vup
            last_vdown, now_vdown = now_vdown, last_vdown
            
            if phyArrParams.ir_drop_less_than_thresholdBreak_enable:
                diff_up = ((now_vup - last_vup).abs()>phyArrParams.ir_drop_less_than_threshold).sum().item()
                diff_down = ((now_vdown - last_vdown).abs()>phyArrParams.ir_drop_less_than_threshold).sum().item()
                if diff_up==0 and diff_down==0:
                    break

            
        # 由于每次迭代进行了交换，迭代完后 last_vup and last_vodwn is the newest.
        current = (last_vup-last_vdown).mul(x) # [bsize, numx, rowsize, numy, colsize] * [numx, rowsize, numy, colsize]
        return current.sum(2).reshape(bsize, numX, -1) # [bsize, numx, true_outsize]
    else:  # batch by batch mode
        for iter in range(phyArrParams.ir_drop_iter_time):

            left = inv.unsqueeze(-1).repeat(1, 1, 1, numY) # left is [bsize, numx, rowSize, numY]
            
            for j in range(arrColSize-1):
                right = last_vup[..., j+1]  # right is [bsize, numx, rowsize, numy]
                sumg = x[..., j].add(g_wire*2) #sumg is [numx, rowsize, numy]
                now_vup[..., j] = ((left+right).mul(g_wire) + x[..., j]*last_vdown[..., j]).mul(beta_scale).div(sumg) + last_vup[..., j].mul(1.0-beta_scale)
                left = now_vup[..., j]
            sumg = x[..., -1].add(g_wire)  # sumg is [numX, rosSize, numY]
            now_vup[..., -1] = (left.mul(g_wire) + x[..., -1]*last_vdown[..., -1]).mul(beta_scale).div(sumg) + last_vup[..., -1].mul(1.0-beta_scale)

            down = inv.new_zeros((bsize, numX, numY, arrColSize)) # if i==phyArrParams.arrRowSize-1 else now_vdown[:, :, i+1, :, j]

            for i in range(phyArrParams.arrRowSize-1, 0, -1):
                up = last_vdown[:, :, i-1, ...] # [bsize, numx, numy, colSize]
                sumg = x[:, i, ...].add(g_wire*2) #[numx, numy, colsize]
                now_vdown[:, :, i, ...] = ((up+down).mul(g_wire) + x[:, i, ...]*now_vup[:, :, i, ...]).mul(beta_scale).div(sumg) + last_vdown[:, :, i, ...].mul(1.0-beta_scale)
                down = now_vdown[:, :, i, ...]
            sumg = x[:, 0, ...].add(g_wire) #[numx, numy, colsize]
            now_vdown[:, :, 0, ...] = (down.mul(g_wire) + x[:, 0, ...]*now_vup[:, :, 0, ...]).mul(beta_scale).div(sumg) + last_vdown[:, :, 0, ...].mul(1.0-beta_scale)

            last_vup, now_vup = now_vup, last_vup
            last_vdown, now_vdown = now_vdown, last_vdown

            if phyArrParams.ir_drop_less_than_thresholdBreak_enable:
                diff_up = ((now_vup - last_vup).abs()>phyArrParams.ir_drop_less_than_threshold).sum().item()
                diff_down = ((now_vdown - last_vdown).abs()>phyArrParams.ir_drop_less_than_threshold).sum().item()
                # print("iter = {}, larger than threshould number = {}".format(iter, diff_up+diff_down))
                ######### debugging info ################
                if debug.detailed:
                    print("iter = {}, larger than threshould number = {}".format(iter, diff_up + diff_down))
                #########################################
                if diff_up == 0 and diff_down == 0:
                    ######### debugging info ################
                    if debug.simple:
                        print(iter)
                    #########################################
                    break

        current = (last_vup-last_vdown).mul(x) # [bsize, numx, rowsize, numy, colsize] * [numx, rowsize, numy, colsize]
        return current.sum(2).reshape(bsize, numX, -1) # [bsize, numx, true_outsize]


# inv [bsize, numX, OURowSize]
# x [numX, OURowSize, numY*OUColSize]
# return [bsize, numX, numY*OUColSize]
def ir_drop_accurate_GSMethod_OU(inv: Tensor, x: Tensor, bsize: int, numX: int, numY: int,
                                 OURowSize: int = phyArrParams.OUSize[0], OUColSize: int = phyArrParams.OUSize[1],
                                 r: int = None, c: int = None) -> Tensor:
    # 注意: 此时阵列大小为OU, 并且需要考虑g_load
    beta_scale = phyArrParams.ir_drop_GS_method_beta_scale
    last_vup = inv.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, numY, OUColSize)
    last_vdown = x.new_zeros((bsize, numX, OURowSize, numY, OUColSize))
    now_vup = last_vup.clone()
    now_vdown = last_vdown.clone()

    x = x.reshape(numX, OURowSize, numY, OUColSize)
    g_wire = 1.0 / phyArrParams.r_wire
    g_load_left_wire = 1.0 / ((OUColSize * c + 1) * phyArrParams.r_wire + phyArrParams.r_load) # load conductance left from input to OU
    g_load_down_wire = 1.0 / ((phyArrParams.arrRowSize - (r + 1) * OURowSize + 1) * phyArrParams.r_wire + phyArrParams.r_load) # load conductance from OU down to output
    # both parameters contain wire conductance at the left/bottom side of OU

    # node by node is not used for OU
    if phyArrParams.ir_drop_GS_method_mode == GS_method.nodeBynode:
        for _ in range(phyArrParams.ir_drop_iter_time):
            for i in range(OURowSize - 1, -1, -1):
                for j in range(OUColSize):
                    left = inv[..., i].unsqueeze(-1).repeat(1, 1, numY) if j == 0 else now_vup[:, :, i, :,
                                                                                       j - 1]  # left is [bsize, numx, numY]
                    last_vup_ij = last_vup[:, :, i, :, j]
                    last_vdown_ij = last_vdown[:, :, i, :, j]

                    other_point = last_vdown_ij  # is [bsize, numX, numY]
                    x_ij = x[:, i, :, j]
                    x_o = x_ij.mul(other_point)

                    last_vup_ij_x1b = last_vup_ij.mul(1.0 - beta_scale)
                    last_vdown_ij_x1b = last_vdown_ij.mul(1.0 - beta_scale)

                    if j == OUColSize - 1:
                        sumg = x_ij.add(g_wire)  # sumg is [numX, numY]
                        now_vup[:, :, i, :, j] = (left.mul(g_wire) + x_o).mul(beta_scale).div(sumg) + last_vup_ij_x1b
                    else:
                        right = last_vup[:, :, i, :, j + 1]
                        sumg = x_ij.add(2 * g_wire)  # sumg is [numX, numY]
                        now_vup[:, :, i, :, j] = ((left + right).mul(g_wire) + x_o).mul(beta_scale).div(
                            sumg) + last_vup_ij_x1b

                    right = inv.new_zeros((bsize, numX, numY)) if i == OURowSize - 1 else now_vdown[:, :,
                                                                                          i + 1, :, j]
                    other_point = now_vup[:, :, i, :, j]  ## 必须要注意nodeGSMethod的计算方式中，不是每次都是取上一次的计算值，是有一种特别的顺序要求
                    # 例如这个位置就是取得now_vup

                    x_o = x_ij.mul(other_point)
                    if i == 0:
                        sumg = x_ij.add(g_wire)  # sumg is [numX, numY]
                        now_vdown[:, :, i, :, j] = (right.mul(g_wire) + x_o).mul(beta_scale).div(
                            sumg) + last_vdown_ij_x1b
                    else:
                        left = last_vdown[:, :, i - 1, :, j]
                        sumg = x_ij.add(2 * g_wire)  # sumg is [numX, numY]
                        now_vdown[:, :, i, :, j] = ((left + right).mul(g_wire) + x_o).mul(beta_scale).div(
                            sumg) + last_vdown_ij_x1b
            last_vup, now_vup = now_vup, last_vup
            last_vdown, now_vdown = now_vdown, last_vdown

            if phyArrParams.ir_drop_less_than_thresholdBreak_enable:
                diff_up = ((now_vup - last_vup).abs() > phyArrParams.ir_drop_less_than_threshold).sum().item()
                diff_down = ((now_vdown - last_vdown).abs() > phyArrParams.ir_drop_less_than_threshold).sum().item()
                if diff_up == 0 and diff_down == 0:
                    break

        # 由于每次迭代进行了交换，迭代完后 last_vup and last_vodwn is the newest.
        current = (last_vup - last_vdown).mul(x)  # [bsize, numx, rowsize, numy, colsize] * [numx, rowsize, numy, colsize]
        return current.sum(2).reshape(bsize, numX, -1)  # [bsize, numx, true_outsize]
    else:  # batch by batch mode
        for iter in range(phyArrParams.ir_drop_iter_time):

            left = inv.unsqueeze(-1).repeat(1, 1, 1, numY)  # left is [bsize, numX, OURowSize, numY]

            for j in range(OUColSize - 1):
                right = last_vup[..., j + 1]  # right is [bsize, numx, OURowSize, numy]
                if j == 0:
                    sumg = x[..., j].add(g_load_left_wire + g_wire)  # sumg is [numx, OURowSize, numy]
                    now_vup[..., j] = (left.mul(g_load_left_wire) + right.mul(g_wire) + x[..., j] * last_vdown[..., j]).mul(beta_scale).div(
                        sumg) + last_vup[..., j].mul(1.0 - beta_scale)
                else:
                    sumg = x[..., j].add(g_wire * 2)  # sumg is [numx, OURowSize, numy]
                    now_vup[..., j] = ((left + right).mul(g_wire) + x[..., j] * last_vdown[..., j]).mul(beta_scale).div(
                        sumg) + last_vup[..., j].mul(1.0 - beta_scale)
                left = now_vup[..., j]
            sumg = x[..., -1].add(g_wire)  # sumg is [numX, OURowSize, numY]
            now_vup[..., -1] = (left.mul(g_wire) + x[..., -1] * last_vdown[..., -1]).mul(beta_scale).div(sumg) + \
                               last_vup[..., -1].mul(1.0 - beta_scale)

            down = inv.new_zeros((bsize, numX, numY, OUColSize))  # if i==OURowSize-1 else now_vdown[:, :, i+1, :, j]

            for i in range(OURowSize - 1, 0, -1):
                up = last_vdown[:, :, i - 1, ...]  # [bsize, numx, numy, OUColSize]
                if i == OURowSize - 1:
                    sumg = x[:, i, ...].add(g_load_down_wire + g_wire)  # [numx, numy, OUColSize]
                    now_vdown[:, :, i, ...] = (up.mul(g_wire) + down.mul(g_load_down_wire) + x[:, i, ...] * now_vup[:, :, i, ...]).mul(
                        beta_scale).div(sumg) + last_vdown[:, :, i, ...].mul(1.0 - beta_scale)
                else:
                    sumg = x[:, i, ...].add(g_wire * 2)  # [numx, numy, OUColSize]
                    now_vdown[:, :, i, ...] = ((up + down).mul(g_wire) + x[:, i, ...] * now_vup[:, :, i, ...]).mul(
                        beta_scale).div(sumg) + last_vdown[:, :, i, ...].mul(1.0 - beta_scale)
                down = now_vdown[:, :, i, ...]
            sumg = x[:, 0, ...].add(g_wire)  # [numx, numy, OUColSize]
            now_vdown[:, :, 0, ...] = (down.mul(g_wire) + x[:, 0, ...] * now_vup[:, :, 0, ...]).mul(beta_scale).div(
                sumg) + last_vdown[:, :, 0, ...].mul(1.0 - beta_scale)

            last_vup, now_vup = now_vup, last_vup
            last_vdown, now_vdown = now_vdown, last_vdown

            if phyArrParams.ir_drop_less_than_thresholdBreak_enable:
                diff_up = ((now_vup - last_vup).abs() > phyArrParams.ir_drop_less_than_threshold).sum().item()
                diff_down = ((now_vdown - last_vdown).abs() > phyArrParams.ir_drop_less_than_threshold).sum().item()
                ######### debugging info ################
                if debug.detailed:
                    print("iter = {}, larger than threshould number = {}".format(iter, diff_up+diff_down))
                #########################################
                if diff_up == 0 and diff_down == 0:
                    ######### debugging info ################
                    if debug.simple:
                        print(iter)
                    #########################################
                    break

        current = (last_vup - last_vdown).mul(x)  # [bsize, numx, OURowSize, numy, OUColSize] * [numx, OURowSize, numy, OUColSize]
        return current.sum(2).reshape(bsize, numX, -1)  # [bsize, numx, numY*OUColSize]]

# inv [batch_size, numX, row_size]
# x [numX, arrRowSize, true_outsize]
# return output {batch_size, numX*num_ou_X, -1}
def ir_drop_OU(inv: Tensor, x: Tensor, bsize: int, numX: int, numY: int, arrType) -> Tensor:
    arrOUnum = phyArrParams.arrOUnum[1] if arrType == phyArrMode.PN else phyArrParams.arrOUnum[1] + 1
    current = inv.new_zeros((bsize, numX, phyArrParams.arrOUnum[0], numY, arrOUnum, phyArrParams.OUSize[1]))

    inv = inv.reshape(bsize, numX, phyArrParams.arrOUnum[0], phyArrParams.OUSize[0]) # [bsize, numX, #OURow, OURowSize]

    x = x.reshape(numX, phyArrParams.arrOUnum[0], phyArrParams.OUSize[0], numY, arrOUnum, phyArrParams.OUSize[1])
    x = torch.transpose(x, 3, 4)
    x = x.reshape(numX, phyArrParams.arrOUnum[0], phyArrParams.OUSize[0],
                  arrOUnum, numY * phyArrParams.OUSize[1]) # [numX, #OURow, OURowSize, #OUCol, numY*OUColSize]

    for i in range(phyArrParams.arrOUnum[0]):
        for j in range(arrOUnum):
            ######### debugging info ################
            if debug.detailed or debug.simple:
                print("i={0}, total={1}; j={2}, total={3}".format(i+1, phyArrParams.arrOUnum[0], j+1, arrOUnum))
            #########################################
            output_currents = ir_drop_accurate_GSMethod_OU(inv[:, :, i, :].squeeze(2), x[:, i, :, j, :].squeeze(1).squeeze(2), bsize, numX, numY, r=i, c=j)
            # output_currents = inv.new_empty((bsize, numX, numY*phyArrParams.OUColSize))
            output_currents = output_currents.reshape(bsize, numX, numY, phyArrParams.OUSize[1])

            current[:, :, i, :, j, :] = output_currents
    # return current, IR_drop_record
    return current.reshape(bsize, numX*phyArrParams.arrOUnum[0], -1)

# inv [bsize, numX, #OURow, numY, #OUCol, OURowSize]
# x [numX, arrRowSize, true_outsize]
# return output {batch_size, numX, #OURow, numY, #OUCol, OUColSize}
def ir_drop_OU_memCMOS(inv: Tensor, x: Tensor, bsize: int, numX: int, numY: int, arrType) -> Tensor:
    # TODO: debugging ir_drop_OU_memCMOS
    arrOUnum = phyArrParams.arrOUnum[1] if arrType == phyArrMode.PN else phyArrParams.arrOUnum[1] + 1
    current = inv.new_zeros((bsize, numX, phyArrParams.arrOUnum[0], numY, arrOUnum, phyArrParams.OUSize[1]))

    x = x.reshape(numX, phyArrParams.arrOUnum[0], phyArrParams.OUSize[0], numY, arrOUnum, phyArrParams.OUSize[1])
    # x [numX, #OURow, OURowSize, numY, #OUCol, OUColSize]

    for i in range(phyArrParams.arrOUnum[0]):
        for j in range(arrOUnum):
            ######### debugging info ################
            if debug.detailed or debug.simple:
                print("i={0}, total={1}; j={2}, total={3}".format(i + 1, phyArrParams.arrOUnum[0], j + 1, arrOUnum))
            #########################################
            for Y in range(numY):
                output_currents = ir_drop_accurate_GSMethod_OU(inv[:, :, i, Y, :].squeeze(2).squeeze(2),
                                                               x[:, i, :, Y, j, :].squeeze(1).squeeze(2).squeeze(2), bsize, numX, 1,
                                                               r=i, c=j)

                current[:, :, i, Y, j, :] = output_currents
    # return current, IR_drop_record
    return current

# inv [bsize, numX, phyArrParams.arrRowSize]
# G [numX, phyArrParams.arrRowSize, true_outsize]
# return [bsize, numX, true_outsize]
# Efficient evaluation model including interconnect resistance effect for large scale RRAM crossbar array matrix computing
# sci china 
def ir_drop_em_fastsolve(inv : Tensor, G : Tensor, bsize: int, numX: int, numY: int, arrColSize: int = phyArrParams.arrColSize) -> Tensor: 
    arrRowSize = phyArrParams.arrRowSize
    r_wire = phyArrParams.r_wire
    r_load = phyArrParams.r_load
    G = G.reshape(numX, arrRowSize, numY, arrColSize).transpose(1, 2) # g is [numX, numY, rowsize, colsize]
    R = 1/G
    g_sum_for_bl = G.sum(2).reshape(numX, numY, 1, arrColSize)  #  [numX, numY, 1, arrColSize]

    add = G.new_tensor(range(arrRowSize, 0, -1)).view(arrRowSize, -1).repeat(1, arrColSize).repeat(numX, numY, 1, 1)*r_wire
    RR = R + add + (g_sum_for_bl*R)*r_load

    RRR = RR.new_empty(RR.shape) #[numX, numY, rowSize, colSize]
    for j in range(arrColSize-1, -1, -1):
        if j==arrColSize-1:
            RRR[..., j] = RR[..., j]+r_wire
        else:
            RRR[..., j] = ((RR[..., j]*RRR[..., j+1])/(RR[..., j]+RRR[..., j+1])) + r_wire
        
    V_wl = RRR.new_empty(bsize, numX, numY, arrRowSize, arrColSize)
    
    for j in range(0, arrColSize, 1):
        if j==0:
            # [numx, numy, rowsize] * [bsize, numx, numy, rowsize]
            V_wl[..., j] = ((RRR[..., j]-r_wire)/RRR[..., j]).reshape(1, numX, numY, arrRowSize)*inv.repeat(1, 1, numY).reshape(bsize, numX, numY, arrRowSize)
        else:
            V_wl[..., j] = ((RRR[..., j]-r_wire)/RRR[..., j]).reshape(1, numX, numY, arrRowSize)*V_wl[..., j-1]
    
    def parallel(x : Tensor, y : Tensor) -> Tensor:
        return x*y/(x+y)
    
    Reqv_up = RRR.new_empty(RRR.shape)
    
    for i in range(0, arrRowSize, 1):
        if i==0:
            Reqv_up[..., i, :] = R[..., i, :] + r_wire
        else:
            Reqv_up[..., i, :] = parallel(Reqv_up[..., i-1, :], R[..., i, :]) + r_wire
    
    Reqv_down = RRR.new_empty(RRR.shape)
    f = Reqv_down.clone()

    for i in range(arrRowSize-1, -1, -1):
        if i==arrRowSize-1:
            Reqv_down[..., i, :] = r_load
            f[..., i, :] = 1
        else:
            temp = parallel(Reqv_down[..., i+1, :], R[..., i+1, :])
            Reqv_down[..., i, :] = r_wire + temp
            f[..., i, :] = f[..., i+1, :] * temp/Reqv_down[..., i, :]
    
    Reqv = Reqv_down.new_empty(Reqv_down.shape)
    for i in range(0, arrRowSize, 1):
        if i==0:
            Reqv[..., i, :] = Reqv_down[..., i, :]
        else:
            Reqv[..., i, :] = parallel(Reqv_up[..., i-1, :], Reqv_down[..., i, :])
    
    # V_wl [bsize, numx, numy, rowsize, colsize]
    # Reqv [numx, numy, rowsize, colsize]
    
    out = V_wl * (Reqv/(Reqv+R)*f).reshape(1, numX, numY, arrRowSize, arrColSize)
    out = out.sum(3)/r_load
    return out.reshape(bsize, numX, -1)
# inv [bsize, numX, numY, arrRowSize]
# x [numX, arrRowSize, true_outsize]
# return output {batch_size, numX, numY, arrColSize}
def ir_drop_memCMOS(inv: Tensor, x: Tensor, bsize: int, numX: int, numY: int) -> Tensor:
    # TODO: debugging ir_drop_OU_memCMOS
    current = inv.new_zeros((bsize, numX, numY, phyArrParams.arrColSize))

    x = x.reshape(numX, phyArrParams.arrRowSize, numY, phyArrParams.arrColSize)
    # x [numX, #OURow, OURowSize, numY, #OUCol, OUColSize]

    for Y in range(numY):
        # inv [bsize, numX, phyArrParams.arrRowSize]
        # x [numX, phyArrParams.arrRowSize, true_outsize]
        # return [bsize, numx, true_outsize]
        output_currents = ir_drop_accurate_GSMethod(inv[:, :, Y, :].squeeze(2), x[:, :, Y, :].squeeze(2), bsize, numX, 1)

        current[:, :, Y, :] = output_currents
    return current



    
def ir_drop_ahk_fastsolve():
    pass

# inv [bsize, numX, phyArrParams.arrRowSize]
# G [numX, phyArrParams.arrRowSize, true_outsize]
# return [bsize, numX, true_outsize]   
# not support ref col mode now.
def ir_drop_scn_model(inv : Tensor, G : Tensor, bsize: int, numX: int, numY: int, arrColSize = phyArrParams.arrColSize):
    with torch.no_grad():
        if arrColSize==phyArrParams.arrColSize:
            G = G.reshape(numX, phyArrParams.arrRowSize, numY, arrColSize).transpose(1, 2) # g is [numX, numY, rowsize, colsize]
            w = (G-phyArrParams.cellMinConduct)/phyArrParams.cellDeltaConduct  #   G = w*deltaC + Gmin, so w = (G-Gmin)/deltaC
            phyArrParams.scn_model.to(G)
            we = phyArrParams.scn_model(w).reshape(numX, numY, phyArrParams.arrRowSize, arrColSize).mul(phyArrParams.cellDeltaConduct)+phyArrParams.cellMinConduct
            inv = torch.transpose(inv, 0, 1)
            we = we.transpose(1, 2).reshape(numX, phyArrParams.arrRowSize, -1)
            return torch.matmul(inv, we).transpose(0, 1)
        else:
            G = G.reshape(numX, phyArrParams.arrRowSize, numY, arrColSize).transpose(1, 2) # g is [numX, numY, rowsize, colsize]
            w = (G-phyArrParams.cellMinConduct)/phyArrParams.cellDeltaConduct  #   G = w*deltaC + Gmin, so w = (G-Gmin)/deltaC
            phyArrParams.scn_model.to(G)
            we = phyArrParams.scn_model(w[..., 0:phyArrParams.arrColSize]).reshape(numX, numY, phyArrParams.arrRowSize, phyArrParams.arrColSize)
            additional_col = we.new_zeros(numX, phyArrParams.arrRowSize, numY, 2)
            additional_col[..., -1] = 1
            we = torch.cat((we, additional_col), 3).mul(phyArrParams.cellDeltaConduct)+phyArrParams.cellMinConduct
            we = we.transpose(1, 2).reshape(numX, phyArrParams.arrRowSize, -1)
            return torch.matmul(inv, we).transpose(0, 1)
            
            


        






    

