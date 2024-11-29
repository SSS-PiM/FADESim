import time
import torch
import numpy as np
from numpy import ndarray
from scipy.sparse import coo_matrix
from torch import Tensor
from fixedPoint.nn.splitArrayArithmetic import splitArr_matmul_f, float64_fifo
from fixedPoint.nn.commonConst import ConstForSplitArray as cfs, phyArrParams, phyArrMode, debug

# Ax = b

# used for symmetric-positive-definite matrix
def cg(A: Tensor, b: Tensor, tol: float, maxit: int, matrix_info: dict = None, bias_info: dict = None, print_iter: int = None) -> Tensor:
    start_time = time.time()

    if b.shape[0] == 1:
        b = b.T

    if print_iter is None:
        print_iter = maxit

    x = torch.zeros_like(b)
    p = r = b
    rho = torch.matmul(r.T, r)
    epsilon_act2 = torch.matmul(b.T, b) * (tol ** 2)

    for j in range(maxit):
        if phyArrParams.scientific_computing_pim_enable:
            p_ = p.cpu().numpy().astype(np.float64).view('int64')
            q = splitArr_matmul_f(p_, matrix_info, p_.shape[0], adcBits=cfs.adc_bit_width, arrType=phyArrMode.Ref, device=A.device)
            q = q.T
        else:
            q = torch.matmul(A, p)
        phi = torch.matmul(q.T, p)
        alpha = rho/phi
        x = x + alpha * p
        r = r - alpha * q
        epsilon = torch.matmul(r.T, r)
        if j % print_iter == 0:
            end_time = time.time()
            print("Iterations: {0}, Precision: {1}, Time: {2}".format(j+1, epsilon.item(), end_time-start_time))
        if epsilon < epsilon_act2:
            end_time = time.time()
            print("Iterations: {0}, Precision: {1}, Time: {2}".format(j + 1, epsilon.item(), end_time - start_time))
            break
        p = r + epsilon/rho * p
        rho = epsilon
    else:
        end_time = time.time()
        print("Iterations: {0}, Precision: {1}, Time: {2}".format(j + 1, epsilon.item(), end_time - start_time))

    return x
    

# used for non symmetric-positive-definite matrix
# TODO: 对于大矩阵bicgstab求不出来测试结果，猜测是non-SPD较难收敛，可能需要实现preconditioner版本
def bicgstab(A: Tensor, b: Tensor, tol: float, maxit: int, matrix_info, bias_info=None) -> Tensor:
    if b.shape[0] == 1:
        b = b.T

    x = torch.zeros_like(b)
    p = r0star = r = b
    rho = torch.matmul(r.T, r0star)
    epsilon_act2 = torch.matmul(b.T, b) * (tol ** 2)

    for j in range(maxit):
        # if ScientificComputing.pim_enable:
        #     raise Exception("Not implemented")
        # else:
        q = torch.matmul(A, p) # q = splitArr_matmul(p, A.T, )
        phi = torch.matmul(q.T, r0star)
        alpha = rho / phi
        s = r - alpha * q
        t = torch.matmul(A, s) # t = splitArr_matmul(s, A.T, )
        omega = torch.matmul(t.T, s) / torch.matmul(t.T, t)
        x = x + alpha * p + omega * s
        r = s - omega * t
        epsilon = torch.matmul(r.T, r)
        if epsilon < epsilon_act2:
            print("Precision: {0}".format(epsilon))
            print("Iterations: {0}".format(j + 1))
            break
        rho = torch.matmul(r.T, r0star)
        p = r + (rho / phi) * (p / omega - q)
    else:
        print("Precision: {0}".format(epsilon))
        print("Iterations: {0}".format(j+1))

    return x

# future implementation
def cgs():
    pass

def pcg():
    pass

def pcgs():
    pass

# future implementation, preconditioner
def ilu0():
    pass

def Jacobi():
    pass

def sparseMatrixParser(A: coo_matrix, device) -> dict:
    A_info = dict()

    A_num = A.count_nonzero()
    A_rowsize, A_colsize = A.get_shape()
    A_row_index, A_col_index = A.nonzero()

    arr_row_size = phyArrParams.arrRowSize

    for i in range(A_num):
        r = A_row_index[i]
        c = A_col_index[i]
        xbr = r // phyArrParams.arrRowSize
        if not xbr in A_info:
            A_info[xbr] = dict()
        if not c in A_info[xbr]: # xbr stands for crossbar rows
            current_input_dim = min((xbr+1)*arr_row_size,A_rowsize) - xbr*arr_row_size
            matrix_block = A.getcol(c)[xbr*arr_row_size:xbr*arr_row_size+current_input_dim].toarray().astype(np.float64).view('int64')
            pos_array = None
            pos_array_len = None
            matrix_pos_expo_max = None
            neg_array = None
            neg_array_len = None
            matrix_neg_expo_max = None
            if (matrix_block > 0).any():
                # 矩阵块中存在正数
                matrix_pos = matrix_block.copy()
                matrix_pos[matrix_pos < 0] = 0
                pos_array, pos_array_len, matrix_pos_expo_max = float64_fifo(matrix_pos, current_input_dim)
                if debug.detailed:
                    print(matrix_pos.view('float64'))
                    print(pos_array)
                    print(pos_array_len)
                    print(matrix_pos_expo_max)
                pos_array = Tensor(pos_array).to(device)
            else:
                pos_array_len = 0
            if (matrix_block < 0).any():
                # 矩阵块中存在负数
                matrix_neg = matrix_block.copy()
                matrix_neg[matrix_neg > 0] = 0
                neg_array, neg_array_len, matrix_neg_expo_max = float64_fifo(matrix_neg, current_input_dim)
                if debug.detailed:
                    print(matrix_neg.view('float64'))
                    print(neg_array)
                    print(neg_array_len)
                    print(matrix_neg_expo_max)
                neg_array = Tensor(neg_array).to(device)
            else:
                neg_array_len = 0
            A_info[xbr][c] = (pos_array, pos_array_len, matrix_pos_expo_max, neg_array, neg_array_len, matrix_neg_expo_max)

    return A_info