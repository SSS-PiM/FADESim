import time

import numpy as np
import torch
import argparse
from scipy.io import mmread
from scipy.sparse import coo_matrix
from scientific_computing_methods import cg, bicgstab, sparseMatrixParser
from fixedPoint.nn.commonConst import phyArrParams


def main():
    parser = argparse.ArgumentParser(description='PyTorch Scientific Computing Example')
    parser.add_argument('--tolerance', type=float, default=1e-9, metavar='T',
                        help='')
    parser.add_argument('--iterations', type=int, default=15000, metavar='I',
                        help='')
    # parser.add_argument('--seed', type=int, default=4, metavar='S',
    #                     help='random seed (default: 1)')
    parser.add_argument('--data-dir', default='data', metavar='DD',
                        help='dir of dataset')
    # SPD: nos1, bcsstk22, sherman3
    # non-SPD: Chebyshev1?, nnc261, sherman3
    parser.add_argument('--data-name', default='bcsstk22', metavar='DN',
                        help='matrix testbench name')
    parser.add_argument('--matrix-filename', default='', metavar='DN',
                        help='matrix testbench name')
    parser.add_argument('--bias-filename', default='', metavar='DN',
                        help='matrix testbench name')
    # cg for SPD, bicgstab for non-SPD
    parser.add_argument('--method', type=int, default=0, metavar='MD',
                        help='use which iteration method (0:cg 1:cgs 2:bicgstab 3:pcg 4:pcgs -1:none)')
    parser.add_argument('--preconditioner', type=int, default=-1, metavar='PC',
                        help='use which iteration method (0:ilu0 1:Jacobi -1:none)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA for acceleration')
    parser.add_argument('--cuda_use_num', type=int, default=0, metavar='CUDA',
                        help='use which cuda (choice: 0-2)')
    args = parser.parse_args()
    print(args)

    # torch.manual_seed(args.seed)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.cuda_use_num) if use_cuda else "cpu")
    print(f"device: {device}")

    if args.matrix_filename:
        matrix_filepath = args.data_dir + '/' + args.data_name + '/' + args.matrix_filename + ".mtx"
    else:
        matrix_filepath = args.data_dir + '/' + args.data_name + '/' + args.data_name + ".mtx"

    matrix = mmread(matrix_filepath)
    matrix_info = None
    if phyArrParams.scientific_computing_pim_enable:
        matrix.eliminate_zeros()
        matrix_info = sparseMatrixParser(matrix.transpose(), device)
    matrix = torch.Tensor(matrix.toarray()).to(device)

    if args.bias_filename:
        bias_filepath = args.data_dir + '/' + args.data_name + '/' + args.bias_filename + ".mtx"
        bias = torch.Tensor(mmread(bias_filepath)).to(device) # 读进来就是列向量
    else:
        bias = (torch.ones(matrix.shape[0]).T).to(device) # 参考已有工作，将b设置为全1向量，转置一下变成列向量

    print(matrix.shape)
    print(bias.shape)
    
    if args.method == 0:
        x = cg(matrix, bias, args.tolerance, args.iterations, matrix_info=matrix_info, print_iter=1)
    elif args.method == 2:
        raise Exception("Bicgstab not converge for many testbenches")
        # x = bicgstab(matrix, bias, args.tolerance, args.iterations, matrix_info=matrix_info)
    else:
        raise Exception("Other iteration methods not implemented")

    print(x)

    error = torch.matmul(matrix.T, x) - bias
    print("Error: ", torch.matmul(error.T, error).item())

if __name__ == '__main__':
    # torch.manual_seed(44)
    # input = torch.rand([1, 64], device="cuda").to(torch.float64)
    # w = torch.rand([64, 8], device="cuda").to(torch.float64)
    # out2 = torch.matmul(input, w)
    
    # T1 = time.time()
    # out1 = fpnn.splitArr_matmul(input, w, 8, 1, 8, 8, 8, fpnn.commonConst.phyArrMode.PN, torch.float64)
    # T2 = time.time()
    # print("use time = {} ms".format((T2-T1)*1000))
    # print(out1)
    # print(out2) 
    main()