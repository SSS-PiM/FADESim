import time
import torch
import argparse
import random
from torch import Tensor
import matplotlib.pyplot as plt
from fixedPoint.nn.commonConst import phyArrParams, debug, GS_method
# from fixedPoint.nn.ir_drop_solve import ir_drop_accurate_GSMethod, ir_drop_accurate_GSMethod_OU, ir_drop_OU


# inv [bsize, numX, phyArrParams.arrRowSize]
# x [numX, phyArrParams.arrRowSize, true_outsize]
# return [bsize, numx, true_outsize]
def ir_drop_accurate_GSMethod(inv: Tensor, x: Tensor, bsize: int, numX: int, numY: int,
                              arrColSize: int = phyArrParams.arrColSize) -> Tensor:
    beta_scale = phyArrParams.ir_drop_GS_method_beta_scale
    last_vup = inv.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, numY, arrColSize)
    last_vdown = x.new_zeros((bsize, numX, phyArrParams.arrRowSize, numY, arrColSize))
    now_vup = last_vup.clone()
    now_vdown = last_vdown.clone()

    x = x.reshape(numX, phyArrParams.arrRowSize, numY, arrColSize)
    g_wire = 1.0 / phyArrParams.r_wire

    if phyArrParams.ir_drop_GS_method_mode == GS_method.nodeBynode:
        for _ in range(phyArrParams.ir_drop_iter_time):
            for i in range(phyArrParams.arrRowSize - 1, -1, -1):
                for j in range(arrColSize):
                    left = inv[..., i].unsqueeze(-1).repeat(1, 1, numY) if j == 0 else now_vup[:, :, i, :,
                                                                                       j - 1]  # left is [bsize, numx, numY]
                    last_vup_ij = last_vup[:, :, i, :, j]
                    last_vdown_ij = last_vdown[:, :, i, :, j]

                    other_point = last_vdown_ij  # is [bsize, numX, numY]
                    x_ij = x[:, i, :, j]
                    x_o = x_ij.mul(other_point)

                    last_vup_ij_x1b = last_vup_ij.mul(1.0 - beta_scale)
                    last_vdown_ij_x1b = last_vdown_ij.mul(1.0 - beta_scale)

                    if j == arrColSize - 1:
                        sumg = x_ij.add(g_wire)  # sumg is [numX, numY]
                        now_vup[:, :, i, :, j] = (left.mul(g_wire) + x_o).mul(beta_scale).div(sumg) + last_vup_ij_x1b
                    else:
                        right = last_vup[:, :, i, :, j + 1]
                        sumg = x_ij.add(2 * g_wire)  # sumg is [numX, numY]
                        now_vup[:, :, i, :, j] = ((left + right).mul(g_wire) + x_o).mul(beta_scale).div(
                            sumg) + last_vup_ij_x1b

                    right = inv.new_zeros((bsize, numX, numY)) if i == phyArrParams.arrRowSize - 1 else now_vdown[:, :,
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
        current = (last_vup - last_vdown).mul(
            x)  # [bsize, numx, rowsize, numy, colsize] * [numx, rowsize, numy, colsize]
        return current.sum(2).reshape(bsize, numX, -1)  # [bsize, numx, true_outsize]
    else:  # batch by batch mode
        for iter in range(phyArrParams.ir_drop_iter_time):

            left = inv.unsqueeze(-1).repeat(1, 1, 1, numY)  # left is [bsize, numx, rowSize, numY]

            for j in range(arrColSize - 1):
                right = last_vup[..., j + 1]  # right is [bsize, numx, rowsize, numy]
                sumg = x[..., j].add(g_wire * 2)  # sumg is [numx, rowsize, numy]
                now_vup[..., j] = ((left + right).mul(g_wire) + x[..., j] * last_vdown[..., j]).mul(beta_scale).div(
                    sumg) + last_vup[..., j].mul(1.0 - beta_scale)
                left = now_vup[..., j]
            sumg = x[..., -1].add(g_wire)  # sumg is [numX, rosSize, numY]
            now_vup[..., -1] = (left.mul(g_wire) + x[..., -1] * last_vdown[..., -1]).mul(beta_scale).div(sumg) + \
                               last_vup[..., -1].mul(1.0 - beta_scale)

            down = inv.new_zeros(
                (bsize, numX, numY, arrColSize))  # if i==phyArrParams.arrRowSize-1 else now_vdown[:, :, i+1, :, j]

            for i in range(phyArrParams.arrRowSize - 1, 0, -1):
                up = last_vdown[:, :, i - 1, ...]  # [bsize, numx, numy, colSize]
                sumg = x[:, i, ...].add(g_wire * 2)  # [numx, numy, colsize]
                now_vdown[:, :, i, ...] = ((up + down).mul(g_wire) + x[:, i, ...] * now_vup[:, :, i, ...]).mul(
                    beta_scale).div(sumg) + last_vdown[:, :, i, ...].mul(1.0 - beta_scale)
                down = now_vdown[:, :, i, ...]
            sumg = x[:, 0, ...].add(g_wire)  # [numx, numy, colsize]
            now_vdown[:, :, 0, ...] = (down.mul(g_wire) + x[:, 0, ...] * now_vup[:, :, 0, ...]).mul(beta_scale).div(
                sumg) + last_vdown[:, :, 0, ...].mul(1.0 - beta_scale)

            last_vup, now_vup = now_vup, last_vup
            last_vdown, now_vdown = now_vdown, last_vdown

            if phyArrParams.ir_drop_less_than_thresholdBreak_enable:
                diff_up = ((now_vup - last_vup).abs() > phyArrParams.ir_drop_less_than_threshold).sum().item()
                diff_down = ((now_vdown - last_vdown).abs() > phyArrParams.ir_drop_less_than_threshold).sum().item()
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
        return last_vup - last_vdown
        # current = (last_vup-last_vdown).mul(x) # [bsize, numx, rowsize, numy, colsize] * [numx, rowsize, numy, colsize]
        # return current.sum(2).reshape(bsize, numX, -1) # [bsize, numx, true_outsize]


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
    g_load_left_wire = 1.0 / ((
                                          OUColSize * c + 1) * phyArrParams.r_wire + phyArrParams.r_load)  # load conductance left from input to OU
    g_load_down_wire = 1.0 / ((phyArrParams.arrRowSize - (
                r + 1) * OURowSize + 1) * phyArrParams.r_wire + phyArrParams.r_load)  # load conductance from OU down to output
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
        current = (last_vup - last_vdown).mul(
            x)  # [bsize, numx, rowsize, numy, colsize] * [numx, rowsize, numy, colsize]
        return current.sum(2).reshape(bsize, numX, -1)  # [bsize, numx, true_outsize]
    else:  # batch by batch mode
        for iter in range(phyArrParams.ir_drop_iter_time):

            left = inv.unsqueeze(-1).repeat(1, 1, 1, numY)  # left is [bsize, numX, OURowSize, numY]

            for j in range(OUColSize - 1):
                right = last_vup[..., j + 1]  # right is [bsize, numx, OURowSize, numy]
                if j == 0:
                    sumg = x[..., j].add(g_load_left_wire + g_wire)  # sumg is [numx, OURowSize, numy]
                    now_vup[..., j] = (left.mul(g_load_left_wire) + right.mul(g_wire) + x[..., j] * last_vdown[
                        ..., j]).mul(beta_scale).div(
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
                    now_vdown[:, :, i, ...] = (up.mul(g_wire) + down.mul(g_load_down_wire) + x[:, i, ...] * now_vup[:,
                                                                                                            :, i,
                                                                                                            ...]).mul(
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
                    print("iter = {}, larger than threshould number = {}".format(iter, diff_up + diff_down))
                #########################################
                if diff_up == 0 and diff_down == 0:
                    ######### debugging info ################
                    if debug.simple:
                        print(iter)
                    #########################################
                    break

        return last_vup - last_vdown
        # current = (last_vup - last_vdown).mul(x)  # [bsize, numx, OURowSize, numy, OUColSize] * [numx, OURowSize, numy, OUColSize]
        # return current.sum(2).reshape(bsize, numX, -1)  # [bsize, numx, numY*OUColSize]]


def main():
    parser = argparse.ArgumentParser(description='PyTorch Scientific Computing Example')
    parser.add_argument('--tolerance', type=float, default=1e-9, metavar='T',
                        help='')
    parser.add_argument('--iterations', type=int, default=15000, metavar='I',
                        help='')
    parser.add_argument('--seed', type=int, default=10, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--data-dir', default='data', metavar='DD',
                        help='dir of dataset')
    # SPD: nos1, bcsstk22
    # non-SPD: Chebyshev1?, nnc261, sherman3
    parser.add_argument('--data-name', default='sherman3', metavar='DN',
                        help='matrix testbench name')
    parser.add_argument('--matrix-filename', default='', metavar='DN',
                        help='matrix testbench name')
    parser.add_argument('--bias-filename', default='sherman3_b', metavar='DN',
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
    random.seed(args.seed)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.cuda_use_num) if use_cuda else "cpu")
    print(f"device: {device}")

    # Vin [bsize=1,numX=1,]
    # inv [bsize, numX, phyArrParams.arrRowSize]
    # x [numX, phyArrParams.arrRowSize, true_outsize]
    Vin = Tensor([[[phyArrParams.inputMaxVoltage for _ in range(phyArrParams.arrRowSize)]]]).to(device)
    G = Tensor([[[phyArrParams.cellMaxConduct for j in range(phyArrParams.arrColSize)] for i in range(phyArrParams.arrRowSize)]]).to(device)

    Vcell = ir_drop_accurate_GSMethod(Vin, G, 1, 1, 1).reshape((phyArrParams.arrRowSize, phyArrParams.arrColSize))

    Vin_ = Vin.reshape((1, 1, phyArrParams.arrOUnum[0], phyArrParams.OUSize[0]))
    G_ = G.reshape((1, phyArrParams.arrOUnum[0], phyArrParams.OUSize[0], phyArrParams.arrOUnum[1], phyArrParams.OUSize[1]))
    VcellOU = Vin_.new_zeros((1, phyArrParams.arrOUnum[0], phyArrParams.OUSize[0], phyArrParams.arrOUnum[1], phyArrParams.OUSize[1]))
    for i in range(phyArrParams.arrOUnum[0]):
        for j in range(phyArrParams.arrOUnum[1]):
            VcellOU_ = ir_drop_accurate_GSMethod_OU(Vin_[:, :, i, :].squeeze(2), G_[:, i, :, j, :].squeeze(1).squeeze(2), 1, 1, 1, r=i, c=j).reshape((1, phyArrParams.OUSize[0], phyArrParams.OUSize[1]))
            VcellOU[:, i, :, j, :] = VcellOU_

    VcellOU = VcellOU.reshape(phyArrParams.arrRowSize, phyArrParams.arrColSize)

    plt.figure(figsize=(10, 10))
    plt.imshow(Vcell.cpu(), interpolation='none')
    plt.clim(0.05, 0.1)
    plt.colorbar()
    plt.savefig("./image.jpg", dpi=600, bbox_inches='tight')

    plt.figure(figsize=(10, 10))
    plt.imshow(VcellOU.cpu(), interpolation='none')
    plt.clim(0.05, 0.1)
    plt.colorbar()
    plt.savefig("./image2.jpg", dpi=600, bbox_inches='tight')

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