from torch import nn
import fixedPoint.nn as fpnn
import argparse
import torch
import time
import torch.nn.functional as F
from trainCommon import split_data_loader, train_model, test_model, create_layer_weight_bit_width_list, \
    load_float_weight_for_fixed_point
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch import Tensor
from fixedPoint.nn.splitArrayArithmetic import splitArr_matmul
from fixedPoint.nn.commonConst import ConstForSplitArray as cfs, phyArrParams, phyArrMode


def readFile(name: str, device: torch.device):
    '''
    @param name
    @return attNum
            classNum
            attriNum [attNum]
            attSum
            Pc [classNum]
            Pxc [attSum, classNum]
            n
            res [n]
            test [n, attNum]
    '''
    fileName = "data" + "/" + "Neuro" + "/" + name + "_Neuro.out"
    with open(fileName, mode='r', encoding='utf-8') as fp:
        attNum, classNum = tuple(fp.readline().strip().rstrip().split(' '))
        attNum, classNum = int(attNum), int(classNum)

        attriNum = fp.readline().strip().rstrip().split(' ')
        attriNum = [int(item) for item in attriNum]

        attSum = 0

        Pc = fp.readline().strip().rstrip().split(' ')
        Pc = [float(item) for item in Pc]

        # 将Pxc合并成为[attNum*attriNum, classNum]矩阵形式，后续可以直接放在阵列上进行计算
        Pxc = list()
        for i in range(attNum):
            attSum += attriNum[i]
            line = fp.readline().strip().rstrip().split(' ')
            for j in range(attriNum[i]):
                Pxc.append(list())
                for c in range(classNum):
                    Pxc[-1].append(float(line[classNum * j + c]))

        n = int(fp.readline().strip().rstrip())

        res = fp.readline().strip().rstrip().split(' ')
        res = [int(item) for item in res]

        # 将测试数据按照attr取值进行编码，如2(5)编码为00010，后续直接加载在阵列输入对应位置
        test = list()
        for i in range(n):
            line = fp.readline().strip().rstrip().split(' ')
            test.append(list())
            for j in range(attNum):
                encodes = list(bin(1 << int(line[j]))[2:].zfill(attriNum[j]))
                encodes.reverse()
                test[-1].extend([int(e) for e in encodes])
        assert(len(test[0]) == attSum)

    return attNum, classNum, Tensor(attriNum).to(device), attSum, Tensor(Pc).to(device), Tensor(Pxc).to(
        device), n, Tensor(res).to(device), Tensor(test).to(device)


def test_nb(activations: Tensor, weights: Tensor, bias: Tensor, bsize: int, inputDim: int, classDim: int,
            attriNum: Tensor, attSum: int) -> Tensor:
    # 输入数据和概率矩阵拆分到不同阵列上进行计算
    output = splitArr_matmul(activations, weights, inputBits=cfs.input_bit_width, dacBits=cfs.dac_bit_width,
                             adcBits=cfs.adc_bit_width, outputBits=cfs.output_bit_width,
                             weightBits=cfs.weight_bit_width, arrType=phyArrParams.defaultArrMode,
                             float_type=torch.float32)
    # 最后把bias加上
    output = output + bias
    # 由于是-log展开，因此取最小值对应的index
    return torch.argmin(output, 1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--seed', type=int, default=4, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--data-dir', default='data', metavar='DD',
                        help='dir of dataset')

    parser.add_argument('--load-filename', default='MNIST', metavar='LF',
                        help='filename of load model')
    parser.add_argument('--train', action='store_true', default=False,
                        help='train the model')
    # parser.add_argument('--half-float', action='store_true', default=True,
    #                     help='For use 16b float')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA training')
    parser.add_argument('--cuda_use_num', type=int, default=0, metavar='CUDA',
                        help='use which cuda (choice: 0-2)')

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.cuda_use_num) if use_cuda else "cpu")
    print(f"device: {device}")

    attNum, classNum, attriNum, attSum, Pc, Pxc, n, res, test = readFile(args.load_filename, device=device)

    if args.train:
        raise Exception("Naive bayes training mode not implemented")
    else:
        # pass
        ans = 0
        batchNum = 10
        bsize = n // batchNum
        current_n = 0
        # 分batch计算，便于IR drop仿真查看进度
        for i in range(batchNum):
            current_bsize = min(bsize, n - i * bsize)
            activations = test[i * bsize:i * bsize + current_bsize]
            labels = res[i * bsize:i * bsize + current_bsize]
            l = test_nb(activations, Pxc, Pc, current_bsize, attNum, classNum, attriNum, attSum)
            ans += (l == labels).type(torch.int).sum(0)
            current_n += current_bsize
            print("test accuracy = {}".format(ans / current_n))


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
