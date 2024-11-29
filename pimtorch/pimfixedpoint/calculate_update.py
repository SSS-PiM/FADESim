import argparse
import time

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision
import json

from fixedPoint import optim as fpOptim
import torch.utils.data

from fixedPoint.nn.fixedPointArithmetic import *
from trainCommon import split_data_loader, train_model, test_model, create_layer_weight_bit_width_list, \
    load_float_weight_for_fixed_point, draw_data_graph, differ_two_model
from VGG_cifar10_model import vgg16, vgg11, vgg13, vgg19, FixedPointVGG8B, VGG8B, fp_vgg16, VGG16ForMotivation, \
    fp_vgg19, fp_vgg11, fp_vgg13


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 Example')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--seed', type=int, default=4, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--data-dir', default='data', metavar='DD',
                        help='dir of dataset')
    parser.add_argument('--model-dir', default='trace/FixedPointVGG_VGG16_full16', metavar='MD',
                        help='dir of load/save model(default: model)')
    parser.add_argument('--mix-precision', action='store_true', default=False,
                        help='mix-precision or not')
    parser.add_argument('--half-float', action='store_true', default=True,
                        help='For use 16b float')
    parser.add_argument('--net', default="VGG16", metavar='NET',
                        help='use which NN model')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA training')
    parser.add_argument('--cuda-use-num', type=int, default=2, metavar='CUDA',
                        help='use which cuda (choice: 0-2)')

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.cuda_use_num) if use_cuda else "cpu")
    print(f"device: {device}")

    if args.mix_precision:
        # conv_input_bit_width
        # conv_output_bit_width
        # conv_weight_bit_width
        # conv_grad_output_bit_width
        # conv_next_grad_output_bit_width
        # conv_compute_weight_bit_width
        # fc_output_bit_width
        # fc_weight_bit_width
        # fc_grad_output_bit_width
        # fc_compute_weight_bit_width
        bit_width_tuple = (16, 16, 16, 16, 16, 8, 16, 16, 16, 8)
    else:
        bit_width_tuple = (16, 16, 16, 16, 16, 16, 16, 16, 16, 16)

    if args.net == "VGG16":
        model_before = fp_vgg16(*bit_width_tuple)
        model_after = fp_vgg16(*bit_width_tuple)
        model_before.to(device)
        model_after.to(device)
    elif args.net == "VGG19":
        model_before = fp_vgg19(*bit_width_tuple)
        model_after = fp_vgg19(*bit_width_tuple)
        model_before.to(device)
        model_after.to(device)
    elif args.net == "VGG11":
        model_before = fp_vgg11(*bit_width_tuple)
        model_after = fp_vgg11(*bit_width_tuple)
        model_before.to(device)
        model_after.to(device)
    elif args.net == "VGG13":
        model_before = fp_vgg13(*bit_width_tuple)
        model_after = fp_vgg13(*bit_width_tuple)
        model_before.to(device)
        model_after.to(device)
    else:
        raise Exception('undefined net: ' + str(args.net))

    # print(model_before)

    # epoch_300_after_mid.pt
    for i in range(args.epochs):
        model_before_filename = args.model_dir + '/epoch_' + str(i + 1) + '_before_mid.pt'
        model_after_filename = args.model_dir + '/epoch_' + str(i + 1) + '_after_mid.pt'

        model_before.load_state_dict(torch.load(model_before_filename))
        model_after.load_state_dict(torch.load(model_after_filename))

        diff = differ_two_model(model_before, model_after)
        with open('log/diff/' + args.net + 'diff_epoch' + str(i + 1).zfill(3), 'w') as f:
            f.write(json.dumps(diff))


def get_result(net, epochs, layer, weightBits):
    for i in range(epochs):
        filename = 'log/diff/' + net + 'diff_epoch' + str(i + 1).zfill(3)
        all_zero = True
        with open(filename, 'r') as f:
            layer_list = json.loads(f.read())

            for layer_item in layer_list:
                for bit_item in layer_item:
                    if bit_item != 0:
                        all_zero = False
                        break

                if all_zero is False:
                    break

        if all_zero:
            print('the epoch' + str(i+1) + ' is all zero')


def analyze_result(net, epochs, layer, weightBits):
    bit_write_times = []

    for _ in range(weightBits):
        bit_write_times.append(0)

    for i in range(epochs):
        filename = 'log/diff/' + net + 'diff_epoch' + str(i + 1).zfill(3)
        with open(filename, 'r') as f:
            layer_list = json.loads(f.read())

            for layer_item in layer_list:
                bit_index = 0
                for bit_item in layer_item:
                    bit_write_times[bit_index] += bit_item
                    bit_index += 1

    times_sum = 0
    sum_list = []
    for times in bit_write_times:
        times_sum += times
        sum_list.append(times_sum)

    for i in range(len(sum_list)):
        sum_list[i] = sum_list[i] / times_sum

    print(sum_list)


if __name__ == "__main__":
    analyze_result('VGG16', 200, 16, 16)
    # get_result('VGG16', 300, 16, 16)
    # main()
