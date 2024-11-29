import torch.utils.data
import pickle
import copy
import math
import os
from torch.utils.data import IterableDataset, Dataset
import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from trainCommon import split_data_loader, train_model, test_model
from fixedPoint.nn.commonConst import phyArrParams
from SCN_data_gen import WeGetModel, get_we
import logging

import torch.nn.functional as F

class IRdropData(Dataset):
    def __init__(self, path, name, use_float64 = False, transform = None, target_transform = None, train = True):
        if path[-1]!='/':
            path += '/'
        self.name = path+name
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.use_float64 = use_float64
        with open(self.name, 'rb') as f:
            line = pickle.load(f)
            self.rowSize, self.colSize, self.inputMaxV, self.dacBits, self.HRS, self.LRS, self.cellBits, self.r_wire, self.sigma = line

            self.rowSize = int(self.rowSize)
            self.colSize = int(self.colSize)
            self.dacBits = int(self.dacBits)
            self.cellBits = int(self.cellBits)

            flag = 0
            line = pickle.load(f) #f.readline()
            self.data = []
            self.target = []
            while line:
                if flag%2==0:
                    self.data.append(list(map(float, line)))
                else:
                    self.target.append(list(map(float, line)))
                flag += 1
                try:
                    line = pickle.load(f) #f.readline()
                except:
                    break
                # line = f.readline()
            
        assert(len(self.data)==len(self.target))

        print("Read ir drop {} dataset Finish!".format("train" if self.train else "test"), flush=True)
        print("Crx size = {}x{}, inputMaxV = {}, dac bit = {},\
            HRS & LRS = {} & {}, cell bit = {}, r_wire = {}, sigma = {}".format(self.rowSize, self.colSize,
                                                                   self.inputMaxV, self.dacBits,
                                                                   self.HRS, self.LRS, self.cellBits,
                                                                   self.r_wire, self.sigma
                                                                   ))

        self.data = torch.tensor(self.data)
        self.target = torch.tensor(self.target)

        train_len = int(len(self.data)*0.8)    #前80%是训练数据，后20%是测试数据
        self.train_data = self.data[:train_len].clone()
        self.train_target = self.target[:train_len].clone()

        self.test_data = self.data[train_len:].clone()
        self.test_target = self.target[train_len:].clone()
    
    def set_transform(self, transform):
        self.transform = transform
        
    def set_target_transform(self, transform):
        self.target_transform = transform

    def __len__(self):
        return len(self.train_data) if self.train else len(self.test_data)
    
    def __getitem__(self, index):
        if self.train:
            data, target = self.train_data[index], self.train_target[index]
        else:
            data, target = self.test_data[index], self.test_target[index]

        data = data.view(-1, self.rowSize, self.colSize)
        target = target.view(-1, self.rowSize, self.colSize)

        if self.transform is not None:
            data = self.transform(data)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.use_float64:
            return data.type(torch.float64).view(self.rowSize*self.colSize), target.type(torch.float64).view(self.rowSize*self.colSize)
        return data.view(self.rowSize*self.colSize), target.view(self.rowSize*self.colSize)
    
class Cmul(nn.Module):
    def __init__(self, *size):
        super().__init__()
        self.size = size
        # self.w = torch.Tensor(*self.size)
        self.w = torch.ones(*self.size)
        # self.reset()
        self.w = nn.Parameter(self.w)
    
    def reset(self, stdv: float = None):
        if stdv is not None:
            stdv = stdv * math.sqrt(3)
        else:
            stdv = 1/math.sqrt(self.w.nelement())
        nn.init.uniform_(self.w, -stdv, stdv)
        # self.w.add_(1)
        # print(self.w)
        logging.debug("cmul stdv = {}, cmul data = {}".format(stdv, self.w))

    
    def forward(self, input: torch.Tensor):
        output = input * self.w
        # print(f"cmul sum = {self.w}")
        return output

# input: weight data quantized after original training of DNN (int)
# output: weight data after affected by IR drop (int)
class SCN(nn.Module):
    def __init__(self, layer_num, channel_size, cbsize):
        super().__init__()
        self.layer_num = layer_num        
        self.channel_size = channel_size        
        self.cbsize = cbsize
        self.c1 = Cmul(1, self.cbsize[0], self.cbsize[1])
        self.net = nn.Sequential()
        self.net.add_module("conv0", nn.Conv2d(1, self.channel_size, kernel_size=3, stride=1, padding=1))
        self.net.add_module("relu0", nn.ReLU())
        for i in range(1, self.layer_num-1):
            self.net.add_module("conv"+str(i), nn.Conv2d(self.channel_size, self.channel_size, kernel_size=3, stride=1, padding=1))
            self.net.add_module("relu"+str(i), nn.ReLU())

        self.net.add_module("conv"+str(self.layer_num-1), nn.Conv2d(self.channel_size, 1, kernel_size=3, stride=1, padding=1))
        
        self.c2 = Cmul(1, self.cbsize[0], self.cbsize[1])
        

    def forward(self, x):
        x = x.view(-1, self.cbsize[0], self.cbsize[1])
        y = self.c1(x)
        y = y.view(-1, 1, self.cbsize[0], self.cbsize[1])
        y = self.net(y)
        y = y.view(-1, self.cbsize[0], self.cbsize[1])
        y = self.c2(y)
        return y.view(-1, self.cbsize[0]*self.cbsize[1])
        

def mytest(model, dataset: IRdropData, device, dtype):

    batch_size = 1
    inNum = 2**dataset.dacBits
    inDeltaV = dataset.inputMaxV/(inNum-1)
    
    cellNum = 2**dataset.cellBits
    conductanceDelta = (1/dataset.LRS - 1/dataset.HRS)/(cellNum-1)
    iter_time = 5000

    temp = torch.empty([1, 2]).to(device, dtype)
    cbsize = [dataset.rowSize, dataset.colSize]
    
    input = torch.randint(0, inNum, (batch_size, 1, cbsize[0]))*inDeltaV
    w_q = torch.randint(0, cellNum, (1, cbsize[0], cbsize[1])).to(temp)
    input = input.to(temp)
    w_q = w_q.to(temp)

    print(f"no ir drop ideal out  = {input.view(1, cbsize[0]).matmul(w_q.view(cbsize[0], cbsize[1])).div(inDeltaV)}")
    w = w_q*conductanceDelta + 1/dataset.HRS
    w = w.to(temp)

    print(f"no ir drop ideal Iout  = {input.view(1, cbsize[0]).matmul(w.view(cbsize[0], cbsize[1]))}")

    print(cbsize)
    y_out =  phyArrParams.cpp_ir_drop_accelerate.ir_drop_gs(input, w, batch_size, cbsize[0], cbsize[1], 1, 1, iter_time, 1/dataset.r_wire, 1.8, False, 1e-9, False, [], False)

    print(f"real I out = {y_out}")

    print(f"real out = {(y_out-1/dataset.HRS*cbsize[0]).div(conductanceDelta*inDeltaV)}")
    # print(input)
    # print(y)

    w_e = model(w_q)
    w_e = w_e.reshape(cbsize[0], cbsize[1]).mul(conductanceDelta) + 1/dataset.HRS
    y_pred_out = input.matmul(w_e)
    print(f"pred I out = {y_pred_out}")
    print(f"pred out = {(y_pred_out-1/dataset.HRS*cbsize[0]).div(conductanceDelta*inDeltaV)}")

    print(f"relative I error sum = {((y_pred_out-y_out)/y_out).abs().sum()}")
    print(f"max relative I error = {((y_pred_out-y_out)/y_out).abs().max()}")

    

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='irdrop SCN Example')
    parser.add_argument('--train-batch-size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 50)')
    parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
                        help='input batch size for testing (default: 50)')
    parser.add_argument('--hidden-channels', type=int, default=32, metavar='N',
                        help='Hidden channel size (default: 32)')
    parser.add_argument('--layer_num', type=int, default=7, metavar='N',
                        help='layer num(default: 7)')
    parser.add_argument('--cbsize', type=int, nargs="+", default=[128, 128], metavar='N',
                        help='default crx size = [128, 128]')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--scheduler', action='store_true', default=True,
                        help='use scheduler or not')
    parser.add_argument('--lr-decay-step', type=int, default=40, metavar='STEP',
                        help='Period of learning rate decay. (default: 30)')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='GAMMA',
                        help='Learning rate step gamma (default: 0.5)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', type=float, default=0,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--seed', type=int, default=999, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--data-dir', default='data/IRdrop', metavar='DD',
                        help='dir of dataset')
    parser.add_argument('--data-name', default='Alox_hfox_32x32_293_5w.binlog', metavar='DD',
                        help='dataset name')
    parser.add_argument('--model-dir', default='model', metavar='MD',
                        help='dir of load/save model')
    parser.add_argument('--load-filename', default='Alox_hfox_32x32_293_5w_checkp.pt', metavar='LF',
                        help='filename of load model')
    parser.add_argument('--save-filename', default='Alox_hfox_32x32_293_5w_checkp.pt', metavar='LF',
                        help='filename of load model')
    parser.add_argument('--load', type=int, default=0,
                        help='load model file or not')
    parser.add_argument('--train', type=int, default=0,
                        help='train the model')
    parser.add_argument('--cuda', type=int, default=0,
                        help='use CUDA training')
    parser.add_argument('--cuda-use-num', type=int, default=2, metavar='CUDA',
                        help='use which cuda (choice: 0-2)')
    parser.add_argument('--use-float64', action='store_true', default=False,
                        help='use float64')
    parser.add_argument('--skip-dataset', action='store_true', default=False,
                        help='if you use mytest, then you can skip dataset read')

    args = parser.parse_args()
    print(args)

    # logging.basicConfig(level=logging.DEBUG)

    torch.manual_seed(args.seed)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.cuda_use_num) if use_cuda else "cpu")
    print(f"device: {device}")

    train_kwargs = {'batch_size': args.train_batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': True}
    if use_cuda:
        train_cuda_kwargs = {'num_workers': 0, 'pin_memory': False}
        test_cuda_kwargs = {'num_workers': 0, 'pin_memory': False}
        train_kwargs.update(train_cuda_kwargs)
        test_kwargs.update(test_cuda_kwargs)

    # dataset
    # transform = transforms.Compose([
    #     transforms.Normalize((0.5), (1))
    # ])

    if not args.skip_dataset:
        train_data = IRdropData(path=args.data_dir, name = args.data_name, train=True, use_float64=args.use_float64)
        test_data = copy.copy(train_data)

        test_data.train = False
        train_loader, test_loader, _ = split_data_loader(train_data, test_data, train_kwargs, test_kwargs)

    # test_data = IRdropData(path=args.data_dir, name = args.data_name, train=False, transform=transform)

    # mean = (1/train_data.HRS+1/train_data.LRS)/2
    # std = math.sqrt(((1/train_data.HRS)-mean)**2+((1/train_data.LRS)-mean)**2)*1e-1

    # transform = transforms.Compose([
    #     transforms.Normalize((mean), (std))
    # ])

    # train_data.set_transform(transform)
    # train_data.set_target_transform(transform)
    # test_data.set_transform(transform)
    # test_data.set_target_transform(transform)


    model = SCN(args.layer_num, args.hidden_channels, args.cbsize).to(device)

    dtype = torch.float32
    if args.use_float64:
        dtype = torch.float64
        model = model.double()
        print("Use float64 for the model.")
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    print(model)

    model_name = type(model).__name__

    model_save_filename = args.model_dir + '/' + model_name + args.save_filename

    model_load_filename = args.model_dir + '/' + model_name + args.load_filename

    try:
        if args.load:
            para = torch.load(model_load_filename)
            model.load_state_dict(para)
    except Exception as e:
        print(e)

    if args.scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.gamma)
    else:
        scheduler = None

    logging.info("[Important Note] that the correct item number and train/test accuracy are meaningless in SCN as they only judge\
        whether the maxarg is equal, we should see the loss to judge the entire output!!!")
    if args.train:
        criterion = nn.MSELoss()
        _, _, _ = train_model(model, device, train_loader, test_loader, criterion, optimizer, args.epochs,
                              model_filename=model_save_filename, score_type='accuracy', scheduler=scheduler, single_target=False)

    criterion = nn.MSELoss()
    test_model(model, device, test_loader, criterion, single_target=False)
    mytest(model, train_data, device, dtype)


if __name__ == '__main__':
    main()

