import torch.nn as nn
import torch.optim as optim
import logging
import torch
import math

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