import torch.utils.data
import pickle
import math
from torch.utils.data import IterableDataset
import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from functools import reduce
import torch
from fixedPoint.nn.commonConst import phyArrParams


class WeGetModel(nn.Module):
    def __init__(self, *sizes, init_We: torch.Tensor = None):
        super().__init__()
        if init_We is None:
            self.We = torch.Tensor(*sizes)
        else:
            self.We = init_We.clone()
        
        self.We = nn.Parameter(self.We)
    
    def forward(self, input: torch.Tensor):
        output = input.matmul(self.We)
        return output

def get_we(init_We: torch.Tensor, X: torch.Tensor, Y: torch.Tensor, device, iter_time: int, lr_decay_step: int, lr: float = 1e-2, dtype = torch.float64):
    model = WeGetModel(*init_We.shape, init_We=init_We).to(device).to(dtype=dtype)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=4e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=0.5)
    criterion = nn.MSELoss()
    avg_loss = 0.
    all_loss = []
    for _ in range(iter_time):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, Y)
        avg_loss += loss
        all_loss.append(loss)
        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_loss /= iter_time
    print("avg loss = {}, first loss = {}, last loss = {}".format(avg_loss, all_loss[0], all_loss[-1]))

    return model.We.clone()

    
    
        


        
    
    
        
        
        

#    self.rowSize, self.colSize, self.inputMaxV, self.dacBits, \
                # self.HRS, self.LRS, self.cellBits, self.sigma = list(map(float, line.split()))

def main():
    parser = argparse.ArgumentParser(description='irdrop SCN Example')
    
    parser.add_argument('--data-num', type=int, default=50000, metavar='N',
                        help='generate data num (default: 10000)')
    parser.add_argument('--cbsize', type=int, nargs='+', default=[128, 128], metavar='N',
                        help='generate data num (default: 10000)')
    parser.add_argument('--inputV', type=float, default=1.0, metavar='N',
                        help='max input voltage (default: 1.0)')
    parser.add_argument('--dacBits', type=int, default=1, metavar='N',
                        help='dac bit = (default: 1)')
    parser.add_argument('--HRS', type=float, default=74867, metavar='N',
                        help='HRS = (default: 2e5)')
    parser.add_argument('--LRS', type=float, default=16.9e3, metavar='N',
                        help='LRS = (default: 1e5)')
    parser.add_argument('--cellBits', type=int, default=1, metavar='N',
                        help='cell bit = (default: 1)')
    parser.add_argument('--r-wire', type=float, default=2.93, metavar='N',
                        help='wire resistance= (default: 2.93)')
    parser.add_argument('--sigma', type=float, default=0, metavar='N',
                        help='program variation sigma = (default: 0)')
    parser.add_argument('--seed', type=int, default=959595, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--data-dir', default='data/IRdrop', metavar='DD',
                        help='dir of dataset')
    parser.add_argument('--data-name', default='Alox_hfox_128x128_293_5w.binlog', metavar='DD',
                        help='dataset name')
    parser.add_argument('--batch', type=int, default=1, metavar='DD',
                        help='data generate batch')
    parser.add_argument('--iter-time', type=int, default=830, metavar='DD',
                        help='ir drop iter time')
    parser.add_argument('--irdropMode', type=int, default=1, metavar='DD',
                        help='0: ir2s,  other: ir_drop_gs')
    parser.add_argument('--ir-drop-GS-beta-scale', type=float, default=1.97608, metavar='DD',
                        help='gs method beta scale')
                    
    
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)

    inNum = 2**args.dacBits
    inDeltaV = args.inputV/(inNum-1)
    
    cellNum = 2**args.cellBits
    conductanceDelta = (1/args.LRS - 1/args.HRS)/(cellNum-1)


    name = args.data_dir if args.data_dir[-1]=='/' else args.data_dir + '/'
    name += args.data_name

    with open(name, "wb") as f:
        header = [args.cbsize[0], args.cbsize[1], args.inputV, args.dacBits, args.HRS, args.LRS, args.cellBits, args.r_wire, args.sigma]
        # header = reduce(lambda x, y: x+y, map(lambda x: str(x)+" ", header))
        pickle.dump(header, f)

        print_flag = 0
        for i in range(0, args.data_num, args.batch):
            if i*args.batch-print_flag>500:
                print("in item {}".format(i), flush=True)
                print_flag = i*args.batch
            batch_size = args.batch
            if i+args.batch>args.data_num:
                batch_size = args.data_num -i
                
            # input_q = torch.randint(0, inNum, (batch_size, 1, args.cbsize[0])) 
            # input_q = torch.ones((batch_size, 1, args.cbsize[0]))
            input_q = torch.nn.functional.one_hot(torch.arange(0, args.cbsize[0]))
            input_q = input_q.reshape(args.cbsize[0], 1, -1)
            # print(input_q)
            input = input_q*inDeltaV
            # input = torch.ones((batch_size, 1, args.cbsize[0]))*args.inputV
            w_q = torch.randint(0, cellNum, (1, args.cbsize[0], args.cbsize[1]))
            w = w_q*conductanceDelta + 1/args.HRS

            device = torch.device("cuda:0")
            input = input.to(device).type(torch.float64)
            w = w.to(device).type(torch.float64)

            if args.irdropMode == 0:
                assert(args.cbsize[0]==args.cbsize[1])
                batch_size = args.cbsize[0]
                Y =  phyArrParams.cpp_ir_drop_accelerate.ir2s(input, w, batch_size, args.cbsize[0], args.cbsize[1], 1, 1, args.iter_time, args.r_wire, False, 1e-9, False, [], False)
            else:
                assert(args.cbsize[0]==args.cbsize[1])
                batch_size = args.cbsize[0]
                Y =  phyArrParams.cpp_ir_drop_accelerate.ir_drop_gs(input, w, batch_size, args.cbsize[0], args.cbsize[1], 1, 1, args.iter_time, 1/args.r_wire, args.ir_drop_GS_beta_scale, False, 1e-9, False, [], False)
            
            w_e = Y

            #### rand test ####
            # input = torch.randint(0, inNum, (1, 1, args.cbsize[0])) *inDeltaV
            # input = input.to(device).type(torch.float64)
            # print(f"input = {input}")
            # print(f"w = {w}")

            # y_out =  phyArrParams.cpp_ir_drop_accelerate.ir_drop_gs(input, w, 1, args.cbsize[0], args.cbsize[1], 1, 1, args.iter_time, 1/args.r_wire, args.ir_drop_GS_beta_scale, False, 1e-9, False, [], False)
            # print(f"spice real I out = {y_out}")

            # y_pred_out = input.view(1, args.cbsize[0]).matmul(w_e.div(inDeltaV).view(args.cbsize[0], args.cbsize[1]))
            # print(f"using Geff pred out = {y_pred_out}")
            # print(((y_out - y_pred_out)/y_out).abs().sum())
            #### rand test end ####
            
            batch_size = 1
            for j in range(batch_size):
                # data = reduce(lambda x, y: x+y, map(lambda x: str(x)+" ", w_q.view(-1).tolist()))
                # f.write(data+"\n")
                pickle.dump(w_q.view(-1).tolist(), f)

                # ans = reduce(lambda x, y: x+y, map(lambda x: str(x)+" ", (w_e.view(-1)-1/args.HRS).div(conductanceDelta*inDeltaV).tolist()))
                # f.write(ans+"\n")
                pickle.dump((w_e.view(-1)-1/args.HRS).div(conductanceDelta*inDeltaV).tolist(), f)
            
            

            





        
        
        

    


if __name__ == '__main__':
    main()