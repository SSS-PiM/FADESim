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


class BNNMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = fpnn.BinarizedLinear(784, 32, bnn_first_layer_flag=True)
        self.bn1 = nn.BatchNorm1d(32)
        self.htanh1 = nn.Hardtanh()
        self.fc2 = fpnn.BinarizedLinear(32, 10)
        self.bn2 = nn.BatchNorm1d(10)
        self.classifier = nn.LogSoftmax()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.classifier(x)
        return x

class ConvMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(fpnn.Conv2d_sp(1, 10, (5, 5)),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                  fpnn.Conv2d_sp(10, 20, (5, 5)),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                  nn.Flatten(),
                                  nn.Dropout(p=0.2),
                                  fpnn.Linear_sp(4 * 4 * 20, 10)
                                  )

    def forward(self, x):
        x = self.conv(x)
        return x


class BNNLeNetMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.bnn_lenet = nn.Sequential(
            fpnn.BinarizedConv2d(1, 6, (5, 5), padding=(2, 2), bnn_first_layer_flag=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(6),
            nn.Hardtanh(),

            fpnn.BinarizedConv2d(6, 16, (5, 5), padding=(2, 2)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(16),
            nn.Hardtanh(),

            nn.Flatten(),
            nn.Dropout(p=0.2),

            fpnn.BinarizedLinear(7*7*16, 120),
            nn.BatchNorm1d(120),
            nn.Hardtanh(),

            fpnn.BinarizedLinear(120, 84),
            nn.BatchNorm1d(84),
            nn.Hardtanh(),

            fpnn.BinarizedLinear(84, 10),
            nn.BatchNorm1d(10),

            nn.LogSoftmax()
        )

    def forward(self, x):
        x = self.bnn_lenet(x)

        return x


class Fc5Mnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = fpnn.Linear_sp(784, 2048)
        self.fc2 = fpnn.Linear_sp(2048, 2048)
        self.fc3 = fpnn.Linear_sp(2048, 2048)
        self.fc4 = fpnn.Linear_sp(2048, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        return x

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    
    parser.add_argument('--scheduler', action='store_true', default=True,
                        help='use scheduler or not')
    parser.add_argument('--lr-decay-step', type=int, default=5, metavar='STEP',
                        help='Period of learning rate decay. (default: 30)')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='GAMMA',
                        help='Learning rate step gamma (default: 0.5)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', type=float, default=5e-4,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--seed', type=int, default=4, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--data-dir', default='data', metavar='DD',
                        help='dir of dataset')
    parser.add_argument('--model-dir', default='model', metavar='MD',
                        help='dir of load/save model')
    
    parser.add_argument('--load-filename', default='BNNMnist_checkpoint.pt', metavar='LF',
                        help='filename of load model')
    parser.add_argument('--train', action='store_true', default=False,
                        help='train the model')
    parser.add_argument('--load-model-type', type=int, default=1, metavar='LD',
                        help='load mode type (0:no 1:load)')
    parser.add_argument('--half-float', action='store_true', default=True,
                        help='For use 16b float')
    parser.add_argument('--net', type=int, default=0, metavar='NET',
                        help='use which model (0:bnnfc_mnist 1:fc5mnist 2:conv_mnist 3:bnnlenet_mnist -1 :none)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA training')
    parser.add_argument('--cuda_use_num', type=int, default=2, metavar='CUDA',
                        help='use which cuda (choice: 0-2)')

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.cuda_use_num) if use_cuda else "cpu")
    print(f"device: {device}")

    train_kwargs = {'batch_size': args.train_batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        train_cuda_kwargs = {'num_workers': 1, 'pin_memory': False, 'shuffle': True}
        test_cuda_kwargs = {'num_workers': 1, 'pin_memory': False, 'shuffle': False}
        train_kwargs.update(train_cuda_kwargs)
        test_kwargs.update(test_cuda_kwargs)

    # dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)

    train_loader, test_loader, _ = split_data_loader(train_data, test_data, train_kwargs, test_kwargs)

    if args.net == 0:
        model = BNNMnist().to(device)
    elif args.net ==1:
        model = Fc5Mnist().to(device)
    elif args.net == 2:
        model = ConvMnist().to(device)
    elif args.net == 3:
        model = BNNLeNetMnist().to(device)
    else:
        raise Exception('undefined net: ' + str(args.net))

    bit_width_list = create_layer_weight_bit_width_list(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    print(model)

    model_name = type(model).__name__

    model_save_filename = args.model_dir + '/' + model_name + '_checkpoint.pt'

    model_load_filename = args.model_dir + '/' + args.load_filename

    try:
        if args.load_model_type == 1:
            para = torch.load(model_load_filename)
            model.load_state_dict(para)
    except Exception as e:
        print(e)

    if args.scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.gamma)
    else:
        scheduler = None

    if args.train:
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.MSELoss()
        _, _, _ = train_model(model, device, train_loader, test_loader, criterion, optimizer, args.epochs,
                              model_filename=model_save_filename, score_type='accuracy', scheduler=scheduler)

    criterion = nn.CrossEntropyLoss(reduction='sum')
    test_model(model, device, test_loader, criterion)


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
