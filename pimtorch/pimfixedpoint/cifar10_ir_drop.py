import argparse

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision import transforms

import fixedPoint.nn as fpnn
from fixedPoint import optim as fpOptim
from fixedPoint.nn.fixedPointArithmetic import *
from fixedPoint.nn.commonConst import phyArrParams

from trainCommon import split_data_loader, train_model, test_model
# from trainCommon import split_data_loader, train_model, test_model, create_layer_bit_width_list, \
#     load_float_weight_for_fixed_point

# class VGG16(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.vgg16 = nn.Sequential(fpnn.Conv2d_sp(3, 64, (3, 3), padding=(1, 1)),
#                                    nn.ReLU(),
#                                    fpnn.Conv2d_sp(64, 64, (3, 3), padding=(1, 1)),
#                                    nn.ReLU(),
#                                    nn.MaxPool2d(kernel_size=(2, 2), stride=2),

#                                    fpnn.Conv2d_sp(64, 128, (3, 3), padding=(1, 1)),
#                                    nn.ReLU(),
#                                    fpnn.Conv2d_sp(128, 128, (3, 3), padding=(1, 1)),
#                                    nn.ReLU(),
#                                    nn.MaxPool2d(kernel_size=(2, 2), stride=2),

#                                    fpnn.Conv2d_sp(128, 256, (3, 3), padding=(1, 1)),
#                                    nn.ReLU(),
#                                    fpnn.Conv2d_sp(256, 256, (3, 3), padding=(1, 1)),
#                                    nn.ReLU(),
#                                    nn.MaxPool2d(kernel_size=(2, 2), stride=2),

#                                    nn.Flatten(),
#                                    nn.Dropout(p=0.2),
#                                    fpnn.Linear_sp(7 * 7 * 16, 120),
#                                    nn.Tanh(),
#                                    fpnn.Linear_sp(120, 84),
#                                    nn.Tanh(),
#                                    fpnn.Linear_sp(84, 10)
#                                    )

#     def forward(self, x):
#         x = self.vgg16(x)
#         return x

class VGG11(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg11 = nn.Sequential(fpnn.Conv2d_sp(3, 64, (3, 3), padding=(1, 1)),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2), stride=2),

                                   fpnn.Conv2d_sp(64, 128, (3, 3), padding=(1, 1)),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2), stride=2),

                                   fpnn.Conv2d_sp(128, 256, (3, 3), padding=(1, 1)),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(),
                                   fpnn.Conv2d_sp(256, 256, (3, 3), padding=(1, 1)),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2), stride=2),

                                   fpnn.Conv2d_sp(256, 512, (3, 3), padding=(1, 1)),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(),
                                   fpnn.Conv2d_sp(512, 512, (3, 3), padding=(1, 1)),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2), stride=2),

                                   fpnn.Conv2d_sp(512, 512, (3, 3), padding=(1, 1)),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(),
                                   fpnn.Conv2d_sp(512, 512, (3, 3), padding=(1, 1)),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2), stride=2),

                                   nn.Flatten(),
                                #    nn.Dropout(p=0.2),
                                   fpnn.Linear_sp(1 * 1 * 512, 512),
                                   nn.ReLU(),
                                   nn.Dropout(),
                                   fpnn.Linear_sp(512, 512),
                                   nn.ReLU(),
                                   nn.Dropout(),
                                   fpnn.Linear_sp(512, 10)
                                   )

    def forward(self, x):
        x = self.vgg11(x)
        return x
    
class BNNVGG11(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            fpnn.BinarizedConv2d(3, 64, (3, 3), padding=(1, 1), bnn_first_layer_flag=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(64),
            nn.Hardtanh(),

            fpnn.BinarizedConv2d(64, 128, (3, 3), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(128),
            nn.Hardtanh(),

            fpnn.BinarizedConv2d(128, 256, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.Hardtanh(),
            fpnn.BinarizedConv2d(256, 256, (3, 3), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(256),
            nn.Hardtanh(),

            fpnn.BinarizedConv2d(256, 512, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.Hardtanh(),
            fpnn.BinarizedConv2d(512, 512, (3, 3), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(512),
            nn.Hardtanh(),

            fpnn.BinarizedConv2d(512, 512, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.Hardtanh(),
            fpnn.BinarizedConv2d(512, 512, (3, 3), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(512),
            nn.Hardtanh()
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
        #    nn.Dropout(p=0.2),
            fpnn.BinarizedLinear(1 * 1 * 512, 512),
            nn.BatchNorm1d(512),
            nn.Hardtanh(),
            nn.Dropout(p=0.5),
            fpnn.BinarizedLinear(512, 512),
            nn.BatchNorm1d(512),
            nn.Hardtanh(),
            nn.Dropout(p=0.5),
            fpnn.BinarizedLinear(512, 10),
            nn.BatchNorm1d(10),
            nn.LogSoftmax()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 Example')
    parser.add_argument('--train-batch-size', type=int, default=64, metavar='TRAIN_BATCH',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='TEST_BATCH',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    
    parser.add_argument('--scheduler', action='store_true', default=True,
                        help='use scheduler or not')
    parser.add_argument('--lr-decay-step', type=int, default=30, metavar='STEP',
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
    
    parser.add_argument('--load-filename', default='VGG11_checkpoint.pt', metavar='LF',
                        help='filename of load model')
    parser.add_argument('--train', action='store_true', default=False,
                        help='train the model')
    parser.add_argument('--load-model-type', type=int, default=1, metavar='LD',
                        help='load mode type (0:no 1:VGG11 2:BNNVGG11')
    parser.add_argument('--half-float', action='store_true', default=True,
                        help='For use 16b float')
    parser.add_argument('--net', type=int, default=0, metavar='NET',
                        help='use which NN model (0:VGG11 1:VGG16 2:BNNVGG11)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA training')
    parser.add_argument('--cuda-use-num', type=int, default=2, metavar='CUDA',
                        help='use which cuda (choice: 0-3)')
 
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:"+str(args.cuda_use_num) if use_cuda else "cpu")
    print(f"device: {device}")

    train_kwargs = {'batch_size': args.train_batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        train_cuda_kwargs = {'num_workers': 1, 'pin_memory': False, 'shuffle': True}
        test_cuda_kwargs = {'num_workers': 1, 'pin_memory': False, 'shuffle': False}
        train_kwargs.update(train_cuda_kwargs)
        test_kwargs.update(test_cuda_kwargs)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_datasets = \
        torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)

    # extra_train_datasets_for_valid = \
    #     torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_test)
    test_datasets = \
        torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)

    train_loader, test_loader, _ = split_data_loader(train_datasets, test_datasets, train_kwargs, test_kwargs)

    # labels_map = {
    #     0: "plane",
    #     1: "car",
    #     2: "bird",
    #     3: "cat",
    #     4: "deer",
    #     5: "dog",
    #     6: "frog",
    #     7: "horse",
    #     8: "ship",
    #     9: "truck",
    # }

    # show_data_img(labels_map, train_datasets)
    if args.net == 0:
        model = VGG11().to(device)
    # elif args.net == 1:
    #     model = Fc5Mnist().to(device)
    elif args.net == 2:
        model = BNNVGG11().to(device)
    # elif args.net == 3:
    #     model = LeNetMnist().to(device)
    else:
        raise Exception('undefined net: ' + str(args.net))
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

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
        # train_loss, valid_loss = train_model(model, device, train_loader, valid_loader, criterion, optimizer,
        #                                      args.epochs, filename=model_save_filename, score_type='loss',
        #                                      scheduler=scheduler)
        _, _, _ = train_model(model, device, train_loader, test_loader, criterion, optimizer, args.epochs,
                              model_filename=model_save_filename, score_type='accuracy', patience=100, scheduler=scheduler)

        # print("retrain use full data")
        #
        # train_loss, valid_loss = train_model(model, device, full_train_loader, valid_loader, criterion, optimizer,
        #                                      args.epochs, filename=model_save_filename)

    criterion = nn.CrossEntropyLoss(reduction='sum')
    test_model(model, device, test_loader, criterion)


if __name__ == "__main__":
    main()
