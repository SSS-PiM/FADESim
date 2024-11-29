import torch.utils.data
import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from fixedPoint import optim as fpOptim
from trainCommon import split_data_loader, train_model, test_model, create_layer_weight_bit_width_list, \
    load_float_weight_for_fixed_point
from mnist_model import PimFcMnist, FixedPointSimpleConvNet, FcMnist, ConvMnist, PimConvMnist, PimDeepFcMnist


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.0125, metavar='LR',
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
    parser.add_argument('--load-model-type', type=int, default=1, metavar='LD',
                        help='load mode type (0:no 1:float point model 2:fixed point model')
    parser.add_argument('--load-filename', default='FcMnist_checkpoint.pt', metavar='LF',
                        help='filename of load model')
    parser.add_argument('--train', action='store_true', default=False,
                        help='train the model')
    parser.add_argument('--fixed-point', action='store_true', default=True,
                        help='For use fixed point')
    parser.add_argument('--half-float', action='store_true', default=True,
                        help='For use 16b float')
    parser.add_argument('--net', type=int, default=1, metavar='NET',
                        help='use which model (0:conv 1:fc 2:deepFc)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA training')
    parser.add_argument('--cuda_use_num', type=int, default=1, metavar='CUDA',
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

    if args.fixed_point:
        if args.net == 0:
            # model = PimConvMnist(args.train_batch_size, device=device).to(device)
            model = PimConvMnist().to(device)
            # model = FixedPointSimpleConvNet(args.train_batch_size, device=device).to(device)
            # model.double()
        elif args.net == 1:
            model = PimFcMnist().to(device)
        elif args.net == 2:
            model = PimDeepFcMnist().to(device)
        else:
            raise Exception('undefined net: ' + str(args.net))

        bit_width_list = create_layer_weight_bit_width_list(model)

        optimizer = fpOptim.SGD(model.named_parameters(), bit_width_list, lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay,
                                run_mode=fpOptim.OptimMode.full_fix)
    else:
        if args.net == 0:
            model = ConvMnist().to(device)
        elif args.net == 1:
            model = FcMnist().to(device)
        else:
            raise Exception('undefined net: ' + str(args.net))

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    print(model)

    model_name = type(model).__name__

    model_save_filename = args.model_dir + '/' + model_name + '_checkpoint.pt'

    model_load_filename = args.model_dir + '/' + args.load_filename

    try:
        if (args.load_model_type == 1 and not args.fixed_point) or (args.load_model_type == 2 and args.fixed_point):
            para = torch.load(model_load_filename)
            model.load_state_dict(para)
        elif args.load_model_type == 1 and args.fixed_point:
            load_float_weight_for_fixed_point(model_load_filename, model)
        elif args.load_model_type:
            print("unsupported load type, load_model_type: " + str(args.load_model_type)
                  + ", but fixed point: " + str(args.fixed_point))
    except Exception as e:
        print(e)

    if args.scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.gamma)
    else:
        scheduler = None

    if args.train:
        criterion = nn.CrossEntropyLoss()
        _, _, _ = train_model(model, device, train_loader, test_loader, criterion, optimizer, args.epochs,
                              model_filename=model_save_filename, score_type='accuracy', scheduler=scheduler)

    criterion = nn.CrossEntropyLoss(reduction='sum')
    test_model(model, device, test_loader, criterion)


if __name__ == '__main__':
    main()
