import fixedPoint as fp
import torch.nn as nn
import torch.nn.functional as F
import fixedPoint.nn as fpnn


class FcMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 32, bias=False)
        self.fc2 = nn.Linear(32, 10, bias=False)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


class PimFcMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.quan = fpnn.Quan(bit_width=6)
        self.fc1 = fpnn.Linear(784, 32, output_bit_width=6, weight_bit_width=4, bias=False)
        self.relu = fpnn.ReLU()
        self.fc2 = fpnn.Linear(32, 10, output_bit_width=6, weight_bit_width=4, bias=False)
        self.dequan = fpnn.DeQuan(back_bit_width=6)

    def forward(self, x):
        x = self.flatten(x)
        x, x_p = self.quan(x)
        x, x_p = self.fc1(x, x_p)
        x, x_p = self.relu(x, x_p)
        x, x_p = self.fc2(x, x_p)
        x = self.dequan(x, x_p)

        return x


class PimDeepFcMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.deepFc = fpnn.MulInputSequential(nn.Flatten(),
                                              fpnn.Quan(),
                                              fpnn.Linear(784, 128),
                                              fpnn.ReLU(),
                                              fpnn.Linear(128, 128),
                                              fpnn.ReLU(),
                                              fpnn.Linear(128, 128),
                                              fpnn.ReLU(),
                                              fpnn.Linear(128, 128),
                                              fpnn.ReLU(),
                                              fpnn.Linear(128, 128),
                                              fpnn.ReLU(),
                                              fpnn.Linear(128, 128),
                                              fpnn.ReLU(),
                                              fpnn.Linear(128, 128),
                                              fpnn.ReLU(),
                                              fpnn.Linear(128, 128),
                                              fpnn.ReLU(),
                                              fpnn.Linear(128, 128),
                                              fpnn.ReLU(),
                                              fpnn.Linear(128, 10),
                                              fpnn.DeQuan())

    def forward(self, x):
        x = self.deepFc(x)

        return x


class ConvMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 10, (5, 5)),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                  nn.Conv2d(10, 20, (5, 5)),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                  nn.Flatten(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(4 * 4 * 20, 10)
                                  )

    def forward(self, x):
        x = self.conv(x)

        return x


class PimConvMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = fpnn.MulInputSequential(fpnn.Conv2d(1, 10, (5, 5)),
                                            nn.ReLU(),
                                            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                            fpnn.Conv2d(10, 20, (5, 5)),
                                            nn.ReLU(),
                                            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                            nn.Flatten(),
                                            fpnn.Quan(),
                                            fpnn.Dropout(p=0.2),
                                            fpnn.Linear(4 * 4 * 20, 10),
                                            fpnn.DeQuan())

    def forward(self, x):
        x = self.conv(x)

        return x


class FixedPointSimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = fpnn.MulInputSequential(fpnn.Conv2d(1, 30, (5, 5)),
                                            nn.ReLU(),
                                            nn.MaxPool2d(kernel_size=2, stride=2),
                                            nn.Flatten(),
                                            fpnn.Quan(),
                                            fpnn.Linear(12 * 12 * 30, 100),
                                            fpnn.ReLU(),
                                            fpnn.Linear(100, 10),
                                            fpnn.DeQuan())

    def forward(self, x):
        x = self.conv(x)

        return x
