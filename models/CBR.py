"""Defines the neural network and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU() #if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class CBR_LargeW(nn.Module):
    def __init__(self):
        super(CBR_LargeW, self).__init__()
        self.conv1 = Conv(c1=3, c2=64, k=7, s=1)
        self.m1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = Conv(c1=64, c2=128, k=7, s=1)
        self.m2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = Conv(c1=128, c2=256, k=7, s=1)
        self.m3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv4 = Conv(c1=256, c2=512, k=7, s=1)
        self.m4 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, 5)

    def forward(self, x):
        x = self.m1(self.conv1(x))
        x = self.m2(self.conv2(x))
        x = self.m3(self.conv3(x))
        x = self.m4(self.conv4(x))
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class CBR_LargeT(nn.Module):
    def __init__(self):
        super(CBR_LargeT, self).__init__()
        self.conv1 = Conv(c1=3, c2=32, k=7, s=1)
        self.m1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = Conv(c1=32, c2=64, k=7, s=1)
        self.m2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = Conv(c1=64, c2=128, k=7, s=1)
        self.m3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv4 = Conv(c1=128, c2=256, k=7, s=1)
        self.m4 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv5 = Conv(c1=256, c2=512, k=7, s=1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, 5)


    def forward(self, x):
        x = self.m1(self.conv1(x))
        x = self.m2(self.conv2(x))
        x = self.m3(self.conv3(x))
        x = self.m4(self.conv4(x))
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class CBR_Small(nn.Module):
    def __init__(self):
        super(CBR_Small, self).__init__()
        self.conv1 = Conv(c1=3, c2=32, k=7, s=1)
        self.m1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = Conv(c1=32, c2=64, k=7, s=1)
        self.m2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = Conv(c1=64, c2=128, k=7, s=1)
        self.m3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv4 = Conv(c1=128, c2=256, k=7, s=1)
        self.m4 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(256, 5)

    def forward(self, x):
        x = self.m1(self.conv1(x))
        x = self.m2(self.conv2(x))
        x = self.m3(self.conv3(x))
        x = self.m4(self.conv4(x))
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class CBR_Tiny(nn.Module):
    def __init__(self):
        super(CBR_Tiny, self).__init__()
        self.conv1 = Conv(c1=3, c2=32, k=5, s=1)
        self.m1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = Conv(c1=32, c2=64, k=5, s=1)
        self.m2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = Conv(c1=64, c2=128, k=5, s=1)
        self.m3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv4 = Conv(c1=128, c2=256, k=5, s=1)
        self.m4 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(256, 5)

    def forward(self, x):
        x = self.m1(self.conv1(x))
        x = self.m2(self.conv2(x))
        x = self.m3(self.conv3(x))
        x = self.m4(self.conv4(x))
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

