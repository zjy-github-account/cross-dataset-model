#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import sys

current_module = sys.modules[__name__]

debug = False
# %% ShallowConvNet
class ShallowConvNet(nn.Module):
    # input size:(Batchsize, 1, Channel, Samples)
    def __init__(
            self,
            num_classes,
            input_size,
            F1 = 40,
            K1 = 25,
            P1_T=75,
            P1_S=15,
            drop_out=0.5
    ):
        super(ShallowConvNet, self).__init__()
        channel_num = input_size[1]
        self.net = nn.Sequential(
            Conv2dWithConstraint(1, F1, (1, K1), max_norm=2),
            Conv2dWithConstraint(F1, F1, (channel_num, 1), bias=False, max_norm=2),
            nn.BatchNorm2d(F1),
            ActSquare(),
            nn.AvgPool2d((1, P1_T), (1, P1_S)),
            ActLog(),
            nn.Dropout(drop_out)
        )
        self.size = self.get_size_flatten(input_size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(self.size[0], num_classes, max_norm=0.5)
        )

    def get_size_flatten(self, input_size):
        # input_size: filter x channel x data point
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        data = self.net(data)
        data = torch.reshape(data, (-1,))
        size = data.size()
        return size

    def forward(self, x):
        x = self.net(x)
        x = self.classifier(x)
        return x
# %% EEGNet8_2
class EEGNet8_2(nn.Module):
    # input size:(Batchsize, 1, Channel, Samples)
    def __init__(
            self,
            num_classes,
            input_size,
            F1 = 8,
            F2 = 16,
            K1 = 64,
            K2 = 32,
            P1=8,
            P2=16,
            drop_out=0.5
    ):
        super(EEGNet8_2, self).__init__()
        channel_num = input_size[1]
        self.TConv = Conv2dWithConstraint(input_size[0], F1,  (1, K1), padding='same', bias=False)
        self.BN_t = nn.BatchNorm2d(F1)
        self.DWConv = Conv2dWithConstraint(F1, F2, (channel_num, 1), padding=0, bias=False, max_norm=1,
                                             groups=F1)
        self.midblock =  nn.Sequential(
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, P1), stride=P1),
            nn.Dropout(p=drop_out)
        )
        self.block2 = nn.Sequential(
            Conv2dWithConstraint(F2, F2, (1, K2), padding='same', bias=False, groups=F2),
            nn.Conv2d(F2, F2, (1, 1), stride=1, bias=False, padding='same'),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, P2), stride=P2),
            nn.Dropout(p=drop_out)
        )
        self.size = self.get_size_flatten(input_size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(self.size[0], num_classes, max_norm=0.25)
        )

    def get_size_flatten(self, input_size):
        # input_size: filter x channel x data point
        x = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        x = self.TConv(x)
        x = self.BN_t(x)
        x = self.DWConv(x)
        x = self.midblock(x)
        x = self.block2(x)
        x = torch.reshape(x, (-1,))
        size = x.size()
        return size

    def forward(self, x):
        x = self.TConv(x)
        x = self.BN_t(x)
        x = self.DWConv(x)
        x = self.midblock(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x

class DeepConvNet(nn.Module):
    # input size:(Batchsize, 1, Channel, Samples)
    def __init__(
            self,
            num_classes,
            input_size,
            F1 = 25,
            F2 = 50,
            F3 = 100,
            F4 = 200,
            K1 = 10,
            P1=3,
            drop_out=0.5
    ):
        super(DeepConvNet, self).__init__()
        channel_num = input_size[1]
        self.net1 = nn.Sequential(
            Conv2dWithConstraint(input_size[0], F1, (1, K1), max_norm=2),
            Conv2dWithConstraint(F1, F1, (channel_num, 1), max_norm=2),
            nn.BatchNorm2d(F1),
            nn.ELU(),
            nn.MaxPool2d((1, P1), stride=P1),
            nn.Dropout(p=drop_out)
        )
        self.net2 = nn.Sequential(
            Conv2dWithConstraint(F1, F2, (1, K1), max_norm=2),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.MaxPool2d((1, P1), stride=P1),
            nn.Dropout(p=drop_out)
        )
        self.net3 = nn.Sequential(
            Conv2dWithConstraint(F2, F3, (1, K1), max_norm=2),
            nn.BatchNorm2d(F3),
            nn.ELU(),
            nn.MaxPool2d((1, P1), stride=P1),
            nn.Dropout(p=drop_out)
        )
        self.net4 = nn.Sequential(
            Conv2dWithConstraint(F3, F4, (1, K1), max_norm=2),
            nn.BatchNorm2d(F4),
            nn.ELU(),
            nn.MaxPool2d((1, P1), stride=P1),
            nn.Dropout(p=drop_out)
        )

        self.size = self.get_size_flatten(input_size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(self.size[0], num_classes, max_norm=0.5)
        )

    def get_size_flatten(self, input_size):
        # input_size: filter x channel x data point
        x = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        x = self.net1(x)
        x = self.net2(x)
        x = self.net3(x)
        x = self.net4(x)
        x = torch.reshape(x, (-1,))
        size = x.size()
        return size

    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)
        x = self.net3(x)
        x = self.net4(x)
        x = self.classifier(x)
        return x

class TDenseNet(nn.Module):
    '''
    input size:(Batchsize, 1, Channel, Samples), TDenseNet has mutiple inputs based on the brain rhythm.
    In the forward function: the input could be(x1,x2,x3)
    Overall band(single input) or bands fusion(four inputs: 4-8,7-13,12-32,overall) perform better than ShallowConvNet,
    DeepConvNet, EEGNet or FBCNet on first session in KU2019datasets.
    '''
    def __init__(
            self,
            num_classes,
            input_size,
            F1 = 40,
            F2 = 54,  # F1 + Dense_layer1 * growth_rate1
            F3 = 68,  # F1 + 2 * Dense_layer1 * growth_rate1
            F4 = 20,
            Dense_layer1 = 2,
            growth_rate1 = 7,
            K1 = 64,
            K2 = 20,
            P1 = 5,
            P2 = 2,
            drop_out=0.5
    ):
        super(TDenseNet, self).__init__()
        channel_num = input_size[1]
        self.net1 = nn.Sequential(
            Conv2dWithConstraint(input_size[0], F1, (1, K1), max_norm=2, padding='same', bias=False),
            nn.BatchNorm2d(F1),
            Conv2dWithConstraint(F1, F1, (channel_num, 1), padding=0, bias=False, max_norm=1,
                                               groups=F1),
            nn.ELU(),
            nn.BatchNorm2d(F1),
            DenseBlock(block=BasicBlock, nb_layers=Dense_layer1, in_planes=F1, growth_rate=growth_rate1, kernel_size=K1,
                       pool_size=P1, pool_stride=P1, dropRate=drop_out),
            DenseBlock(block=BasicBlock, nb_layers=Dense_layer1, in_planes=F2, growth_rate=growth_rate1, kernel_size=K2,
                       pool_size=P1, pool_stride=P1, dropRate=drop_out),
            Conv2dWithConstraint(F3, F4, (1, 1), max_norm=2, padding='same', bias=False),
            nn.ELU(),
            nn.BatchNorm2d(F4),
            nn.AvgPool2d((1, P2), stride=P2)
        )
        self.net2 = nn.Sequential(
            Conv2dWithConstraint(input_size[0], F1, (1, K1), max_norm=2, padding='same', bias=False),
            nn.BatchNorm2d(F1),
            Conv2dWithConstraint(F1, F1, (channel_num, 1), padding=0, bias=False, max_norm=1,
                                 groups=F1),
            nn.ELU(),
            nn.BatchNorm2d(F1),
            DenseBlock(block=BasicBlock, nb_layers=Dense_layer1, in_planes=F1, growth_rate=growth_rate1, kernel_size=K1,
                       pool_size=P1, pool_stride=P1, dropRate=drop_out),
            DenseBlock(block=BasicBlock, nb_layers=Dense_layer1, in_planes=F2, growth_rate=growth_rate1, kernel_size=K2,
                       pool_size=P1, pool_stride=P1, dropRate=drop_out),
            Conv2dWithConstraint(F3, F4, (1, 1), max_norm=2, padding='same', bias=False),
            nn.ELU(),
            nn.BatchNorm2d(F4),
            nn.AvgPool2d((1, P2), stride=P2)
        )
        self.net3 = nn.Sequential(
            Conv2dWithConstraint(input_size[0], F1, (1, K1), max_norm=2, padding='same', bias=False),
            nn.BatchNorm2d(F1),
            Conv2dWithConstraint(F1, F1, (channel_num, 1), padding=0, bias=False, max_norm=1,
                                 groups=F1),
            nn.ELU(),
            nn.BatchNorm2d(F1),
            DenseBlock(block=BasicBlock, nb_layers=Dense_layer1, in_planes=F1, growth_rate=growth_rate1, kernel_size=K1,
                       pool_size=P1, pool_stride=P1, dropRate=drop_out),
            DenseBlock(block=BasicBlock, nb_layers=Dense_layer1, in_planes=F2, growth_rate=growth_rate1, kernel_size=K2,
                       pool_size=P1, pool_stride=P1, dropRate=drop_out),
            Conv2dWithConstraint(F3, F4, (1, 1), max_norm=2, padding='same', bias=False),
            nn.ELU(),
            nn.BatchNorm2d(F4),
            nn.AvgPool2d((1, P2), stride=P2)
        )
        self.size = self.get_size_flatten(input_size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(self.size[0], num_classes, max_norm=0.25)
        )

    def get_size_flatten(self, input_size):
        # input_size: filter x channel x data point
        x = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        x1 = self.net1(x)
        x2 = self.net2(x)
        x3 = self.net3(x)
        x4 = self.net4(x)
        x = torch.cat([x1, x2, x3, x4], 1)
        x = torch.reshape(x, (-1,))
        size = x.size()
        return size

    def forward(self, x1,x2,x3,x4):
        x1 = self.net(x1)
        x2 = self.net(x2)
        x3 = self.net(x3)
        x4 = self.net(x4)
        x = torch.cat([x1, x2, x3, x4], 1)
        x = self.classifier(x)
        return x

class ActSquare(nn.Module):
    def __init__(self):
        super(ActSquare, self).__init__()
        pass

    def forward(self, x):
        return torch.square(x)

class ActLog(nn.Module):
    def __init__(self, eps=1e-06):
        super(ActLog, self).__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

# %% Support classes for FBNet Implementation
class VarLayer(nn.Module):
    '''
    The variance layer: calculates the variance of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(VarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.var(dim=self.dim, keepdim=True)

class StdLayer(nn.Module):
    '''
    The standard deviation layer: calculates the std of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(StdLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.std(dim=self.dim, keepdim=True)

class LogVarLayer(nn.Module):
    '''
    The log variance layer: calculates the log variance of the data along given 'dim'
    (natural logarithm)
    '''
    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(x.var(dim=self.dim, keepdim=True))

class MeanLayer(nn.Module):
    '''
    The mean layer: calculates the mean of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(MeanLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim=self.dim, keepdim=True)

class MaxLayer(nn.Module):
    '''
    The max layer: calculates the max of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(MaxLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        ma, ima = x.max(dim=self.dim, keepdim=True)
        return ma

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, dropRate=0.5):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2dWithConstraint(in_planes, out_planes, (1, kernel_size), padding='same', bias=False,max_norm=1)
        self.elu = nn.ELU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.dropout = nn.Dropout(p=dropRate)
    def forward(self, x):
        out = self.bn1(self.elu(self.conv1(x)))
        out = self.dropout(out)
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, block, nb_layers, in_planes, growth_rate,kernel_size,pool_size,pool_stride, dropRate=0.5):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, nb_layers, in_planes, growth_rate, kernel_size, dropRate)
        self.bn = nn.BatchNorm2d(in_planes+nb_layers*growth_rate)
        self.avg_pooling = nn.AvgPool2d((1,pool_size),(1, pool_stride))
    def _make_layer(self, block, nb_layers, in_planes, growth_rate, kernel_size, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes=in_planes+i*growth_rate, out_planes=growth_rate,
                                kernel_size=kernel_size, dropRate=dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.layer(x)
        x = self.bn(x)
        x = self.avg_pooling(x)
        return x