#!/usr/bin/env python
# coding: utf-8
import math
import torch
import torch.nn as nn
import numpy as np
import sys
import torch.nn.functional as F
from layers import GraphConvolution, GCN_layer
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

current_module = sys.modules[__name__]

debug = False

class IGNNNet(nn.Module):
    def __init__(
            self,
            num_classes,
            input_size,
            F1=10,
            F2=30,
            F3=50,  # F2 + nb_layers1 * growth_rate1
            F4=70,
            F5=140,
            growth_rate1=10,
            nb_layers1=2,
            out_graph=100,
            drop_out=0.5,
            window=[0.25, 0.125, 0.08],
            sampling_rate=250
    ):
        super(IGNNNet, self).__init__()
        self.channel_num = input_size[1]
        self.Tception1 = Conv2dWithConstraint(input_size[0], F1,
                                              (1, int(window[0] * sampling_rate)),
                                              padding='same', bias=False, max_norm=2)
        self.Tception2 = Conv2dWithConstraint(input_size[0], F1,
                                              (1, int(window[1] * sampling_rate)),
                                              padding='same', bias=False, max_norm=2)
        self.Tception3 = Conv2dWithConstraint(input_size[0], F1,
                                              (1, int(window[2] * sampling_rate)),
                                              padding='same', bias=False, max_norm=2)
        self.BN_t = nn.BatchNorm2d(F2)
        self.D_Block1 = DenseBlock(block=BasicBlock, nb_layers=nb_layers1, in_planes=F2, growth_rate=growth_rate1,
                                   kernel_size=63, pool_size=2, pool_stride=2, dropRate=0.5)
        self.D_Block2 = DenseBlock(block=BasicBlock, nb_layers=nb_layers1, in_planes=F3, growth_rate=growth_rate1,
                                   kernel_size=32, pool_size=2, pool_stride=2, dropRate=0.5)
        self.DWConv = Conv2dWithConstraint(F4, F4, (1, 1), stride=1, bias=False, padding='same', max_norm=1)
        # self.size = self.get_size_flatten(input_size)
        # self.local_filter_weight = nn.Parameter(torch.FloatTensor(self.channel_num, self.size[-1]),
        #                                         requires_grad=True)
        # nn.init.xavier_uniform_(self.local_filter_weight)
        # self.local_filter_bias = nn.Parameter(torch.zeros((1, self.channel_num, 1), dtype=torch.float32),
        #                                       requires_grad=True)
        self.global_adj = nn.Parameter(torch.FloatTensor(self.channel_num, self.channel_num), requires_grad=True)
        nn.init.xavier_uniform_(self.global_adj)
        self.BN_1 = nn.BatchNorm2d(F4)
        self.BN_2 = nn.BatchNorm2d(F4)
        self.GCN1 = GraphConvolution(F4, F4)
        self.GCN2 = GraphConvolution(F4, F4)
        self.Dropout1 = nn.Dropout(p=drop_out)
        self.Dropout2 = nn.Dropout(p=drop_out)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # LinearWithConstraint(int(self.channel_num * out_graph), num_classes, max_norm=0.25)
            LinearWithConstraint(980, num_classes, max_norm=0.25)
        )
        self.DWConv1 = Conv2dWithConstraint(F4, F4, (8, 1), padding=0, bias=False, max_norm=1,
                                           groups=F4)
        self.DWConv2 = Conv2dWithConstraint(F4, F4, (8, 1), padding=0, bias=False, max_norm=1,
                                           groups=F4)
        self.midblock1 = nn.Sequential(
            nn.BatchNorm2d(F4),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=(1, 4)),
            nn.Dropout(p=drop_out)
        )
        self.midblock2 = nn.Sequential(
            nn.BatchNorm2d(F4),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=(1, 4)),
            nn.Dropout(p=drop_out)
        )
        self.block1 = nn.Sequential(
            Conv2dWithConstraint(F4, F4, (1, 8), padding='same', bias=False, groups=F4),
            nn.Conv2d(F4, F4, (1, 1), stride=1, bias=False, padding='same'),
            nn.BatchNorm2d(F4),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=(1, 8)),
            nn.Dropout(p=drop_out)
        )
        self.block2 = nn.Sequential(
            Conv2dWithConstraint(F4, F4, (1, 8), padding='same', bias=False, groups=F4),
            nn.Conv2d(F4, F4, (1, 1), stride=1, bias=False, padding='same'),
            nn.BatchNorm2d(F4),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=(1, 8)),
            nn.Dropout(p=drop_out)
        )

    def get_size_flatten(self, input_size):
        # input_size: filter x channel x data point
        x = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        x1 = self.Tception1(x)
        x2 = self.Tception2(x)
        x3 = self.Tception3(x)
        out = torch.cat((x1, x2, x3), dim=1)
        out = self.BN_t(out)
        out = self.D_Block1(out)
        out = out.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        size = out.size()
        return size

    def forward(self, x):
        x1 = self.Tception1(x)
        x2 = self.Tception2(x)
        x3 = self.Tception3(x)
        out = torch.cat((x1, x2, x3), dim=1)
        out = self.BN_t(out)
        out = self.D_Block1(out)
        out = self.D_Block2(out)

        adj = self.get_adj(out)
        out1 = self.GCN1(out, adj)
        out1 = self.BN_1(out1)
        out1 = self.Dropout1(out1)

        channel_8 = [10, 14, 4, 12, 7, 20, 13, 17]
        out1 = out1[:, :, np.array(channel_8), :]
        out = out[:, :, np.array(channel_8), :]
        out1 = self.DWConv1(out1)
        out1 = self.midblock1(out1)
        out1 = self.block1(out1)
        out = self.DWConv2(out)
        out = self.midblock2(out)
        out = self.block2(out)
        out = torch.cat([out, out1], dim=1)

        out = self.classifier(out)
        return out

    def local_filter_fun(self, x, w):
        w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
        x = F.relu(torch.mul(x, w) - self.local_filter_bias)  #mul为逐元素相乘
        return x

    def get_adj(self, x, self_loop=True):
        # x: b, node, feature
        batch, filter, channel, samples = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = torch.permute(x,(0,2,1,3))
        x = x.reshape(batch, channel, -1)
        adj = self.self_similarity(x)   # b, n, n
        num_nodes = adj.shape[-1]
        adj = F.relu(adj * (self.global_adj * self.global_adj.transpose(1, 0)))
        # adj = F.relu(adj)
        if self_loop:
            adj = adj + torch.eye(num_nodes).to(DEVICE)

        adj = adj.cpu().detach().numpy()
        # 指定不考虑的下标
        skip_indices = [10, 14, 4, 12, 7, 20, 13, 17]
        # 复制以防修改原始矩阵
        modified_zzz = np.copy(adj)
        reduce_chan = 6
        for i in np.arange(batch):
            for j in skip_indices:
                # 获取第三行和第三列中不在排除列表中的最小的9个数的索引
                row_indices = [i for i in range(62) if i not in skip_indices]
                col_indices = [i for i in range(62) if i not in skip_indices]

                min_row_indices = np.argpartition(modified_zzz[i, j, row_indices], reduce_chan)[:reduce_chan]
                min_col_indices = np.argpartition(modified_zzz[i, col_indices, j], reduce_chan)[:reduce_chan]
                # modified_zzz[i, j, np.array(row_indices)] = 0
                # modified_zzz[i, np.array(col_indices), j] = 0
                # 将对应的值设为 0
                modified_zzz[i, j, np.array(row_indices)[min_row_indices]] = 0
                modified_zzz[i, np.array(col_indices)[min_col_indices], j] = 0
        adj = torch.tensor(modified_zzz)
        device = torch.device('cuda')
        adj = adj.to(device)

        rowsum = torch.sum(adj, dim=-1)
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj

    def self_similarity(self, x):
        # x: b, node, feature
        x_ = x.permute(0, 2, 1)
        s = torch.bmm(x, x_)

        # 按照批次和通道计算皮尔逊相关系数
        correlation_matrices = torch.zeros(x.shape[0],x.shape[1],x.shape[1])
        for batch in range(x.shape[0]):
            batch_data = x[batch]
            correlation_matrices[batch]= torch.corrcoef(batch_data)
        return correlation_matrices.to(torch.device("cuda:0"))
        # return s


class IGNNNet_stu(nn.Module):
    def __init__(
            self,
            num_classes,
            input_size,
            F1=10,
            F2=30,
            F3=50,  # F2 + nb_layers1 * growth_rate1
            F4=70,
            growth_rate1=10,
            nb_layers1=2,
            out_graph=100,
            drop_out=0.5,
            window=[0.25, 0.125, 0.08],
            sampling_rate=250
    ):
        super(IGNNNet_stu, self).__init__()
        self.channel_num = input_size[1]
        self.Tception1 = Conv2dWithConstraint(input_size[0] , F1,
                                              (1, int(window[0] * sampling_rate)),
                                              padding='same', bias=False, max_norm=2)
        self.Tception2 = Conv2dWithConstraint(input_size[0] , F1,
                                              (1, int(window[1] * sampling_rate)),
                                              padding='same', bias=False, max_norm=2)
        self.Tception3 = Conv2dWithConstraint(input_size[0] , F1,
                                              (1, int(window[2] * sampling_rate)),
                                              padding='same', bias=False, max_norm=2)
        self.BN_t = nn.BatchNorm2d(F2)
        self.D_Block1 = DenseBlock(block=BasicBlock, nb_layers=nb_layers1, in_planes=F2, growth_rate=growth_rate1,
                                   kernel_size=63, pool_size=2, pool_stride=2, dropRate=0.5)
        self.D_Block2 = DenseBlock(block=BasicBlock, nb_layers=nb_layers1, in_planes=F3, growth_rate=growth_rate1,
                                   kernel_size=32, pool_size=2, pool_stride=2, dropRate=0.5)
        self.DWConv = Conv2dWithConstraint(F4, F4, (1, 1), stride=1, bias=False, padding='same', max_norm=1)
        # self.local_filter_weight = nn.Parameter(torch.FloatTensor(self.channel_num, self.size[-1]),
        #                                         requires_grad=True)
        # nn.init.xavier_uniform_(self.local_filter_weight)
        # self.local_filter_bias = nn.Parameter(torch.zeros((1, self.channel_num, 1), dtype=torch.float32),
        #                                       requires_grad=True)
        self.global_adj = nn.Parameter(torch.FloatTensor(self.channel_num, self.channel_num), requires_grad=True)
        nn.init.xavier_uniform_(self.global_adj)
        self.BN_1 = nn.BatchNorm2d(F4)
        self.BN_2 = nn.BatchNorm2d(F4)
        self.GCN1 = GraphConvolution(F4, F4)
        self.GCN2 = GraphConvolution(F4, F4)
        self.Dropout1 = nn.Dropout(p=drop_out)
        self.Dropout2 = nn.Dropout(p=drop_out)

        # self.size = self.get_size_flatten(input_size)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # LinearWithConstraint(int(self.channel_num * out_graph), num_classes, max_norm=0.25)
            # LinearWithConstraint(980, num_classes, max_norm=0.25)   # 980或14700
            LinearWithConstraint(980, num_classes, max_norm=0.25)  # 980或13720
        )
        self.DWConv1 = Conv2dWithConstraint(F4, F4, (8, 1), padding=0, bias=False, max_norm=1,
                                            groups=F4)
        self.DWConv2 = Conv2dWithConstraint(F4, F4, (8, 1), padding=0, bias=False, max_norm=1,
                                            groups=F4)
        self.midblock1 = nn.Sequential(
            nn.BatchNorm2d(F4),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=(1, 4)),
            nn.Dropout(p=drop_out)
        )
        self.midblock2 = nn.Sequential(
            nn.BatchNorm2d(F4),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=(1, 4)),
            nn.Dropout(p=drop_out)
        )
        self.block1 = nn.Sequential(
            Conv2dWithConstraint(F4, F4, (1, 8), padding='same', bias=False, groups=F4),
            nn.Conv2d(F4, F4, (1, 1), stride=1, bias=False, padding='same'),
            nn.BatchNorm2d(F4),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=(1, 8)),
            nn.Dropout(p=drop_out)
        )
        self.block2 = nn.Sequential(
            Conv2dWithConstraint(F4, F4, (1, 8), padding='same', bias=False, groups=F4),
            nn.Conv2d(F4, F4, (1, 1), stride=1, bias=False, padding='same'),
            nn.BatchNorm2d(F4),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=(1, 8)),
            nn.Dropout(p=drop_out)
        )

    def get_size_flatten(self, input_size):
        # input_size: filter x channel x data point
        x = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        x1 = self.Tception1(x)
        x2 = self.Tception2(x)
        x3 = self.Tception3(x)
        out = torch.cat((x1, x2, x3), dim=1)
        out = self.BN_t(out)
        out = self.D_Block1(out)
        out = self.D_Block2(out)

        adj = self.get_adj(out)
        out1 = self.GCN1(out, adj)
        out1 = self.BN_1(out1)
        out1 = self.Dropout1(out1)

        out1 = self.DWConv1(out1)
        out1 = self.midblock1(out1)
        out1 = self.block1(out1)
        out = self.DWConv2(out)
        out = self.midblock2(out)
        out = self.block2(out)
        out = torch.cat([out, out1], dim=1)

        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        size = out.size()
        return size

    def forward(self, x):
        x1 = self.Tception1(x)
        x2 = self.Tception2(x)
        x3 = self.Tception3(x)
        out = torch.cat((x1, x2, x3), dim=1)
        out = self.BN_t(out)
        out = self.D_Block1(out)
        out = self.D_Block2(out)

        adj = self.get_adj(out)
        out1 = self.GCN1(out, adj)
        out1 = self.BN_1(out1)
        out1 = self.Dropout1(out1)

        out1 = self.DWConv1(out1)
        out1 = self.midblock1(out1)
        out1 = self.block1(out1)

        out = self.DWConv2(out)
        out = self.midblock2(out)
        out = self.block2(out)
        out = torch.cat([out, out1], dim=1)

        out = self.classifier(out)
        return out

    def get_adj(self, x, self_loop=True):
        # x: b, node, feature
        batch, filter, channel, samples = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = torch.permute(x, (0, 2, 1, 3))
        x = x.reshape(batch, channel, -1)
        adj = self.self_similarity(x)  # b, n, n
        num_nodes = adj.shape[-1]
        adj = F.relu(adj * (self.global_adj * self.global_adj.transpose(1, 0)))
        # adj = F.relu(adj)
        if self_loop:
            adj = adj + torch.eye(num_nodes).to(DEVICE)
        rowsum = torch.sum(adj, dim=-1)
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj

    def self_similarity(self, x):
        # x: b, node, feature
        x_ = x.permute(0, 2, 1)
        s = torch.bmm(x, x_)

        # 按照批次和通道计算皮尔逊相关系数
        correlation_matrices = torch.zeros(x.shape[0], x.shape[1], x.shape[1])
        for batch in range(x.shape[0]):
            batch_data = x[batch]
            correlation_matrices[batch] = torch.corrcoef(batch_data)
        return correlation_matrices.to(torch.device("cuda:0"))


class Adjacent_create():
    def __init__(self, channel):
        # chan_in_area: a list of the number of channels within each area
        self.channel = channel
        self.aj = np.ones((channel, channel))
    def datanorm(self,x):
        for i in range(np.shape(x)[0]):
            x[i] = (x[i] - np.min(x[i])) / (np.max(x[i]) - np.min(x[i]))
        return x
    def normalize_adj(self,adj):
        d = np.diag(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d)
        return a_norm
    def forward(self):
        A = self.normalize_adj(self.aj)
        A = np.float32(A)
        A = torch.from_numpy(A)
        A = A.cuda()
        return A[0,:].view(1, -1)

class Aggregator():

    def __init__(self, idx_area):
        # chan_in_area: a list of the number of channels within each area
        self.chan_in_area = idx_area
        self.idx = self.get_idx(idx_area)
        self.area = len(idx_area)

    def forward(self, x):
        # x: batch x channel x data
        data = []
        for i, area in enumerate(range(self.area)):
            if i < self.area - 1:
                data.append(self.aggr_fun(x[:, self.idx[i]:self.idx[i + 1], :], dim=1))
            else:
                data.append(self.aggr_fun(x[:, self.idx[i]:, :], dim=1))
        return torch.stack(data, dim=1)

    def get_idx(self, chan_in_area):
        idx = [0] + chan_in_area
        idx_ = [0]
        for i in idx:
            idx_.append(idx_[-1] + i)
        return idx_[1:]

    def aggr_fun(self, x, dim):
        # return torch.max(x, dim=dim).values
        return torch.mean(x, dim=dim)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, dropRate=0.5):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2dWithConstraint(in_planes, out_planes, (1, kernel_size), padding='same', bias=False,
                                          max_norm=1)
        self.elu = nn.ELU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.dropout = nn.Dropout(p=dropRate)

    def forward(self, x):
        out = self.elu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, block, nb_layers, in_planes, growth_rate, kernel_size, pool_size, pool_stride, dropRate=0.5):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, nb_layers, in_planes, growth_rate, kernel_size, dropRate)
        self.bn = nn.BatchNorm2d(in_planes + nb_layers * growth_rate)
        self.avg_pooling = nn.AvgPool2d((1, pool_size), (1, pool_stride))

    def _make_layer(self, block, nb_layers, in_planes, growth_rate, kernel_size, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes=in_planes + i * growth_rate, out_planes=growth_rate,
                                kernel_size=kernel_size, dropRate=dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)
        x = self.bn(x)
        x = self.avg_pooling(x)
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
