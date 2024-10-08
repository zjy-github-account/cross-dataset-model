import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    simple GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight, gain=1.414)
        if bias:
            self.bias = Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        # output = torch.matmul(x, self.weight)-self.bias
        batch, filter, channel, samples = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        output = torch.permute(x, (0, 2, 3, 1))
        # output = output.reshape(-1, filter)
        # output = torch.matmul(output, self.weight)-self.bias
        # output = output.reshape(batch, channel, samples, self.out_features)
        output = output.reshape(batch, channel, -1)
        output = F.elu(torch.matmul(adj, output))
        output = output.reshape(batch, channel, samples, self.out_features)
        output = torch.permute(output, (0, 3, 1, 2))
        return output


class GCN_layer(nn.Module):

    def __init__(self, signal_shape, in_features=None, out_features=None, bias=False):
        super(GCN_layer, self).__init__()
        # input_shape=(node,timestep)
        self.W = nn.Parameter(torch.ones(signal_shape[0], signal_shape[0]), requires_grad=True)
        self.theta = nn.Parameter(torch.randn(signal_shape[1]), requires_grad=True)
        self.b = nn.Parameter(torch.zeros([1, 1, 1, signal_shape[1]]), requires_grad=True)
        self.bias = bias
        # self.params = nn.ParameterDict({
        #         'W': nn.Parameter(torch.rand(signal_shape[0], signal_shape[0]), requires_grad=True),
        #         'theta': nn.Parameter(torch.rand(signal_shape[1]), requires_grad=True)
        # })
        self.in_features = in_features
        self.out_features = out_features
        if in_features != None and out_features != None:
            self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
            torch.nn.init.xavier_uniform_(self.weight, gain=1.414)
        # if bias:
        #     self.bias = Parameter(torch.zeros((1, 1, self.out_features), dtype=torch.float32))
        # else:
        #     self.register_parameter('bias', None)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv, stdv)

    def forward(self, Adj_matrix, input_features):
        # G = torch.from_numpy(Adj_matrix).type(torch.FloatTensor)
        if self.in_features != None and self.out_features != None:
            input_features = input_features.permute(0, 2, 3, 1)
            input_features = torch.matmul(input_features, self.weight)
            input_features = input_features.permute(0, 3, 1, 2)
        aggregate = torch.einsum("ce,abed->abcd", Adj_matrix, input_features)
        # output = torch.einsum("abcd,d->abcd", aggregate, self.theta)
        output = aggregate

        if self.bias == True:
            output = output + self.b

        return output

