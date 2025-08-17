import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

class GraphConvolution(Module):
    '''
    根据传统的gcn设计的网络

    '''
    def __init__(self,
                 inputFeats, 
                 outputFeats,
                 adj,
                 normalInit = True):
        '''
        inputFeats: 输入特征的维度
        outputFeats: 输出特征的维度
        adj: 邻接矩阵
        normalInit: 是否使用正态分布初始化权重
        '''
        super(GraphConvolution,self).__init__()

        self.inputFeats = inputFeats
        self.outputFeats = outputFeats
        self.normalInit = normalInit

        # 初始化权重矩阵

        if self.normalInit:
            self.weightMatrix = Parameter(torch.Tensor(inputFeats, outputFeats))
            torch.nn.init.normal_(self.weightMatrix, mean=0.0, std=0.01)
        else:
            self.weightMatrix = Parameter(torch.Tensor(inputFeats, outputFeats))

        self.adj = adj


    def forward(self,
                HMatrix):
        '''
        adj : 邻接矩阵,已经做了归一化处理
        HMatrix : 特征矩阵
        '''

        weightMatrix = self.weightMatrix

        #计算A_hat * H^{l} * W^{l}

        support = torch.mm(self.adj, torch.mm(HMatrix, weightMatrix))

        # 原本应该有relu激活函数，但是我们将relu激活层嵌入到module模型中

        return support


if __name__ == "__main__":
    dense_A = torch.tensor([
    [1., 0., 2.],
    [0., 0., 3.],
    [4., 0., 0.]
])
    dense_H = torch.tensor([
    [1., 1., 1.],
    [0., 1., 0.],
    [0., 0., 1.]
])
    dense_W = torch.tensor([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.]
])
    gcn_layer = GraphConvolution(inputFeats=3, 
                                 outputFeats=3,
                                 adj=dense_A)
    output = gcn_layer(dense_H)
    print(output)