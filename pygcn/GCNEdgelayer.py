import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

class GCNEdgeConvolution(Module):

    def __init__(self, 
                 inputFeats, 
                 outputFeats,
                 normalize = 'col'):
        '''
        normalize : 归一化方式，'col'表示按列归一化，'abs'表示按绝对值归一化
        '''
        super(GCNEdgeConvolution, self).__init__()
        self.inputFeats = inputFeats
        self.outputFeats = outputFeats
        self.weight = Parameter(torch.Tensor(2,inputFeats, outputFeats))
        self.normalize = normalize

    def inputFeatsCalculate(self,adj,edgeFeatures):
        '''
        adj : 邻接矩阵 [N,N]
        edgeFeatures : [N,N,inputFeats]
        '''

        ADJ = adj.unsqueeze(0).expand(edgeFeatures.shape[0], adj.shape[0], adj.shape[1])
        renewEdgeFeatures = torch.bmm(ADJ,edgeFeatures)

        return renewEdgeFeatures # renewEdgeFeatures : [N,N,inputFeats]

    def outputFeatsCalculate(self, adj,edgeFeatures):
        '''
        adj : 邻接矩阵 [N,N]
        edgeFeatures : [N,N,inputFeats]
        '''
        ADJT = adj.t().unsqueeze(0).expand(edgeFeatures.shape[0], adj.shape[0], adj.shape[1])
        renewEdgeFeatures = torch.bmm(ADJT,edgeFeatures)

        return renewEdgeFeatures # renewEdgeFeatures : [N,N,inputFeats]
    
    def nomalizeCol(self, edgeFeatures):
        '''
        edgeFeatures : [N,N,inputFeats]
        edgeFeatures的每个矩阵按照列归一化
        '''
        colMax = edgeFeatures.max(dim=1, keepdim=True).values  # [N, 1, inputFeats]
        edgeFeaturesNorm = edgeFeatures / (colMax + 1e-8) 
        
        return edgeFeaturesNorm

    def nomalizeAbs(self, edgeFeatures):
        '''
        edgeFeatures : [N,N,inputFeats]
        edgeFeatures的每个矩阵按照绝对值归一化
        '''
        colSum = edgeFeatures.abs().sum(dim=1, keepdim=True)  # [N,1,inputFeats]
        edgeFeaturesNorm = edgeFeatures / (colSum + 1e-8)

        return edgeFeaturesNorm

    def forward(self, adj,edgeFeatures):
        '''
        edgeFeatures: [N, N, inputFeats]  E:边的数量  M:边的特征维度
        '''
        weightIn = self.weight[0] # weightIn : [inputFeats, outputFeats]
        weightOut = self.weight[1] # weightOut : [inputFeats, outputFeats]

        # 对于weightIn和weightOut进行扩张
        weightIn = weightIn.unsqueeze(0).expand(edgeFeatures.shape[0], 
                                                weightIn.shape[0], 
                                                weightIn.shape[1])
        weightOut = weightOut.unsqueeze(0).expand(edgeFeatures.shape[0],
                                                  weightOut.shape[0],
                                                  weightOut.shape[1])

        # 对于edgeFeatures进行计算InFeature 和 outFeature
        inFeatures = torch.bmm(edgeFeatures, weightIn)
        outFeatures = torch.bmm(edgeFeatures, weightOut)

        # 对于新边特征进行计算
        inFeatures = self.inputFeatsCalculate(adj, inFeatures)
        outFeatures = self.outputFeatsCalculate(adj, outFeatures)

        renewEdgeFeatures = inFeatures - outFeatures

        # 对于新特征做归一化
        if self.normalize == 'col':
            renewEdgeFeatures = self.nomalizeCol(renewEdgeFeatures)
        elif self.normalize == 'abs':
            renewEdgeFeatures = self.nomalizeAbs(renewEdgeFeatures)

        return renewEdgeFeatures
