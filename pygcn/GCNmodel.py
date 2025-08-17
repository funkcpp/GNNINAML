import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

from GCNlayers import GraphConvolution

class GCN(Module):
    def __init__(self,
                 inFeats,
                 hiddenFeats,
                 outFeats,
                 adj,
                 usingSparse = True,
                 normalInit = True,
                 layers = 2):
        '''
        inFeats : 输入特征的维度，即每个节点的节点特征
        hiddenFeats : 隐藏层的特征维度，在这里我们固定每个隐藏层的特征维度
        outFeats : 输出特征的维度，即每个节点的输出特征
        adj : 邻接矩阵，在静态网络中我们确保adj不变，所以将normalization放入init中做为图结构 N*N
        usingSparse : 是否使用稀疏矩阵
        normalInit : 是否使用正态分布初始化权重
        layers : GCN的层数
        '''
        super(GCN, self).__init__()

        # 归一化adj
        adjNorm = self.normalize(adj, spares=usingSparse)
        self.adjNorm = adjNorm

        # 构造层级序列
        layersList = []
        for i in range(layers):
            if i == 0:
                layersList.append(GraphConvolution(inputFeats=inFeats, 
                                                   outputFeats = hiddenFeats,
                                                   adj=adjNorm,
                                                   normalInit=normalInit))
            elif i == layers - 1:
                layersList.append(GraphConvolution(inputFeats = hiddenFeats, 
                                                   outputFeats = outFeats,
                                                   adj = adjNorm,
                                                   normalInit=normalInit))
            else:
                layersList.append(GraphConvolution(inputFeats = hiddenFeats, 
                                                   outputFeats = hiddenFeats,
                                                   adj = adjNorm,
                                                   normalInit = normalInit))
        
        self.layers = torch.nn.ModuleList(layersList)


    def normalize(self,adj,spares = True):
        """
            归一化邻接矩阵 D^{-1/2} (A+I) D^{-1/2}
            返回:
                A_norm: 归一化后的邻接矩阵
        """
        adjMatrix = adj

        N = adjMatrix.size(0)

        if spares:

            AHat = adjMatrix + torch.eye(N,device=adjMatrix.device)
            AHat = AHat.to_sparse()

                # 计算度矩阵
            degreeMatrix = torch.sparse.sum(AHat, dim=1).to_dense()
            degreeMatrix = degreeMatrix.pow(-0.5)
            degreeMatrix[degreeMatrix == float('inf')] = 0
            DHat = torch.diag(degreeMatrix)

            ANorm = torch.sparse.mm(DHat, torch.sparse.mm(AHat, DHat))
        else:
            device = adj.device
            adj = adj + torch.eye(N, device=device)  # 加自环
            deg = torch.sum(adj, dim=1)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.
            D_inv_sqrt = torch.diag(deg_inv_sqrt)
            ANorm = D_inv_sqrt @ adj @ D_inv_sqrt

        return ANorm

    def forward(self, HMatrix):
        """
            前向传播
            参数:
                HMatrix: 节点特征矩阵 
                初始化输入的是节点特征 N * inputFeats
                而后是 hiddenFeats * hiddenFeats
                最后是 hiddenFeats * outFeats
            返回:
                H': 更新后的节点特征矩阵
        """

        Layers = self.layers
        H = HMatrix
        for i in range(len(Layers)):
            layer = Layers[i]

            if i < len(Layers) - 1:
                H = layer(H)
                H = torch.relu(H)
            
            # 只要求最后一层是softmax输出
            else:
                H = layer(H)
                
                H = torch.softmax(H, dim = -1)
        
        return H

if __name__ == "__main__":

    # 测试一下
    
    # 构造邻接矩阵
    dense_A = torch.tensor([[1, 3, 1], [3, 1, 9], [1, 9, 1]], dtype=torch.float32)
    N = dense_A.size(0)

    # 构造节点特征
    dense_H = torch.tensor([
        [0,1,2,3],
        [1,2,3,4],
        [2,3,4,5]
      ],dtype=torch.float32)
    
    gcn = GCN(
        inFeats=4,
        hiddenFeats = 8,
        outFeats=2,
        adj = dense_A,
        usingSparse=True,
        normalInit=True,
        layers=2
    )

    output = gcn(dense_H)

    print(output)