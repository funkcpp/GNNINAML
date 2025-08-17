import torch
import torch_geometric

import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

from GCNlayers import GraphConvolution
from GCNmodel import GCN



if __name__ == "__main__":
    
    # 创造初始边界条件
    
    numNodes = 100 #节点个数
    numEdges = 2000 #边的个数

    # 设置随机种子
    torch.manual_seed(0)

    row = torch.randint(0,numNodes,(numEdges,))
    col = torch.randint(0,numNodes,(numEdges,))
    
    edgeIndex = torch.stack([row, col], dim=0) #按照列合并 [2,numEdges]
    print(edgeIndex.shape)

    edgeIndex = torch.cat([edgeIndex, edgeIndex.flip(0)], dim=1) # 按照每行翻转后按行cat
    print(edgeIndex.shape)


    
    # 去掉重复的
    edgeIndex = edgeIndex.unique(dim=1)

    # 利用pyg生成网络
    data = torch_geometric.data.Data(edge_index=edgeIndex)

    adj = torch_geometric.utils.to_dense_adj(data.edge_index)
    print(adj.shape) # [1,numNodes,numNodes]

    # 构建随机节点特征模型
    nodeFeatures = 16
    nodeTarget = torch.randn((numNodes,nodeFeatures))

    # 利用pyg做 trainMask 和 testMask
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    data.train_mask[:int(0.8 * numNodes)] = 1
    data.test_mask[int(0.8 * numNodes):] = 1

    adjTrain = adj[:, data.train_mask, :][:, :, data.train_mask]

    # 构建模型
    model = GCN(
        inFeats=nodeFeatures,
        hiddenFeats=32,
        outFeats=2,
        adj=adj[0]
    )

    out = model(nodeTarget)

    print(out)