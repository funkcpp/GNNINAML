import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

import torch

def match(edgeInfor,edges):
    '''
    adj : [N, N]
    edgeInfor : [N, N, M] 
    N : 节点个数
    M : 边的特征维度
    '''
    # 找出所有非零的边 (i, j)

    if edges.shape[0] == 0:
        return torch.empty((0, edgeInfor.shape[2]))

    # 直接索引 edgeInfor
    i_idx, j_idx = edges[:, 0], edges[:, 1]
    COO = edgeInfor[i_idx, j_idx]  # shape = [E, M]

    return COO

if __name__ == "__main__":
    N, M = 4, 3
    adj = torch.tensor([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    edgeInfor = torch.arange(N*N*M).reshape(N, N, M)
    print(edgeInfor)

    out = match(adj, edgeInfor)
    print(out.shape)  # [E, M]
    print(out)