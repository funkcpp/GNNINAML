import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

from GCNlayers import GraphConvolution

# 设置随机种子
seed = 42
torch.manual_seed(seed)

edgeFeatures = torch.randn(2, 3, 4)
print(edgeFeatures)

colMax = edgeFeatures.max(dim=1, keepdim=True).values  # [N, 1, inputFeats]
edgeFeatures_norm = edgeFeatures / colMax 
print(edgeFeatures_norm)

col_sum = edgeFeatures.abs().sum(dim=1, keepdim=True)  # [N, 1, M]
edgeFeatures_norm = edgeFeatures / (col_sum + 1e-8)

print(edgeFeatures_norm)