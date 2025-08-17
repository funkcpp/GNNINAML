import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

from GCNlayers import GraphConvolution

inFeats = 16
outFeats = 32
A = Parameter(torch.Tensor(inFeats, outFeats))
torch.nn.init.normal_(A, mean=0.0, std=0.01)
print(A)