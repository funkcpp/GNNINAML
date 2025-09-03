# 建立baseline模型
# 将卷积层和边信息结合，在这里我们先不考虑时间顺序，只是搭建一个baseline模型
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

'''
infeat:GCN层输入维数
hiddenFeat:GCN层隐含维数
outFeat:GCN层输出维数
embeddingDims:属性数据嵌入维数
fchidden:属性数据输出隐藏层维数
vdim:数值数据维数
finalDim:最后的输出层维数
'''
class GCNWithEdge(nn.Module):
    def __init__(self,
                 inFeat,
                 hiddenFeat,
                 outFeat,
                 embeddingDims = [20,20],
                 fchidden = 32,
                 vdim = 2,
                 finalDim = 2,
                 dropout = 0.5):
        super(GCNWithEdge, self).__init__()
        
        # 由于数据集较大，故我们对于GCN要求的层数不能过多，在这里做为baseline我们只要求有两层gcn链接
        self.conv1 = GCNConv(inFeat, hiddenFeat)
        self.conv2 = GCNConv(hiddenFeat, outFeat)

        # 构建处理属性数据层
        self.embeddingList = nn.ModuleList()
        for dim in embeddingDims:
            self.embeddingList.append(nn.Embedding(dim, fchidden))
        
        self.finalLayer = nn.Sequential(
            nn.Linear(outFeat + len(embeddingDims)*fchidden + vdim, finalDim),
            nn.ReLU(),
            nn.Dropout(dropout),    
            nn.Linear(finalDim, 2)  # 假设二分类问题
        )
    
    def forward(self, 
                data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # gcn得到节点特征
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index) # x : [num_node,outFeat]

        # 将节点数据转换为边数据
        edgeIndex = data.edge_index
        start,end = edgeIndex
        node2Edge = x[start] - x[end]  #node2Edge : [num_edge,outFeat]

        # 提取属性数据
        edgeAttr = edge_attr.T # edgeAttr : [num_edge, num_attr]
        vlData = edgeAttr[:, :2].float()

        for i in range(len(self.embeddingList)):
            emb = self.embeddingList[i]
            vlData = torch.cat([vlData, emb(edgeAttr[:, i+2].long())], dim=1)
        
        total = torch.cat([node2Edge, vlData], dim=1)

        out = self.finalLayer(total)

        return F.log_softmax(out, dim=1)
