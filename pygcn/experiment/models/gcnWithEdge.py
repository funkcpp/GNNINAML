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
                 fc_hidden = 32,
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
            self.embeddingList.append(nn.Embedding(dim, fc_hidden))
        
        self.finalLayer = nn.Sequential(
            nn.Linear(outFeat + len(embeddingDims)*fc_hidden + vdim, finalDim),
            nn.ReLU(),
            nn.Dropout(dropout),    
            nn.Linear(finalDim, 2)  # 假设二分类问题
        )
    
    def forward(self, 
                data):
        ''' 
        data = Data(
        x = node_feat,
        edge_index=edgeIndex,
        edge_attr=torch.cat([edge_cat_feat, edge_val_feat], dim=1),
        edge_cat = edge_cat_feat,
        edge_val = edge_val_feat,
        y=torch.tensor(data[y_label].values, dtype=torch.float)
    )
        '''
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        edge_cat = data.edge_cat # [num_edge,num_cat_feat]
        edge_val = data.edge_val # [num_edge,num_val_feat]


        # gcn得到节点特征
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index) # x : [num_node,outFeat]

        # 将节点数据转换为边数据
        edgeIndex = data.edge_index
        start,end = edgeIndex
        node2Edge = x[start] - x[end]  #node2Edge : [num_edge,outFeat]

        # 提取属性数据
        

        for i in range(len(self.embeddingList)):
            emb = self.embeddingList[i]
            edge_val = torch.cat([edge_val, emb(edge_cat[:,i].long())], dim=1)
        
        total = torch.cat([node2Edge, edge_val], dim=1)

        out = self.finalLayer(total)

        return out
    
if __name__ == '__main__':
    path = 'D:/IBMAML/simulate_data.pt'
    data = torch.load(path)
    model = GCNWithEdge(inFeat=data.x.shape[1], 
                        hiddenFeat=64, 
                        outFeat=32, 
                        embeddingDims=[10,10,10,10,10], 
                        fc_hidden=32, 
                        vdim=data.edge_val.shape[1], 
                        finalDim=2, 
                        dropout=0.5)
    out = model(data)
    print(out)

