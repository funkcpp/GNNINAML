import torch_geometric 
from torch_geometric.data import Data
import pandas as pd
import torch
def csv2pt(path, 
           label_col,
           save_path):
    
    ''' 
    path : csv文件路径
    label_col : 标签列名，字典
    '''
    data = pd.read_csv(path)

    # 读取相关列名
    edge_feat_name = label_col['edge_feat_name']
    node_feat_name = label_col['node_feat_name']
    edge_cat_feat_name = label_col['edge_cat_feat_name']
    edge_val_feat_name = label_col['edge_val_feat_name']
    y_label = label_col['y_label']

    start = edge_feat_name[0]
    end = edge_feat_name[1]

    # 构建边索引
    all_nodes = pd.unique(data[edge_feat_name].values.ravel())
    node2idx = {node: idx for idx, node in enumerate(all_nodes)}
    
    # 构建边
    edgeIndex = torch.tensor([
        [node2idx[nid] for nid in data[start]],  # 起点索引
        [node2idx[nid] for nid in data[end]]   # 终点索引
    ], dtype=torch.long)

    # 构建节点特征
    node_feat = torch.tensor(data[node_feat_name].values, dtype=torch.float)
    print(node_feat.shape)

    # 构建边属性特征
    edge_cat_feat = torch.tensor(data[edge_cat_feat_name].values, dtype=torch.long)
    print(edge_cat_feat.shape)

    # 构建边数值特征
    edge_val_feat = torch.tensor(data[edge_val_feat_name].values, dtype=torch.float)
    print(edge_val_feat.shape)

    # 构建图数据
    graph = Data(
        x = node_feat,
        edge_index=edgeIndex,
        edge_attr=torch.cat([edge_cat_feat, edge_val_feat], dim=1),
        edge_cat = edge_cat_feat,
        edge_val = edge_val_feat,
        y=torch.tensor(data[y_label].values, dtype=torch.long).squeeze()
    )

    # 保存图数据
    torch.save(graph,save_path)

    return 

if __name__ == '__main__':
    parh = 'D:/IBMAML/simulate_data.csv'
    label_col = {
        'edge_feat_name': ['start', 'end'],
        'node_feat_name': [f'node_feat_start_{i}' for i in range(2)]+[f'node_feat_end_{i}' for i in range(2)],
        'edge_cat_feat_name' : [f'edge_cat_feat_{i}' for i in range(5)],
        'edge_val_feat_name' : [f'edge_val_feat_{i}' for i in range(8)],
        'y_label' : ['y_label']
        }
    save_path = 'D:/IBMAML/simulate_data.pt'
    csv2pt(path=parh, label_col=label_col, save_path=save_path)
