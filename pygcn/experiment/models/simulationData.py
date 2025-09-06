import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import os
import numpy as np
import random
import pandas as pd

# 用来加密节点数据
class hidden_node_mlp(nn.Module):
    def __init__(self, 
                 input_dim ,#节点初始数据维数, 
                 hidden_dim ,# 隐藏层维数, 
                 output_dim, # 节点加密后数据维度
                 seed = 42, # 随机种子
                 ):
        super(hidden_node_mlp, self).__init__()

        torch.manual_seed(seed)

        self.seq = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # 初始化权重参数
        for layer in self.seq:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=1)
                nn.init.normal_(layer.bias, mean=0, std=1)
        
        # 打印并保存当前参数
        print('初始化参数保存中...')

        save_dir = 'params'
        os.makedirs(save_dir,exist_ok=True)
        save_path = os.path.join(save_dir,'node_pram.pt')
        torch.save(self.state_dict(),save_path)

        print('参数保存完毕')
    
    def forward(self, x):
        return self.seq(x)

# 用来加密边数据
class hidden_edge_mlp(nn.Module):
    def __init__(self, 
                 input_val_dim,
                 hidden_val_dim,
                 output_val_dim,
                 nums_cat_dim,
                 hidden_cat_dim,
                 output_cat_dim,
                 output_dim,
                 seed = 42):
        super(hidden_edge_mlp,self).__init__()

        # 处理val_feat
        self.val_seq = nn.Sequential(
            nn.Linear(input_val_dim,hidden_val_dim),
            nn.ReLU(),
            nn.Linear(hidden_val_dim,output_val_dim)
        )

        # 处理cat_feat
        # 为每个分类特征创建embedding层
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_cat, hidden_cat_dim) for n_cat in nums_cat_dim
        ])
        
        # 分类特征处理
        total_cat_dim = len(nums_cat_dim) * hidden_cat_dim
        self.cat_processor = nn.Sequential(
            nn.Linear(total_cat_dim, output_cat_dim),
            nn.ReLU()
        )

        # cat以后重新加密
        self.fc = nn.Linear(output_cat_dim + output_val_dim,output_dim)

        torch.manual_seed(seed)
        # 初始化val_feat参数
        for layer in self.val_seq:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=1)
                nn.init.normal_(layer.bias, mean=0, std=1)
        
        # 初始化cat_feat参数
        for layer in self.embeddings:
            if isinstance(layer, nn.Embedding):
                nn.init.normal_(layer.weight, mean=0, std=1)
        
        # 初始化cat_processor参数
        
        for layer in self.cat_processor:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=1)
                nn.init.normal_(layer.bias, mean=0, std=1)
        
        # 初始化outputlayer
        nn.init.normal_(self.fc.weight, mean=0, std=1)

        print('初始化参数保存中...')
        save_dir = 'params'
        os.makedirs(save_dir,exist_ok=True)
        save_path = os.path.join(save_dir,'edge_pram.pt')
        torch.save(self.state_dict(),save_path)
        print('参数保存完毕')

    def forward(self, x_cat, x_val):
        # 确保x_val是float类型
        print(x_val.shape)
        if x_val.dtype != torch.float32:
            x_val = x_val.float()
        
        # 处理连续特征
        val_output = self.val_seq(x_val)  # [batch_size, output_val_dim]
        
        # 处理分类特征 - 确保是long类型
        if x_cat.dtype != torch.long:
            x_cat = x_cat.long()
        else:
            x_cat = x_cat  # 已经是long类型
        
        # 处理每个分类特征
        embedded_features = []
        for i in range(x_cat.shape[1]):
            feat = x_cat[:, i].long()
            embedded = self.embeddings[i](feat)
            embedded_features.append(embedded)
        
        # 合并所有embedding特征
        cat_combined = torch.cat(embedded_features, dim=1)
        cat_output = self.cat_processor(cat_combined)
        
        combined = torch.cat([val_output, cat_output], dim=1)
        return self.fc(combined)

class hidden_mlp(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim = 1,
                 random_state = 42):
        super(hidden_mlp,self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,output_dim)
        )

        # 固定随机种子
        torch.manual_seed(random_state)

        # 初始化权重参数
        for layer in self.seq:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=1)
                nn.init.normal_(layer.bias, mean=0, std=1)
        
        # 打印并保存当前参数
        print('初始化参数保存中...')

        save_dir = 'params'
        os.makedirs(save_dir,exist_ok=True)
        save_path = os.path.join(save_dir,'mlp_pram.pt')
        torch.save(self.state_dict(),save_path)

        print('参数保存完毕')

    def forward(self,x):
        out = self.seq(x)
        return out

def shuffled_data(x,random_state):
    np.random.seed(random_state)
    col_index = np.random.permutation(x.shape[1])
    print(col_index)
    x_shuffled = x[:,col_index]
    return x_shuffled

def cat_int(x,nums_cat):
    # 先将x映射到cat区间内
    max_ = np.max(x)
    min_ = np.min(x)
    x = (x - min_)/(max_ - min_)
    x_scaled = (nums_cat - 1) * x
    x_int = np.floor(x_scaled).astype(int)
    return x_int

def node_feat_creat(nums_node_feat,node_feat_dim):
    # 随机生成节点特征
    node_feat = {}
    
    for i in range(nums_node_feat):
        feat = np.random.rand(1,node_feat_dim).squeeze()
        node_feat[i] = feat
    
    return node_feat

def simulationLinear(
        n_samples = 20000,
        nums_node = 1000,
        n_features = 15,
        n_informative = 10,
        n_redundant = 2,
        n_classes = 2,
        weights = [0.5, 0.5],
        random_state = 42,
        node_dims = 2,
        edge_cat_dims = 5,
        edge_val_dims = 10,
        edge_cats = [10,10,10,10,10]
        ):
    '''
    生成图数据的函数
    参数介绍：
    n_samples : int, default=100
        样本总数
    nums_node : int, default=1000
        图中节点数
    n_features : int, default=15
        特征维度
    n_informative : int, default=10
        有效特征
    n_redundant : int, default=2
        冗余特征
    n_classes : int, default=2
        类别数
    weights : list, default=[0.5, 0.5]
        类别平衡
    random_state : int, default=42
        随机种子
    node_dims : int, default = 2
        节点特征维数
    edge_cat_dims : int,default = 5
        边属性特征维数
    edge_val_dims : int,default = 8
        边数值特征维数
    edge_cats : list
        边特征的每个特征初始维度
    '''

    X, y = make_classification(
    n_samples=n_samples,      # 总样本数
    n_features=n_features,       # 特征维度
    n_informative=n_informative,     # 有效特征
    n_redundant=n_redundant,       # 冗余特征
    n_classes=n_classes,         # 类别数
    weights=weights,  # 类别平衡
    random_state=random_state
    )

    # X : [n_samples,n_features]
    # y : [n_samples,]

    # 随机打乱列
    x_shuffled = shuffled_data(X,random_state = random_state)

    # 在这里我们将 x_shuffled 的前两维特征当作其节点的特征
    edge_cat_feat,edge_val_feat = [i for i in range(0,edge_cat_dims)], \
                                    [i for i in range(edge_cat_dims,n_features)]
    # 确定边属性特征
    # 将每个边属性数据映射到对于属性区间内后取整

    for col in edge_cat_feat:
        nums_cats = edge_cats[col] # 得到对应每个属性数据的种类
        x_shuffled[:,col] = cat_int(x_shuffled[:,col],nums_cats)
    
    data_feat = pd.DataFrame(x_shuffled)
    # 重命名列名
    for col in range(len(data_feat.columns)):
        if col in edge_cat_feat:
            data_feat.rename(columns={col:f'edge_cat_feat_{col}'},inplace=True)
        elif col in edge_val_feat:
            data_feat.rename(columns={col:f'edge_val_feat_{col-edge_cat_dims}'},inplace=True)

    # 制造图结点数据
    start = np.random.randint(0,nums_node,size = n_samples)
    end = np.random.randint(0,nums_node,size = n_samples)

    # 随机生成节点数据特征
    node_feat_dic = node_feat_creat(nums_node_feat = nums_node,node_feat_dim = node_dims)
    
    # 构建节点特征
    start_feat_data = pd.DataFrame([node_feat_dic[i] for i in start])

    # 重命名
    for col in range(len(start_feat_data.columns)):
        start_feat_data.rename(columns={col:f'node_feat_start_{col}'},inplace=True)
    
    
    end_feat_data = pd.DataFrame([node_feat_dic[i] for i in end])
    # 重命名
    for col in range(len(end_feat_data.columns)):
        end_feat_data.rename(columns={col:f'node_feat_end_{col}'},inplace=True)
    
    node_feat_data = pd.concat([start_feat_data,end_feat_data],axis=1)

    
    data_ = {'start' : start,
             'end' : end}   
    # 合并数据
    data = pd.DataFrame(data = data_ )
    data = pd.concat([data,node_feat_data,data_feat],axis=1)
    
    data_loader = {
        'data' : data,
        'data_node_feat':node_dims,
        'data_edge_cat_feat' : edge_cat_dims,
        'data_edge_val_feat' : edge_val_dims
    }

    return data_loader


def y_label_simulate(
        data_loader,
        confuse_model = [hidden_node_mlp,hidden_edge_mlp,hidden_mlp],
        data_pram_dic = {},
        weight = [0.5,0.5]
    ):
        
        node_feat = data_loader['data_node_feat']
        edge_cat_feat = data_loader['data_edge_cat_feat']
        edge_val_feat = data_loader['data_edge_val_feat']
        
        if data_pram_dic == {}:
            print('请传入参数模型')
            return
        
        # 随机种子
        seed = data_pram_dic['seed']

        # 初始化节点加密模型
        node_input_dim = data_pram_dic['node_input_dim']
        node_hidden_dim = data_pram_dic['node_hidden_dim']
        node_output_dim = data_pram_dic['node_output_dim']

        model_node = confuse_model[0](node_input_dim * 2,
                                      node_hidden_dim,
                                      node_output_dim,
                                      seed)

        # 初始化边加密模型
        edge_cat_input_dim = data_pram_dic['edge_cat_input_dim']
        edge_cat_hidden_dim = data_pram_dic['edge_cat_hidden_dim']
        edge_cat_output_dim = data_pram_dic['edge_cat_output_dim']

        edge_val_input_dim = data_pram_dic['edge_val_input_dim']
        edge_val_hidden_dim = data_pram_dic['edge_val_hidden_dim']
        edge_val_output_dim = data_pram_dic['edge_val_output_dim']
        
        edge_output_dim = data_pram_dic['edge_output_dim']

        model_edge = confuse_model[1](input_val_dim = edge_val_input_dim,
                                      hidden_val_dim = edge_val_hidden_dim,
                                      output_val_dim = edge_val_output_dim,
                                      nums_cat_dim = edge_cat_input_dim,
                                      hidden_cat_dim = edge_cat_hidden_dim,
                                      output_cat_dim = edge_cat_output_dim,
                                      output_dim = edge_output_dim,
                                      seed = seed)
        

        # 初始化最后加密模型
        mlp_input_dim = data_pram_dic['mlp_input_dim']
        mlp_hidden_dim = data_pram_dic['mlp_hidden_dim']
        mlp_output_dim = data_pram_dic['mlp_output_dim']

        model_mlp = confuse_model[2]( input_dim = mlp_input_dim,
                                      hidden_dim = mlp_hidden_dim,
                                      output_dim = mlp_output_dim,
                                      random_state = seed)
        
        # 加载相关特征
        node_feat_data = data_loader['data'].filter(regex='node_feat_')
        print('node_feat_data.shape:',node_feat_data.shape)
        edge_cat_feat_data = data_loader['data'].filter(regex='edge_cat_feat_')
        print('edge_cat_feat_data.shape:',edge_cat_feat_data.shape)
        edge_val_feat_data = data_loader['data'].filter(regex='edge_val_feat_')
        print('edge_val_feat_data.shape:',edge_val_feat_data.shape)


        # 将数据转为torch.tensor
        node_feat_data = torch.tensor(node_feat_data.values,dtype=torch.float)
        edge_cat_feat_data = torch.tensor(edge_cat_feat_data.values,dtype=torch.long)
        edge_val_feat_data = torch.tensor(edge_val_feat_data.values,dtype=torch.float)

        # 将数据传入到模型中
        confuse_node_feat_data = model_node(node_feat_data)
        confuse_edge_feat_data = model_edge(edge_cat_feat_data,edge_val_feat_data)
        confuse_output = model_mlp(torch.cat([confuse_node_feat_data,confuse_edge_feat_data],dim=1))
        print(confuse_output)

        # 标签生成，利用分位数
        # 两样本之间的比例
        ratio = weight[0]/(weight[1] + weight[0])
        y_ratio = torch.quantile(confuse_output,q = torch.tensor(ratio))
        y_label = (confuse_output > y_ratio).long()

        # 制造标签
        data_loader['label'] = y_label.numpy().reshape(-1)

        return data_loader




if __name__ == '__main__':

    simulate_data = simulationLinear(
        n_samples = 20000,
        nums_node = 1000,
        n_features = 15,
        n_informative = 10,
        n_redundant = 2,
        n_classes = 2,
        weights = [0.5, 0.5],
        random_state = 42,
        node_dims = 2,
        edge_cat_dims = 5, 
        edge_val_dims = 10,
        edge_cats = [10,10,10,10,10]
    )

    parm_dic = {
        'seed' : 41,
        'node_input_dim' : 2,
        'node_hidden_dim' : 10,
        'node_output_dim' : 5,
        'edge_cat_input_dim' : [10,10,10,10,10],
        'edge_cat_hidden_dim' : 20,
        'edge_cat_output_dim' : 20,
        'edge_val_input_dim' : 10,
        'edge_val_hidden_dim' : 20,
        'edge_val_output_dim' : 5,
        'edge_output_dim' : 10,
        'mlp_input_dim' : 15,
        'mlp_hidden_dim' : 20,
        'mlp_output_dim' : 1
    }

    simulate_data = y_label_simulate(simulate_data,
                                     confuse_model = [hidden_node_mlp,hidden_edge_mlp,hidden_mlp],
                                     data_pram_dic = parm_dic,
                                     weight = [0.5,0.5])

    print(simulate_data)
    data = simulate_data['data']
    data['y_label'] = simulate_data['label']
    data.to_csv('simulate_data.csv',index=False)
    print(simulate_data['label'].sum())