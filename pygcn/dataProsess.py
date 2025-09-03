import pandas as pd
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric 
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
import random


import sklearn
import random

import numpy as np
'''
这个类主要是用于分批处理图数据，因为图数据一般不太适合用普通的batch处理
'''

class DataProcessor:
    def __init__(self, 
                 file_path,
                 edgeStarts,
                 edgeEnds,
                 edgeStartFeats,
                 edgeEndFeats):
        
        self.file_path = file_path
        self.edgeStarts = edgeStarts
        self.edgeEnds = edgeEnds
        self.edgeStartFeats = edgeStartFeats
        self.edgeEndFeats = edgeEndFeats
        
        
        
        # 存储图的列表
        self.graphs = self.createGraphs()

    def createGraphs(self,
                      randomDivide = 10,
                      isRandom = 'random'):
        '''
        randomDivide: 将图起点随即划分成为 randomDivide个部分
        '''
        graphs = []
        edgeStarts = self.edgeStarts
        
        # 随机划分edgeStarts
        edgeStartsList = np.array_split(edgeStarts, randomDivide)

        # 对于每个划分节点，从中随机挑选K个邻居做为其子图
        if isRandom == 'random':
            for edgeStart in edgeStartsList:
                subgraph = self.sampleNeighbors(edgeStart, K=5)
                graphs.append(subgraph)
        return graphs
    

    def sample_neighbors_from_csv(csv_path, edgeStart, K=5, chunksize=100000):
        """
        从大规模 CSV 分块读取边表，为 edgeStart 节点构建子图并采样邻居
        ----------
        csv_path : str
            边表 csv 文件路径 (columns=['src','dst','weight'])
        edgeStart : list[int]
            起始节点集合
        K : int
            每个节点采样的邻居数
        chunksize : int
            每次读取的行数
        """
        edge_index_list = []
        edge_attr_list = []

        edgeStart = set(edgeStart)  # 提高查询速度

        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            # 只保留起点在 edgeStart 的边
            mask = chunk['src'].isin(edgeStart)
            filtered = chunk[mask]

            if not filtered.empty:
                # 对每个起点随机采样 K 个邻居
                sampled = (
                    filtered.groupby('src', group_keys=False)
                            .apply(lambda x: x.sample(n=min(K, len(x)), 
                                                    random_state=random.randint(0,1000)))
                )

                edges = torch.tensor([sampled.src.values, sampled.dst.values], dtype=torch.long)
                edge_index_list.append(edges)

                if 'weight' in sampled.columns:
                    edge_attr_list.append(torch.tensor(sampled.weight.values, dtype=torch.float))

        if edge_index_list:
            edge_index = torch.cat(edge_index_list, dim=1)
            edge_attr = torch.cat(edge_attr_list) if edge_attr_list else None
        else:
            edge_index = torch.empty((2,0), dtype=torch.long)
            edge_attr = None

        # 构造 PyG Data 子图
        data = Data(edge_index=edge_index, edge_attr=edge_attr)
        return data


    def preprocess(self):
        # Implement your preprocessing steps here
        pass
