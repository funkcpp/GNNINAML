# train 
# 分类的train
import math

import torch
import torch.nn as nn

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

from GCNlayers import GraphConvolution
from GCNmodel import GCN


class Trainer:
    def __init__(self, model, 
                 optimizer, 
                 criterion, 
                 device="cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def crossEntropyLoss(self,yPred,yTrue):
        '''
        yPred : 预测值  yPred = [0,1,0,1,2,0,1,2,....]
        yTrue : 真实值

        这个损失函数是交叉熵损失函数
        '''

        # y_pred: [N, C] logits
        # y_true: [N] int64 label
        
        loss = self.criterion(yPred, yTrue) 
        return loss
    

    def train(self, data, epochs=100):
        self.model.train()
        data = data.to(self.device)
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            loss = self.criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            self.optimizer.step()
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    def test(self, data, mask):
        self.model.eval()
        data = data.to(self.device)
        out = self.model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = pred[mask].eq(data.y[mask]).sum().item()
        acc = correct / mask.sum().item()
        return acc


