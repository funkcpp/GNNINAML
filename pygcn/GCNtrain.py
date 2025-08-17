# train 
# 分类的train
import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

from GCNlayers import GraphConvolution
from GCNmodel import GCN

class Trainer:
    def __init__(self, model, optimizer, criterion, device="cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def crossEntropyLoss(self,yTrain,yTrue):
        
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


