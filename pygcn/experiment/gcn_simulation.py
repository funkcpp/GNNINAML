import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader

from torch.utils.data import Dataset
import numpy as np

from models import GCNWithEdge
from models import train

path = 'D:/IBMAML/simulate_data.pt'
data = torch.load(path)

print(data.y.shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


model = GCNWithEdge(inFeat=data.x.shape[1], 
                    hiddenFeat=64, 
                    outFeat=32, 
                    embeddingDims=[10,10,10,10,10], 
                    fc_hidden=32, 
                    vdim=data.edge_val.shape[1], 
                    finalDim=2, 
                    dropout=0.1).to(device)


train_loader = DataLoader(
    dataset=[data],  # 将单个 Data 对象包装在列表中
    batch_size=1,
    shuffle=False
)

print("DataLoader 初始化完成")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

train(
    train_loader = train_loader,
    model = model,
    criterion = criterion,
    optimizer = optimizer,
    device = device,
    num_epochs = 1000
)