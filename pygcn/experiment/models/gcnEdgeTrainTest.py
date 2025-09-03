import torch
from typing import Tuple, Dict, Any, Optional, Union, List
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def calculate_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, average: str = 'binary') -> Dict[str, float]:
    """
    计算分类指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        average: 多分类时的平均方式 ('binary', 'micro', 'macro', 'weighted')
    
    Returns:
        dict: 包含各种指标的字典
    """
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    metrics = {}
    
    try:
        # 精确率
        metrics['precision'] = precision_score(y_true_np, y_pred_np, average=average, zero_division=0)
        # 召回率
        metrics['recall'] = recall_score(y_true_np, y_pred_np, average=average, zero_division=0)
        # F1分数
        metrics['f1'] = f1_score(y_true_np, y_pred_np, average=average, zero_division=0)
        # 准确率
        metrics['accuracy'] = (y_pred == y_true).float().mean().item()
        
        # 混淆矩阵（对于二分类）
        if len(torch.unique(y_true)) == 2:
            tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred_np).ravel()
            metrics.update({
                'true_negative': tn,
                'false_positive': fp,
                'false_negative': fn,
                'true_positive': tp
            })
            
    except Exception as e:
        print(f"计算指标时出错: {e}")
        metrics.update({
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0
        })
    
    return metrics


def train_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, Any]]:
    """
    使用DataLoader训练模型
    """
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    for data in data_loader:
        data = data.to(device)
        
        # 确保标签是long类型
        if data.y.dtype != torch.long:
            data.y = data.y.long()
        
        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        
        # 计算损失
        loss = criterion(output, data.y)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计信息
        total_loss += loss.item()
        
        # 收集预测和真实标签
        pred = output.argmax(dim=1)
        all_predictions.append(pred)
        all_targets.append(data.y)
    
    # 计算指标
    all_pred = torch.cat(all_predictions)
    all_target = torch.cat(all_targets)
    metrics = calculate_metrics(all_target, all_pred)
    metrics['loss'] = total_loss / len(data_loader)
    metrics['num_graphs'] = len(data_loader)
    metrics['total_samples'] = all_target.size(0)
    
    return metrics['loss'], metrics


def test_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    return_predictions: bool = False
) -> Tuple[float, Dict[str, Any]]:
    """
    使用DataLoader测试模型性能
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            
            # 确保标签是long类型
            if data.y.dtype != torch.long:
                data.y = data.y.long()
            
            # 前向传播
            output = model(data)
            
            # 计算损失
            loss = criterion(output, data.y)
            total_loss += loss.item()
            
            # 收集预测和真实标签
            pred = output.argmax(dim=1)
            all_predictions.append(pred)
            all_targets.append(data.y)
            all_outputs.append(output)
    
    # 计算指标
    all_pred = torch.cat(all_predictions)
    all_target = torch.cat(all_targets)
    metrics = calculate_metrics(all_target, all_pred)
    metrics['loss'] = total_loss / len(data_loader)
    metrics['num_graphs'] = len(data_loader)
    metrics['total_samples'] = all_target.size(0)
    
    if return_predictions:
        metrics['predictions'] = all_pred.cpu()
        metrics['targets'] = all_target.cpu()
        metrics['outputs'] = torch.cat(all_outputs).cpu()
    
    return metrics['loss'], metrics


def print_detailed_metrics(metrics: Dict[str, Any], phase: str = "Test"):
    """
    打印详细的评估指标
    """
    print(f"\n{phase} 详细指标:")
    print(f"损失: {metrics.get('loss', 0):.4f}")
    print(f"准确率: {metrics.get('accuracy', 0):.4f}")
    print(f"精确率: {metrics.get('precision', 0):.4f}")
    print(f"召回率: {metrics.get('recall', 0):.4f}")
    print(f"F1分数: {metrics.get('f1', 0):.4f}")
    
    # 如果是二分类，打印混淆矩阵
    if 'true_positive' in metrics:
        print(f"\n混淆矩阵:")
        print(f"真阳性(TP): {metrics['true_positive']}")
        print(f"假阳性(FP): {metrics['false_positive']}")
        print(f"假阴性(FN): {metrics['false_negative']}")
        print(f"真阴性(TN): {metrics['true_negative']}")
        
        # 计算额外指标
        if metrics['true_positive'] + metrics['false_positive'] > 0:
            precision = metrics['true_positive'] / (metrics['true_positive'] + metrics['false_positive'])
            print(f"精确率(手动计算): {precision:.4f}")
        
        if metrics['true_positive'] + metrics['false_negative'] > 0:
            recall = metrics['true_positive'] / (metrics['true_positive'] + metrics['false_negative'])
            print(f"召回率(手动计算): {recall:.4f}")
