Graph Convolutional Networks in PyTorch
====

PyTorch implementation of Graph Convolutional Networks (GCNs) for semi-supervised classification [1].

For a high-level introduction to GCNs, see:

Thomas Kipf, [Graph Convolutional Networks](http://tkipf.github.io/graph-convolutional-networks/) (2016)

![Graph Convolutional Networks](figure.png)

Note: There are subtle differences between the TensorFlow implementation in https://github.com/tkipf/gcn and this PyTorch re-implementation. This re-implementation serves as a proof of concept and is not intended for reproduction of the results reported in [1].

This implementation makes use of the Cora dataset from [2].



这个store主要是为了处理这样一个问题，对于图问题而言会存在着这样一种图：他们的节点数据会随着边的变化而变化，而且对于这种图往往我们是对其边的label进行分类任务。对于小图而言你可以将每条边做为虚拟节点，然后再构造新的包含这些虚拟节点的零阶矩阵，使用GCN来进行分类任务,也可以利用`PyG`的`NNConv`包将边数据导入进去。或者使用GTA可以直接将边的数据输入进去。而我们在这里提出一种全新的架构，利用信息流和GCN的概念，将每条边的信息定义为 起点信息和-终点信息和 。

## Installation

```python setup.py install```

## Requirements

  * PyTorch 0.4 or 0.5
  * Python 2.7 or 3.6

## Usage

```python train.py```

## References

[1] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907)

[2] [Sen et al., Collective Classification in Network Data, AI Magazine 2008](http://linqs.cs.umd.edu/projects/projects/lbc/)

## Cite

Please cite our paper if you use this code in your own work:

```
@article{kipf2016semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N and Welling, Max},
  journal={arXiv preprint arXiv:1609.02907},
  year={2016}
}
```
