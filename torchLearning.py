# 随机模块
import random

# 绘图模块
import matplotlib as mpl
import matplotlib.pyplot as plt

# numpy
import numpy as np

# pytorch
import torch
from torch import nn,optim
import torch.nn.functional as F
from torch.utils.data import Dataset,TensorDataset,DataLoader

def tensorGenReg(num_examples = 1000,w = [2,-1,1],bias = True,delta = 0.01,deg = 1):
    """回归类数据集创建函数
    
    param num_examples:创建数据集的数据量
    param w:包括截距的特征系数向量
    param bias:是否需要截距
    param delta:扰动项取值
    param def:方程次数
    return :生成的特征张量和标签张量
    
    该函数无法创建带有交叉项的方程 eg:deg>=2 y=x1^2 + x1x2(交叉项) + x2^2
    
    """
    if bias == True:
        num_inputs = len(w) - 1 # 特征数量
        features_true = torch.randn(num_examples,num_inputs) # 不包含全是1的列的特征张量
        w_true = torch.tensor(w[:-1]).reshape(-1,1).float() # 自变量系数
        b_true = torch.tensor(w[-1]).float() # 截距
        if num_inputs == 1: # 如果特征只有一个，不能使用矩阵乘法，需要特殊处理
            labels_true = torch.pow(features_true,deg) * w_true + b_true
        else:
            labels_true = torch.mm(torch.pow(features_true,deg),w_true) + b_true
        features = torch.cat((features_true,torch.ones(len(features_true),1)),1) # 在特征张量的最后添加一列全是1的列
        labels = labels_true + torch.randn(size = labels_true.shape) * delta
        
    else:
        num_inputs = len(w)
        features = torch.randn(num_examples,num_inputs)
        w_true = torch.tensor(w).reshape(-1,1).float()
        if num_inputs ==1:
            labels_true = torch.pow(features,deg) + w_true
        else:
            labels_true = torch.mm(torch.pow(features,deg),w_true)
        labels = labels_true + torch.randn(size = labels_true.shape) * delta
        
    return features,labels

def tensorGenCla(num_examples = 500,num_inputs = 2,num_class = 3,deg_dispersion = [4,2],bias = False):
    """分类数据集创建函数
    
    param num_examples:每个类别的数据量
    param num_inputs:数据集特征数量
    param num_class:数据集标签类别总数
    param deg_dispersion:数据分布离散程度参数，需要输入一个列表，其中第一个参数表示每个类别数组均值的参考、第二个参数表示随机数组标准差。
    param bias:建立模型逻辑回归模型时，是否带入截距
    return :生成的特征张量和标签张量，其中特征张量是浮点型二维数组，标签张量是长正型二维数组
    
    """
    
    cluster_1 = torch.empty(num_examples,1) #每一类标签张量的形状
    mean_ = deg_dispersion[0] #每一类特征张量的均值的参考值
    std_ = deg_dispersion[1] #每一类特征张量的方差
    lf = [] #用于存储每一类特征张量的列表容器
    ll = [] #用于存储每一类标签张量的列表容器
    k = mean_ * (num_class-1) / 2 #每一类特征张量均值的惩罚引子
    
    for i in range(num_class):
        data_temp = torch.normal(i*mean_ - k, std_, size=(num_examples, num_inputs)) #生成每一类张量
        lf.append(data_temp) #将每一类张量添加到lf中
        labels_temp = torch.full_like(cluster_1,i) #生成类的标签
        ll.append(labels_temp) #将每一类标签添加到ll中
        
    features = torch.cat(lf).float()
    labels = torch.cat(ll).long()
    
    if bias == True:
        features = torch.cat((features,torch.ones(len(features),1)),1) #在特征张量中添加全是1的列
    
    return features,labels

def data_iter(batch_size,features,labels):
    """数据切分函数
    
    param batch_size: 每个子数据集包含多少数据
    param features: 输入的特征张量
    param labels: 输入的标签张量
    return l: 包含batch_size个列表，每个列表由切分后的特征和标签所组成
    
    """
    
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    l = []
    
    for i in range(0,num_examples,batch_size): # 等差数列 ，i的间隔为batch_size
        j = torch.tensor(indices[i:min(i+batch_size,num_examples)])
        # torch.index_select(input, dim, index, out=None) 
        l.append([torch.index_select(features,0,j),torch.index_select(labels,0,j)])
        
    return l