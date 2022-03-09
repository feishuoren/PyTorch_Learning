# 随机模块
import random

# 绘图模块
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# numpy
import numpy as np

# pytorch
import torch
from torch import nn,optim
import torch.nn.functional as F
from torch.utils.data import Dataset,TensorDataset,DataLoader,random_split

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

def linreg(X,w):
    """线性模型
    """
    return torch.mm(X,w)

def logistic(X,w):
    """回归模型
    """
    return sigmoid(torch.mm(X,w))

def softmax(X,w):
    """softmax
    对所有数据前向传播后的f*w,计算exp() 以第一个数据为例 exp_data1 = [exp(y1),exp(y2),exp(y3)]
    对该条数据exp()计算后的结果求和 sum_data1 = exp(y1) + exp(y2) + exp(y3)
    softmax_data1 = exp_data1 / sum_data1
    对前向传播计算结果进行放缩 [y1,y2,y3] -> [new_y1,new_y2,new_y3]
    """
    m = torch.exp(torch.mm(X,w))
    # torch.sum(m,1),参数1代表对行计算，0为对列计算
    sp = torch.sum(m,1).reshape(-1,1)
    
    return m / sp

def squared_loss(y_hat,y):
    num_ = y.numel()
    sse = torch.sum((y_hat.reshape(-1,1) - y.reshape(-1,1)) ** 2)
    
    return sse/num_

def cross_entropy(sigma,y):
    
    return (-(1/y.numel()) * torch.sum((1-y)*torch.log(1-sigma) + y*torch.log(sigma)))

def m_cross_entropy(soft_z,y):
    y = y.long()
    # 每个样本最大分类可能性 eg: soft_z = [[0.6,0.5,0.3],[0.3,0.4,0.2]]  pob_real=[[0.6],[0.4]]
    prob_real = torch.gather(soft_z,1,y)
    # torch.prod(prob_real) = 0.6 * 0.4
    # torch.log(torch.prod(prob_real)) = log(0.6*0.4) = log(0.6) +log(0.4) 先相乘再log，内部取值可能过小可能会log0，因此先log再相加
    return (-(1 / y.numel()) * torch.log(torch.prod(prob_real)))

def sgd(params,lr):
    params.data -= lr*params.grad
    params.grad.zero_()

class GenData(Dataset):
    """针对手动创建数据的数据类
    """
    def __init__(self,features,labels): # 创建该类时需要输入的数据集
        self.features = features
        self.labels  = labels
        self.lens = len(features)
        
    def __getitem__(self,index):
        return self.features[index,:],self.labels[index]
    
    def __len__(self):
        return self.lens
    
def split_loader(features,labels,batch_size=10,rate=0.7):
    """数据封装、切分和加载数据
    
    param features: 输入的特征
    param labele: 数据集标签张量
    param batch_size: 数据加载时的每一个小批数据量
    param rate: 训练集数据占比
    return : 加载好的训练集和测试集
    
    """
    data = GenData(features,labels)
    num_train = int(data.lens * 0.7)
    num_test = data.lens - num_train
    
    data_train,data_test = random_split(data,[num_train,num_test])
    train_loader = DataLoader(data_train,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(data_test,batch_size=batch_size,shuffle=False)
    
    return (train_loader,test_loader)
    
def fit(net,criterion,optimizer,batchData,epochs=3,cla=False):
    """模型训练函数
    param net: 待训练的模型
    param citerion: 损失函数
    param optimizer: 优化算法
    param batchData: 训练数据集
    param cla: 是否是分类问题
    param epochs: 遍历数据次数
    
    """
    
    for epoch in range(epochs):
        for X,y in batchData:
            if cla == True:
                y = y.flatten().long() # 如果是分类问题，要对y进行整数转化
            yhat = net.forward(X)
            loss = criterion(yhat,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
def mse_cal(data_loader,net):
    """mse计算函数
    
    param data_Loader:加载好的数据
    param net:模型
    return :根据输入的数据，输出其MSE计算的结果
    
    """
    data = data_loader.dataset # 还原 Dataset类
    x = data[:][0] # 还原数据的特征
    y = data[:][1] # 还原数据的标签
    yhat = net(x)
    
    return F.mse_loss(yhat,y)


def accuracy_cal(data_loader,net):
    """准确率
    param data_loader: 加载好的数据
    param net: 数据
    return : 根据输入的数据，输出其准确率计算结果
    
    """
    data = data_loader.dataset # 还原 Dataset 类
    X = data[:][0] # 还原数据的特征
    y = data[:][1] # 还原数据的标签
    zhat = net(X) # 默认是分类问题，并且输出结果是未经softmax转化的结果
    soft_z = F.softmax(zhat,1) # 进行softmax转化
    acc_bool = torch.argmax(soft_z,1).flatten() == y.flatten() # 每条数据最大值结果所属的类别与标签是否一致 1 列
    acc = torch.mean(acc_bool.float())
    
    return acc

def model_train_test(model,train_data,test_data,num_epochs=20,criterion=nn.MSELoss(),optimizer=optim.SGD,lr=0.03,cla=False,eva=mse_cal):
    """模型误差测试函数
    
    param model: 模型
    param train_data: 训练数据
    param test_data: 测试数据
    param num_epochs: 迭代次数
    param criterion: 损失函数
    param optimizer: 优化方法
    param lr: 学习率
    param cla: 是否是分类模型
    return :MSE列表
    
    """
    
    # 模型评估指标列表
    train_l = []
    test_l = []
    # 模型训练过程
    for epochs in range(num_epochs):
        fit(net = model,
            criterion = criterion,
            optimizer = optimizer(model.parameters(),lr=lr),
            batchData = train_data,
            epochs = epochs,
            cla = cla
        )
        train_l.append(eva(train_data, model).detach())
        test_l.append(eva(test_data, model).detach())
        
    return train_l, test_l

def model_comparison(model_l,name_l,train_data,test_data,num_epochs=20,criterion=nn.MSELoss(),optimizer=optim.SGD,lr=0.03,cla=False,eva=mse_cal):
    """模型对比函数
    
    param model_l: 模型序列
    param name_l: 模型名称序列
    param train_data: 训练数据
    param test_data: 测试数据
    param num_epochs: 迭代次数
    param criterion: 损失函数
    param lr: 学习率
    param cla: 是否是分类模型
    return :MSE张量矩阵
    
    """
    
    # 模型评估指标矩阵
    train_l = torch.zeros(len(model_l),num_epochs)
    test_l = torch.zeros(len(model_l),num_epochs)
    
    # 模型训练过程
    for epochs in range(num_epochs):
        for i,model in enumerate(model_l):
            fit(net=model,
               criterion=criterion,
               optimizer=optimizer(model.parameters(),lr=lr),
               batchData = train_data,
               epochs = epochs,
               cla=cla)
            train_l[i][epochs] = eva(train_data, model).detach()
            test_l[i][epochs] = eva(test_data, model).detach()
            
    return train_l,test_l

def weights_vp(model, att="grad"):
    """观察各层参数取值和梯度的小提琴图绘图函数
    
    param model: 观察对象（模型）
    param att: 选择参数梯度（grad）还是参数取值（weights）进行观察
    return : 对应att的小提琴图
    
    """
    
    vp =[] # 创建空列表用于存储每一层参数的难度

    for i,m in enumerate(model.modules()):
        if isinstance(m, nn.Linear):
            if att == "grad":
                vp_x = m.weight.grad.detach().reshape(-1,1).numpy() # 每一层参数梯度
            else:
                vp_x = m.weight.detach().reshape(-1,1).numpy() # 每一层参数权重
            
            vp_y = np.full_like(vp_x, i) # 对层进行标记
            vp_a = np.concatenate((vp_x, vp_y), 1)
            vp.append(vp_a)

    vp_r = np.concatenate((vp), 0) # 拼接行

    ax = sns.violinplot(y = vp_r[:,0],x = vp_r[:,1])
    ax.set(xlabel='num_hidden',title=att)