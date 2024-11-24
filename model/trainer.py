import torch
import torch.nn as nn
import random
import copy
from torchdiffeq import odeint
import numpy as np

def shuffle_obs(Z, P):
    """Shuffle the observations."""
    combined = list(zip(Z, P))  # 将 z 和 p 组合
    random.shuffle(combined)      # 打乱组合的列表
    return combined

def each_batch(data, mb):
    """Generate batches from data."""
    Z, P = zip(*data)
    num_samples = len(Z)
    # 通过批次大小分割数据
    rangeSize = range(0, num_samples, mb)
    for i in range(0, num_samples, mb):
        # 每次取出一个批次
        yield Z[i:i+mb], P[i:i+mb]
    


# Xavier初始化函数
def xavier_uniform(shape):
    weight = torch.empty(shape)
    nn.init.xavier_uniform_(weight)
    return weight
# 定义 FitnessLayer 层
class FitnessLayer(nn.Module):
    def __init__(self, N):
        super(FitnessLayer, self).__init__()
        self.W = nn.Parameter(xavier_uniform((N, N)),requires_grad=True)

    def forward(self,t,p):
        #微分方程右边
        a = self.W#*10
        f = torch.matmul(a, p) #10*10   10*1 = 10*1
        g = torch.matmul(torch.ones(p.shape[0],1),p.T) #10*1 1*10 = 10*10
        # # 10*10 10*1 = 10*1
        dx = p * (torch.matmul(a, p) - torch.matmul(g,f)) #10*1 10*1 
        return dx


#封装 Neural ODE 模型
class NeuralODE(nn.Module):
    def __init__(self, N):
        super(NeuralODE, self).__init__()
        self.layer = FitnessLayer(N)
    def forward(self,p):
        # 使用 solve_ivp 求解 ODE
        #p = p.to(dtype=torch.float64)
        if len(p.shape) == 1:
            p = p.view(p.shape[0], 1)
        p_dot = odeint(self.layer,p,torch.linspace(0, 1.0, 100))
        return p_dot

def getModel(N):
    neural_ode = NeuralODE(N)
    return neural_ode

def _loss(cnode, z, p):
   # p = torch.tensor(p)
    if len(p.shape) == 1:
        p = p.to(dtype=cnode.layer.W.dtype).view(p.shape[0], 1)

    preds = predict(cnode,z)  # 获取模型预测值
    numerator = torch.sum(torch.abs(p - preds))  # 计算分子
    denominator = torch.sum(torch.abs(p + preds))  # 计算分母
    val =  numerator / denominator
    return val # 返回损失值

def loss(cnode, Z, P):
    Z,P = np.array(Z), np.array(P)
    Z,P = torch.from_numpy(Z).to(dtype=cnode.layer.W.dtype),torch.from_numpy(P).to(dtype=cnode.layer.W.dtype)
    
    # 使用列表推导计算每个样本的损失，并取平均值
    if len(Z.shape) == 1:
        Z = Z.view(Z.shape[0], 1)
    if len(P.shape) == 1:
        P = P.view(P.shape[0], 1) 
    if(Z.shape[0]!=cnode.layer.W.shape[0]):
            Z =Z.T
            P =P.T      
    losses = torch.stack([_loss(cnode, Z[:, i], P[:, i]) for i in range(Z.shape[1])])
    return torch.mean(losses)

def apply_opt_out(v, update):
    return v + 0.01 * update

def predict(cnode,z):
    #调用NeuralODE 的forward方法，返回对Z的预测
    if type(z) != torch.Tensor:
        z = torch.tensor(z)
    if len(z.shape)==1:
        z = z.to(dtype=cnode.layer.W.dtype).view(z.shape[0], 1)
    if z.shape[0] == 1:
       z = z.T.to(dtype=cnode.layer.W.dtype)     
    conde = cnode(z)
    return conde[-1]


#使用Reptile元学习算法训练cNODE。

def train_reptile(cnode,
            epochs,
            mb,
            LR,
            Ztrn, Ptrn,
            Zval, Pval,
            Ztst, Ptst,
            report,
            early_stoping):
    loss_train = []
    loss_test = []
    loss_val = []
    stoping = []
    EarlyStoping = []
    es = 0

    W =  cnode.layer.W.detach().numpy() #10*10
    opt_in = torch.optim.Adam([cnode.layer.W],lr=LR[0])
    opt_out = torch.optim.Adam([cnode.layer.W],lr=LR[1])
    i=1
    for e in range(epochs):
        V = copy.deepcopy(W)
        shuffled_data = shuffle_obs(Ztrn, Ptrn) #463组
        batchSize= each_batch(shuffled_data, mb)
        for z,p in batchSize:  #每次批传入五个数组
            opt_in.zero_grad()
            loss_value  = loss(cnode, z, p)
            loss_value.backward()
            opt_in.step() #参数更新
        newW = cnode.layer.W.detach().numpy()
        V = V
        for w, v in zip(newW, V):
           dv = w - v  # 计算 w 和 v 的差
           v = apply_opt_out(v, dv)  # 使用更新函数更新 v
           w = v + dv  # 更新 w 为 v + dv 
        loss_train.append(loss(cnode, Ztrn, Ptrn))   
        loss_test.append(loss(cnode,Ztst,Ptst))   
        loss_val.append(loss(cnode, Zval,Pval))   
        stoping.append(loss(cnode,Ztrn,Ptrn))
        print(f"循环次数：{e}")
        if e > early_stoping:
            mean_value = torch.mean(stoping[-early_stoping:-1])  # 计算均值
            comparison = mean_value <= stoping[-1]           # 与 stoping 的最后一个值比较
            counts = int(comparison) / Ztrn.shape[1] 
            if counts>0.5:
                es +=1
                if es>10:
                    print("Early stoping at $e...")
                    break
        elif e>0:
            print(f"{e}---train={loss_train[-1]},test={loss_test[-1]}")        
            
    return W, loss_train, loss_val, loss_test   
 