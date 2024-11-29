import torch
import torch.nn as nn
import random
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
   #1.  5*0  2. 5*0 
    if len(p.shape) == 1 or len(z.shape) == 1  :
        p,z = p.view(p.shape[0], 1),z.view(z.shape[0], 1)
    preds = predict(cnode,z)  # 获取模型预测值
    numerator = torch.sum(torch.abs(p - preds))  # 计算分子
    denominator = torch.sum(torch.abs(p + preds))  # 计算分母
    val =  numerator / denominator
    return val # 返回损失值

def loss(cnode, Z, P):
    if isinstance(Z,tuple):  
        Z,P = torch.tensor(Z).to(dtype=cnode.layer.W.dtype), torch.tensor(P).to(dtype=cnode.layer.W.dtype)
    if isinstance(Z,np.ndarray): 
        Z,P = torch.tensor(Z).to(dtype=cnode.layer.W.dtype), torch.tensor(P).to(dtype=cnode.layer.W.dtype)
    if len(Z.shape) == 1 or len(P.shape) == 1:
        Z,P = Z.view(Z.shape[0], 1), P.view(P.shape[0], 1)
    if(Z.shape[0]!=cnode.layer.W.shape[0]):
        Z,P =Z.T,P.T 
    losses = torch.stack([_loss(cnode, Z[:, i], P[:, i]) for i in range(Z.shape[1])]) 

    return torch.mean(losses)

def apply_opt_out(v, update):
    return v + 0.01 * update

def predict(cnode,z):
    if isinstance(z,np.ndarray): 
      z = torch.tensor(z).to(dtype=cnode.layer.W.dtype)
    if(z.shape[0]!=cnode.layer.W.shape[0]):
        z =z.T   
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
    W = list(cnode.layer.parameters())
    opt_in = torch.optim.Adam(W,lr=LR[0])
    #训练次数 epochs 100
    for e in range(epochs): 
        V= [v.clone().detach() for v in W] 
        #  25*5  1*5 syn:817*10 将所有训练集分批次训练[1,5,10]
        shuffled_data = shuffle_obs(Ztrn, Ptrn) 
        batchSize= each_batch(shuffled_data, mb)
        
        #内层参数优化
        for z,p in batchSize:  
            opt_in.zero_grad()
            #1.z,p 为tuple
            loss_value  = loss(cnode, z, p)
            loss_value.backward()
            opt_in.step() 
        print(f"Epoch {e }: 内层更新后权重 = {[w.data for w in W]}")

        #外层参数优化
        grads = []
        for w, v in zip(W[0], V[0]):
            dv = w.data - v.data
            grads.append(dv)
        mean_grad = torch.mean(torch.stack(grads),dim=0)
        with torch.no_grad():
            for w in W:
               w += LR[1] * mean_grad
        print(f"Epoch {e }: 外层更新后权重 = {[w.data for w in W]}")   
       
        loss_train.append(loss(cnode, Ztrn, Ptrn))   
        loss_test.append(loss(cnode,Ztst,Ptst))   #1*5
        loss_val.append(loss(cnode, Zval,Pval))  #1*5
        stoping.append([loss(cnode,Ztrn[k,:],Ptrn[k,:])  for k in range(Ptrn.shape[0])])
        if e > early_stoping:
            stoping_tensor = torch.tensor(stoping)
            mean_value = torch.mean(stoping_tensor[-early_stoping:-1])  # 计算均值
            comparison = mean_value <= stoping_tensor[-1]     
            a =  torch.sum(comparison).item()      # 与 stoping 的最后一个值比较
            counts = a / Ztrn.shape[0] 
            print(f"counts:{counts},comparison:{comparison}")
            if counts>0.5:
                es +=1
                if es>5:
                    print("Early stoping at $e...")
                    break
        if e>0:
            print(f"{e}---train={loss_train[-1]},test={loss_test[-1]}")        
            
    return W, loss_train, loss_val, loss_test   
 