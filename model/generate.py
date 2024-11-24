import numpy as np
import random
from scipy.sparse import csr_matrix,coo_matrix
from scipy.integrate import solve_ivp
import os
import setting
S = [.1,.15,.2]  #相互作用强度
C = [0.5,0.75,1] #连通性
U = [0.1,1,2]
E = [.001,.01,.025] #噪声强度
R = [.01,.1,.25]

#参数：xx：表示物种的初始种群数量；
# 求解时间区间的终点
# 相互作用矩阵
# 内在增长率向量  
# 返回值：求解GLV模型在给定参数条件下的稳态解。稳态解指的是系统在时间趋于无穷大时的状态，通常表示为种群数量的恒定值

def getSteadyState(xx,τ,A,r):
    t_eval = np.linspace(0, 10.0, 1000)  # 1000个时间点
    pars = np.hstack((A, r))
    solution = solve_ivp(getGLV, (0.0, τ), xx, args=(pars,), t_eval=t_eval, method='RK45')
    # 提取解的最后时刻的解（稳态解）
    xf = solution.y[:, -1]  # 选择最后时刻的解   
    return xf
   
#根据相互作用矩阵 A 和内在生长率 r 来计算物种数量的变化率   
def getGLV(t,x,pars):
    A = pars[:, :-1]  # 相互作用矩阵 A
    r = pars[:, -1]   # 生长率向量 r

    # 检查物种数量是否小于零
    x[x < 0] = 0    # 将负值设为零 
    # 计算物种数量的变化率
    dx = np.diag(x) @ (A @ x + r) 
    # 对于负数量的物种，变化率设为零
    dx[x < 0] = 0.0
    return dx

#生成具有通用动态的数据集
#参数：N:物种数量A的维数；C连通度：两个物种之间的相互作用的概率；σ：标准差，用于从正态分布中采样随机数，控制非对角元素的值
#返回相互作用矩阵和内在增长率（每个物种在无竞争情况下的增长率）
def getParameters(N,C,σ):
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                A[i, j] = -1.0  # 对角线元素 自我竞争，每个物种对自身的自我抑制作用
            elif random.random() < C:
                A[i, j] = np.random.normal(0, σ**2)  # 从均值为 0、方差为 σ^2 的正态分布中生成一个数
            else:
                A[i, j] = 0.0  # 不存在相互作用的元素设为 0
    #表示在没有其他物种干扰时，各物种的增长率
    r = np.random.rand(N, 1) 
    return A, r 


#得到物种‘ z ’:每列代表一个物种集合的“存在1/不存在0”状态
#数值初始条件‘ xx ’的集合中包含了每个物种随机初始化的数值
def getIC(N,M):
   z = np.zeros((N,M),dtype=int)
   inx = []
   while len(inx) < M :
        rand_indices = np.random.randint(1, N+1,size=np.random.randint(1, N+1))  # 随机选择列数，范围是 1 到 N
        unique_indices = np.unique(rand_indices) # 将矩阵中的元素展平，并去除重复元素
        inx.append(np.sort(unique_indices)) # 排序
   inx = list(map(np.unique, inx))
   for i in range(M):
      z[inx[i]-1,i] = 1
   xx = z * np.random.rand(N, M)
   return z,xx

  #返回重新连接交互矩阵。
def getRewiredNetwork(A,p):
    N = A.shape[0]
    result = np.diag(np.ones(N)) + A
    sA = csr_matrix(result)
    rows, cols = sA.nonzero()  # 获取非零元素的行列索引
    s = sA.data  # 获取非零元素的数值
    if p > 0.0:
        random_vals = np.random.rand(sA.nnz)
        condition = random_vals < p
        indices = np.where(condition)[0] #返回稀疏矩阵中所有非零值的索引
        for i in indices:
           alreadyconnected = rows[cols == cols[i]]
           possibletargets = set(range(1, N+1)) - alreadyconnected
           if len(possibletargets) >= 1:
              rows[i] = random.choice(possibletargets)
    sparse_matrix = coo_matrix((s, (rows, cols)), shape=(N, N))
    result = sparse_matrix.toarray() - np.ones((N, N))
    
    return result


#c 连通性：两个物种直接相互作用的可能性；
#σ：典型相互作用强度σ≥0，表示一个物种相互作用对另一个物种的人均生长率的典型影响。
# 参数：参与竞争的物种数量；种群个数；生态网络数；扫描参数值的个数 ；参数
# S = [.1,.15,.2]  #相互作用强度
# C = [0.5,0.75,1] #连通性
# U = [0.1,1,2] #正态分布的强度
# E = [.001,.01,.025] #噪声强度
# R = [.01,.1,.25] #随机重新连接边缘的比例

#参数：参与竞争的物种数量；种群个数；生态网络数；扫描参数值的个数 ；不同GLV类型的参数
def generate_data (N,M,repetitions,values,params):
   #repetitions:生态网络数量 
   reps = [i for i in range(repetitions)] #10
   num = [i for i in range(values)] #3
   for par in params:
      inx = [ (r,p) for r in reps for p in num ]
      for i in inx:
         #验证robustness
         # Changing parameters
         rep,p = i
         if par == "Strength":
            σ,c,η,ϵ,ρ = S[p],C[0],0,0,0.
         elif par == "Connectivity":
            σ,c,η,ϵ,ρ = S[0],C[p],0,0,0.
         elif par == "Universality": #η值越大普遍动力学损失越严重
            σ,c,η,ϵ,ρ = S[0],C[0],U[p],0,0.
         elif par == "Noise":#ε 对物种相对丰度测量噪声的robustness
            σ,c,η,ϵ,ρ = S[0],C[0],0,E[p],0.
         elif par == "Rewiring":
            σ,c,η,ϵ,ρ = S[0],C[0],0,0,R[p]
         print( σ,c,η,ϵ,ρ)
         # Allocate
         X = np.full((N,M), 0, dtype=np.float64)   
         A = np.full((N,N), 0, dtype=np.float64)   
         r = np.full((N,1), 0, dtype=np.float64) 

         # Get initial values  生成具有通用动态的数据集 
         #生成相互作用矩阵和固有增长率（每个物种在无竞争情况下的增长率）
         #参数：N:物种数量A的维数；C连通度：两个物种之间的相互作用的概率；σ：标准差，用于从正态分布中采样随机数，控制非对角元素的值
         Aa, r = getParameters(N,c,σ) 
         #Z生成不同物种的组合10个物种。就有2^10个01搭配
         #xx随机搭配
         Z,xx = getIC(N,M)

         conv = 0
         j = 0
         while j < M:
            #根据概率ρ重新连接物种之间的连线，来模拟不同的生态网络
            A = getRewiredNetwork(Aa,ρ)
            # Non-Universality: noise in non-zero entries 增加噪声的扰动 生成具有非通用动态的数据集。
            if η > 0:
                noise = np.random.normal(0, σ**2, (N, N)) #正态分布
                mask = np.where(A != 0, 1, 0)
                A += η * (noise * mask)
            #
            #参数：xx：表示物种的初始种群数量；
            # 求解时间区间的终点
            # 相互作用矩阵
            # 固有增长率向量  
            # 返回值：求解GLV模型在给定参数条件下的稳态解。稳态解指的是系统在时间趋于无穷大时的状态，通常表示为种群数量的恒定值   
            x = getSteadyState(xx[:,j], 1000., A, r)    
            #收敛性分析
            conv += np.sum(np.diag(x) @ (A @ x + r))
            #ϵ：GLV模型中引入 噪声扰动，模拟系统中种群数量的随机波动。这种噪声可能代表自然界中的不确定因素，例如环境变化、资源波动等。
            #随机扰动会影响物种数量的变化，使得系统表现出更多的动态行为，帮助研究系统的鲁棒性和稳定性
            x += ϵ * (np.random.uniform(0, np.mean(x[x > 0]), N) * np.random.choice([-1, 1], N) * Z[:, j])
            x[x < 0] = 0
            X[:, j] = x
            j += 1
            if j % 100 == 0:
               print("··")
         path = f"/home/liufei/yangyixue/cNODEPy/data/synthetic/{N}/{par}/" if setting.env_key else  f"./data/synthetic/{N}/{par}/"
         if not os.path.exists(path):
               os.makedirs(path)
         print(f"··{conv/M}")
         # 保存数据到 CSV 文件
         np.savetxt(os.path.join(path, f"A{p}_{rep}.csv"), np.hstack((Aa, r)), delimiter=',') #相互作用矩阵和固有增长率
         np.savetxt(os.path.join(path, f"Z{p}_{rep}.csv"), Z, delimiter=',') #不同初始条件下，物种的数据
         np.savetxt(os.path.join(path, f"P{p}_{rep}.csv"), X, delimiter=',') #对种群进行鲁棒性分析后的数据
     
          