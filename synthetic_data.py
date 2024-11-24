from model import generate,loader,trainer
import itertools
import os
import numpy as np
import setting
import pandas as pd
 

num_workers = 4
################################################################################
#       1. DATA GENERATION
################################################################################

repetitions = 10         # number of ecological networks 生态网络数量 10
vals = 3                 # number of values for sweeping parameter 扫描参数值的个数 3
N = 10                  # number of species 物种数量 10
M = (2**N)-1             # number of communities 种群数量
#强度、连通性、普性适、噪声、重组
synthetic_data = ["Strength","Connectivity","Universality","Noise","Rewiring"]
pars = synthetic_data[:3]
# Generate GLV data 
# 参数：参与竞争的物种数量；种群个数；生态网络数；扫描参数值的个数 ；参数
#generate.generate_data(N,M,repetitions,vals,pars)
#print("结束1")

################################################################################
#       2. HYPERPARAMETER SEARCH
################################################################################

LearningRates = [[0.01,0.01],[0.01,0.05],[0.01,0.1]]
Minibatches = [5,10]
# Parameters for run
max_epochs = 1000
early_stoping = 100
report = 10

#超参数、参数和重复的迭代器

inx = list(itertools.product(
    enumerate(LearningRates), 
    enumerate(Minibatches), 
    range(repetitions)
))
#enumerate：生成包含索引即对应元素的数组，product计算两个枚举的笛卡尔积，collect[:]转成一维数组
v = 0
DATA = synthetic_data[0]
# 使用低强度值Strength
for it in inx:
    ((j,lr),(k,mb),rep) = it
    path = f"/home/liufei/yangyixue/cNODEPy/data/synthetic/{N}/{DATA}/P{v}_{rep}.csv" if setting.env_key else f"./data/synthetic/{N}/{DATA}/P{v}_{rep}.csv"
    Z,P = loader.import_data(path,1)
    ztrn,ptrn,zval,pval,ztst,ptst = loader.split_data(Z,P,1,0.8) 
    cnode = trainer.getModel(N) #10
    mb_size = min(mb, ztrn.shape[1])
    print("进入2")
     # Train cNODE
    W, loss_train, loss_val, loss_test = trainer.train_reptile(
                                            cnode, max_epochs,
                                            mb_size, lr,
                                            ztrn, ptrn,
                                            zval, pval,
                                            ztst, ptst,
                                            report, early_stoping
                                            )
    #Save
    result = f"/home/liufei/yangyixue/cNODEPy/data/synthetic/{N}/hyperparameters/" if setting.env_key else f"./results/synthetic/{N}/hyperparameters/"
    if not os.path.exists(path):
            os.makedirs(path)
    np.savetxt(os.path.join(path, f"train_loss_{j}{k}_{rep}.csv"), loss_train[-1], delimiter=',') 
    np.savetxt(os.path.join(path, f"val_loss_{j}{k}_{rep}.csv"), loss_val[-1], delimiter=',') 
    np.savetxt(os.path.join(path, f"test_loss_{j}{k}_{rep}.csv"), loss_test[-1], delimiter=',') 
              
# ################################################################################
# #       3. HYPERPARAMETER SELECTION
# ################################################################################
# root = f"/home/liufei/yangyixue/cNODEPy/results/synthetic/{N}/hyperparameters/val_loss_" if setting.env_key else f"./results/synthetic/100/hyperparameters/val_loss_"
# val_hp = np.empty((len(LearningRates), len(Minibatches)), dtype=np.float64) #3,2
# for i in range(1,len(LearningRates)+1):
#     print("进入3")
#     for j in range(1,len(Minibatches)+1):
#         val = [
#             pd.read_csv(f"{root}{i}{j}_0{r}.csv", header=None).iloc[-1].values[0]
#             for r in range(repetitions)
#         ]
#         val_hp[i-1,j-1] = sum(val)/len(val)
# min_value = np.min(val_hp)
# min_index = np.where(val_hp == min_value)
# best_hp = min_index  # 找出最优超参数索引  2,1  

# combinations  = list(itertools.product(
#     LearningRates, 
#     Minibatches
# ))

# if best_hp[0]<=1: item = 0 
# else:  item = best_hp[0]-1
# index = len(LearningRates) * item +best_hp[1]
# lr,mb = combinations[index[0]-1]
    
# ################################################################################
# #       4. VALIDATION OF ALL DATASETS
# ################################################################################
# #percentage = [.001,.01,0.1,0.25,0.5,0.75,1]

# percentage = [0.1,0.25,0.5,0.75,1]

# inx = list(itertools.product(
#     pars, 
#     range(vals), 
#     range(repetitions)
# ))

# for it in inx:
#     (DATA,i,j) = it
#     path = f"/home/liufei/yangyixue/cNODEPy/results/data/synthetic/{N}/{DATA}/P{i}_{j}.csv" if setting.env_key else  f"./data/synthetic/{N}/{DATA}/P{i}_{j}.csv"
#     Z, P = loader.import_data(path,1)
#     LossTrain = np.zeros(len(percentage), dtype=np.float64)
#     LossVal = np.zeros(len(percentage), dtype=np.float64)
#     LossTest = np.zeros(len(percentage), dtype=np.float64)   
#     print("进入4")
#     results = f"/home/liufei/yangyixue/cNODEPy/results/synthetic/{N}/{DATA}/"  if setting.env_key else f"./results/synthetic/{N}/{DATA}/" 
#     if not os.path.exists(results):
#         os.makedirs(results)
#     if not os.path.exists(os.path.join(results, "loss_epochs/")):
#         os.makedirs(os.path.join(results, "loss_epochs/"))  
#     if not os.path.exists(os.path.join(results, "loss_percentage/")):
#         os.makedirs(os.path.join(results, "loss_percentage/"))        
#     for p,item in enumerate(percentage):
#         ztrn,ptrn,zval,pval,ztst,ptst = loader.split_data(Z,P,1,percentage[p])  
#         cnode = trainer.getModel(N)
#         mb_size = min(mb, ztrn.shape[1])
#         W, loss_train, loss_val, loss_test = trainer.train_reptile(
#                                             cnode, max_epochs,
#                                             mb_size, lr,
#                                             ztrn, ptrn,
#                                             zval, pval,
#                                             ztst, ptst,
#                                             report, early_stoping
#                                             )
#         LossTrain[p] = loss_train[-1]
#         LossVal[p] = loss_val[-1]
#         LossTest[p] = loss_test[-1]
#         if percentage[p] == percentage[-1]:
#             np.savetxt(os.path.join(results, f"loss_epochs/train_{i}{j}.csv"), loss_train, delimiter=',')
#             np.savetxt(os.path.join(results, f"loss_epochs/val_{i}{j}.csv"), loss_val, delimiter=',')
#             np.savetxt(os.path.join(results, f"loss_epochs/test_{i}{j}.csv"), loss_test, delimiter=',') 

#     np.savetxt(os.path.join(results, f"loss_percentage/train_{i}{j}.csv"), LossTrain, delimiter=',')
#     np.savetxt(os.path.join(results, f"loss_percentage/val_{i}{j}.csv"), LossVal, delimiter=',')
#     np.savetxt(os.path.join(results, f"loss_percentage/test_{i}{j}.csv"), LossTest, delimiter=',')
