import torch
import itertools
import os
import pandas as pd
import numpy as np
from model import loader
from model import trainer
from sklearn.model_selection import KFold
import setting

num_workers = 4

real_data = ["Drosophila_Gut","Soil_Vitro","Soil_Vivo","Human_Gut","Human_Oral","Ocean"]
################################################################################
#       1. HYPERPARAMETER SEARCH 超参数搜索
################################################################################

#初始化参数

LearningRates = [[0.001,0.0025],[0.001,0.005],[0.001,0.01],[0.01,0.025],[0.01,0.05],[0.01,0.1]]
Minibatches = [1,5,10]
# Parameters for run
max_epochs = 100 #1000
early_stoping = 20
report = 50
inx = list(itertools.product(enumerate(LearningRates),enumerate(Minibatches)))[:]  
#enumerate：生成包含索引即对应元素的数组，product计算两个枚举的笛卡尔积，collect[:]转成一维数组
pars = real_data[0:1]


for (i,DATA) in enumerate(pars):
    path = f"/home/liufei/yangyixue/cNODEPy/data/real/{DATA}/P.csv" if setting.env_key else f"./data/real/{DATA}/P.csv" #5*26
   
    Z,P = loader.import_data(path,2/3) 
    print(f"import_Z={Z}")
    print(f"import_P={P}")
    if not isinstance(Z, np.ndarray):
       Z,P = np.array(Z),np.array(P)
       print(f"arrayZ={Z}")
       print(f"arrayP={P}")
    N = Z.shape[0] #获取矩阵z的行;5
    M = Z.shape[1] #获取矩阵z的列,16

    print(f"Z={Z}")
    print(f"P={P}")

    for it in inx:
        #j,k:学习率和Minibatches的索引;lr,mb学习率和Minibatches的值
        ((j,lr),(k,mb)) = it
        mb = Minibatches[k]
        Qtst =np.zeros((M,N), dtype=np.float64) #创建5*26的数组，用于储存数据
        Ptst = np.zeros((M,N), dtype=np.float64)#创建5*26的数组，用于储存数据
        LossTrain = np.zeros(M, dtype=np.float64)
        LossTest = np.zeros(M, dtype=np.float64)
        print(f"M={M};N={N}")
        #K折交叉验证
        kf = KFold(n_splits=M)
        kf_indices = []
       
        for train_index, test_index in kf.split(Z.T):
          I = Z.T
          J = P.T
          #组合为[(训练集特征，训练集标签),(测试集特征，测试集便签)]   
          Z_train, Z_test = I[train_index], I[test_index]
          P_train, P_test = J[train_index], J[test_index] 
          kf_indices.append([(Z_train,P_train),(Z_test,P_test)])
        # 使用 enumerate
        enumerated_kf = list(enumerate(kf_indices))   
        for l in enumerated_kf:
            (l,[(ztrn,ptrn),(ztst,ptst)]) = l
            #CNODE模块
            cnode = trainer.getModel(N) 
            #ztrn 15*5   ztst 1*5
            W, loss_train, loss_val, loss_test = trainer.train_reptile(
                                                cnode, max_epochs,
                                                mb, lr,
                                                ztrn, ptrn, ztst, ptst, ztst, ptst,
                                                report, early_stoping
                                            )
             # Save values
            Ptst[l,:] = ptst
            ravelTensor = trainer.predict(cnode,ztst)
            Qtst[l,:] = ravelTensor.detach().numpy().copy().ravel()
            LossTrain[l] = loss_train[-1].detach().numpy().copy()
            LossTest[l] = loss_test[-1].detach().numpy().copy()
            
        # Save results
        results =  f"/home/liufei/yangyixue/cNODEPy/results/real/{DATA}/hyperparameters/"  if setting.env_key else  f"./results/real/{DATA}/hyperparameters/"    
        np.savetxt(os.path.join(results, f"real_sample_{j}{k}.csv"), Ptst, delimiter=',')
        np.savetxt(os.path.join(results, f"pred_sample_{j}{k}.csv"), Qtst, delimiter=',')
        np.savetxt(os.path.join(results, f"test_loss_{j}{k}.csv"), LossTest, delimiter=',')
        np.savetxt(os.path.join(results, f"train_loss_{j}{k}.csv"), LossTrain, delimiter=',')

################################################################################
#       2. Experimental Validation
################################################################################        

# for (i,DATA) in enumerate(pars):
#     path = f"/home/liufei/yangyixue/cNODEPy/data/real/{DATA}/P.csv"  if setting.env_key else f"./data/real/{DATA}/P.csv"
#     Z,P = loader.import_data(path,2/3)
#     N = Z.shape[0] #获取矩阵z的行;5
#     M = Z.shape[1] #获取矩阵z的列,16
#     results = f"/home/liufei/yangyixue/cNODEPy/results/real/{DATA}/hyperparameters/"  if setting.env_key else f"./results/real/{DATA}/hyperparameters/"
#     #读取多个 CSV 文件中的测试损失数据，计算它们的平均值，并找到平均损失最小的索引
#     _mean = []
#     for i in range(1, 7):  # 1 到 6
#         row_means = []
#         for j in range(1, 4):  # 1 到 3
#             file_path = f"{results}test_loss_{i}{j}.csv"
#             data = pd.read_csv(file_path, header=None)  # 读取 CSV 文件
#             row_mean = data.mean().values[0]  # 计算平均值
#             row_means.append(row_mean)
#         _mean.append(row_means)   
#     _mean = np.array(_mean)
#     _mean = np.unravel_index(np.argmin(_mean), _mean.shape)     
#     mb = Minibatches[_mean[1]]
#     lr = LearningRates[_mean[0]]
#     Qtst =np.zeros((M,N), dtype=np.float64) #创建M*N的数组，用于储存数据
#     Ptst = np.zeros((M,N), dtype=np.float64)
#     LossTrain = np.zeros((M,N), dtype=np.float64)
#     LossTest = np.zeros((M,N), dtype=np.float64)
#     results = f"/home/liufei/yangyixue/cNODEPy/results/real/{DATA}/"  if setting.env_key else f"./results/real/{DATA}/"
#     # 枚举每一折，并将其收集到列表中
#     kf = KFold(n_splits=M)
#     kf_indices = []
    
#     for train_index, test_index in kf.split(Z.T):
#         I = Z.T
#         J = P.T
#         #组合为[(训练集特征，训练集标签),(测试集特征，测试集便签)]   
#         Z_train, Z_test = I[train_index], I[test_index]
#         P_train, P_test = J[train_index], J[test_index] 
#         kf_indices.append([(Z_train,P_train),(Z_test,P_test)])
#     # 使用 enumerate
#     enumerated_kf = list(enumerate(kf_indices))   
    
#     for l in enumerated_kf:
#         (l,((ztrn,ptrn),(ztst,ptst))) = l
#         #CONDE模块
#         cnode = trainer.getModel(N) 
#         W, loss_train, loss_val, loss_test = trainer.train_reptile(
#                                             cnode, max_epochs,
#                                             mb, lr,
#                                             ztrn, ptrn, ztst, ptst, ztst, ptst,
#                                             report, early_stoping
#                                         )
#         # Save values
#         Ptst[l,:] = ptst

#         ravelTensor = trainer.predict(cnode,ztst)
#         Qtst[l,:] = ravelTensor.detach().numpy().copy().ravel()
#         LossTrain[l] = loss_train[-1].detach().numpy().copy()
#         LossTest[l] = loss_test[-1].detach().numpy().copy()
#         # Save realization
#         results = f"/home/liufei/yangyixue/cNODEPy/results/real/{DATA}/loss_epochs/" if setting.env_key else f"./results/real/{DATA}/loss_epochs/"    
#         if not os.path.exists(results):
#              os.makedirs(results)
#         loss_train_array= [i.detach().numpy() for i in loss_train ]
#         loss_test_array= [i.detach().numpy() for i in loss_test ]
#         loss_val_array= [i.detach().numpy() for i in loss_val ]
#         np.savetxt(os.path.join(results, f"train{l}.csv"), loss_train_array, delimiter=',')
#         np.savetxt(os.path.join(results, f"test{l}.csv"), loss_test_array, delimiter=',')
#         np.savetxt(os.path.join(results, f"val{l}.csv"), loss_val_array, delimiter=',')
    
#     # Write full results
#     np.savetxt(os.path.join(results, f"real_sample.csv"), Ptst, delimiter=',')
#     np.savetxt(os.path.join(results, f"pred_sample.csv"), Qtst, delimiter=',')
#     np.savetxt(os.path.join(results, f"test_loss.csv"), LossTest, delimiter=',')
#     np.savetxt(os.path.join(results, f"train_loss.csv"), LossTrain, delimiter=',')




