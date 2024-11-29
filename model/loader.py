import numpy as np
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

#导入物种集合 `Z` 和组合 `P`。
def import_data(path,p):

  P = np.loadtxt(path, delimiter=',') #10*1023 原始样本矩阵
  Z = np.where(P > 0, 1, 0) #物种集合，样本存在z=1,不存在z=0
  print(f"old_Z={Z}")
  print(f"old_Z={P}")
    # 根据索引对 P 和 Z 进行筛选
  reps = []
  for zz in Z.T:  # Z.T 转置矩阵以便按列遍历
    indices = set(idx for idx, z in enumerate(Z.T) if np.array_equal(zz, z))
    if indices not in reps:
        reps.append(indices)
  # 去重 reps 列表中的集合
  # inx = [random.choice(list(r)) for r in reps]          
  # P = P[:, inx]  
  # Z = Z[:, inx]     
  P = normalize(P, axis=0, norm='l1')  # l1 归一化每一列 
  Z = normalize(Z, axis=0, norm='l1')  # l1 归一化每一列

  if p < 1.0:
      Z_train, Z_test, P_train, P_test= train_test_split(Z.T,P.T, test_size=1-p, shuffle=True)
      print(f"Z_train={Z_train.T}")
      print(f"P_train={P_train.T}")
      return Z_train.T, P_train.T
  return Z, P
#p:1,q:0.8
def split_data(Z,P,p,q):
    Z, P = shuffle(Z.T, P.T, random_state=42)

    # 将数据分为训练集和测试集（q分割）
    Z_train, Z_test, P_train, P_test = train_test_split(Z, P, test_size=1-q, shuffle=True)

    # 重新打乱训练数据，并将其分成训练集和验证集
    Z_train_final, Z_val, P_train_final, P_val = train_test_split(Z_train, P_train, test_size= 1e-5, shuffle=True)

    # 打乱测试数据并将其分成验证集和测试集（1/2分割） //整除向下取整
    
    Z_val_test, Z_test_final, P_val_test, P_test_final = train_test_split(Z_test, P_test, test_size=len(Z_test) // 2, shuffle=True)

    return Z_train_final, P_train_final, Z_val_test, P_val_test, Z_test_final, P_test_final
   