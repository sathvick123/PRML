

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy as sc
from numpy import linalg as LA
from PIL import Image
from scipy.stats import multivariate_normal
import seaborn as sns; sns.set_theme()


import math
import sys

f1 = open("trian.txt","r")
X11 = []
X12 = []
X21 = []
X22 = []
X31 = []
X32 = []


for i in f1.readlines():
  x1,x2,t = i.strip().split(',')
  if(t == "1"):
    X11.append(np.longdouble(x1))
    X12.append(np.longdouble(x2))
  elif(t == "2"):
    X21.append(np.longdouble(x1))
    X22.append(np.longdouble(x2))
  else :
    X31.append(np.longdouble(x1))
    X32.append(np.longdouble(x2))
    

 

X11 = np.array(X11)
X12 = np.array(X12)
X21 = np.array(X21)
X22 = np.array(X22)
X31 = np.array(X31)
X32 = np.array(X32)


f2 = open("dev.txt","r")
X1f_d = []
X2f_d = []
X11_d = []
X12_d = []
X21_d = []
X22_d = []
X31_d = []
X32_d = []


for i in f2.readlines():
  x1,x2,t = i.strip().split(',')
  X1f_d.append(np.longdouble(x1))
  X2f_d.append(np.longdouble(x2))
  if(t == "1"):
    X11_d.append(np.longdouble(x1))
    X12_d.append(np.longdouble(x2))
  elif(t == "2"):
    X21_d.append(np.longdouble(x1))
    X22_d.append(np.longdouble(x2))
  else :
    X31_d.append(np.longdouble(x1))
    X32_d.append(np.longdouble(x2))

X1f_d = np.array(X1f_d)
X2f_d = np.array(X2f_d)
X11_d = np.array(X11_d)
X12_d = np.array(X12_d)
X21_d = np.array(X21_d)
X22_d = np.array(X22_d)
X31_d = np.array(X31_d)
X32_d = np.array(X32_d)

def mean(X):
  ans = 0
  l  = np.size(X)
  for i in range(l):
    ans = (ans+X[i])
  ans = ans/l
  return ans


def cov(X,Y):
  ans = 0
  l = np.size(X)
  m1 = mean(X)
  m2 = mean(Y)
  for i in range(np.size(X)):
    ans = ans + (X[i]-m1)*(Y[i]-m2)
  ans = ans/(l-1)
  return ans
  
def covmat(X,Y):
  ans = np.zeros((2,2))
  ans[0][0] = cov(X,X)
  ans[0][1] = cov(X,Y)
  ans[1][0] = cov(Y,X)
  ans[1][1] = cov(Y,Y)
  return ans

def get_prob(X,C,M):
  const = 1/math.pi
  det = math.sqrt(LA.det(C))
  det = det*2
  const = const/det
  xmat = np.subtract(X,M)
  xmat_t = xmat.transpose()
  C_inv = LA.inv(C)
  const2 = xmat_t @ C_inv @ xmat
  const2 = const2/2
  const2 = const2*(-1)
  ans = const*(math.exp(const2))
  ans = ans/3
  return ans


def get(X):
  ans = np.zeros(np.size(X))
  for i in range(np.size(X)):
    ans[i] = X[i]
  return ans


def maxi(X):
  ans = X[0]
  for i in range(np.size(X)):
    ans = max(ans,X[i])
  return ans

def mini(X):
  ans = X[0]
  for i in range(np.size(X)):
    ans = min(ans,X[i])
  return ans

def getm(a,b):
  ans = np.zeros((2,1))
  ans[0][0]  = a
  ans[1][0]  = b
  return ans

def gettm(a,b):
  ans = np.zeros((1,2))
  ans[0][0] = a
  ans[0][1] = b
  return ans
 
def identify(X,Mean1,Mean2,Mean3,Cov1,Cov2,Cov3):
  ans = np.zeros(3)

  const = 1/math.sqrt(LA.det(Cov1))
  Cov_i1 = LA.inv(Cov1)
  x_mat1 = np.subtract(X,Mean1)
  x_mat1_t = x_mat1.transpose()
  const2 = x_mat1_t @ Cov_i1 @ x_mat1
  const2 = const2/2
  const2 = const2*(-1)
  ans[0] = const*(math.exp(const2))

  const = 1/math.sqrt(LA.det(Cov2))
  Cov_i2 = LA.inv(Cov2)
  x_mat2 = np.subtract(X,Mean2)
  x_mat2_t = x_mat2.transpose()
  const2 = x_mat2_t @ Cov_i2 @ x_mat2
  const2 = const2/2
  const2 = const2*(-1)
  ans[1] = const*(math.exp(const2))

  const = 1/math.sqrt(LA.det(Cov3))
  Cov_i3 = LA.inv(Cov3)
  x_mat3 = np.subtract(X,Mean3)
  x_mat3_t = x_mat3.transpose()
  const2 = x_mat3_t @ Cov_i3 @ x_mat3
  const2 = const2/2
  const2 = const2*(-1)
  ans[2] = const*(math.exp(const2))
  
  if(ans[0] >= ans[1] and ans[0] >= ans[2]):
    return 1
  elif(ans[1]>=ans[0] and ans[1]>=ans[2]):
    return 2
  else:
    return 3

Mean1 = getm(mean(X11),mean(X12))
Mean2 = getm(mean(X21),mean(X22))
Mean3 = getm(mean(X31),mean(X32))

Mean1_t = Mean1.transpose()
Mean2_t = Mean2.transpose()
Mean3_t = Mean3.transpose()




def ROC(C1,C2,C3):
  TPR  = []
  FPR  = []
  S    = []
  
  for i in range(np.size(X1f_d)):
    X = getm(X1f_d[i],X2f_d[i])
    for j in range(3):
      if(j == 0):
        temp = get_prob(X,C1,Mean1)
      elif(j == 1):
        temp = get_prob(X,C2,Mean2)
      else:
        temp = get_prob(X,C3,Mean3)
      S.append(temp)
      
  S = np.array(S)
  S = np.sort(S)

  for k in range(np.size(S)):
    thre = S[k]
    TP = 0
    FP = 0
    TN = 0
    FN = 0
   
    for i in range(np.size(X11_d)):
      X = getm(X11_d[i],X12_d[i])

      for j in range(3):
        if(j == 0):
          temp = get_prob(X,C1,Mean1)
        elif(j == 1):
          temp = get_prob(X,C2,Mean2)
        else:
          temp = get_prob(X,C3,Mean3)

        if(temp >= thre):
          if(j == 0):
            TP = TP + 1
          else:
            FP = FP + 1 
        else:
          if(j == 0):
            FN = FN + 1
          else:
            TN = TN + 1

     
    for i in range(np.size(X21_d)):
      X = getm(X21_d[i],X22_d[i])

      for j in range(3):
        if(j == 0):
          temp = get_prob(X,C1,Mean1)
        elif(j == 1):
          temp = get_prob(X,C2,Mean2)
        else:
          temp = get_prob(X,C3,Mean3)

        if(temp >= thre):
          if(j == 1):
            TP = TP + 1
          else:
            FP = FP + 1 
        else:
          if(j == 1):
            FN = FN + 1
          else:
            TN = TN + 1

    for i in range(np.size(X31_d)):
      X = getm(X31_d[i],X32_d[i])
    
      for j in range(3):

        if(j == 0):
          temp = get_prob(X,C1,Mean1)
        elif(j == 1):
          temp = get_prob(X,C2,Mean2)
        else:
          temp = get_prob(X,C3,Mean3)

        if(temp >= thre):
          if(j == 2):
            TP = TP + 1
          else:
            FP = FP + 1 
        else:
          if(j == 2):
            FN = FN + 1
          else:
            TN = TN + 1

    t1 = TP/(TP+FN)
    t2 = FP/(FP+TN)
    TPR.append(t1)
    FPR.append(t2)

  TPR = np.array(TPR)
  FPR = np.array(FPR)
  res = [TPR,FPR]
  return res


def plot_eig(P,D):
  x = np.zeros(1000)
  y = np.zeros(1000)
  for i in range(1000):
    x[i] = P[0] + D[0]*(i)
    y[i] = P[1] + D[1]*(i)
  plt.plot(x,y)

def Fn(X):
  ans = np.zeros(np.size(X))
  for i in range(np.size(X)):
    ans[i] = 1 - X[i]
  return ans

#case-2 

Cov1 = covmat(X11,X12)
Cov2 = covmat(X21,X22)
Cov3 = covmat(X31,X32)

E_v1,E_1 = LA.eig(Cov1)
E_v2,E_2 = LA.eig(Cov2)
E_v3,E_3 = LA.eig(Cov3)

P1 = [mean(X11),mean(X12)]
P2 = [mean(X21),mean(X22)]
P3 = [mean(X31),mean(X32)]



plt.style.use('seaborn-dark')
plt.rcParams['figure.figsize']=14,6
fig = plt.figure()

random_seed=1000
pdf_lis2 = []



for k in range(3):
  if(k == 0):
   mean_mat = P1
   cov_mat2 = Cov1
  elif(k == 1):
    mean_mat = P2
    cov_mat2 = Cov2
  else:
    mean_mat = P3
    cov_mat2 = Cov3

  sigma_1 = cov_mat2[0][0]
  sigma_2 = cov_mat2[1][1]
  mean_1  = mean_mat[0]
  mean_2  = mean_mat[1]
  distr = multivariate_normal(cov = cov_mat2, mean = mean_mat,seed = random_seed)
  x = np.linspace(0, 1500, num=100)
  y = np.linspace(0, 2500, num=100)
  X, Y = np.meshgrid(x,y)
  pdf = np.zeros(X.shape)
  for i in range(X.shape[0]):
    for j in range(X.shape[1]):
      pdf[i,j] = distr.pdf([X[i,j], Y[i,j]])
  key = 131+k
  ax = fig.add_subplot(key, projection = '3d')
  ax.plot_surface(X, Y, pdf, cmap = 'viridis')
  plt.xlabel("x1")
  plt.ylabel("x2")
  plt.title(f'PDF of Class {k+1}')
  pdf_lis2.append(pdf)
  ax.axes.zaxis.set_ticks([])
  if(k == 0):
    X1 = X
    X2 = Y
  elif(k == 1):
    Y1 = X
    Y2 = Y
  else:
    Z1 = X
    Z2 = Y
plt.tight_layout()
plt.show()


for idx, val in enumerate(pdf_lis2):
  if(idx == 0):
    X = X1
    Y = X2
  elif(idx == 1):
    X = Y1
    Y = Y2
  else:
    X = Z1
    Y = Z2
  plt.subplot(1,3,idx+1)
  plt.contour(X, Y, val, cmap='viridis')
  plt.xlabel("x1")
  plt.ylabel("x2")
  plt.title(f'Constant Density curve of Class {idx}')
  if(idx == 0):
    plot_eig(P1,E_1[:,0])
    plot_eig(P1,E_1[:,1])
  elif(idx == 1):
    plot_eig(P2,E_2[:,0])
    plot_eig(P2,E_2[:,1])
  else:
    plot_eig(P3,E_3[:,0])
    plot_eig(P3,E_3[:,1])

plt.tight_layout()
plt.show()


ans = np.zeros((3,3))

for i in range(np.size(X11_d)):
  temp = getm(X11_d[i],X12_d[i])
  cl = identify(temp,Mean1,Mean2,Mean3,Cov1,Cov2,Cov3)
  cl = cl-1
  ans[cl][0] = ans[cl][0]+1

for i in range(np.size(X21_d)):
  temp = getm(X21_d[i],X22_d[i])
  cl = identify(temp,Mean1,Mean2,Mean3,Cov1,Cov2,Cov3)
  cl = cl-1
  ans[cl][1] = ans[cl][1]+1

for i in range(np.size(X31_d)):
  temp = getm(X31_d[i],X32_d[i])
  cl = identify(temp,Mean1,Mean2,Mean3,Cov1,Cov2,Cov3)
  cl = cl-1
  ans[cl][2] = ans[cl][2]+1

x_l = [1,2,3]
ax = sns.heatmap(ans, annot=True, fmt="f",xticklabels = x_l,yticklabels=x_l)
ax.set_xlabel('Actual Class',fontsize = 24)
ax.set_ylabel('Predicted Class',fontsize = 24)
ax.set_title('Confusion Matrix',fontsize = 30)


R2 = ROC(Cov1,Cov2,Cov3)

TPR2 = R2[0]
FPR2 = R2[1]
FNR2 = Fn(TPR2)

#case-5


Cov1 = covmat(X11,X12)
Cov2 = covmat(X21,X22)
Cov3 = covmat(X31,X32)
Cov1[0][1] = 0
Cov1[1][0]  = 0
Cov2[0][1] = 0
Cov2[1][0]  = 0
Cov3[0][1] = 0
Cov3[1][0]  = 0

E_v1,E_1 = LA.eig(Cov1)
E_v2,E_2 = LA.eig(Cov2)
E_v3,E_3 = LA.eig(Cov3)

P1 = [mean(X11),mean(X12)]
P2 = [mean(X21),mean(X22)]
P3 = [mean(X31),mean(X32)]



plt.style.use('seaborn-dark')
plt.rcParams['figure.figsize']=14,6
fig = plt.figure()

random_seed=1000
pdf_lis5 = []



for k in range(3):
  if(k == 0):
   mean_mat = P1
   cov_mat5 = Cov1
  elif(k == 1):
    mean_mat = P2
    cov_mat5 = Cov2
  else:
    mean_mat = P3
    cov_mat5 = Cov3

  sigma_1 = cov_mat5[0][0]
  sigma_2 = cov_mat5[1][1]
  mean_1  = mean_mat[0]
  mean_2  = mean_mat[1]
  distr = multivariate_normal(cov = cov_mat5, mean = mean_mat,seed = random_seed)
  x = np.linspace(0, 1500, num=100)
  y = np.linspace(0, 2500, num=100)
  X, Y = np.meshgrid(x,y)
  pdf = np.zeros(X.shape)
  for i in range(X.shape[0]):
    for j in range(X.shape[1]):
      pdf[i,j] = distr.pdf([X[i,j], Y[i,j]])
  key = 131+k
  ax = fig.add_subplot(key, projection = '3d')
  ax.plot_surface(X, Y, pdf, cmap = 'viridis')
  plt.xlabel("x1")
  plt.ylabel("x2")
  plt.title(f'PDF of Class {k+1}')
  pdf_lis5.append(pdf)
  ax.axes.zaxis.set_ticks([])
  if(k == 0):
    X1 = X
    X2 = Y
  elif(k == 1):
    Y1 = X
    Y2 = Y
  else:
    Z1 = X
    Z2 = Y
plt.tight_layout()
plt.show()


for idx, val in enumerate(pdf_lis5):
  if(idx == 0):
    X = X1
    Y = X2
  elif(idx == 1):
    X = Y1
    Y = Y2
  else:
    X = Z1
    Y = Z2
  plt.subplot(1,3,idx+1)
  plt.contour(X, Y, val, cmap='viridis')
  plt.xlabel("x1")
  plt.ylabel("x2")
  plt.title(f'Constant Density curve of  Class {idx}')
  if(idx == 0):
    plot_eig(P1,E_1[:,0])
    plot_eig(P1,E_1[:,1])
  elif(idx == 1):
    plot_eig(P2,E_2[:,0])
    plot_eig(P2,E_2[:,1])
  else:
    plot_eig(P3,E_3[:,0])
    plot_eig(P3,E_3[:,1])

plt.tight_layout()
plt.show()




ans = np.zeros((3,3))

for i in range(np.size(X11_d)):
  temp = getm(X11_d[i],X12_d[i])
  cl = identify(temp,Mean1,Mean2,Mean3,Cov1,Cov2,Cov3)
  cl = cl-1
  ans[cl][0] = ans[cl][0]+1

for i in range(np.size(X21_d)):
  temp = getm(X21_d[i],X22_d[i])
  cl = identify(temp,Mean1,Mean2,Mean3,Cov1,Cov2,Cov3)
  cl = cl-1
  ans[cl][1] = ans[cl][1]+1

for i in range(np.size(X31_d)):
  temp = getm(X31_d[i],X32_d[i])
  cl = identify(temp,Mean1,Mean2,Mean3,Cov1,Cov2,Cov3)
  cl = cl-1
  ans[cl][2] = ans[cl][2]+1

x_l = [1,2,3]
ax = sns.heatmap(ans, annot=True, fmt="f",xticklabels = x_l,yticklabels=x_l)
ax.set_xlabel('Actual Class',fontsize = 24)
ax.set_ylabel('Predicted Class',fontsize = 24)
ax.set_title('Confusion Matrix',fontsize = 30)



R5 = ROC(Cov1,Cov2,Cov3)
TPR5 = R5[0]
FPR5 = R5[1]
FNR5 = Fn(TPR5)

#case-1



Xf = np.zeros(np.size(X11)+np.size(X21)+np.size(X31))
Yf = np.zeros(np.size(X12)+np.size(X22)+np.size(X32))
for i in range(np.size(X11)):
  Xf[i] = X11[i]
  Yf[i] = X12[i]
for i in range(np.size(X21)):
  Xf[i+np.size(X11)] = X21[i]
  Yf[i+np.size(X11)] = X22[i]
for i in range(np.size(X31)):
  Xf[i+np.size(X21)+np.size(X11)] = X31[i]
  Yf[i+np.size(X21)+np.size(X11)] = X32[i]

cov_mat1 = covmat(Xf,Yf)
Cov1 = covmat(Xf,Yf)
Cov2 = covmat(Xf,Yf)
Cov3 = covmat(Xf,Yf)

E_v1,E_1 = LA.eig(Cov1)
E_v2,E_2 = LA.eig(Cov2)
E_v3,E_3 = LA.eig(Cov3)

P1 = [mean(X11),mean(X12)]
P2 = [mean(X21),mean(X22)]
P3 = [mean(X31),mean(X32)]



plt.style.use('seaborn-dark')
plt.rcParams['figure.figsize']=14,6
fig = plt.figure()

random_seed=1000
pdf_lis1 = []

for k in range(3):
  if(k == 0):
    D1 = get(X11)
    D2 = get(X12)
  elif(k == 1):
    D1 = get(X21)
    D2 = get(X22)
  else:
    D1 = get(X31)
    D2 = get(X32)
  mean_1 = mean(D1)
  mean_2 = mean(D2)
  mean_mat = np.array([mean_1,mean_2])
  sigma_1 = cov(D1,D1)
  sigma_2 = cov(D2,D2)
  
  distr = multivariate_normal(cov = cov_mat1, mean = mean_mat,seed = random_seed)
  x = np.linspace(0, 1500, num=100)
  y = np.linspace(0, 3500, num=100)
  X, Y = np.meshgrid(x,y)
  pdf = np.zeros(X.shape)
  for i in range(X.shape[0]):
    for j in range(X.shape[1]):
      pdf[i,j] = distr.pdf([X[i,j], Y[i,j]])
  key = 131+k
  ax = fig.add_subplot(key, projection = '3d')
  ax.plot_surface(X, Y, pdf, cmap = 'viridis')
  plt.xlabel("x1")
  plt.ylabel("x2")
  plt.title(f'PDF of Class {k+1}')
  pdf_lis1.append(pdf)
  ax.axes.zaxis.set_ticks([])
  if(k == 0):
    X1 = X
    X2 = Y
  elif(k == 1):
    Y1 = X
    Y2 = Y
  else:
    Z1 = X
    Z2 = Y

plt.tight_layout()
plt.show()


for idx, val in enumerate(pdf_lis1):
  if(idx == 0):
    X = X1
    Y = X2
  elif(idx == 1):
    X = Y1
    Y = Y2
  else:
    X = Z1
    Y = Z2
  plt.subplot(1,3,idx+1)
  plt.contour(X, Y, val, cmap='viridis')
  plt.xlabel("x1")
  plt.ylabel("x2")
  plt.title(f'Constant Density curve of Class {idx}')
  if(idx == 0):
    plot_eig(P1,E_1[:,0])
    plot_eig(P1,E_1[:,1])
  elif(idx == 1):
    plot_eig(P2,E_2[:,0])
    plot_eig(P2,E_2[:,1])
  else:
    plot_eig(P3,E_3[:,0])
    plot_eig(P3,E_3[:,1])

plt.tight_layout()
plt.show()




ans = np.zeros((3,3))

for i in range(np.size(X11_d)):
  temp = getm(X11_d[i],X12_d[i])
  cl = identify(temp,Mean1,Mean2,Mean3,Cov1,Cov2,Cov3)
  cl = cl-1
  ans[cl][0] = ans[cl][0]+1

for i in range(np.size(X21_d)):
  temp = getm(X21_d[i],X22_d[i])
  cl = identify(temp,Mean1,Mean2,Mean3,Cov1,Cov2,Cov3)
  cl = cl-1
  ans[cl][1] = ans[cl][1]+1

for i in range(np.size(X31_d)):
  temp = getm(X31_d[i],X32_d[i])
  cl = identify(temp,Mean1,Mean2,Mean3,Cov1,Cov2,Cov3)
  cl = cl-1
  ans[cl][2] = ans[cl][2]+1

x_l = [1,2,3]
ax = sns.heatmap(ans, annot=True, fmt="f",xticklabels = x_l,yticklabels=x_l)
ax.set_xlabel('Actual Class',fontsize = 24)
ax.set_ylabel('Predicted Class',fontsize = 24)
ax.set_title('Confusion Matrix',fontsize = 30)




R1 = ROC(Cov1,Cov2,Cov3)
TPR1 = R1[0]
FPR1 = R1[1]
FNR1 = Fn(TPR1)

#case 4

cov_mat4 = covmat(Xf,Yf)
cov_mat4[0][1] = 0
cov_mat4[1][0] = 0 


E_v1,E_1 = LA.eig(cov_mat4)
E_v2,E_2 = LA.eig(cov_mat4)
E_v3,E_3 = LA.eig(cov_mat4)

P1 = [mean(X11),mean(X12)]
P2 = [mean(X21),mean(X22)]
P3 = [mean(X31),mean(X32)]



plt.style.use('seaborn-dark')
plt.rcParams['figure.figsize']=14,6
fig = plt.figure()

random_seed=1000
pdf_lis4 = []

for k in range(3):
  if(k == 0):
    D1 = get(X11)
    D2 = get(X12)
  elif(k == 1):
    D1 = get(X21)
    D2 = get(X22)
  else:
    D1 = get(X31)
    D2 = get(X32)
  mean_1 = mean(D1)
  mean_2 = mean(D2)
  mean_mat = np.array([mean_1,mean_2])
  sigma_1 = cov(D1,D1)
  sigma_2 = cov(D2,D2)
  
  distr = multivariate_normal(cov = cov_mat4, mean = mean_mat,seed = random_seed)
  x = np.linspace(0, 1500, num=100)
  y = np.linspace(0, 3500, num=100)
  X, Y = np.meshgrid(x,y)
  pdf = np.zeros(X.shape)
  for i in range(X.shape[0]):
    for j in range(X.shape[1]):
      pdf[i,j] = distr.pdf([X[i,j], Y[i,j]])
  key = 131+k
  ax = fig.add_subplot(key, projection = '3d')
  ax.plot_surface(X, Y, pdf, cmap = 'viridis')
  plt.xlabel("x1")
  plt.ylabel("x2")
  plt.title(f'PDF of Class {k+1}')
  pdf_lis4.append(pdf)
  ax.axes.zaxis.set_ticks([])
  if(k == 0):
    X1 = X
    X2 = Y
  elif(k == 1):
    Y1 = X
    Y2 = Y
  else:
    Z1 = X
    Z2 = Y
plt.tight_layout()
plt.show()


for idx, val in enumerate(pdf_lis4):
  if(idx == 0):
    X = X1
    Y = X2
  elif(idx == 1):
    X = Y1
    Y = Y2
  else:
    X = Z1
    Y = Z2
  plt.subplot(1,3,idx+1)
  plt.contour(X, Y, val, cmap='viridis')
  plt.xlabel("x1")
  plt.ylabel("x2")
  plt.title(f'Constant Density curve of Class {idx}')
  if(idx == 0):
    plot_eig(P1,E_1[:,0])
    plot_eig(P1,E_1[:,1])
  elif(idx == 1):
    plot_eig(P2,E_2[:,0])
    plot_eig(P2,E_2[:,1])
  else:
    plot_eig(P3,E_3[:,0])
    plot_eig(P3,E_3[:,1])

plt.tight_layout()
plt.show()





ans = np.zeros((3,3))

for i in range(np.size(X11_d)):
  temp = getm(X11_d[i],X12_d[i])
  cl = identify(temp,Mean1,Mean2,Mean3,cov_mat4,cov_mat4,cov_mat4)
  cl = cl-1
  ans[cl][0] = ans[cl][0]+1

for i in range(np.size(X21_d)):
  temp = getm(X21_d[i],X22_d[i])
  cl = identify(temp,Mean1,Mean2,Mean3,cov_mat4,cov_mat4,cov_mat4)
  cl = cl-1
  ans[cl][1] = ans[cl][1]+1

for i in range(np.size(X31_d)):
  temp = getm(X31_d[i],X32_d[i])
  cl = identify(temp,Mean1,Mean2,Mean3,cov_mat4,cov_mat4,cov_mat4)
  cl = cl-1
  ans[cl][2] = ans[cl][2]+1

x_l = [1,2,3]
ax = sns.heatmap(ans, annot=True, fmt="f",xticklabels = x_l,yticklabels=x_l)
ax.set_xlabel('Actual Class',fontsize = 24)
ax.set_ylabel('Predicted Class',fontsize = 24)
ax.set_title('Confusion Matrix',fontsize = 30)




R4 = ROC(Cov1,Cov2,Cov3)
TPR4 = R4[0]
FPR4 = R4[1]
FNR4 = Fn(TPR4)

#case-3

cov3 = cov(Xf,Xf)*(np.size(Xf)-1) + cov(Yf,Yf)*(np.size(Yf) -1)
cov3 = cov3/(np.size(Xf)+np.size(Yf) - 1)

cov_mat3 = np.zeros((2,2))
cov_mat3[0][0]  = cov3
cov_mat3[1][1]  = cov3



E_v1,E_1 = LA.eig(cov_mat3)
E_v2,E_2 = LA.eig(cov_mat3)
E_v3,E_3 = LA.eig(cov_mat3)

P1 = [mean(X11),mean(X12)]
P2 = [mean(X21),mean(X22)]
P3 = [mean(X31),mean(X32)]


plt.style.use('seaborn-dark')
plt.rcParams['figure.figsize']=14,6
fig = plt.figure()

random_seed=1000
pdf_lis3 = []

for k in range(3):
  if(k == 0):
    D1 = get(X11)
    D2 = get(X12)
  elif(k == 1):
    D1 = get(X21)
    D2 = get(X22)
  else:
    D1 = get(X31)
    D2 = get(X32)
  mean_1 = mean(D1)
  mean_2 = mean(D2)
  mean_mat = np.array([mean_1,mean_2])
  sigma_1 = cov(D1,D1)
  sigma_2 = cov(D2,D2)
  
  distr = multivariate_normal(cov = cov_mat3, mean = mean_mat,seed = random_seed)
  x = np.linspace(-500, 1500, num=100)
  y = np.linspace(0, 3500, num=100)
  X, Y = np.meshgrid(x,y)
  pdf = np.zeros(X.shape)
  for i in range(X.shape[0]):
    for j in range(X.shape[1]):
      pdf[i,j] = distr.pdf([X[i,j], Y[i,j]])
  key = 131+k
  ax = fig.add_subplot(key, projection = '3d')
  ax.plot_surface(X, Y, pdf, cmap = 'viridis')
  plt.xlabel("x1")
  plt.ylabel("x2")
  plt.title(f'PDF of Class {k+1}')
  pdf_lis3.append(pdf)
  ax.axes.zaxis.set_ticks([])
  if(k == 0):
    X1 = X
    X2 = Y
  elif(k == 1):
    Y1 = X
    Y2 = Y
  else:
    Z1 = X
    Z2 = Y
plt.tight_layout()
plt.show()


for idx, val in enumerate(pdf_lis3):
  if(idx == 0):
    X = X1
    Y = X2
  elif(idx == 1):
    X = Y1
    Y = Y2
  else:
    X = Z1
    Y = Z2
  plt.subplot(1,3,idx+1)
  plt.contour(X, Y, val, cmap='viridis')
  plt.xlabel("x1")
  plt.ylabel("x2")
  plt.title(f'Constant Density curve of Class {idx}')
  if(idx == 0):
    plot_eig(P1,E_1[:,0])
    plot_eig(P1,E_1[:,1])
  elif(idx == 1):
    plot_eig(P2,E_2[:,0])
    plot_eig(P2,E_2[:,1])
  else:
    plot_eig(P3,E_3[:,0])
    plot_eig(P3,E_3[:,1])

plt.tight_layout()
plt.show()



ans = np.zeros((3,3))

for i in range(np.size(X11_d)):
  temp = getm(X11_d[i],X12_d[i])
  cl = identify(temp,Mean1,Mean2,Mean3,cov_mat3,cov_mat3,cov_mat3)
  cl = cl-1
  ans[cl][0] = ans[cl][0]+1

for i in range(np.size(X21_d)):
  temp = getm(X21_d[i],X22_d[i])
  cl = identify(temp,Mean1,Mean2,Mean3,cov_mat3,cov_mat3,cov_mat3)
  cl = cl-1
  ans[cl][1] = ans[cl][1]+1

for i in range(np.size(X31_d)):
  temp = getm(X31_d[i],X32_d[i])
  cl = identify(temp,Mean1,Mean2,Mean3,cov_mat3,cov_mat3,cov_mat3)
  cl = cl-1
  ans[cl][2] = ans[cl][2]+1

x_l = [1,2,3]
ax = sns.heatmap(ans, annot=True, fmt="f",xticklabels = x_l,yticklabels=x_l)
ax.set_xlabel('Actual Class',fontsize = 24)
ax.set_ylabel('Predicted Class',fontsize = 24)
ax.set_title('Confusion Matrix',fontsize = 30)




R3 = ROC(Cov1,Cov2,Cov3)
TPR3 = R3[0]
FPR3 = R3[1]
FNR3 = Fn(TPR3)

plt.plot(FPR1,TPR1)
plt.plot(FPR2,TPR2)
plt.plot(FPR3,TPR3)
plt.plot(FPR4,TPR4)
plt.plot(FPR5,TPR5)
plt.title("ROC curve")
plt.show()

from sklearn.metrics import DetCurveDisplay
fig,ax_det = plt.subplots(1,1,figsize = (20,10))
display = DetCurveDisplay(fpr=FPR1,fnr=FNR1).plot(ax = ax_det)
display = DetCurveDisplay(fpr=FPR2,fnr=FNR2).plot(ax = ax_det)
display = DetCurveDisplay(fpr=FPR3,fnr=FNR3).plot(ax = ax_det)
display = DetCurveDisplay(fpr=FPR4,fnr=FNR4).plot(ax = ax_det)
display = DetCurveDisplay(fpr=FPR5,fnr=FNR5).plot(ax = ax_det)
plt.show()