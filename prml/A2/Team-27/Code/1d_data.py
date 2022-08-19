import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy as sc
from numpy import linalg as LA
from PIL import Image


import math
import sys

f1 = open("1d_team_27_dev.txt","r")
X_d = []
T_d = []
for i in f1.readlines():
  x,t = i.strip().split(' ')
  X_d.append(np.longdouble(x))
  T_d.append(np.longdouble(t))

X_d = np.array(X_d)
T_d = np.array(T_d)



f = open("1d_team_27_train.txt","r")
X = []
T = []
for i in f.readlines():
  x,t = i.strip().split(' ')
  X.append(np.longdouble(x))
  T.append(np.longdouble(t))

X = np.array(X)
T = np.array(T)

def basis(x,m) :
    temp = np.zeros(m)
    for i in range(m):
      temp[i] = pow(x,i)
    return temp

def reduce(Mat,siz):
  res = np.zeros(siz)
  for i in range(siz):
    res[i] = Mat[i]
  return res



def get(N,M,P):
  X_trnn = reduce(X,N)
  T_trnn = reduce(T,N)

  def calc(m):
    temp = []
    for i in range(N):
      temp.append(pow(X_trnn[i],m))
    temp = np.array(temp)
    return temp
  
 
   
  phi = np.zeros((N,M))

  for i in range(phi.shape[1]):
    phi[:,i] = calc(i)

  phi_t = phi.transpose()
  phi_1 = phi_t @ phi

  LI = np.zeros((M,M))
  for i in range(M):
     LI[i][i] = P 
 
  phi_2 = np.add(phi_1,LI)
  phi_inv2 = LA.inv(phi_2)
  W  = phi_inv2 @ phi_t @ T_trnn
  return W

def calc_reg(N,M,P,c):
  X_dev = reduce(X_d,N)
  T_dev = reduce(T_d,N)
  X_trn = reduce(X,N)
  T_trn = reduce(T,N)


  W = get(N,M,P)
  W_trans = W.transpose()
  T_caln = np.zeros(N)
  T_cald = np.zeros(N)
  

  for i in range(N):
    temp = basis(X_dev[i],M)
    temp2 = basis(X_trn[i],M)
    T_cald[i] = np.dot(W_trans,temp)
    T_caln[i] = np.dot(W_trans,temp2)
    
  if(c == 0):
    return T_caln
  else :
    return T_cald

def plot_reg(N,M,P):
  T_dev = reduce(T_d,N)
  T_trn = reduce(T,N)
  T_caln = calc_reg(N,M,P,0)
  T_cald = calc_reg(N,M,P,1)
  X_axis = np.zeros(N)
  for i in range(N):
    X_axis[i] = i

  plt.title("Training data plot with regularisation = "+str(P)+",N = "+str(N)+",M = "+str(M))
  plt.plot(X_axis,T_caln,'r')
  plt.plot(X_axis,T_trn,'g')
  plt.show()

  plt.title("Testing data plot with regularisation = "+str(P)+",N = "+str(N)+",M = "+str(M))
  plt.plot(X_axis,T_cald,'r')
  plt.plot(X_axis,T_dev,'g')
  
  plt.show()

def E_rms(N,M,P,c):
  T_dev = reduce(T_d,N)
  T_trn = reduce(T,N)
  T_caln = calc_reg(N,M,P,0)
  T_cald = calc_reg(N,M,P,1)
  E_tr = 0
  E_dv = 0
  for i in range(N):
    E_tr = E_tr + pow((T_trn[i] - T_caln[i]),2)
    E_dv = E_dv + pow((T_d[i] - T_cald[i]),2)
  W = get(N,M,P)
  norm = LA.norm(W)
  LC = (P/2)*(norm)*(norm)
  E_tr = E_tr +  LC
  E_dv = E_dv + LC
  E_tr = E_tr/N
  E_dv = E_dv/N
  E_tr = math.sqrt(E_tr)
  E_dv = math.sqrt(E_dv)

  if(c == 0):
    return E_tr
  else:
    return E_dv

E_tr = np.zeros(8)
E_dv = np.zeros(8)
X_axis = np.zeros(8)

for i in range(8):
  E_tr[i] = E_rms(200,i,0,0)
  E_dv[i] = E_rms(200,i,0,1)
  X_axis[i] = i

plt.xlabel("value of M")
plt.ylabel("Erms value")
plt.plot(X_axis,E_tr,color = 'b',label = 'training data')
plt.plot(X_axis,E_dv,color = 'r',label = 'development data')
plt.title('Erms Vs M graph')
plt.legend()
plt.show()

E_tr = np.zeros(8)
E_dv = np.zeros(8)
X_axis = np.zeros(8)
for i in range(8):
  P = math.exp(0 - 5*i)
  E_tr[i] = E_rms(200,9,P,0)
  E_dv[i] = E_rms(200,9,P,1)
  X_axis[i] = 0 - 5*i

plt.xlabel("value of ln(regularisation value)")
plt.ylabel("Erms value")
plt.plot(X_axis,E_tr,'b',label = 'training data')
plt.plot(X_axis,E_dv,'r',label = 'development data')
plt.title('Erms Vs ln(regularisation value) graph')
plt.legend()
plt.show()