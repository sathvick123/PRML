
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy as sc
from numpy import linalg as LA
from PIL import Image
import math
import sys

f1 = open("2d_team_27_dev.txt","r")
X1_d = []
X2_d = []
T_d = []
for i in f1.readlines():
  x1,x2,t = i.strip().split(' ')
  X1_d.append(np.longdouble(x1))
  X2_d.append(np.longdouble(x2))
  T_d.append(np.longdouble(t))

X1_d = np.array(X1_d)
X2_d = np.array(X2_d)
T_d = np.array(T_d)



f = open("2d_team_27_train.txt","r")
X1 = []
X2 = []
T = []
for i in f.readlines():
  x1,x2,t = i.strip().split(' ')
  X1.append(np.longdouble(x1))
  X2.append(np.longdouble(x2))
  T.append(np.longdouble(t))

X1 = np.array(X1)
X2 = np.array(X2)
T = np.array(T)

def basis(x1,x2,m) :
    m_1 = m*(m+1)//2
    temp = np.zeros(m_1)
    j_dum = 0
    k_dum = 0
    for i in range(m_1):
      ta = pow(x1,j_dum)
      tb = pow(x2,k_dum)
      temp[i] = ta*tb
      if(j_dum == 0):
        j_dum = k_dum+1
        k_dum = 0
      else :
        j_dum = j_dum -1
        k_dum = k_dum + 1
    return temp

def reduce(Mat,siz):
  res = np.zeros(siz)
  for i in range(siz):
    res[i] = Mat[i]
  return res



def get(N,M,P):
  X1_trnn = reduce(X1,N)
  X2_trnn = reduce(X2,N)
  T_trnn = reduce(T,N)

  def calc(x,y):
    temp = []
    for i in range(N):
      a = pow(X1_trnn[i],x)
      b = pow(X2_trnn[i],y)
      temp.append(a*b)
    temp = np.array(temp)
    return temp
  
 
  M_1 = M*(M+1)//2
  phi = np.zeros((N,M_1))
  j_dum = 0
  k_dum = 0

  for i in range(phi.shape[1]):
    phi[:,i] = calc(j_dum,k_dum)

    if(j_dum == 0):
      j_dum = k_dum + 1
      k_dum = 0
    else :
      j_dum = j_dum - 1
      k_dum = k_dum + 1

  phi_t = phi.transpose()
  phi_1 = phi_t @ phi

  LI = np.zeros((M_1,M_1))
  for i in range(M_1):
     LI[i][i] = P 
 
  phi_2 = np.add(phi_1,LI)
  phi_inv2 = LA.inv(phi_2)
  W  = phi_inv2 @ phi_t @ T_trnn
  return W

def calc_reg(N,M,P,c):
  X1_dev = reduce(X1_d,N)
  X2_dev = reduce(X2_d,N)
  T_dev = reduce(T_d,N)
  X1_trn = reduce(X1,N)
  X2_trn = reduce(X2,N)
  T_trn = reduce(T,N)


  W = get(N,M,P)
  W_trans = W.transpose()
  T_caln = np.zeros(N)
  T_cald = np.zeros(N)
  

  for i in range(N):
    temp = basis(X1_dev[i],X2_dev[i],M)
    temp2 = basis(X1_trn[i],X2_trn[i],M)
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

  plt.title("Training data plot scatter with regularisation= "+str(P)+",N = "+str(N)+",M = "+str(M))
  plt.xlabel('Calculated data point')
  plt.ylabel('Training data points')
  plt.scatter(T_caln,T_trn)
  plt.show()


  
  plt.title("Development data plot scatter with regularisation= "+str(P)+",N = "+str(N)+",M = "+str(M))
  plt.xlabel('Calculated data point')
  plt.ylabel('Development points')
  plt.scatter(T_cald,T_dev)
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

plt.title('Erms Vs M value')
plt.xlabel('M value')
plt.ylabel('Erms value')
plt.plot(X_axis,E_tr,'b',label = 'Training data')
plt.plot(X_axis,E_dv,'r',label = 'development data')
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

plt.title('Erms Vs ln(regularisation value)')
plt.xlabel('ln(regularisation value)')
plt.ylabel('Erms value')
plt.plot(X_axis,E_tr,'b',label = 'Training data')
plt.plot(X_axis,E_dv,'r',label = 'Development data')
plt.legend()
plt.show()