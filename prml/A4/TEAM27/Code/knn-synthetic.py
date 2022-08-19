import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy as sc
import pandas as pd
from numpy import linalg as LA
from PIL import Image
import math
import sys
import random
from scipy.stats import multivariate_normal
import seaborn as sns;
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.image as mpimg
import seaborn as sn
sn.set_theme()
from sklearn.metrics import DetCurveDisplay

f1 = open("train.txt", "r")

X_t = []
Y_t = []
l_t = []

for i in f1.readlines():
    x1, x2, t = i.strip().split(',')
    p=int(t)
    X_t.append(np.longdouble(x1))
    Y_t.append(np.longdouble(x2))
    l_t.append(p-1)

f2 = open("dev.txt", "r")
X_d = []
Y_d = []
l_d=[]
l_p=[]
for i in f2.readlines():
    x1, x2, t = i.strip().split(',')
    X_d.append(np.longdouble(x1))
    Y_d.append(np.longdouble(x2))
    l_d.append((int)(t)-1)


X=[]
Y=[]
scores=[]
k=10
#for k in range(10,201,10):
count=0
n=len(X_d)
###
res=0
for i in range(len(X_d)):
    dist=[]
    for j in range(len(X_t)):
        diss = math.dist([X_t[j],Y_t[j]],[X_d[i],Y_d[i]])
        dist.append((diss,l_t[j]))

    dist.sort()
    count=np.zeros(2)
    for p in range(k):
        q = dist[p][1]
        count[q] += 1
    ind = np.argmax(count)
    if ind ==l_d[i]:
        res += 1
    l_p.append(ind)
    for p in range(2):
        count[p]=count[p]/k
    scores.append(count)
p=np.size(X_d)
acc=(res*100)/p
print("Accuracy is:",acc)
print(len(scores),len(l_p))
#X.append(k)
#Y.append(acc)
#plt.plot(X,Y)
#plt.show()

def plot_ROC_curve(scores,n):
    scores = np.array(scores)
    scores_mod = scores.flatten()
    scores_mod = np.sort(scores_mod)
    tpr = np.array([])
    FPR = np.array([])
    FNR = np.array([])
    tnr = np.array([])
    for threshold in scores_mod:
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(n):
            ground_truth = l_d[i]
            for j in range(2):
                if (scores[i][j] >= threshold):
                    if ground_truth == j:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if ground_truth == j:
                        fn += 1
                    else:
                        tn += 1
        tpr = np.append(tpr, tp / (tp + fn))
        FPR = np.append(FPR, fp / (fp + tn))
        FNR = np.append(FNR, fn / (tp + fn))
        tnr = np.append(tnr, tn / (tn + fp))
    plt.plot(FPR, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    fig, ax_det = plt.subplots(1, 1, figsize=(10, 10))
    display = DetCurveDisplay(fpr=FPR, fnr=FNR).plot(ax=ax_det)
    plt.show()
    return (FPR, tpr)

def confmatr(n):
    matr=[[0 for j in range(2)] for i in range(2)]
    for cl in range(n):
        i=l_d[cl]
        j=l_p[cl]
        matr[i][j]=matr[i][j]+1
    x_l = [1,2]
    ax = sn.heatmap(matr, annot=True, fmt="f", xticklabels=x_l, yticklabels=x_l)
    ax.set_xlabel('Actual Class', fontsize=24)
    ax.set_ylabel('Predicted Class', fontsize=24)
    ax.set_title('Confusion Matrix', fontsize=30)
    plt.show()

nn=len(scores)
plot_ROC_curve(scores,nn)
confmatr(nn)
