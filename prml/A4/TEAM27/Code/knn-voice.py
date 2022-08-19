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

trn=["2/train","3/train","8/train","o/train","z/train"]
dev=["2/dev","3/dev","8/dev","o/dev","z/dev"]
t=[]
d=[]
l_t=[]
l_d=[]
l_p=[]
scores=[]
def point_sizes(directory):
    l=[]
    for filename in os.listdir(directory):
        (filename,extension) = os.path.splitext(filename)
        if extension == '.mfcc':
            f=open(directory+'/'+filename+extension,"r")
            m=[]
            lines = f.readlines()
            x=len(lines)-1
            l.append(x)
    return l

def extractData(directory,k):
    data=[]
    for filename in os.listdir(directory):
        (filename,extension) = os.path.splitext(filename)
        if extension == '.mfcc':
            f=open(directory+'/'+filename+extension,"r")
            m=[]
            lines = f.readlines()
            for line in lines[1:]:
                x=line.split()
                x = list(map(float, x))
                m.append(x)
            z=[]
            for i in range(k):
                n = len(m[i])
                temp = np.zeros(n)
                j=i+len(m)-k
                for p in range(n):
                    for s in range(i, j + 1):
                        temp[p] += m[s][p]
                    temp[p] = temp[p] / (j - i + 1)
                z.extend(temp)
            data.append(z)
    return data

temp=[]

for i in range(10):
    if i<5:
      l=point_sizes(trn[i])
      temp.extend(l)
    else:
      l = point_sizes(dev[i-5])
      temp.extend(l)
k=min(temp)

for i in range(5):
    l=extractData(trn[i],k)
    t.append(l)
    for j in range(len(l)):
        l_t.append(i)
points=0
for i in range(5):
    l=extractData(dev[i],k)
    d.append(l)
    points+=len(l)
    for j in range(len(l)):
        l_d.append(i)

#for k in range(20,101,20):
k=20
res=0
for di in range(5):
    for dp in d[di]:
        dist = []
        for tj in range(5):
            for tp in t[tj]:
                diss=math.dist(tp,dp)
                dist.append((diss,tj))
        dist.sort()
        count=np.zeros(5)
        for p in range(k):
            q=dist[p][1]
            count[q]+=1
        ind=np.argmax(count)
        if ind==di:
            res+=1
        for i in range(5):
            count[i]=count[i]/k
        scores.append(count)
        l_d.append(di)
        l_p.append(ind)
acc=(res*100)/points
print("k=",k,"accuracy is=",acc,"%")

########################################################

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
            for j in range(5):
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
    matr=[[0 for j in range(5)] for i in range(5)]
    for cl in range(n):
        i=l_d[cl]
        j=l_p[cl]
        matr[i][j]=matr[i][j]+1
    x_l = [1, 2, 3,4,5]
    ax = sn.heatmap(matr, annot=True, fmt="f", xticklabels=x_l, yticklabels=x_l)
    ax.set_xlabel('Actual Class', fontsize=24)
    ax.set_ylabel('Predicted Class', fontsize=24)
    ax.set_title('Confusion Matrix', fontsize=30)
    plt.show()

nn=len(scores)
plot_ROC_curve(scores,nn)
confmatr(nn)
