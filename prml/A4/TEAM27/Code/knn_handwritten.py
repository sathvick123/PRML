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

trn=["ai/train","bA/train","dA/train","tA/train","iA/train"]
deve=["ai/dev","bA/dev","dA/dev","tA/dev","iA/dev"]

def findmin(directory):
    l=[]
    for filename in os.listdir(directory):
        (filename,extension) = os.path.splitext(filename)
        if extension == '.txt':
            f=open(directory+'/'+filename+extension,"r")
            m=[]
            la = f.readlines()[0]
            line=la.split()
            lines=(int)(line[0])
            l.append(lines)
    return l

def avg(l,i,j):
    sum=[]
    n=len(l[i])
    for k in range(n):
        sum.append(l[i][k])
    for k in range(i+1,j+1):
        sum = [sum[p] + l[k][p] for p in range(len(sum))]
    for k in range(len(sum)):
        sum[k]=sum[k]/(j-i+1)
    return sum

def extractData(directory,k):
    data=[]
    for filename in os.listdir(directory):
        (filename,extension) = os.path.splitext(filename)
        if extension == '.txt':
            f=open(directory+'/'+filename+extension,"r")
            m=[]
            lines = f.readlines()
            s=lines[0]
            l=s.split()
            for i in range(1,len(l),2):
                x = float(l[i])
                y = float(l[i+1])
                m.append([x,y])
            d=pd.DataFrame(m)
            Min=d.min()
            Max=d.max()
            x_min=Min[0]
            y_min=Min[1]
            x_max=Max[0]
            y_max=Max[1]
            for i in range(len(m)):
                m[i][0]=(m[i][0]-x_min)/(x_max-x_min)
                m[i][1] = (m[i][1] - y_min) / (y_max - y_min)
            #n = len(m) - a
            z = []
            for i in range(k):
                n = len(m[i])
                temp = np.zeros(n)
                j = i + len(m) - k
                for p in range(n):
                    for s in range(i, j + 1):
                        temp[p] += m[s][p]
                    temp[p] = temp[p] / (j - i + 1)
                z.extend(temp)
            data.append(z)
    return data


arr=[]
for i in range(5):
    l=findmin(trn[i])
    arr.extend(l)

for i in range(5):
    l=findmin(deve[i])
    arr.extend(l)

mini=min(arr)

train=[]
dev=[]

for i in range(5):
    l=extractData(trn[i],mini)
    train.append(l)
points=0
for i in range(5):
    l=extractData(deve[i],mini)
    dev.append(l)
    points+=len(l)
r=[]
for i in range(5):
    r.extend(train[i])
scl=MinMaxScaler().fit(r)

scores=[]
l_d=[]
l_p=[]
#for k in range(20,101,20):
k=20
res=0
for di in range(5):
    for dp in dev[di]:
        dist = []
        for tj in range(5):
            for tp in train[tj]:
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