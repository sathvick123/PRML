import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy as sc
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

train=["./coast/train","./forest/train","./highway/train","./mountain/train","./opencountry/train"]
devv=["./coast/dev","./forest/dev","./highway/dev","./mountain/dev","./opencountry/dev"]
l_t=[]
l_d=[]
l_p=[]

def extractData(directory,x,y):
    ans=[]
    for filename in os.listdir(directory):
        f = os.path.join(directory,filename)
        f=open(f,"r")
        res=[]
        for i in f.readlines():
            numbers = list(map(float, i.split()))
            res = res + numbers
        ans.append(res)
        if y==1:
          l_t.append(x)
        elif y==2:
          l_d.append(x)
    return(ans)
t=[]
d=[]

for i in range(5):
    t1=extractData(train[i],i,1)
    t.extend(t1)

for i in range(5):
    d1 = extractData(devv[i],i,2)
    d.extend(d1)

scl = MinMaxScaler().fit(t)
t=scl.transform(t)
d=scl.transform(d)

X=[]
Y=[]
scores=[]
points=len(d)

k=10
#for k in range(10,201,10):
res=0
for i in range(len(d)):
    dist = []
    for j in range(len(t)):
        diss=math.dist(d[i],t[j])
        dist.append((diss,l_t[j]))
    dist.sort()
    count=np.zeros(5)
    for p in range(k):
        q=dist[p][1]
        count[q]+=1

    ind=np.argmax(count)
    l_p.append(ind)
    if ind==l_d[i]:
        res+=1
    for i in range(5):
        count[i]=count[i]/k
    scores.append(count)

acc=(res*100)/points
print("k=",k,"accuracy is=",acc)

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
