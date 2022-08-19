import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import scipy as sc
import pandas as pd
from numpy import linalg as LA
from PIL import Image
import math
import sys
import random
from scipy.stats import multivariate_normal
from sklearn.preprocessing import MinMaxScaler
import matplotlib.image as mpimg
import seaborn as sn
sn.set_theme()
from sklearn.metrics import DetCurveDisplay
f1 = "train.txt"
f2 = "dev.txt"

def fetch_data(f):
    file = open(f, "r")
    X1 = []
    X2 = []
    C = []
    X = []
    while True:
        line = file.readline()
        if not line:
            break
        x, y, c = line.strip().split(',')
        if c == '1':
            X1.append((np.longdouble(x), np.longdouble(y)))
        elif c == '2':
            X2.append((np.longdouble(x), np.longdouble(y)))
        C.append(int(c)-1)
        X.append((np.longdouble(x), np.longdouble(y)))

    X1 = np.array(X1)
    X2 = np.array(X2)
    return (C,X1,X2, X)

c_train,x1_train,x2_train, x_train = fetch_data(f1)
c_test,x1_test,x2_test, x_test = fetch_data(f2)

def Plot_scatter(X1, X2):
    (A, B) = X1[:,0],X1[:,1]
    (C, D) = X2[:,0],X2[:,1]
    plt.scatter(A, B, c="blue")
    plt.scatter(C, D, c="red")
    plt.suptitle("Decision Boundaries with contours")
    plt.title("Blue - Class1,Red - Class2")
    plt.xlabel("dimension1")
    plt.ylabel("dimension2")

def plot_svm_boundary(model):
    Plot_scatter(x1_train, x2_train)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)
    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')
    plt.show()


model = SVC(kernel="rbf", C=20,probability=True)
model.fit(x_train,c_train)
plot_svm_boundary(model)
prediction = model.predict(x_test)
p = list(prediction)
count = 0
scores = model.predict_proba(x_test)


for i in range(len(p)):
    if p[i] == c_test[i]:
        count += 1
print(f"Accuracy is : {count / len(p) * 100} %")


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
            ground_truth = c_test[i]
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
        i=c_test[cl]
        j=p[cl]
        #print(j)
        matr[i][j]=matr[i][j]+1
    for i in range(2):
        for j in range(2):
            matr[i][j]=(matr[i][j]*100)/n
    x_l = [1,2]
    ax = sn.heatmap(matr, annot=True, fmt="f", xticklabels=x_l, yticklabels=x_l)
    ax.set_xlabel('Actual Class', fontsize=24)
    ax.set_ylabel('Predicted Class', fontsize=24)
    ax.set_title('Confusion Matrix', fontsize=30)
    plt.show()

nn=len(scores)
plot_ROC_curve(scores,nn)
confmatr(nn)
