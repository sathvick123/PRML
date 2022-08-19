import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy as sc
from numpy import linalg as LA
from PIL import Image
import math
import sys
import random
import pandas as pd
from scipy.stats import multivariate_normal
import seaborn as sns
from sklearn.metrics import DetCurveDisplay
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
onehot_encoder = OneHotEncoder(sparse=False)

sns.set_theme()


f1 = open("train.txt", "r")
X_t = []
Y_t = []




for i in f1.readlines():
    x1, x2, t = i.strip().split(',')
    temp = []
    temp.append(np.longdouble(x1))
    temp.append(np.longdouble(x2))
    X_t.append(temp)
    Y_t.append(np.longdouble(t)-1)



f2 = open("dev.txt", "r")
X_d = []
Y_d = []

for i in f2.readlines():
    x1, x2, t = i.strip().split(',')
    temp = []
    temp.append(np.longdouble(x1))
    temp.append(np.longdouble(x2))
    X_d.append(temp)
    Y_d.append(np.longdouble(t) - 1)

def convert(X):
    ans = np.zeros((len(X),2))
    for i in range(len(X)):
        ans[i][0] = X[i][0]
        ans[i][1] = X[i][1]
    return ans

X_t = convert(X_t)
X_d = convert(X_d)

Y_t = np.array(Y_t)
Y_d = np.array(Y_d)



def accuracy(y_true, y_pred):
    ans = 0
    for i in range(np.size(y_true)):
        if(y_pred[i] == y_true[i]):
            ans+=1
    ans = ans/(np.size(y_true))
    return ans*(100)



def loss(X, Y, W):
    Z = - X @ W
    n = X.shape[0]
    ans = 1/n * (np.trace(X @ W @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
    return ans


def gradient(X, Y, W, m):
    Z = - X @ W
    prob = softmax(Z, axis=1)
    n = X.shape[0]
    ans = 1/n * (X.T @ (Y - prob)) + 2 * m * W
    return ans




def gradient_descent(X, Y, iter,lrate,mu):
    Y_onehot = onehot_encoder.fit_transform(Y.reshape(-1, 1))
    W = np.zeros((X.shape[1], Y_onehot.shape[1]))
    step = 0
    step_lst = []
    loss_lst = []
    W_lst = []

    for step in range(iter):
        W -= lrate * gradient(X, Y_onehot, W, mu)
        step_lst.append(step+1)
        W_lst.append(W)
        loss_lst.append(loss(X, Y_onehot, W))

    df = pd.DataFrame({
        'step': step_lst,
        'loss': loss_lst
    })
    return df, W


def Logisitic_Regression(Xt ,Yt ,Xd ,Yd ,iter,lrate,mu):
    loss_steps, W = gradient_descent(Xt, Yt,iter,lrate,mu)
    Z = - (Xd) @ W
    prob = softmax(Z, axis=1)
    Y_predicted =  np.argmax(prob, axis=1)
    print(accuracy(Yd,Y_predicted))
    mat = np.zeros((2, 2))
    for i in range(len(Yd)):
        mat[Y_predicted[i]][int(Yd[i])] += 1
    x_l = [1, 2]
    ax = sns.heatmap(mat, annot=True, fmt="f", xticklabels=x_l, yticklabels=x_l)
    ax.set_xlabel('Actual Class', fontsize=24)
    ax.set_ylabel('Predicted Class', fontsize=24)
    ax.set_title('Confusion Matrix', fontsize=30)
    plt.show()
    temp = []
    for i in range(prob.shape[0]):
        for j in range(prob.shape[1]):
            temp.append(prob[i][j])

    temp = np.array(temp)
    temp = np.sort(temp)

    TPR = []
    FPR = []
    FNR = []

    for kk in temp:
        TP = FP = TN = FN = 0
        for i in range(prob.shape[0]):
            for j in range(prob.shape[1]):
                if (prob[i][j] >= kk):
                    if (Yd[i] == j):
                        TP += 1
                    else:
                        FP += 1
                else:
                    if (Yd[i] == j):
                        FN += 1
                    else:
                        TN += 1

        TPR.append(TP / (TP + FN))
        FPR.append(FP / (FP + TN))
        FNR.append(1 - TP / (TP + FN))
    plt.title("ROC curve for Learning rate = " + str(lrate))
    plt.plot(FPR, TPR)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()

    fig, ax_det = plt.subplots(1, 1, figsize=(20, 10))
    plt.title('DET Curve')
    display = DetCurveDisplay(fpr=FPR, fnr=FNR).plot(ax=ax_det)
    plt.show()



Logisitic_Regression(X_t,Y_t,X_d,Y_d,1000,0.0001,0.01)