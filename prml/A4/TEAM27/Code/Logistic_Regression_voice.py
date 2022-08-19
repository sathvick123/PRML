import numpy as np
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import matplotlib.image as mpimg
from sklearn.metrics import DetCurveDisplay
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
onehot_encoder = OneHotEncoder(sparse=False)


trn=["2/train","3/train","8/train","o/train","z/train"]
dev=["2/dev","3/dev","8/dev","o/dev","z/dev"]
t=[]
d=[]
l_t=[]
l_d=[]
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
    t.extend(l)
    for j in range(len(l)):
        l_t.append(i)

for i in range(5):
    l=extractData(dev[i],k)
    d.extend(l)
    for j in range(len(l)):
        l_d.append(i)

X_t = np.zeros((len(t),len(t[0])))
X_d = np.zeros((len(d),len(d[0])))
Y_t = np.array(l_t)
Y_d = np.array(l_d)

for i in range(len(t)):
    for j in range(len(t[0])):
        X_t[i][j] = t[i][j]

for i in range(len(d)):
    for j in range(len(d[0])):
        X_d[i][j] = d[i][j]


def accuracy(y_true, y_pred):
    ans = 0
    for i in range(np.size(y_true)):
        if(y_pred[i] == y_true[i]):
            ans+=1
    ans = ans/(np.size(y_true))
    return ans*(100)

def PCA(c):
    X = np.zeros((X_t.shape[0] + X_d.shape[0], X_t.shape[1]))
    for i in range(X_t.shape[0]):
        for j in range(X_t.shape[1]):
            X[i][j] = X_t[i][j]

    for i in range(X_d.shape[0]):
        for j in range(X_d.shape[1]):
            X[i + X_t.shape[0]][j] = X_d[i][j]

    X_m = X - np.mean(X, axis=0)
    cov_mat = np.cov(X_m, rowvar=False)
    eig_val,eig_vec = np.linalg.eigh(cov_mat)

    index_arr = np.argsort(eig_val)[::-1]
    eig_val_new = eig_val[index_arr]
    eig_vec_new = eig_vec[:, index_arr]
    eig_vec_final = eig_vec_new[:,0:c]
    ans = np.dot(eig_vec_final.transpose(),X_m.transpose()).transpose()
    X_nt = np.zeros((X_t.shape[0], c))
    X_nd = np.zeros((X_d.shape[0], c))

    for i in range(X_t.shape[0]):
        for j in range(c):
            X_nt[i][j] = ans[i][j]

    for i in range(X_d.shape[0]):
        for j in range(c):
            X_nd[i][j] = ans[i + X_t.shape[0]][j]
    return X_nt,X_nd

def LDA(comp):
    X = np.zeros((X_t.shape[0] + X_d.shape[0], X_t.shape[1]))
    Y = np.zeros(X_t.shape[0]+X_d.shape[0])
    for i in range(X_t.shape[0]):
        Y[i] = Y_t[i]
        for j in range(X_t.shape[1]):
            X[i][j] = X_t[i][j]

    for i in range(X_d.shape[0]):
        Y[i+X_t.shape[0]] = Y_d[i]
        for j in range(X_d.shape[1]):
            X[i + X_t.shape[0]][j] = X_d[i][j]
    ld = None
    clabels = np.unique(Y)
    X_m = np.mean(X, axis=0)
    mn = X.shape[1]
    s_w = np.zeros((mn,mn))
    s_b = np.zeros((mn, mn))
    for c in clabels:
        X_c = X[Y == c]
        mean_c = np.mean(X_c, axis=0)
        s_w += np.dot((X_c - mean_c).T,(X_c - mean_c))
        n_c = X_c.shape[0]
        mean_diff = (mean_c - X_m).reshape(mn, 1)
        s_b += n_c * (mean_diff).dot(mean_diff.T)
    A = np.linalg.inv(s_w).dot(s_b)
    eig_val, eig_vec = np.linalg.eigh(A)
    eig_vec = eig_vec.T
    index_arr = np.argsort(abs(eig_val))[::-1]
    eig_val = eig_val[index_arr]
    eig_vec = eig_vec[index_arr]
    ld = eig_vec[0:comp]
    ans = np.dot(X, ld.T)
    X_nt = np.zeros((X_t.shape[0], comp))
    X_nd = np.zeros((X_d.shape[0], comp))

    for i in range(X_t.shape[0]):
        for j in range(comp):
            X_nt[i][j] = ans[i][j]

    for i in range(X_d.shape[0]):
        for j in range(comp):
            X_nd[i][j] = ans[i + X_t.shape[0]][j]
    return X_nt, X_nd




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
        W = W - lrate * gradient(X, Y_onehot, W, mu)
        step_lst.append(step+1)
        W_lst.append(W)
        loss_lst.append(loss(X, Y_onehot, W))

    df = pd.DataFrame({
        'step': step_lst,
        'loss': loss_lst
    })
    return df, W


def Logisitic_Regression(Xt ,Yt ,Xd ,Yd ,iter,lrate,mu,d):
    if(d == True):
        st = "PCA"
    else:
        st = "LDA"
    loss_steps, W = gradient_descent(Xt, Yt,iter,lrate,mu)
    Z = - (Xd) @ W
    prob = softmax(Z, axis=1)
    Y_predicted =  np.argmax(prob, axis=1)
    print(accuracy(Yd,Y_predicted))
    mat = np.zeros((5, 5))
    for i in range(len(Yd)):
        mat[Y_predicted[i]][int(Yd[i])] += 1
    x_l = [1, 2,3,4,5]
    ax = sns.heatmap(mat, annot=True, fmt="f", xticklabels=x_l, yticklabels=x_l)
    ax.set_xlabel('Actual Class', fontsize=24)
    ax.set_ylabel('Predicted Class', fontsize=24)
    ax.set_title('Confusion Matrix for '+st, fontsize=30)
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
    ans = []
    ans.append(FPR)
    ans.append(TPR)
    ans.append(FNR)
    return ans

X1t,X1d = LDA(64)
X2t,X2d = PCA(64)





ans1 = Logisitic_Regression(X1t,Y_t,X1d,Y_d,1000,0.1,0.01,False)
ans2 = Logisitic_Regression(X2t,Y_t,X2d,Y_d,1000,0.1,0.01,True)


plt.title("ROC curve")
plt.plot(ans1[0],ans1[1],label = 'LDA 64-dimension')
plt.plot(ans2[0],ans2[1],label = 'PCA 64-dimension')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()

fig, ax_det = plt.subplots(1, 1, figsize=(20, 10))
display = DetCurveDisplay(fpr=ans1[0], fnr=ans1[2]).plot(ax=ax_det,label = 'LDA 64-dimension')
display = DetCurveDisplay(fpr=ans2[0], fnr=ans2[2]).plot(ax=ax_det,label = 'PCA 64-dimension')
plt.title("DET Curve")
plt.legend()
plt.show()


