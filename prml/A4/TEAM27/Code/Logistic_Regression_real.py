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


def extractData(directory):
    ans=[]
    for filename in os.listdir(directory):
        f = os.path.join(directory,filename)
        f=open(f,"r")
        res=[]
        for i in f.readlines():
            numbers = list(map(float, i.split()))
            res = res + numbers
        ans.append(res)
    return(ans)


coast_train=extractData("./coast/train")
forest_train=extractData("./forest/train")
highway_train=extractData("./highway/train")
mountain_train=extractData("./mountain/train")
opencountry_train=extractData("./opencountry/train")



X_t = coast_train+ forest_train+ highway_train+ mountain_train+ opencountry_train
Y_t = np.zeros(len(X_t))
k = len(coast_train)
for i in range(len(forest_train)):
    Y_t[k] = 1
    k+=1
for i in range(len(highway_train)):
    Y_t[k] = 2
    k+=1
for i in range(len(mountain_train)):
    Y_t[k] = 3
    k+=1
for i in range(len(opencountry_train)):
    Y_t[k] = 4
    k+=1



coast_dev=extractData("./coast/dev")
forest_dev=extractData("./forest/dev")
highway_dev=extractData("./highway/dev")
mountain_dev=extractData("./mountain/dev")
opencountry_dev=extractData("./opencountry/dev")


X_d = coast_dev + forest_dev + highway_dev + mountain_dev + opencountry_dev
Y_d = np.zeros(len(X_d))
k = len(coast_dev)
for i in range(len(forest_dev)):
    Y_d[k] = 1
    k+=1
for i in range(len(highway_dev)):
    Y_d[k] = 2
    k+=1
for i in range(len(mountain_dev)):
    Y_d[k] = 3
    k+=1
for i in range(len(opencountry_dev)):
    Y_d[k] = 4
    k+=1

def convert(X):
    ans = np.zeros((len(X),len(X[0])))
    for i in range(len(X)):
        for j in range(len(X[0])):
            ans[i][j] = X[i][j]
            ans[i][j] = X[i][j]
    return ans

X_t = convert(X_t)
X_d = convert(X_d)



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





ans1 = Logisitic_Regression(X1t,Y_t,X1d,Y_d,20000,0.01,0.01,False)
ans2 = Logisitic_Regression(X2t,Y_t,X2d,Y_d,20000,0.01,0.01,True)


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


