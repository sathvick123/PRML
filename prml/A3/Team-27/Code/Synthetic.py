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

sns.set_theme()

f1 = open("train.txt", "r")
X11 = []
X12 = []
X21 = []
X22 = []
X1f = []
X2f = []

for i in f1.readlines():
    x1, x2, t = i.strip().split(',')
    X1f.append(np.longdouble(x1))
    X2f.append(np.longdouble(x2))
    if (t == "1"):
        X11.append(np.longdouble(x1))
        X12.append(np.longdouble(x2))
    elif (t == "2"):
        X21.append(np.longdouble(x1))
        X22.append(np.longdouble(x2))

X11 = np.array(X11)
X12 = np.array(X12)
X21 = np.array(X21)
X22 = np.array(X22)
X1f = np.array(X1f)
X2f = np.array(X2f)

f2 = open("dev.txt", "r")
X_d = []
Y_d = []
X1_d = []
Y1_d = []
X2_d = []
Y2_d = []

for i in f2.readlines():
    x1, x2, t = i.strip().split(',')
    X_d.append(np.longdouble(x1))
    Y_d.append(np.longdouble(x2))
    if (t == "1"):
        X1_d.append(np.longdouble(x1))
        Y1_d.append(np.longdouble(x2))
    elif (t == "2"):
        X2_d.append(np.longdouble(x1))
        Y2_d.append(np.longdouble(x2))

X_d = np.array(X_d)
Y_d = np.array(Y_d)
X1_d = np.array(X1_d)
Y1_d = np.array(Y1_d)
X2_d = np.array(X2_d)
Y2_d = np.array(Y2_d)


def get_mean(K, X, Y):
    siz = np.size(X)
    temp = np.linspace(0, siz, siz, endpoint=False).astype(int)
    Mean = np.zeros((2, K))
    for i in range(K):
        ind = random.choice(temp)
        Mean[0][i] = X[ind]
        Mean[1][i] = Y[ind]
    return (Mean)


def samee(m1, m2):
    n = np.size(m1[1])
    for i in range(n):
        if m1[0][i] != m2[0][i] or m1[1][i] != m2[1][i]:
            return (False)
    return (True)


def get(K, X, Y):
    m_new = np.zeros((2, K))
    m_old = np.zeros((2, K))
    m_old = get_mean(K, X, Y)
    n = np.size(X)

    while (True):
        dic = {}
        for i in range(n):
            dist = np.zeros(K)
            for j in range(K):
                dist[j] = math.dist([X[i], Y[i]], [m_old[0][j], m_old[1][j]])
            ind = np.argmin(dist)
            if ind in dic:
                dic[ind].append(i)
            else:
                dic[ind] = [i]
                # i th point will be in ind indexed cluster

        for i in range(K):
            if i not in dic.keys():
                continue
            dic[i] = np.array(dic[i])
            s = np.size(dic[i])
            for j in range(s):
                ind = dic[i][j]
                m_new[0][i] += X[ind]
                m_new[1][i] += Y[ind]
            m_new[0][i] /= s
            m_new[1][i] /= s

        if (samee(m_new, m_old) == True):
            break
        for i in range(K):
            m_old[0][i] = m_new[0][i]
            m_old[1][i] = m_new[1][i]
    return m_old


def minclus(X, Y, Mean, K):
    dist = np.zeros(K)
    for j in range(K):
        dist[j] = math.dist([X, Y], [Mean[0][j], Mean[1][j]])
    ind = np.argmin(dist)
    return ind


def mincluss(X, Y, phi, Mean, cov, K):
    dist = np.zeros(K)
    x = np.zeros((2, 1))
    x[0][0] = X
    x[1][0] = Y
    for j in range(K):
        mean = np.zeros((2, 1))
        mean[0][0] = Mean[0][j]
        mean[1][0] = Mean[1][j]
        dist[j] = phi[j] * (Gauss(x, mean, cov[j]))
    ind = np.argmax(dist)
    return ind


def mean(X):
    ans = 0
    l = np.size(X)
    for i in range(l):
        ans = (ans + X[i])
    if (l == 0):
        l = 1
    ans = ans / l
    return ans


def cov(X, Y):
    ans = 0
    l = np.size(X)
    m1 = mean(X)
    m2 = mean(Y)
    for i in range(np.size(X)):
        ans = ans + (X[i] - m1) * (Y[i] - m2)
    if (l == 1):
        l = 2
    ans = ans / (l - 1)
    return ans


def covmat(X, Y):
    ans = np.zeros((2, 2))
    ans[0][0] = cov(X, X)
    ans[0][1] = cov(X, Y)
    ans[1][0] = cov(Y, X)
    ans[1][1] = cov(Y, Y)
    return ans


def getp(M, ind, X, Y, K):
    Xl = []
    Yl = []
    for i in range(np.size(X)):
        index = minclus(X[i], Y[i], M, np.size(M[1]))
        if (index == ind):
            Xl.append(X[i])
            Yl.append(Y[i])
    Xl = np.array(Xl)
    Yl = np.array(Yl)
    lst = []
    lst.append(Xl)
    lst.append(Yl)
    return lst


def getpp(X, Y, index, P, M, C, K):
    Xl = []
    Yl = []
    ans = []
    for i in range(np.size(X)):
        ind = mincluss(X[i], Y[i], P, M, C, K)
        if (ind == index):
            Xl.append(X[i])
            Yl.append(Y[i])
    ans.append(Xl)
    ans.append(Yl)
    return ans


def covlist(M, X, Y, K):
    ans = []
    for i in range(K):
        lst = getp(M, i, X, Y, K)
        Xlst = lst[0]
        Ylst = lst[1]
        cov = covmat(Xlst, Ylst)
        ans.append(cov)
    return ans


plt.style.use('seaborn-dark')
fig = plt.figure()


def db(Xx, Yy, mean, K):
    X_g = np.linspace(-20, 20, 100)
    Y_g = np.linspace(-20, 20, 100)
    X, Y = np.meshgrid(X_g, Y_g)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            ind = minclus(X[i, j], Y[i, j], mean, 2 * K)
            if (ind < K):
                plt.plot([X[i, j]], [Y[i, j]], color='green', marker='o')
            else:
                plt.plot([X[i, j]], [Y[i, j]], color='orange', marker='o')


def Gauss(X, M, C):
    const = 1 / math.pi
    det = math.sqrt(LA.det(C))
    det = det * 2
    const = const / det
    xmat = np.subtract(X, M)
    xmat_t = xmat.transpose()
    C_inv = LA.inv(C)
    const2 = xmat_t @ C_inv @ xmat
    const2 = const2 / 2
    const2 = const2 * (-1)
    ans = const * (math.exp(const2))
    return ans


def Gnk(X, K, M, C, P, ind):
    s = 0
    for i in range(K):
        mean = np.zeros((2, 1))
        mean[0][0] = M[0][i]
        mean[1][0] = M[1][i]
        s = s + P[i] * (Gauss(X, mean, C[i]))
    m1 = np.zeros((2, 1))
    m1[0][0] = M[0][ind]
    m1[1][0] = M[1][ind]
    ans = P[ind] * Gauss(X, m1, C[ind])
    ans = ans / s
    return ans


random_seed = 100



def score(X,Y,Mean,cl,K):
    if(cl == 1):
        t = 0
    else :
        t = K
    M = np.zeros((2,K))
    for i in range(K):
        M[0][i] = Mean[0][t+i]
        M[1][i] = Mean[1][i+t]
    ind = minclus(X,Y,M,K)
    ans = math.dist([X, Y], [M[0][ind], M[1][ind]])
    ans = 1 /(ans)
    return ans

def scoree(X,Y,Mean,Cov,Phi,K,cl):
    x = np.zeros((2,1))
    x[0][0] = X
    x[1][0] = Y
    if (cl == 1):
        t = 0
    else:
        t = K
    M = np.zeros((2, K))
    C = []
    P = np.zeros(K)
    temp = 0
    for i in range(K):
        M[0][i] = Mean[0][t + i]
        M[1][i] = Mean[1][i + t]
        mm = np.zeros((2,1))
        mm[0][0] = M[0][i]
        mm[1][0] = M[1][i]
        C.append(Cov[i+t])
        P[i] = Phi[i+t]
        temp += P[i]*(Gauss(x,mm,Cov[i+t]))
    ind = mincluss(X,Y,P,M,C,K)
    mm = np.zeros((2, 1))
    mm[0][0] = M[0][ind]
    mm[1][0] = M[1][ind]
    ans = P[ind]* Gauss(x,mm,C[ind])
    ans = ans/temp
    return ans


def ROC(Mean,K):

    N = np.size(X_d)
    N1 = np.size(X1_d)
    S = np.zeros((N,2))
    temp = []

    for i in range(N):
        for j in range(2):
            S[i][j] = score(X_d[i],Y_d[i],Mean,j+1,K)
            temp.append(S[i][j])

    temp = np.array(temp)
    temp = np.sort(temp)

    TPR = []
    FPR = []
    FNR = []
    for kk in temp:
        TP = FP = TN = FN = 0

        for i in range(N):
            for j in range(2):
                if(S[i][j] >= kk):
                    if(i<N1 and j == 0):
                        TP = TP+1
                    elif(i>=N1 and j == 1):
                        TP = TP+1
                    else:
                        FP = FP+1
                else:
                    if(i<N1 and j == 0):
                        FN = FN + 1
                    elif (i >= N1 and j == 1):
                        FN = FN+1
                    else:
                        TN = TN + 1

        TPR.append(TP/(TP+FN))
        FPR.append(FP/(FP+TN))
        FNR.append(1 - TP/(TP+FN))
    ans = []
    ans.append(TPR)
    ans.append(FPR)
    ans.append(FNR)
    return ans

def diag(C):
    C2 = []
    for i in range(len(C)):
        te = C[i]
        temp = np.zeros((2,2))
        temp[0][0] = te[0][0]
        temp[1][1] = te[1][1]
        C2.append(temp)
    return C2


def ROCC(Mean,Phi,Cov,K):
    N = np.size(X_d)
    N1 = np.size(X1_d)
    S = np.zeros((N, 2))
    T = np.zeros((N,2))
    Cov2 = diag(Cov)
    temp1 = []
    temp2 = []
    for i in range(N):
        for j in range(2):
            S[i][j] = scoree(X_d[i],Y_d[i],Mean,Cov,Phi,K,j+1)
            T[i][j] = scoree(X_d[i],Y_d[i],Mean,Cov2,Phi,K,j+1)
            temp1.append(S[i][j])
            temp2.append(T[i][j])

    temp1 = np.array(temp1)
    temp1 = np.sort(temp1)

    temp2 = np.array(temp2)
    temp2 = np.sort(temp2)

    TPR = []
    FPR = []
    FNR = []
    for kk in temp1:
        TP = FP = TN = FN = 0

        for i in range(N):
            for j in range(2):
                if (S[i][j] >= kk):
                    if (i < N1 and j == 0):
                        TP = TP + 1
                    elif (i >= N1 and j == 1):
                        TP = TP + 1
                    else:
                        FP = FP + 1
                else:
                    if (i < N1 and j == 0):
                        FN = FN + 1
                    elif (i >= N1 and j == 1):
                        FN = FN + 1
                    else:
                        TN = TN + 1

        TPR.append(TP / (TP + FN))
        FPR.append(FP / (FP + TN))
        FNR.append(1 - TP / (TP + FN))

    TPR1 = []
    FPR1 = []
    FNR1 = []
    for kk in temp2:
        TP = FP = TN = FN = 0

        for i in range(N):
            for j in range(2):
                if (S[i][j] >= kk):
                    if (i < N1 and j == 0):
                        TP = TP + 1
                    elif (i >= N1 and j == 1):
                        TP = TP + 1
                    else:
                        FP = FP + 1
                else:
                    if (i < N1 and j == 0):
                        FN = FN + 1
                    elif (i >= N1 and j == 1):
                        FN = FN + 1
                    else:
                        TN = TN + 1

        TPR1.append(TP / (TP + FN))
        FPR1.append(FP / (FP + TN))
        FNR1.append(1 - TP / (TP + FN))
    ans = []
    ans.append(TPR1)
    ans.append(FPR1)
    ans.append(FNR1)
    return ans



def classify_K(K,c2):
    Mean1 = get(K, X11, X12)
    Mean2 = get(K, X21, X22)
    Mean = np.zeros((2, 2 * K))
    for i in range(K):
        Mean[0][i] = Mean1[0][i]
        Mean[1][i] = Mean1[1][i]
    for i in range(K):
        Mean[0][K + i] = Mean2[0][i]
        Mean[1][K + i] = Mean2[1][i]
    Clusters = []
    for i in range(2 * K):
        Clusters.append([])
    for i in range(np.size(X_d)):
        ind = minclus(X_d[i], Y_d[i], Mean, 2 * K)
        Clusters[ind].append(i)

    CM = np.zeros((2, 2))
    for i in range(2 * K):
        for j in range(np.size(Clusters[i])):
            x = Clusters[i][j]
            x1 = 1
            y1 = 1
            if (i < K):
                x1 = 0
            if (x < np.size(X1_d)):
                y1 = 0
            CM[x1][y1] = CM[x1][y1] + 1

    accuracy = (CM[0][0] + CM[1][1]) / (CM[0][0] + CM[1][1] + CM[1][0] + CM[0][1]) * 100
    #print(accuracy)
    if(c2):
        strr = "Diagonal covariance matrices"
    else:
        strr = "Non diagonal covariance matrices"

    for k in range(2 * K):
        meann = np.zeros(2)
        meann[0] = Mean[0][k]
        meann[1] = Mean[1][k]
        if (k < K):
            lst = getp(Mean, k, X11, X12, K)
        else:
            lst = getp(Mean, k, X21, X22, K)
        Xlst = lst[0]
        Ylst = lst[1]
        covv = covmat(Xlst, Ylst)
        if(c2):
            covv[1][0] = 0
            covv[0][1] = 0
        distr = multivariate_normal(cov=covv, mean=meann, seed=random_seed)
        x = np.linspace(min(Xlst) - 1, max(Xlst) + 1, num=100)
        y = np.linspace(min(Ylst) - 1, max(Ylst) + 1, num=100)
        X, Y = np.meshgrid(x, y)
        pdf = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pdf[i, j] = distr.pdf([X[i, j], Y[i, j]])
        # plt.subplot(1, 2*K, k + 1)
        plt.contour(X, Y, pdf, cmap='viridis')
    plt.scatter(X_d, Y_d,color = "red")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Contours of K means cluster with K = "+str(K)+" and "+strr)
    plt.tight_layout()
    plt.show()
    ans = ROC(Mean,K)
    return ans


def get_GMM(K, X, Y):
    Mean = get(K, X, Y)
    N = np.size(X)
    covl = covlist(Mean, X, Y, K)
    Phi = np.zeros(K)

    for i in range(K):
        temp = getp(Mean, i, X, Y, K)
        Phi[i] = np.size(temp[0]) / np.size(X)

    G = np.zeros((N, K))

    for ind in range(3):

        for i in range(np.size(X)):
            for j in range(K):
                x = np.zeros((2, 1))
                x[0][0] = X[i]
                x[1][0] = Y[i]
                G[i][j] = Gnk(x, K, Mean, covl, Phi, j)

        for i in range(K):
            s = 0
            s1 = 0
            s2 = 0

            for j in range(np.size(X)):
                s = s + G[j][i]
                s1 = s1 + (G[j][i] * X[j])
                s2 = s2 + (G[j][i] * Y[j])
            Phi[i] = s / np.size(X)
            Mean[0][i] = s1 / s
            Mean[1][i] = s2 / s
            s3 = 0
            for kk in range(N):
                x = np.zeros((2, 1))
                mm = np.zeros((2, 1))
                x[0][0] = X[kk]
                x[1][0] = Y[kk]
                mm[0][0] = Mean[0][i]
                mm[1][0] = Mean[1][i]
                temp = np.subtract(x, mm) @ ((np.subtract(x, mm)).transpose())
                s3 = s3 + (temp) * (G[kk][i])
            covl[i] = s3 / s

    ans = []
    ans.append(Phi)
    ans.append(Mean)
    ans.append(covl)
    return ans


def classify_GMM(K,c1):
    if (c1):
        strr = "Diagonal covariance matrices"
    else:
        strr = "Non diagonal covariance matrices"
    lst1 = get_GMM(K, X11, X12)
    lst2 = get_GMM(K, X21, X22)
    Mean = np.zeros((2, 2 * K))
    covl = []

    Phi = np.zeros(2 * K)
    for i in range(2 * K):
        if (i < K):
            Phi[i] = lst1[0][i]
            if(c1):
                lst1[2][i][0][1] = 0
                lst1[2][i][1][0] = 0

            covl.append(lst1[2][i])
            Mean[0][i] = lst1[1][0][i]
            Mean[1][i] = lst1[1][1][i]
        else:
            Phi[i] = lst2[0][i - K]
            if (c1):
                lst2[2][i-K][0][1] = 0
                lst2[2][i-K][1][0] = 0

            covl.append(lst2[2][i - K])
            Mean[0][i] = lst2[1][0][i - K]
            Mean[1][i] = lst2[1][0][i - K]

    db(X11, X12, Mean, K)
    for k in range(2 * K):
        meann = np.zeros(2)
        meann[0] = Mean[0][k]
        meann[1] = Mean[1][k]
        covv = covl[k]

        if (k < K):
            temp = getp(Mean, k, X11, X12, K)
        else:
            temp = getp(Mean, k, X21, X22, K)

        if (np.size(temp[0]) > 0):
            distr = multivariate_normal(cov=covv, mean=meann, seed=random_seed)
            x = np.linspace(min(temp[0]) - 2, max(temp[0]) + 2, num=100)
            y = np.linspace(min(temp[1]) - 2, max(temp[1]) + 2, num=100)
            X, Y = np.meshgrid(x, y)
            pdf = np.zeros(X.shape)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    pdf[i, j] = distr.pdf([X[i, j], Y[i, j]])
            # plt.subplot(1, 2*K, k + 1)
            plt.contour(X, Y, pdf, cmap='viridis')
    plt.title("Contours of GMM cluster with K = " + str(K)+" and "+strr)
    plt.tight_layout()
    plt.show()
    ans = ROCC(Mean,Phi,covl,K)
    return ans






ans3 = classify_K(15,True)
ans4 = classify_K(15,False)
ans5 = classify_K(20,True)
ans6 = classify_K(20,False)


plt.plot(ans3[1],ans3[0],label = 'K = 15 and Diagonal Covariance matrices')
plt.plot(ans4[1],ans4[0],label = 'K = 15 and Non-Diagonal Covariance matrices')
plt.plot(ans5[1],ans5[0],label = 'K = 20 and Diagonal Covariance matrices')
plt.plot(ans6[1],ans6[0],label = 'K = 20 and Non-Diagonal Covariance matrices')
plt.title("ROC curve for K = 15 and 20")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()

from sklearn.metrics import DetCurveDisplay
fig, ax_det = plt.subplots(1, 1, figsize=(20, 10))
display = DetCurveDisplay(fpr=ans3[1], fnr=ans3[2]).plot(ax=ax_det,label = 'K = 15 and Diagonal Covariance matrices')
display = DetCurveDisplay(fpr=ans4[1], fnr=ans4[2]).plot(ax=ax_det,label = 'K = 15 and Non-Diagonal Covariance matrices')
display = DetCurveDisplay(fpr=ans5[1], fnr=ans5[2]).plot(ax=ax_det,label = 'K = 20 and Diagonal Covariance matrices')
display = DetCurveDisplay(fpr=ans6[1], fnr=ans6[2]).plot(ax=ax_det,label = 'K = 20 and Non-Diagonal Covariance matrices')
plt.title("DET curve for K = 15 and 20")
plt.legend()
plt.show()