import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import MinMaxScaler
import seaborn as sn
sn.set_theme()
from sklearn.metrics import DetCurveDisplay

l_t=[]
l_d=[]
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
        if(y==1):
            l_t.append(x)
        else:
            l_d.append(x)
    return(ans)

t=[]
d=[]

X1_t= extractData("./coast/train",0,1)
t.extend(X1_t)
X1_d = extractData("./coast/dev",0,2)
d.extend(X1_d)
X2_t = extractData("./forest/train",1,1)
t.extend(X2_t)
X2_d = extractData("./forest/dev",1,2)
d.extend(X2_d)
X3_t = extractData("./highway/train",2,1)
t.extend(X3_t)
X3_d = extractData("./highway/dev",2,2)
d.extend(X3_d)
X4_t = extractData("./mountain/train",3,1)
t.extend(X4_t)
X4_d = extractData("./mountain/dev",3,2)
d.extend(X4_d)
X5_t = extractData("./opencountry/train",4,1)
t.extend(X5_t)
X5_d = extractData("./opencountry/dev",4,2)
d.extend(X5_d)


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
    X_nt = []
    X_nd = []

    for i in range(X_t.shape[0]):
        X_nt.append(ans[i])

    for i in range(X_d.shape[0]):
        X_nd.append(ans[i+X_t.shape[0]])
    return X_nt,X_nd



def LDA(comp):
    X = np.zeros((X_t.shape[0] + X_d.shape[0], X_t.shape[1]))
    Y = np.zeros(X_t.shape[0]+X_d.shape[0]);
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

    X_nt = []
    X_nd = []

    for i in range(X_t.shape[0]):
        X_nt.append(ans[i])

    for i in range(X_d.shape[0]):
        X_nd.append(ans[i + X_t.shape[0]])
    return X_nt, X_nd


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


def ANN(t,l_t,d,l_d,dd):
    clf = MLPClassifier(random_state=10, max_iter=5000).fit(t, l_t)

    prediction = clf.predict(d)
    p = list(prediction)
    scores = clf.predict_proba(d)

    n = len(p)
    x = 0

    for i in range(n):
        if p[i] == l_d[i]:
            x += 1

    print("Accuracy is :", x / n * 100, "%")

    if(dd == True):
        st = "PCA"
    else:
        st = "LDA"
    matr = [[0 for j in range(5)] for i in range(5)]
    for cl in range(n):
        i = l_d[cl]
        j = p[cl]
        matr[i][j] = matr[i][j] + 1
    x_l = [1, 2, 3, 4, 5]
    ax = sn.heatmap(matr, annot=True, fmt="f", xticklabels=x_l, yticklabels=x_l)
    ax.set_xlabel('Actual Class', fontsize=24)
    ax.set_ylabel('Predicted Class', fontsize=24)
    ax.set_title('Confusion Matrix for '+st, fontsize=30)
    plt.show()

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
    ans = []
    ans.append(FPR)
    ans.append(tpr)
    ans.append(FNR)
    return ans




X1t,X1d = LDA(64)
X2t,X2d = PCA(64)





ans1 = ANN(X1t,l_t,X1d,l_d,False)
ans2 = ANN(X2t,l_t,X2d,l_d,True)


plt.title("ROC curve")
plt.plot(ans1[0],ans1[1],label = 'LDA 4-dimension')
plt.plot(ans2[0],ans2[1],label = 'PCA 4-dimension')
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
