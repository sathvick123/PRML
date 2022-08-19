import numpy as np
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import seaborn as sns
sns.set_theme()

def extractData(directory, c):
    res1 = []
    res2 = []
    for fn in os.listdir(directory):
        (fn, e) = os.path.splitext(fn)
        if (e == '.mfcc'):
            f = open(directory + '/' + fn + e, "r")
            l = f.readlines()
            temp = []
            for t in l[1:]:
                d = t.split()
                d = list(map(float, d))
                res1 += [d]
                temp.append(d)
            
            res2.append(np.array(temp))

    if (c == True):
        return res1
    else:
        return res2


train_d = []
train_d += extractData("./2/train", True)
train_d += extractData("./3/train", True)
train_d += extractData("./8/train", True)
train_d += extractData("./o/train", True)
train_d += extractData("./z/train", True)
train_data = np.array(train_d)





S = 3
A = ["./2/train", "./3/train", "./8/train", "./o/train", "./z/train"]
B = ["./2/dev", "./3/dev", "./8/dev", "./o/dev", "./z/dev"]





def get_prob(a, b, c, d, e):
    n = len(a)
    s_c = len(b)
    sy_c = len(d)
    al = np.zeros(n + 1)
    al[0] = 1
    for i in range(n):
        temp = np.zeros(n + 1)
        for j in range(s_c):
            temp[j] += al[j] * b[j] * d[j][a[i]]
            if j > 0:
                temp[j] += al[j - 1] * c[j - 1] * e[j - 1][a[i]]
        al, temp = temp, al
    return np.sum(al)



def ROC(kmeans,current,curr_prob,next,next_prob):
    tt = []
    siz = 0
    for j in range(5):
        tt.append(extractData(B[j], False))
        siz= siz+len(tt[j])
    
    
    temp = np.zeros((siz,5))
    thre = []
    ind = 0
    for i in range(5):
        for j in range(len(tt[i])):
            for k in range(5):
                kmeansl = kmeans.predict(tt[i][j])
                temp[ind][k] = get_prob(kmeansl,current[k],next[k],curr_prob[k],next_prob[k])
                thre.append(temp[ind][k])
            ind = ind+1
    thre = np.array(thre)
    thre = np.sort(thre)
   
    TPR = []
    FPR = []
    FNR = []
   
    for thres in thre:
        
        TP = FP = TN = FN = 0
        ind = 0
        for i in range(5):
            for j in range(len(tt[i])):
                for k in range(5):
                    if(temp[ind][k] >= thres):
                        if(i == k):
                            TP+=1
                        else:
                            FP+=1
                    else:
                        if(i == k):
                            FN+=1
                        else:
                            TN+=1
                ind = ind+1           
    
                        
        TPR.append(TP/(TP+FN))
        FPR.append(FP/(FP+TN))
        FNR.append(1 - TP/(TP+FN))
   
    ans = []
    ans.append(TPR)
    ans.append(FPR)
    ans.append(FNR)
    return ans
       


def get(K):
    kmeans = KMeans(n_clusters=K, max_iter=50, random_state=0).fit(train_data)
    lab = kmeans.labels_
    current = []
    next = []
    curr_prob = []
    next_prob = []
    ind = 0
    for i in range(5):
         r = A[i]
         aa = []
         bb = []
         cc = []
         dd = []

         f = open("C.seq", "w")

         for fn in os.listdir(r):
             
             (fn, e) = os.path.splitext(fn)

             if (e == '.mfcc'):
                 ff = open(r + '/' + fn + e, "r")
                 l = ff.readlines()
                 d = l[0].split()
                 d = list(map(float, d))

                 for iii in range(int(d[1])):
                     s = str(lab[ind]) + " "
                     ind = ind + 1
                     f.write(s)

                 f.write("\n")

         f.close()
         command = "./train_hmm C.seq 100000 " + str(S) + " " + str(K) + " 0.001"
         os.system(command)
         f1 = open("C.seq.hmm", "r")
         fr = f1.readlines()
         x = 0
         bbb = []
         ddd = []
         for t in fr[2:]:
             bb = []
             dd = []
             if t.strip() == '':
                 continue
             d = t.split()
             d = list(map(float, d))
             if (x % 2 == 0):
                 for j in range(len(d)):       
                     if (j == 0):
                         aa.append(d[j])
                     else:
                         bb.append(d[j])
             else:
                 for j in range(len(d)):
                     if (j == 0):
                         cc.append(d[j])
                     else:
                         dd.append(d[j])
             x = x + 1
             if(len(bb)>0):
                 bbb.append(bb)
             if(len(dd)>0):
                 ddd.append(dd)
         current.append(aa)
         curr_prob.append(bbb)
         next.append(cc)
         next_prob.append(ddd)
         f1.close()
    
    covmat = np.zeros((5,5))



    for i in range(5):
        tt =  extractData(B[i], False)
        for ii in range(len(tt)):
            ttt = tt[ii]
            kmeansp = kmeans.predict(ttt)
            temp = []
            for j in range(5):
                temp.append(get_prob(kmeansp,current[j],next[j],curr_prob[j],next_prob[j]))

            ind = np.argmax(temp)
            covmat[ind][i] = covmat[ind][i]+1


    x_l = [1,2,3,4,5]
    ax = sns.heatmap(covmat, annot=True, fmt="f",xticklabels = x_l,yticklabels=x_l)
    ax.set_xlabel('Actual Class',fontsize = 24)
    ax.set_ylabel('Predicted Class',fontsize = 24)
    ax.set_title('Confusion Matrix for K = '+str(K),fontsize = 30)
    plt.show()
    ans = ROC(kmeans,current,curr_prob,next,next_prob)
    return ans
  
ans1 = get(10)
ans2 = get(15)
ans3 = get(20)
ans4 = get(25)
ans5 = get(30)
plt.plot(ans1[1],ans1[0],label = "K = 10")
plt.plot(ans2[1],ans2[0],label = "K = 15")
plt.plot(ans3[1],ans3[0],label = "K = 20")
plt.plot(ans4[1],ans4[0],label = "K = 25")
plt.plot(ans5[1],ans5[0],label = "K = 30")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for K = 10,15,20,25,30")
plt.legend()
plt.show()

from sklearn.metrics import DetCurveDisplay
fig,ax_det = plt.subplots(1,1,figsize = (20,10))
display = DetCurveDisplay(fpr = ans1[1],fnr = ans1[2]).plot(ax = ax_det,label = "K = 10")
display = DetCurveDisplay(fpr = ans2[1],fnr = ans2[2]).plot(ax = ax_det,label = "K = 15")
display = DetCurveDisplay(fpr = ans3[1],fnr = ans3[2]).plot(ax = ax_det,label = "K = 20")
display = DetCurveDisplay(fpr = ans4[1],fnr = ans4[2]).plot(ax = ax_det,label = "K = 25")
display = DetCurveDisplay(fpr = ans5[1],fnr = ans5[2]).plot(ax = ax_det,label = "K = 30")
plt.legend()
plt.show()
