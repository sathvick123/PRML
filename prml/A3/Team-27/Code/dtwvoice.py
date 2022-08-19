import numpy as np
import os
import math
import matplotlib.pyplot as plt
import seaborn as sn
sn.set_theme()
import matplotlib.image as mpimg
from sklearn.metrics import DetCurveDisplay



def extractData(directory):
    res=[]
    for filename in os.listdir(directory):
      (filename,ext)= os.path.splitext(filename)
      if ext!='.mfcc':
          continue
      f=open(directory+'/'+filename+ext,"r")
      m=[]
      line = f.readlines()
      for lis in line[1:]:
        l=lis.split()
        l=list(map(float,l))
        m.append(l)
      res.append(m)
    return(np.array(res))

def dtw(s,t):
    m,n=len(s),len(t)
    ans=np.zeros((m+1,n+1))
    for i in range(m+1):
        for j in range(n+1):
            ans[i,j]=np.inf
    ans[0,0]=0
    s=np.array(s)
    t=np.array(t)
    for i in range(1,m+1):
        for j in range(1,n+1):
            x=s[i-1]-t[j-1]
            cost=np.dot(x.T,x)
            cost=math.sqrt(cost)
            ans[i,j]=min(ans[i-1,j-1],ans[i-1,j],ans[i,j-1])+cost
    return(ans[m,n])

tw_train=extractData("./2/train")
th_train=extractData("./3/train")
e_train=extractData("./8/train")
o_train=extractData("./o/train")
z_train=extractData("./z/train")

tw_dev=extractData("./2/dev")
th_dev=extractData("./3/dev")
e_dev=extractData("./8/dev")
o_dev=extractData("./o/dev")
z_dev=extractData("./z/dev")

anss=[]
C_dev=[]
C_pred=[]


def dtw_classify(dev,pres,a,b,c,d,e,k):

    lis=[]
    n=len(dev)
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)
    d=np.array(d)
    e=np.array(e)

    res=0
    for i in range(n):
        dis1 = []
        dis2 = []
        dis3 = []
        dis4 = []
        dis5 = []
        C_dev.append(pres)
        m = np.size(a)

        for j in range(m):
            dev[i]=np.array(dev[i])
            a[j]=np.array(a[j])
            dd=dtw(dev[i],a[j])
            dis1.append(dd)
        m = np.size(b)
        for j in range(m):
            dev[i] = np.array(dev[i])
            b[j] = np.array(b[j])
            dd =dtw(dev[i],b[j])
            dis2.append(dd)
        m = np.size(c)
        for j in range(m):
            dev[i] = np.array(dev[i])
            c[j] = np.array(c[j])
            dd = dtw(dev[i],c[j])
            dis3.append(dd)
        m = np.size(d)
        for j in range(m):
            dev[i] = np.array(dev[i])
            d[j] = np.array(d[j])
            dd = dtw(dev[i],d[j])
            dis4.append(dd)
        m = np.size(e)
        for j in range(m):
            dev[i] = np.array(dev[i])
            e[j] = np.array(e[j])
            dd = dtw(dev[i],e[j])
            dis5.append(dd)
        dis1 = sorted(dis1)
        dis2 = sorted(dis2)
        dis3 = sorted(dis3)
        dis4 = sorted(dis4)
        dis5 = sorted(dis5)

        temp=[]

        x=0
        for i in range(k):
           x+=dis1[i]
        temp.append(1/x)
        x = 0
        for i in range(k):
            x += dis2[i]
        temp.append(1 / x)
        x = 0
        for i in range(k):
            x += dis3[i]
        temp.append(1 / x)
        x = 0
        for i in range(k):
            x += dis4[i]
        temp.append(1 / x)
        x = 0
        for i in range(k):
            x += dis5[i]
        temp.append(1 / x)

        anss.append(temp)
        temp1=[]
        for i in range(5):
          temp1.append([temp[i],i])

        temp1=sorted(temp1)
        C_pred.append(temp1[0][1])
        if temp1[4][1]==pres:
            res+=1

    return(res)



def confmatr(n):
    matr=[[0 for j in range(5)] for i in range(5)]
    for cl in range(n):
        i=C_dev[cl]
        j=C_pred[cl]
        matr[i][j]=matr[i][j]+1
    x_l = [1, 2, 3,4,5]
    ax = sn.heatmap(matr, annot=True, fmt="f", xticklabels=x_l, yticklabels=x_l)
    ax.set_xlabel('Actual Class', fontsize=24)
    ax.set_ylabel('Predicted Class', fontsize=24)
    ax.set_title('Confusion Matrix', fontsize=30)
    plt.show()

def plot_ROC_curve(scores, n):
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
            ground_truth = C_dev[i]
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
    # plt.show()


    fig, ax_det = plt.subplots(1, 1, figsize=(20, 10))
    display = DetCurveDisplay(fpr=FPR, fnr=FNR).plot(ax=ax_det)


    return (FPR, tpr)

#
# for k in range(5,30,5):
#     anss=[]
#     C_dev=[]
#     C_pred=[]
#
#     a=len(ai_dev)
#     b=len(bA_dev)
#     c=len(iA_dev)
#     d=len(dA_dev)
#     e=len(tA_dev)
#
#     p=dtw_classify(ai_dev,0,ai_train,bA_train,iA_train,dA_train,tA_train,k)
#     q=dtw_classify(bA_dev,1,ai_train,bA_train,iA_train,dA_train,tA_train,k)
#     r=dtw_classify(iA_dev,2,ai_train,bA_train,iA_train,dA_train,tA_train,k)
#     s=dtw_classify(dA_dev,3,ai_train,bA_train,iA_train,dA_train,tA_train,k)
#     t=dtw_classify(tA_dev,4,ai_train,bA_train,iA_train,dA_train,tA_train,k)
#
#     n = len(anss)
#     plot_ROC_curve(anss, n)
#     confmatr(n)
k=5
anss=[]
C_dev=[]
C_pred=[]

a=len(tw_dev)
b=len(th_dev)
c=len(e_dev)
d=len(o_dev)
e=len(z_dev)

p=dtw_classify(tw_dev,0,tw_train,th_train,e_train,o_train,z_train,k)
q=dtw_classify(th_dev,1,tw_train,th_train,e_train,o_train,z_train,k)
r=dtw_classify(e_dev,2,tw_train,th_train,e_train,o_train,z_train,k)
s=dtw_classify(o_dev,3,tw_train,th_train,e_train,o_train,z_train,k)
t=dtw_classify(z_dev,4,tw_train,th_train,e_train,o_train,z_train,k)


# print(str(a)+" "+str(p))
# print(str(b)+" "+str(q))
# print(str(c)+" "+str(r))
# print(str(d)+" "+str(s))
# print(str(e)+" "+str(t))

n = len(anss)
plot_ROC_curve(anss, n)
confmatr(n)
plt.show()
acc=(p+q+r+s+t)/(a+b+c+d+e)
print("Accuracy is :"+str(acc*100))


