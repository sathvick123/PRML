Synthetic.py:- 

On calling classify_K(K,C) where K = no of clusters and C = true if covariance is diagonal otherwise C = false K_means contours will be displayed 
and also it returns a list which contains TPR , FPR and FNR. In the same way call classify_GMM(K,C) to get GMM contours .ROC and DET curves can be plotted using
the last segment of code in the file.


dtwhand.py and dtwvoice.py:-

For K th cluster change the global value K and run the code.


hmmhand.py and hmmvoice.py:-

On calling get(K) where K = no of clusters confusion matrix for that K will be displayed and also it returns a list which contains TPR,FPR and FNR.
ROC and DET curves can be plotted using last segement of code in the file.