import tqdm

import sklearn
from sklearn import metrics
from scipy import sparse
import scipy
import src.algorithms.alg_utils as alg_utils
import time
from sklearn.linear_model import LogisticRegression

def training(P, labels, max_iter):
    '''
    *P* : features corresponding to the samples in the training set
    *labels*: labels corresponding to the samples in the training set
    '''
    
    logReg_clf =  LogisticRegression(max_iter=max_iter)
    logReg_clf.fit(P, labels)
    
    return logReg_clf

def testing(clf, P):
    
    predict = clf.decision_function(P)
    return predict
