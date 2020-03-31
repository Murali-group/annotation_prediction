import tqdm

import sklearn
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from scipy import sparse
import scipy
import seaborn as sns
import src.algorithms.alg_utils as alg_utils
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
import time
#from scikit.learn.svm.sparse import SVC


def training(P, labels, max_iter):
    '''
    *P* : features corresponding to the samples in the training set
    *labels*: labels corresponding to the samples in the training set
    '''
    
    
    clf = LinearSVC(max_iter=max_iter)
    clf.fit(P, labels)
    

    return clf

def testing(clf, P):
    
    predict = clf.decision_function(P)
    return predict
