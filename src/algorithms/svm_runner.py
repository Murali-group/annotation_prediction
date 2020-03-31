import tqdm
from tqdm import tqdm, trange
#import scikit
from rpy2.robjects import *
import subprocess
from rpy2 import robjects as ro
import numpy as np
from scipy import sparse
import src.algorithms.fastsinksource_runner as fastsinksource
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
import logging


import src.algorithms.svm as svm

#from scikit.learn.svm.sparse import SVC

def setupInputs(run_obj):
    run_obj.ann_matrix = run_obj.ann_obj.ann_matrix
    run_obj.goids = run_obj.ann_obj.goids
    run_obj.prots = run_obj.ann_obj.prots
    run_obj.hpoidx = run_obj.ann_obj.goid2idx
    run_obj.protidx = run_obj.ann_obj.node2idx
    if run_obj.net_obj.weight_swsn:
        W, process_time = run_obj.net_obj.weight_SWSN(run_obj.ann_matrix)
        run_obj.params_results['%s_weight_time'%(run_obj.name)] += process_time
    else:
        W = run_obj.net_obj.W
    '''
    if run_obj.net_obj.influence_mat:
        run_obj.P = alg_utils.influenceMatrix(W, ss_lambda=run_obj.params.get('lambda', None))
    else:
    '''
    run_obj.P = W

    return

def setup_params_str(weight_str, params, name):
    max_iter=params['num_iter']
    return "{}-{}-maxi{}".format(weight_str, name, str_(max_iter))

def setupOutputs(run_obj):
    return

def run(run_obj):
    
    params_results = run_obj.params_results
    P, alg, params = run_obj.P, run_obj.name, run_obj.params
    max_iter = params['num_iter']

    # get the labels matrix and transpose it to have label names as columns
    ann_mat = run_obj.ann_matrix
    labels = ann_mat.transpose()    # genes x hpo

    
    if run_obj.train_mat is not None and run_obj.test_mat is not None:
        print("Performing Cross validation")
        run_obj.cv = True
        train_mat = run_obj.train_mat
        test_mat = run_obj.test_mat
    else:
        run_obj.cv = False
        train_mat = ann_mat
        test_mat = ann_mat


    scores = sparse.lil_matrix(ann_mat.shape, dtype=np.float)        #   dim: hpo x genes
    
    combined_scores = sparse.lil_matrix(ann_mat.shape, dtype=np.float) # dim: hpo x genes terms
    
    for term in tqdm(run_obj.goids_to_run):
        
        idx = run_obj.hpoidx[term]

        # get the training positive, negative sets for current fold 
        train_pos, train_neg = alg_utils.get_goid_pos_neg(train_mat,idx)
        train_set = sorted(list(set(train_pos)|set(train_neg)))
        
        if len(train_pos)==0:
            print("Skipping term, 0 positive examples")
            continue
        
        if run_obj.cv:
            # if cross validation, then obtain the test gene set on which classifier should be tested
            test_pos, test_neg = alg_utils.get_goid_pos_neg(test_mat, idx)
            test_set = set(test_pos) | set(test_neg)
            test_set = sorted(list(test_set))
        else:
            # set all unlabeled genes to the test set
            test_set = sorted(list(set(run_obj.protidx.values()) - set(train_set)))


         
        # obtain the feature vector only for the genes in the training set
        X_train = P[train_set, :]
        # obtain the feature vector only for the genes in the testing set
        X_test = P[test_set, :]
        # obtain the labels matrix corresponding to genes in the training set
        y_train = train_mat.transpose()[train_set, :]
        y_train = sparse.lil_matrix(y_train) 
        
        
        classifier = svm.training(X_train, y_train[:,idx].toarray().flatten(), max_iter)
        score_testSet = svm.testing(classifier, X_test)
        
        predict = score_testSet.tolist()
        
        # get the current scores for the given label l in current fold
        curr_score = scores[idx].toarray().flatten()
        # for the test indices of the current label, set the scores
        curr_score[test_set] = predict

        curr_score[train_pos] = 1

        # add the scores produced by predicting on the current label of test set to a combined score matrix
        scores[idx] = curr_score


    run_obj.goid_scores = scores
    run_obj.params_results = params_results

def str_(s):
    return str(s).replace('.','_')

