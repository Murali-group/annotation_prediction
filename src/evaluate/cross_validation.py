# needed for cross-validation
#import run_eval_algs
import os
import src.setup_sparse_networks as setup
import src.algorithms.alg_utils as alg_utils
import src.utils.file_utils as utils
import src.evaluate.eval_utils as eval_utils
from tqdm import tqdm, trange
import numpy as np
from scipy import sparse
try:
    from sklearn.model_selection import KFold
except ImportError:
    pass


def run_cv_all_terms(
        alg_runners, ann_obj, folds=5, num_reps=1, 
        cv_seed=None, **kwargs):
    """
    Split the positives and negatives into folds across all terms
    and then run the algorithms on those folds.
    Algorithms are all run on the same split of data. 
    *num_reps*: Number of times to repeat cross-validation. 
    An output file will be written for each repeat
    *cv_seed*: Seed to use for the random number generator when splitting the annotations into folds
        If *num_reps* > 1, the seed will be incremented by 1 each time
    """
    ann_matrix = ann_obj.ann_matrix
    terms, prots = ann_obj.terms, ann_obj.prots

    # set the cv_seed if specified
    # 2019-06-26 BUG: If there are a different number of terms, or the order of the terms changed, then the results won't be the same
    #if cv_seed is not None:
    #    print("\nSetting the Random State seed to %d" % (cv_seed))
    #    np.random.seed(cv_seed)

    # first check to see if the algorithms have already been run
    # and if the results should be overwritten
    if kwargs['forcealg'] is True or len(terms) == 1:
        # runners_to_run is a list of runners for each repitition
        runners_to_run = {i: alg_runners for i in range(1,num_reps+1)}
    else:
        runners_to_run = {}
        # a different file is stored for each repitition, so check each one
        for rep in range(1,num_reps+1):
            curr_runners_to_run = [] 
            curr_seed = cv_seed
            if curr_seed is not None:
                # add the current repitition number to the seed
                curr_seed += rep-1
            for run_obj in alg_runners:
                out_file = "%s/cv-%dfolds-rep%d%s%s.txt" % (
                    run_obj.out_dir, folds, rep,
                    "-seed%s"%curr_seed if curr_seed is not None else "", run_obj.params_str)
                if os.path.isfile(out_file):
                    print("%s already exists. Use --forcealg to overwite" % (out_file))
                else:
                    curr_runners_to_run.append(run_obj)
            runners_to_run[rep] = curr_runners_to_run

    # repeat the CV process the specified number of times
    for rep in range(1,num_reps+1):
        if len(runners_to_run[rep]) == 0:
            continue
        curr_seed = cv_seed
        if curr_seed is not None:
            # add the current repitition number to the seed
            curr_seed += rep-1
        # split the annotation matrix into training and testing matrices K times
        ann_matrix_folds = split_cv_all_terms(ann_obj, folds=folds, seed=curr_seed, **kwargs)

        for run_obj in runners_to_run[rep]:
            # because each fold contains a different set of positives, and combined they contain all positives,
            # store all of the prediction scores from each fold in a matrix
            combined_fold_scores = sparse.lil_matrix(ann_matrix.shape, dtype=np.float)
            for curr_fold, (train_ann_mat, test_ann_mat) in enumerate(ann_matrix_folds):
                print("*  "*20)
                print("Fold %d" % (curr_fold+1))
    
                # change the annotation matrix to the current fold
                curr_ann_obj = setup.Sparse_Annotations(run_obj.ann_obj.dag_matrix, train_ann_mat, terms, prots)
                # replace the ann_obj in the runner with the current fold's annotations  
                run_obj.ann_obj = curr_ann_obj
                run_obj.train_mat = train_ann_mat
                run_obj.test_mat = test_ann_mat
                #alg_runners = run_eval_algs.setup_runners([alg], alg_settings, net_obj, curr_ann_obj, **kwargs)
                # now setup the inputs for the runners
                run_obj.setupInputs()
                # run the alg
                run_obj.run()
                # parse the outputs. Only needed for the algs that write output files
                run_obj.setupOutputs()

                # store only the scores of the test (left out) positives and negatives
                for i in range(len(terms)):
                    test_pos, test_neg = alg_utils.get_term_pos_neg(test_ann_mat, i)
                    curr_term_scores = run_obj.term_scores[i].toarray().flatten()
                    curr_comb_scores = combined_fold_scores[i].toarray().flatten()
                    curr_comb_scores[test_pos] = curr_term_scores[test_pos]
                    curr_comb_scores[test_neg] = curr_term_scores[test_neg]
                    combined_fold_scores[i] = curr_comb_scores 

            # replace the term_scores in the runner to combined_fold_scores to evaluate
            run_obj.term_scores = combined_fold_scores 

            # now evaluate the results and write to a file
            out_file = "%s/cv-%dfolds-rep%d%s%s.txt" % (
                run_obj.out_dir, folds, rep,
                "-seed%s"%curr_seed if curr_seed is not None else "", run_obj.params_str)
            utils.checkDir(os.path.dirname(out_file)) 
            eval_utils.evaluate_ground_truth(
                run_obj, ann_obj, out_file,
                #non_pos_as_neg_eval=opts.non_pos_as_neg_eval,
                alg=run_obj.name, append=False, **kwargs)

    print("Finished running cross-validation")
    return


def split_cv_all_terms(ann_obj, folds=5, seed=None, **kwargs):
    """
    Split the positives and negatives into folds across all terms
    *seed*: the seed used by the random number generator when generating the folds. If None, the np.random RandomState will be used
    *returns*: a list of tuples containing the (train pos, train neg, test pos, test neg)
    """
    ann_matrix = ann_obj.ann_matrix
    terms, prots = ann_obj.terms, ann_obj.prots
    print("Splitting all annotations into %d folds by splitting each terms annotations into folds, and then combining them" % (folds))
    # TODO there must be a better way to do this than getting the folds in each term separately
    # but thi at least ensures that each term has evenly split annotations
    # list of tuples containing the (train pos, train neg, test pos, test neg) 
    ann_matrix_folds = []
    for i in range(folds):
        train_ann_mat = sparse.lil_matrix(ann_matrix.shape, dtype=np.float)
        test_ann_mat = sparse.lil_matrix(ann_matrix.shape, dtype=np.float)
        ann_matrix_folds.append((train_ann_mat, test_ann_mat))

    for i in trange(ann_matrix.shape[0]):
        term = terms[i]
        positives, negatives = alg_utils.get_term_pos_neg(ann_matrix, i)
        # if there are less positives or negatives than there are folds, this will give an error
        if len(positives) < folds or len(negatives) < folds:
            continue
        # print("%d positives, %d negatives for term %s" % (len(positives), len(negatives), term))
        # split the set of positives and the set of negatives into K folds separately
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        kf.get_n_splits(positives)
        kf.get_n_splits(negatives)
        fold = 0

        # now combine the positive and negative sets into a single array, and store it in the corresponding training or testing matrix
        for (pos_train_idx, pos_test_idx), (neg_train_idx, neg_test_idx) in zip(kf.split(positives), kf.split(negatives)):
            train_pos, test_pos= positives[pos_train_idx], positives[pos_test_idx]
            train_neg, test_neg = negatives[neg_train_idx], negatives[neg_test_idx]
            train_ann_mat, test_ann_mat = ann_matrix_folds[fold]
            fold += 1
            # build an array of positive and negative assignments and set it in corresponding annotation matrix
            for pos, neg, mat in [(train_pos, train_neg, train_ann_mat), (test_pos, test_neg, test_ann_mat)]:
                pos_neg_arr = np.zeros(len(prots))
                pos_neg_arr[list(pos)] = 1
                pos_neg_arr[list(neg)] = -1
                mat[i] = pos_neg_arr

    return ann_matrix_folds
