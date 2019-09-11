# Python implementation of SinkSource

import time
import src.algorithms.alg_utils as alg_utils
import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse import linalg


def FastSinkSource(P, f, max_iters=1000, eps=0.0001, a=0.8, verbose=False):
    """ Power iteration until all of the scores of the unknowns have converged
    *max_iters*: maximum number of iterations
    *eps*: Epsilon convergence cutoff. If the maximum node score change from one iteration to the next is < *eps*, then stop
    *a*: alpha parameter

    *returns*: scores array, process_time, wall_time
        # of iterations needed to converge
    """
    # UPDATE 2018-12-21: just start at zero so the time taken measured here is correct
    s = np.zeros(len(f))
    prev_s = s.copy()

    wall_time = time.time()
    # this measures the amount of time taken by all processors
    # only available in Python 3
    process_time = time.process_time()
    for iters in range(1,max_iters+1):
        update_time = time.time()
        # s^(i+1) = aPs^(i) + f
        s = a*csr_matrix.dot(P, prev_s) + f

        if eps != 0:
            max_d = (s - prev_s).max()
            if verbose:
                print("\t\titer %d; %0.4f sec to update scores, max score change: %0.6e" % (iters, time.time() - update_time, max_d))
            if max_d <= eps:
                # converged!
                break
            prev_s = s.copy()
        else:
            prev_s = s
            if verbose:
                print("\t\titer %d; %0.4f sec to update scores" % (iters, time.time() - update_time))

    wall_time = time.time() - wall_time
    process_time = time.process_time() - process_time
    if verbose:
        print("SinkSource converged after %d iterations (%0.3f wall time (sec), %0.3f process time)" % (iters, wall_time, process_time))

    return s, process_time, wall_time, iters


def runFastSinkSource(P, positives, negatives=None, max_iters=1000, eps=0.0001, a=0.8, verbose=False):
    """
    *P*: Network as a scipy sparse matrix. Should already be normalized
    *positives*: numpy array of node ids to be used as positives
    *negeatives*: numpy array of node ids to be used as negatives. 
        If not given, will be run as SinkSourcePlus. 
        For SinkSourcePlus, if the lambda parameter is desired, it should already have been included in the graph normalization process. 
        See the function normalizeGraphEdgeWeights in alg_utils.py 
    *max_iters*: max # of iterations to run SinkSource. 
        If 0, use spsolve to solve the equation directly 
    """
    num_nodes = P.shape[0]
    if len(positives) == 0:
        print("WARNING: No positive examples given. Skipping.")
        return np.zeros(len(num_nodes)), 0,0,0
    # remove the positive and negative nodes from the graph 
    # and setup the f vector which contains the influence from positive and negative nodes
    newP, f, = alg_utils.setup_fixed_scores(
        P, positives, negatives, a=a, remove_nonreachable=False)

    if max_iters > 0:
        s, process_time, wall_time, num_iters = FastSinkSource(
            newP, f, max_iters=max_iters, eps=eps, a=a, verbose=verbose)
    else:
        # Solve for s directly. Scipy uses the form Ax=b to solve for x
        # SinkSource equation: (I - aP)s = f
        # TODO add an option to specify which solver to use
        # spsolve basically stalls for denser networks (e-value cutoff 0.1)
        solver = "spsolve"
        # eye is the identity matrix
        A = eye(P.shape[0]) - a*P
        # this measures the amount of time taken by all processors
        start_process_time = time.process_time()
        # this measures the amount of time that has passed
        start_wall_time = time.time()
        if solver == 'spsolve':
            num_iters = 0
            s = linalg.spsolve(A, f)
        process_time = time.process_time() - start_process_time 
        wall_time = time.time() - start_wall_time
        if verbose:
            print("Solved SS using %s (%0.3f sec, %0.3f process time). %s" % (
                solver, wall_time, process_time, 
                "iters: %d, max_iters: %d" % (num_iters, 1000) if solver == 'cg' else ''))

    # map back from the indices after deleting pos/neg to the original indices
    ## the positives will be left as 1, and the rest of the unknown examples will get their score below
    #scores_arr = np.ones(num_nodes)
    ## leave the negative examples at 0
    #if negatives is not None:
    #    scores_arr[negatives] = 0
    #indices = [idx2node[n] for n in range(len(s))]
    #scores_arr[indices] = s
    # keep the positive examples at 1
    s[positives] = 1

    return s, process_time, wall_time, num_iters


def runLocal(P, positives, negatives=None):
    """
    Baseline method where each node's score is the average score of its neighbors.
        Essentially one iteration of SinkSource.
    *P*: Network as a scipy sparse matrix. Should already be normalized
    *positives*: numpy array of node ids to be used as positives
    *negeatives*: numpy array of node ids to be used as negatives. 
        If not given, will be run as "LocalPlus". 
    """
    f = np.zeros(P.shape[0])
    f[positives] = 1
    if negatives is not None:
        f[negatives] = -1

    start_wall_time = time.time()
    start_process_time = time.process_time()
    s = csr_matrix.dot(P, f)
    wall_time = time.time() - start_wall_time
    process_time = time.process_time() - start_process_time

    return s, process_time, wall_time
