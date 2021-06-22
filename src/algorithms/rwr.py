# Python implementation of SinkSource

import time
import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve, cg, spilu, LinearOperator
from tqdm import tqdm
import sys
from . import alg_utils


def RWR(P, y, max_iters=1000, eps=0.0001, a=0.95, verbose=False):
    """ Power iteration until all of the scores of the unknowns have converged
    *max_iters*: maximum number of iterations
    *eps*: Epsilon convergence cutoff. If the maximum node score change from one iteration to the next is < *eps*, then stop
    *a*: alpha parameter
    *y*: restart vector

    *returns*: scores array, process_time, wall_time
        # of iterations needed to converge
    """
    # UPDATE 2018-12-21: just start at zero so the time taken measured here is correct
    s = np.zeros(len(y))
    prev_s = s.copy()

    wall_time = time.time()
    # this measures the amount of time taken by all processors
    # only available in Python 3
    process_time = time.process_time()
    for iters in range(1,max_iters+1):
        update_time = time.time()
        # s^(i+1) = aPs^(i) + (1-a)y
        s = a*csr_matrix.dot(P, prev_s) + (1-a)*y

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
            #if verbose:
            #    print("\t\titer %d; %0.4f sec to update scores" % (iters, time.time() - update_time))

    wall_time = time.time() - wall_time
    process_time = time.process_time() - process_time
    #if verbose:
    #    print("SinkSource converged after %d iterations (%0.3f wall time (sec), %0.3f process time)" % (iters, wall_time, process_time))

    return s, process_time, wall_time, iters


def LazyRW(P, y, a=0.95, tol=1e-05, verbose=False):
    """
    *y*: vector of positive and negative assignments. 
         If y does not contain negatives, will be run as GeneManiaPlus, also known as Regularized Laplacian (RL). 
    *alpha*: parameter between 0 and 1 to control the influence of neighbors in the network.
        0 would ignore the network completely, and nodes would get their original score.
    *tol*: Conjugate Gradient tolerance for convergance

    *returns*: scores array, process_time, wall_time
        # of iterations needed to converge
    """
    # this measures the amount of time taken by all processors
    # and Conjugate Gradient is paralelized
    start_process_time = time.process_time()
    # this measures the amount of time that has passed
    start_wall_time = time.time()
    # I - aP
    M = eye(P.shape[0]) - a*P

    # keep track of the number of iterations
    num_iters = 0
    def callback(xk):
        # keep the reference to the variable within the callback function (Python 3)
        nonlocal num_iters
        num_iters += 1

    # use scipy's conjugate gradient solver
    s, info = cg(M, y, tol=tol, callback=callback)
    process_time = time.process_time() - start_process_time
    wall_time = time.time() - start_wall_time
    if verbose:
        print("Solved LazyRW using conjugate gradient (%0.2f sec, %0.2f sec process_time). Info: %s, iters=%d" % (
            wall_time, process_time, str(info), num_iters))

    return s, process_time, wall_time, num_iters


# def runRWR(
#         P, positives, negatives=None,
#         max_iters=1000, eps=0.0001, a=0.8,
#         tol=1e-5, solver=None, Milu=None, verbose=False):
#     """
#     *P*: Network as a scipy sparse matrix. Should already be normalized
#     *positives*: numpy array of node ids to be used as positives
#     *negatives*: numpy array of node ids to be used as negatives. 
#         If not given, will be run as FastSinkSourcePlus. 
#         For FastSinkSourcePlus, if the lambda parameter is desired, it should already have been included in the graph normalization process. 
#         See the function normalizeGraphEdgeWeights in alg_utils.py 
#     *max_iters*: max # of iterations to run SinkSource. 
#         If 0, use spsolve to solve the equation directly 
#     """
#     num_nodes = P.shape[0]
#     if len(positives) == 0:
#         print("WARNING: No positive examples given. Skipping.")
#         return np.zeros(num_nodes), 0,0,0
#     # remove the positive and negative nodes from the graph 
#     # and setup the f vector which contains the influence from positive and negative nodes
#     newP, f, = alg_utils.setup_fixed_scores(
#         P, positives, negatives, a=a, remove_nonreachable=False)

#     if solver is None:
#         s, process_time, wall_time, num_iters = FastSinkSource(
#             newP, f, max_iters=max_iters, eps=eps, a=a, verbose=verbose)
#     else:
#         # Solve for s directly. Scipy uses the form Ax=b to solve for x
#         # SinkSource equation: (I - aP)s = f
#         #solvers = ['bicg', 'bicgstab', 'cg', 'cgs', 'gmres', 'lgmres', 'minres', 'qmr', 'gcrotmk']#, 'lsmr']
#         # eye is the identity matrix
#         #M = eye(P.shape[0]) - a*P
#         M = eye(newP.shape[0]) - a*newP

#         # keep track of the number of iterations
#         def callback(xk):
#             # keep the reference to the variable within the callback function (Python 3)
#             nonlocal num_iters
#             num_iters += 1

#         # this measures the amount of time taken by all processors
#         start_process_time = time.process_time()
#         # this measures the amount of time that has passed
#         start_wall_time = time.time()
#         num_iters = 0
#         # spsolve basically stalls for large or dense networks (e.g., e-value cutoff 0.1)
#         if solver == 'spsolve':
#             s = linalg.spsolve(M, f)
#         elif solver == 'bicg':
#             s, info = linalg.bicg(M, f, tol=tol, maxiter=max_iters, M=Milu, callback=callback)
#         elif solver == 'bicgstab':
#             s, info = linalg.bicgstab(M, f, tol=tol, maxiter=max_iters, M=Milu, callback=callback)
#         elif solver == 'cg':
#             s, info = linalg.cg(M, f, tol=tol, maxiter=max_iters, M=Milu, callback=callback)
#         elif solver == 'cgs':
#             s, info = linalg.cgs(M, f, tol=tol, maxiter=max_iters, M=Milu, callback=callback)
#         elif solver == 'gmres':
#             s, info = linalg.gmres(M, f, tol=tol, maxiter=max_iters, M=Milu, callback=callback)
#         elif solver == 'lgmres':
#             s, info = linalg.lgmres(M, f, tol=tol, maxiter=max_iters, M=Milu, callback=callback)
#         elif solver == 'minres':
#             s, info = linalg.minres(M, f, tol=tol, maxiter=max_iters, M=Milu, callback=callback)
#         elif solver == 'qmr':
#             s, info = linalg.qmr(M, f, tol=tol, maxiter=max_iters, M=Milu, callback=callback)
#         elif solver == 'gcrotmk':
#             s, info = linalg.gcrotmk(M, f, tol=tol, maxiter=max_iters, M=Milu, callback=callback)
#         #elif solver == 'lsmr':
#         #    s, info = linalg.lsmr(M, f, maxiter=max_iters, callback=callback)

#         process_time = time.process_time() - start_process_time 
#         wall_time = time.time() - start_wall_time
#         if verbose:
#             print("Solved SS using %s (%0.3f sec, %0.3f process time). %s" % (
#                 solver, wall_time, process_time, 
#                 "iters: %d, max_iters: %d, info: %s" % (
#                     num_iters, 1000, info) if solver != 'spsolve' else ''))

#     # keep the positive examples at 1
#     s[positives] = 1

#     return s, process_time, wall_time, num_iters

