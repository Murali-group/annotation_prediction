
import time
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix, vstack, hstack, eye
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve
from tqdm import tqdm, trange
from .. import alg_utils as alg_utils

__author__ = "Jeff Law"
# python implementation of the BirgRank algorithm 
# see https://github.rcac.purdue.edu/mgribsko/aptrank for the original matlab implementation
# full citation:
#Jiang, B., Kloster, K., Gleich, D. F., & Gribskov, M., Aptrank: an adaptive pagerank model for protein function prediction on bi-relational graphs, Bioinformatics, 33(12), 1829-1836 (2017).  http://dx.doi.org/10.1093/bioinformatics/btx029


def birgRank(G, Rtrain, dH, alpha=.5, theta=.5, mu=.5, 
        eps=0, max_iters=0, nodes=None, verbose=False):
    """
    *Rtrain*: Matrix containing 0s and 1s for annotations. 
        Rows are nodes, columns are goids
    *alpha*: teleportation parameter in Personalized PageRank.
    *theta*: (1-theta) percent of Rtrain used in seeding vectors.
    *mu*: (1-mu) percent of random walkers diffuse from G via Rtrain to H.
    *eps*: use power iteration to approximate the scores, and iterate until x_i - x_i-1 < eps. 
        Default is 0 which means the solution will be solved directly.
        Useful as scipy's solver struggles with these large matrices
    *max_iters*: maximum # of iterations to run power iteration.
    *nodes*: set of nodes for which to run RWR and get GO term scores.
        Only used if eps or max_iters is not 0

    *returns*: Xh - m*n matrix of scores for each GO term, protein pair, same size as Rtrain
    """

    print("Starting birgRank")
    # m: # nodes, n: # terms
    m, n = Rtrain.shape

    # make seeding vectors (matrix)
    # update: specify in B the restart nodes  
    #B = vstack([theta*eye(m), (1-theta)*Rtrain.T]).tocsc()
    if nodes is None:
        nodes = np.arange(m)
    restart_vec = np.zeros(m)
    restart_vec[nodes] = 1 
    restart_nodes_diag = sp.diags(restart_vec)
    # limit Rtrain to the nodes for which we want scores
    restart_ann_mat = Rtrain.T.dot(restart_nodes_diag)
    B = vstack([theta*restart_nodes_diag, 
                (1-theta)*restart_ann_mat]).tocsc()

    # column normalize the matrix
    B = alg_utils.normalizeGraphEdgeWeights(B, axis=0)
    B = (1-alpha)*B

    # make transition matrix
    #G = G/2 + G.T/2  # make sure G is symmetric
    # final matrix is:
    # [G  0]
    # [RT H]
    P = vstack([hstack([mu*G, csc_matrix((m,n))]), hstack([(1-mu)*Rtrain.T, dH])]).tocsc()
    # try reversing the R connection and see if that makes a difference
    #P = vstack([hstack([mu*G, (1-mu)*Rtrain]), hstack([csc_matrix((n,m)), dH])]).tocsc()
    # normalize using the same normalization as AptRank
    P = alg_utils.normalizeGraphEdgeWeights(P, axis=0)  # column normalization
    P = alpha*P
    # make sure they're in csc format for the solvers
    # update: try csr for matrix multiplication since there's more rows than columns
    P = P.tocsr()
    B = B.tocsr()

    start_time = time.process_time()
    if eps != 0 or max_iters != 0:
        #print("\tstarting power iteration over each node individually")
        # Version of birgrank using a power iteration. Useful as scipy's solver struggles with these large matrices
        # looks like the real problem is the amount of ram needed to store the results in the large matrix X
        # rather than power iterate with the entire matrix B, run power iteration for column of B individually
        # and then merge only the Xh results.
        # much faster to only compute scores for a subset of nodes.
        # also uses less RAM because the RWR scores aren't kept
        Xh = lil_matrix((m,n))
        for i in tqdm(nodes):
            e = B[:,i].toarray().flatten()
            x = e.copy()
            prev_x = x.copy()
            for iters in range(1,max_iters+1):
                x = P.dot(prev_x) + e

                max_d = (x - prev_x).max()
                #if verbose:
                #    tqdm.write("\t\tmax score change: %0.6f" % (max_d))
                if max_d < eps:
                    break
                prev_x = x.copy()
            Xh[i] = x[m:]
            if verbose:
                print("\tbirgRank converged after %d iterations. max_d: %0.2e, eps: %0.2e" % (iters, max_d, eps))
        Xh = Xh.T
        # UPDATE: run power iteration for all nodes 
        """
        X = csc_matrix(B.shape)
        prev_X = csc_matrix(B.shape)
        for iters in trange(1,max_iters+1):
            X = P.dot(prev_X) + B
            max_d = (X - prev_X).max()
            if verbose:
                tqdm.write("\t\t%d iters: max score change: %0.6f" % (iters, max_d))
            if max_d < eps:
                break
            prev_X = X
        # now extract the term rows which correspond to the gene-term scores
        Xh = X[m:,:]
        """

        total_time = time.process_time() - start_time
        # this only shows the # of iterations for the last prot
        print("\tbirgRank converged after %d iterations (%0.2f sec)" % (iters, total_time))
    else:
        A = eye(m+n) - P
        A = A.tocsc()
        # solve PageRank linear system using 3-block solver
        print("\tsolving for Xg")
        # now solve the linear system X = A/B
        # split it up into two equations to solve
        # (I-alpha*G)Xg = (1-alpha)I
        Xg = spsolve(A[:m,:][:,:m], B[:m,:])
        print("\tsolving for Xh")
        # alpha*RT*Xg = (I - alpha*H)Xh
        Xh = spsolve(A[m:,:][:,m:], (B[m:,:] - A[m:,:][:,:m]*Xg))

        total_time = time.process_time() - start_time
        print("\tsolved birgRank using sparse linear system (%0.2f sec)" % (total_time))

    # transpose Xh so it has the same dimensions as Rtrain
    # prot rows, goid columns
    return Xh.T, total_time
