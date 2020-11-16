
#import numpy as np
import numpy as np
import time
#import sys
from tqdm import tqdm

__author__ = "Jeff Law"
# python implementation of the Asynchronous Random Walk algorithm 
# see http://mlda.swu.edu.cn/codes.php?name=AsyRW for the original matlab implementation
# full citation:
# Yingwen Zhao, Jun Wang, Maozu Guo, Xiangliang Zhang and Guoxian Yu, Cross-Species Protein Function Prediction with Asynchronous-Random Walk, IEEE CBB, (2019).
# https://doi.org/10.1109/tcbb.2019.2943342


def AsyncRW(P, SSN, H, Y, alpha=.1, min_len=0, max_len=10, verbose=False):
    """
    *P*: Intra-species network matrix. Should already be normalized
    *SSN*: Inter-species sequence similarity network matrix. Should already be normalized
    *H*: directed and weighted Hierarchy matrix. Should already be normalized
    *Y*: Matrix containing 0s and 1s for annotations. 
        Rows are terms, columns are nodes
    *alpha*: RWR restart parameter 
    *min_len*: minimum # of steps to take. I added this option for the case of making predictions for a species with no annotations
    *max_len*: maximum # of steps to take (corresponds to l in the paper)

    *returns*: Y matrix of scores for each GO term, protein pair, same shape as Y
    """

    print("Starting AsyncRW")
    a = alpha

    start_time = time.process_time()
    # First compute the walk lengths
    print("computing P walk lengths")
    P_walk_lens = compute_walk_lengths(Y, P, min_len=min_len, max_len=max_len)
    # O stands for Orthology
    print("computing SSN walk lengths")
    O_walk_lens = compute_walk_lengths(Y, SSN, min_len=min_len, max_len=max_len)
    # use the same function to compute the walk lengths on the hierarchy, but treat the terms as the proteins
    # to get the walk length for each term node
    print("computing H walk lengths")
    H_walk_lens = compute_walk_lengths(Y.T, H, min_len=min_len, max_len=max_len)

    # final function prediction scores
    # These should be nodes x terms for the matrix multiplication to work out
    Y = Y.T
    # start the scores at the first iteration of RWR to match the AsyncRW code
    S = (1-a)*Y
    S_prev = (1-a)*Y

    # Compute the random walk prediction scores
    for curr_len in range(1,max_len+1):
        # intra-species network walk
        S_P = a*P*S_prev + (1-a)*Y
        # replace the scores of the nodes which have passed its walk length with the previous iteration
        # TODO could be faster to remove rows in P rather than replace the score
        S_P = apply_walk_len_max(S_P, S_prev, P_walk_lens, curr_len)

        # inter-species network walk
        S_O = a*SSN*S_prev + (1-a)*Y
        # replace the scores of the nodes which have passed its walk length with the previous iteration
        S_O = apply_walk_len_max(S_O, S_prev, O_walk_lens, curr_len)

        # walk on the hierarchy
        # transpose to terms x nodes so we can apply the walk length limitation to the terms
        S_H = a*H*S_prev.T + (1-a)*Y.T
        # replace the scores of the nodes which have passed its walk length with the previous iteration
        S_H = apply_walk_len_max(S_H, S_prev.T, H_walk_lens, curr_len)
        S_H = S_H.T

        S_1 = elementwise_max(S_P, S_O)
        S = elementwise_max(S_1, S_H)
        S_prev = S

    total_time = time.process_time() - start_time
    print("\tfinished AsyncRW (%0.2f sec)" % (total_time))

    return S.T, total_time


def apply_walk_len_max(S, S_prev, max_walk_lens, curr_len):
    # replace the scores of the nodes which have passed its walk length with the previous iteration
    idx_to_keep = (max_walk_lens >= curr_len).astype(int).reshape(len(max_walk_lens), 1)
    idx_exceeded = (max_walk_lens < curr_len).astype(int).reshape(len(max_walk_lens), 1)
    # get the node scores that haven't passed their limit yet, plus the rest of the node scores from the previous iteration
    S = (S.multiply(idx_to_keep)) + (S_prev.multiply(idx_exceeded))
    return S


def compute_walk_lengths(Y, W, min_len=0, max_len=10):
    """
    For each node, compute the length of the walk, which is roughly the
        network-weighted co-annotation of each node with its direct neighbors.
    """

    num_terms, num_prots = Y.shape
    num_nodes = np.count_nonzero(W.sum(axis=0))
    walk_lens = np.zeros(num_prots)
    if num_nodes == 0:
        print("\tempty network.")
        return walk_lens
    # rather than all prots, loop through the nodes that have at least 1 annotation
    #for i in trange(num_prots):
    prots_with_ann = Y.sum(axis=0).nonzero()[1]
    for i in tqdm(prots_with_ann):
        # get the annotations of i
        Y_i = Y[:,i]
        # extract the row of network to get the neighbors
        row = W[i,:]
        nbrs = (row > 0).nonzero()[1]
        if len(nbrs) > 0:
            # get the annotations of those neighbors
            Y_nbrs = Y[:,nbrs]
            # now get the size of the intersection of those annotations with i's annotations
            Y_int_size = np.ravel(Y_nbrs.multiply(Y_i).sum(axis=0))
            nbr_weights = np.ravel(row[:,nbrs].A)
            # multiply by the weight of the edge in the network to get the walk length
            walk_len = (Y_int_size * nbr_weights).sum()
            # store the result
            walk_lens[i] = walk_len
            #print("Y_int_size: %s" % str(Y_int_size))
            #print("nbrs: %s" % str(nbrs))
            #print(nbr_weights)

    # finally, normalize the walk length between 0 and the max length (10)
    walk_lens = scale_between(walk_lens, a=min_len, b=max_len)
    # and round up
    walk_lens = np.ceil(walk_lens)
    print("\tmean: %s, median: %s" % (np.mean(walk_lens), np.median(walk_lens)))
    print("\twalk_len\tfrac nodes (/%d)" % (num_nodes))
    for i in range(min_len+1, max_len+1):
        print("\t%d\t%s" % (
            i, (len(np.where(walk_lens == i)[0]) / float(num_nodes))))
    return walk_lens


def scale_between(x, a=0, b=1):
    """
    scale a vector between two values a and b
    """
    if max(x) == 0 and min(x) == 0:
        return x
    else:
        scaled_x = ((x - min(x))/(max(x) - min(x))) * (b-a) + a
        return scaled_x


def elementwise_max(A, B):
    """
    return the elementwise maximum between the two sparse matrices A and B
    """
    B_is_max = (B > A).astype(int)
    return A - A.multiply(B_is_max) + B.multiply(B_is_max)
