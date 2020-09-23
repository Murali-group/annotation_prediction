
import time
from scipy import sparse as sp
import numpy as np
#from tqdm import tqdm, trange

import src.algorithms.alg_utils as alg_utils
import src.setup_sparse_networks as setup
import src.algorithms.async_rw as async_rw


def setupInputs(run_obj):
    # extract the variables out of the annotation object
    run_obj.ann_matrix = run_obj.ann_obj.ann_matrix
    run_obj.hierarchy_mat = run_obj.ann_obj.dag_matrix
    run_obj.goids = run_obj.ann_obj.goids
    goid2idx = run_obj.ann_obj.goid2idx
    terms_to_run_idx = [goid2idx[g] for g in run_obj.goids_to_run]

    # setup the matrices
    if run_obj.kwargs.get('verbose'):
        print("Setting up the Async RW hierarchy annotation matrix")
    # get only the positive examples from the ann_matrix
    run_obj.pos_mat = (run_obj.ann_matrix > 0).astype(int)
    # propagate the annotations up the DAG before running to match how they ran their method in their paper
    if run_obj.kwargs.get('verbose'):
        print("\tpropagating annotations up the DAG before running")
    # make sure the annotations are propagated up the DAG
    run_obj.pos_mat = setup.propagate_ann_up_dag(run_obj.pos_mat, run_obj.hierarchy_mat)

    # Build the matrix of transitional probabilities from a term to its child terms
    run_obj.P_H = build_transition_prob_H_mat(run_obj.hierarchy_mat, run_obj.pos_mat)
    # the first network is the SSN, so pull that out on its own
    net_obj = run_obj.net_obj
    try:
        # the SSN should already have been added in experiments.py
        run_obj.SSN = alg_utils.normalizeGraphEdgeWeights(run_obj.net_obj.SSN)
    except AttributeError:
        # if not, then extract it here
        if net_obj.multi_net:
            run_obj.SSN = net_obj.normalized_nets[0] 
        else:
            run_obj.SSN = net_obj.W 
            # just give an empty intra-species network if there is none
            run_obj.P = sp.csr_matrix(run_obj.SSN.shape)
            return

    if run_obj.net_obj.weight_gmw:
        print("WARNING: Async RW cannot use the GMW weighting method since scores are computed for all terms simultaneously. Using SWSN instead.")
        run_obj.net_obj.weight_swsn = True 
        run_obj.net_obj.weight_str = "-swsn"
        run_obj.net_obj.weight_method = "swsn"
        run_obj.net_obj.weight_gmw = False 
    if run_obj.net_obj.weight_swsn:
        # remove the SSN to just get the species networks
        orig_nets, orig_names = net_obj.normalized_nets, net_obj.net_names
        run_obj.net_obj.normalized_nets = net_obj.normalized_nets[1:]
        run_obj.net_obj.net_names = net_obj.net_names[1:]
        W, process_time = run_obj.net_obj.weight_SWSN(run_obj.ann_matrix)
        run_obj.P = alg_utils.normalizeGraphEdgeWeights(W)
        run_obj.params_results['%s_weight_time'%(run_obj.name)] += process_time
        run_obj.net_obj.normalized_nets, run_obj.net_obj.net_names = orig_nets, orig_names
    elif run_obj.net_obj.multi_net is False:
        run_obj.P = alg_utils.normalizeGraphEdgeWeights(run_obj.net_obj.W)

    return


def build_transition_prob_H_mat(H, pos_mat):
    """
    Build a matrix where for each child term s and parent term t pair, we compute
        p(s|t) = (s_num_ann / t_num_ann) + (IC(s) / sum_{v in ch(t)} IC(v))
        which is then normlized 
    *H*: hierarchy matrix, with edges pointed from child -> parent
    *pos_mat*: matrix (term x protein) of annotations per term
    """
    # the information content here is annotation agnostic
    ic_vec = compute_stuctures_IC(H)
    # here we will get p(s|t) = s_num_ann / t_num_ann
    """
    example with six terms and six edges (2->1, 3->1, 4->2, 5->2, 5->3, 6->3)
    |   | 1 | 2 | 3 | 4 | 5 | 6 |   | #ann |
    |---+---+---+---+---+---+---|   |------|
    | 1 |   |   |   |   |   |   |   |    9 |
    | 2 | 1 |   |   |   |   |   |   |    5 |
    | 3 | 1 |   |   |   |   |   | x |    4 |
    | 4 |   | 1 |   |   |   |   |   |    2 |
    | 5 |   | 1 | 1 |   |   |   |   |    3 |
    | 6 |   |   | 1 |   |   |   |   |    1 |
      = 
    |   | 1 | 2 | 3 | 4 | 5 | 6 |
    |---+---+---+---+---+---+---|
    | 1 |   |   |   |   |   |   |
    | 2 | 5 |   |   |   |   |   |
    | 3 | 4 |   |   |   |   |   |
    | 4 |   | 2 |   |   |   |   |
    | 5 |   | 3 | 3 |   |   |   |
    | 6 |   |   | 1 |   |   |   |

    Transpose and multiply by 1/num ann 

    |   | 1 |   2 |   3 |   4 | 5   |   6 |
    |---+---+-----+-----+-----+-----+-----|
    | 1 |   | 5/9 | 4/9 |     |     |     |
    | 2 |   |     |     | 2/5 | 3/5 |     |
    | 3 |   |     |     |     | 3/4 | 1/4 |
    | 4 |   |     |     |     |     |     |
    | 5 |   |     |     |     |     |     |
    | 6 |   |     |     |     |     |     |
    """
    num_ann_per_term = np.ravel(pos_mat.sum(axis=1))
    H_num_ann_mat = H.multiply(num_ann_per_term)
    ann_trans_prob = H_num_ann_mat.T.multiply(np.divide(1.,num_ann_per_term)).T

    # now compute sum_{v in ch(t)} IC(v)
    ic_child_terms = H.T.multiply(ic_vec).T
    # and then use it to get this: (IC(s) / sum_{v in ch(t)} IC(v))
    ic_child_terms.data = 1/ic_child_terms.data
    ic_trans_prob = H.multiply(ic_vec).multiply(ic_child_terms)

    # add to get the final transition probabilities
    trans_prob = ann_trans_prob + ic_trans_prob

    # and finally, normalize by the sum of prob of child terms
    deg = np.ravel(trans_prob.sum(axis=0))
    deg = np.divide(1., deg)
    P_H = trans_prob.multiply(deg)
    P_H = P_H.tocsc()

    return P_H


def expand_hierarchy(H):
    """
    Connect every term to all of its ancestors
    """
    H_full = H.copy()
    H_prop = H
    # To get the all ancestors of each term, keep walking up the DAG until we have reached the top (i.e., all 0s)
    while True:
        H_prop = H_prop * H
        H_full += H_prop
        if H_prop.nnz == 0:
            break
    # put all of the edges back to 0 or 1
    H_full = H_full.astype(bool).astype(int)

    return H_full


def get_term_num_descendants(H):
    """
    Build a vector of the # descendants per term
    """
    H_full = expand_hierarchy(H)
    # print out the number of descendants of each term
    num_descendants_vec = np.ravel(H_full.sum(axis=0))
    return num_descendants_vec


def compute_stuctures_IC(H):
    """
    Compute the structure Information Content (IC)
    """
    num_descendants_vec = get_term_num_descendants(H)
    num_terms = H.shape[0]
    # it works!
    # print("%d total terms" % (num_terms))
    # print(num_descendants_vec)
    # for i, num_desc in enumerate(num_descendants_vec):
    #     if num_desc > 200:
    #         print("%s: %d descendants" % (ann_obj.goids[i], num_desc))
    # for each term t, the IC is 1 - log(|desc(t)|)/log(num_all_terms)
    ic_vec = 1 - (np.log(num_descendants_vec) / np.log(num_terms))
    return ic_vec


def get_alg_type():
    return "term-based"


# setup the params_str used in the output file
def setup_params_str(weight_str, params, name):
    if weight_str == "gmw":
        weight_str = "swsn"
    params_str = "%s" % (weight_str)
    if name == 'async_rw':
        a, min_len, max_len = params['alpha'], params['min_len'], params['max_len']
        params_str += '-a%s-minl%s-maxl%s' % (
            str_(a), str_(min_len), str_(max_len))
    return params_str


def str_(s):
    return str(s).replace('.','_')


# nothing to do here
def setupOutputs(run_obj, **kwargs):
    return


def run(run_obj):
    """
    Function to run AptRank and BirgRank
    """
    params_results = run_obj.params_results
    goid_scores = sp.lil_matrix(run_obj.ann_matrix.shape, dtype=np.float)
    P, SSN, P_H, pos_mat = run_obj.P, run_obj.SSN, run_obj.P_H, run_obj.pos_mat
    alg, params = run_obj.name, run_obj.params

    # direct the hierarchy downwards
    #dH = hierarchy_mat.T
    params['alpha'] = params.get('alpha',0.1)
    params['min_len'] = params.get('min_len',0)
    params['max_len'] = params.get('max_len',10)
    print("Running %s with these parameters: %s" % (alg, params))

    alpha, min_len, max_len = params['alpha'], params['min_len'], params['max_len']
    S, process_time = async_rw.AsyncRW(
            P, SSN, P_H, pos_mat, 
            alpha=alpha, min_len=min_len, max_len=max_len,
            verbose=run_obj.kwargs.get('verbose', False))

    print("\t%s finished after %0.3f sec (process_time)" % (alg, process_time))

    # also keep track of the time it takes for each of the parameter sets
    #params_results["%s_wall_time"%alg] += wall_time
    alg_name = "%s%s" % (alg, run_obj.params_str)
    params_results["%s_process_time"%alg_name] += process_time

    # limit the scores matrix to only the GOIDs for which we want the scores
    if len(run_obj.goids_to_run) < goid_scores.shape[0]:
        for goid in run_obj.goids_to_run:
            idx = run_obj.ann_obj.goid2idx[goid]
            goid_scores[idx] = S[idx]
    else:
        goid_scores = S

    run_obj.goid_scores = goid_scores
    run_obj.params_results = params_results
    return

