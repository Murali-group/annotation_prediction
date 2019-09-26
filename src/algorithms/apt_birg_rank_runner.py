
import time
#import src.algorithms.aptrank_birgrank.run_birgrank as run_birgrank
import src.algorithms.aptrank_birgrank.birgrank as birgrank
#import src.algorithms.aptrank_birgrank.aptrank as aptrank
import src.algorithms.alg_utils as alg_utils
import src.setup_sparse_networks as setup
from scipy import sparse as sp
import numpy as np
#from tqdm import tqdm, trange


def setupInputs(run_obj):
    # extract the variables out of the annotation object
    run_obj.ann_matrix = run_obj.ann_obj.ann_matrix
    run_obj.hierarchy_mat = run_obj.ann_obj.dag_matrix
    run_obj.goids = run_obj.ann_obj.goids

    # setup the matrices
    if run_obj.kwargs.get('verbose'):
        print("Setting up the Birg/AptRank annotation matrix")
    # get only the positive examples from the ann_matrix
    run_obj.pos_mat = (run_obj.ann_matrix > 0).astype(int)
    # make sure the annotations are propagated up the DAG
    run_obj.pos_mat = setup.propagate_ann_up_dag(run_obj.pos_mat, run_obj.hierarchy_mat)
    # make sure there's no 0s leftover
    run_obj.pos_mat.eliminate_zeros()
    assert (run_obj.pos_mat.shape[0] == run_obj.hierarchy_mat.shape[0]), \
        "Error: annotation and hierarchy matrices " + \
        "do not have the same number of rows (terms): %d, %d" % (
            run_obj.pos_mat.shape[0], run_obj.hierarchy_mat.shape[0])

    if run_obj.net_obj.weight_gmw:
        # Cannot be used by birgrank. Change to weight_swsn for now
        print("WARNING: Apt/BirgRank cannot use the gmw weighting method since scores are computed for all terms simultaneously. Using SWSN instead.")
        run_obj.net_obj.weight_swsn = True 
        run_obj.net_obj.weight_gmw = False 
    if run_obj.net_obj.weight_swsn:
        W, process_time = run_obj.net_obj.weight_SWSN(run_obj.ann_matrix)
        run_obj.P = alg_utils.normalizeGraphEdgeWeights(W)
        run_obj.params_results['%s_weight_time'%(run_obj.name)] += process_time
    else:
        run_obj.P = alg_utils.normalizeGraphEdgeWeights(run_obj.net_obj.W)

    # Default is to run birgrank on all nodes
    run_obj.nodes_to_run = set(list(range(run_obj.P.shape[0])))
    # if a subset of nodes is specified, then run birgrank on only those nodes.
    # does not apply for aptrank
    if run_obj.kwargs.get('nodes_to_run') is not None:
        run_obj.nodes_to_run = run_obj.kwargs['nodes_to_run']
        print("\trunning on only the %d nodes with a pos/neg annotation" % (len(run_obj.nodes_to_run)))

    return


def get_alg_type():
    return "gene-based"


# setup the params_str used in the output file
def setup_params_str(weight_str, params, name):
    params_str = "%s" % (weight_str)
    if name == 'birgrank':
        alpha, theta, mu, br_lambda = params['alpha'], params['theta'], params['mu'], params['lambda'] 
        params_str += '-a%s-t%s-m%s-l%s-eps%s-maxi%s' % (
            str_(alpha), str_(theta), str_(mu), str_(br_lambda), str_(params['eps']), str_(params['max_iters']))
    elif name == 'aptrank':
        br_lambda, k, s, t, diff_type = params['lambda'], params['k'], params['s'], params['t'], params['diff_type'] 
        params_str += '-l%s-k%s-s%s-t%s-%s' % (
            str_(br_lambda), str_(k), str_(s), str_(t), diff_type)
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
    P, hierarchy_mat, pos_mat = run_obj.P, run_obj.hierarchy_mat, run_obj.pos_mat
    alg, params, br_lambda = run_obj.name, run_obj.params, run_obj.params['lambda']

    # UPDATE 2019-01-04: Include the birgrank lambda parameter which controls the direction of the flow within the hierarchy
    # dH = l * H + (1-l)H^T
    # a value of 1 would be only upwards, while 0 would be downwards
    dH = (br_lambda * hierarchy_mat) + ((1-br_lambda) * hierarchy_mat.T) 

    if alg == 'birgrank':
        theta, mu = params['theta'], params['mu']
        alpha, eps, max_iters = params['alpha'], float(params['eps']), params['max_iters']
        start_time = time.process_time()
        Xh = birgrank.birgRank(
                P, pos_mat.transpose(), dH,
                alpha=alpha, theta=theta, mu=mu, eps=eps, max_iters=max_iters,
                nodes=run_obj.nodes_to_run, verbose=run_obj.kwargs.get('verbose', False))
        process_time = time.process_time() - start_time
    elif alg == 'aptrank':
        k, s, t, diff_type = params['k'], params['s'], params['t'] , params['diff_type'] 
        # make sure aptrank is imported
        # it has a special dependency
        import src.algorithms.aptrank_birgrank.aptrank as aptrank
        num_cores = 12
        start_time = time.process_time()
        runner = aptrank.AptRank(P, pos_mat.transpose(), dH,
                    K=k, S=s, T=t, NCores=num_cores, diffusion_type=diff_type)
        Xh = runner.algorithm()
        process_time = time.process_time() - start_time
    Xh = Xh.T

    print("\t%s finished after %0.3f sec (process_time)" % (alg, process_time))

    # also keep track of the time it takes for each of the parameter sets
    #params_results["%s_wall_time"%alg] += wall_time
    params_results["%s_process_time"%alg] += process_time

    # limit the scores matrix to only the GOIDs for which we want the scores
    for i in range(len(run_obj.goids_to_run)):
        idx = run_obj.ann_obj.goid2idx[goid]
        goid_scores[idx] = Xh[idx]

    run_obj.goid_scores = Xh
    run_obj.params_results = params_results
    return

