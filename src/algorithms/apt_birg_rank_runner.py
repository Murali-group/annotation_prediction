
import time
#import src.algorithms.aptrank_birgrank.run_birgrank as run_birgrank
import src.algorithms.aptrank_birgrank.birgrank as birgrank
#import src.algorithms.aptrank_birgrank.aptrank as aptrank
import src.algorithms.alg_utils as alg_utils
#from tqdm import tqdm, trange


def setupInputs(run_obj):
    # extract the variables out of the annotation object
    run_obj.ann_matrix = run_obj.ann_obj.ann_matrix
    run_obj.goids = run_obj.ann_obj.goids
    # the annotation also needs to have these three variables
    run_obj.hierarchy_mat = run_obj.ann_obj.dag_matrix
    run_obj.pos_mat = run_obj.ann_obj.pos_matrix
    run_obj.dag_goids  = run_obj.ann_obj.dag_goids 
    run_obj.dag_goids2idx = {g: i for i, g in enumerate(run_obj.dag_goids)}

    if run_obj.net_obj.weight_gm2008:
        # Cannot be used by birgrank. Change to weight_swsn for now
        print("WARNING: Apt/BirgRank cannot use the gm2008 weighting method since scores are computed for all terms simultaneously. Using SWSN instead.")
        run_obj.net_obj.weight_swsn = True 
    if run_obj.net_obj.weight_swsn:
        W, process_time = run_obj.net_obj.weight_SWSN(run_obj.ann_matrix)
        run_obj.P = alg_utils.normalizeGraphEdgeWeights(W)
        run_obj.params_results['%s_weight_time'%(run_obj.name)] += process_time
    else:
        run_obj.P = alg_utils.normalizeGraphEdgeWeights(run_obj.net_obj.W)

    # this may be changed by another function, but for now, just run birgrank on all nodes
    #run_obj.test_nodes = set(list(range(run_obj.P.shape[0])))
    print("WARNING: running birgrank using only 10 nodes")
    run_obj.test_nodes = set(list(range(run_obj.P.shape[0]))[:10])

    return


# setup the params_str used in the output file
def setup_params_str(run_obj):
    params = run_obj.params
    params_str = "-%s" % (run_obj.weight_str)
    if run_obj.name == 'birgrank':
        alpha, theta, mu, br_lambda = params['alpha'], params['theta'], params['mu'], params['lambda'] 
        params_str += 'a%s-t%s-m%s-l%s-eps%s-maxi%s' % (
            str_(alpha), str_(theta), str_(mu), str_(br_lambda), str_(params['eps']), str_(params['max_iters']))
    elif run_obj.name == 'aptrank':
        br_lambda, k, s, t, diff_type = params['lambda'], params['k'], params['s'], params['t'], params['diff_type'] 
        params_str += 'l%s-k%s-s%s-t%s-%s' % (
            str_(br_lambda), str_(k), str_(s), str_(t), diff_type)
    return params_str


def str_(s):
    return str(s).replace('.','_')


# nothing to do here
def setupOutputs(run_obj):
    return


def run(run_obj):
    """
    Function to run AptRank and BirgRank
    """
    params_results = run_obj.params_results 
    goid_scores = run_obj.goid_scores 
    P = run_obj.P
    hierarchy_mat, pos_mat = run_obj.hierarchy_mat, run_obj.pos_mat

    alg = run_obj.name
    params = run_obj.params
    br_lambda = params['lambda']

    assert (run_obj.pos_mat.shape[0] == run_obj.hierarchy_mat.shape[0]), \
        "Error: annotation and hierarchy matrices " + \
        "do not have the same number of rows (terms): %d, %d" % (
            run_obj.pos_mat.shape[0], run_obj.hierarchy_mat.shape[0])

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
                nodes=run_obj.test_nodes, verbose=run_obj.kwargs.get('verbose', False))
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

    print("\t%s finished after %0.3f sec" % (alg, process_time))

    # also keep track of the time it takes for each of the parameter sets
    #params_results["%s_wall_time"%alg] += wall_time
    params_results["%s_process_time"%alg] += process_time

    # limit the scores matrix to only the GOIDs for which we have annotations
    for i in range(len(run_obj.goids)):
        goid_scores[i] = Xh[run_obj.dag_goids2idx[run_obj.goids[i]]]

    run_obj.goid_scores = goid_scores
    run_obj.params_results = params_results
    return

