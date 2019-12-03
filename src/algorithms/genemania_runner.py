
import time
import src.algorithms.genemania as genemania
import src.algorithms.alg_utils as alg_utils
from tqdm import tqdm, trange
from scipy import sparse as sp
from scipy.sparse.linalg import spilu, LinearOperator
import numpy as np


def setupInputs(run_obj):
    # setup the output file

    # may need to make sure the inputs match
    ## if there are more annotations than nodes in the network, then trim the extra pos/neg nodes
    #num_nodes = self.P.shape[0] if self.weight_gmw is False else self.normalized_nets[0].shape[0]
    #if len(self.prots) > num_nodes: 
    #    positives = positives[np.where(positives < num_nodes)]
    #    negatives = negatives[np.where(negatives < num_nodes)]

    # extract the variables out of the annotation object
    run_obj.ann_matrix = run_obj.ann_obj.ann_matrix
    run_obj.goids = run_obj.ann_obj.goids

    # Build the laplacian(?)
    if run_obj.net_obj.weight_swsn:
        W, process_time = run_obj.net_obj.weight_SWSN(run_obj.ann_matrix)
        run_obj.L = genemania.setup_laplacian(W)
        run_obj.params_results['%s_weight_time'%(run_obj.name)] += process_time
    elif run_obj.net_obj.weight_gmw:
        # this will be handled on a GO term by GO term basis
        run_obj.L = None
    else:
        run_obj.L = genemania.setup_laplacian(run_obj.net_obj.W)

    return


# setup the params_str used in the output file
def setup_params_str(weight_str, params, name='genemania'):
    params_str = "%s-tol%s" % (
        weight_str, str(params['tol']).replace('.','_'))
    return params_str


def get_alg_type():
    return "term-based"


# nothing to do here
def setupOutputs(run_obj, **kwargs):
    return


def compute_preconditioner(L):
    print("running spilu")
    start_process_time = time.process_time()
    start_wall_time = time.time()
    M = sp.eye(L.shape[0]) + L
    ilu = spilu(M)
    process_time = time.process_time() - start_process_time 
    wall_time = time.time() - start_wall_time
    print("finished in %0.3f sec, %0.3f process time" % (process_time, wall_time))
    Mx = lambda x: ilu.solve(x)
    Milu = LinearOperator(L.shape, Mx)
    return Milu


def run(run_obj):
    """
    Function to run GeneMANIA
    *goids_to_run*: goids for which to run the method. 
        Must be a subset of the goids present in the ann_obj
    """
    params_results = run_obj.params_results 
    # make sure the goid_scores matrix is reset
    # because if it isn't empty, overwriting the stored scores seems to be time consuming
    goid_scores = sp.lil_matrix(run_obj.ann_matrix.shape, dtype=np.float)
    L, alg = run_obj.L, run_obj.name
    print("Running %s with these parameters: %s" % (alg, run_obj.params))
    if len(run_obj.target_prots) != len(run_obj.net_obj.nodes):
        print("\tstoring scores for only %d target prots" % (len(run_obj.target_prots)))
    # using a preconditioner actually makes it slower, so don't use it
    #Milu = compute_preconditioner(L)
    Milu = None

    # run GeneMANIA on each GO term individually
    for goid in tqdm(run_obj.goids_to_run):
        idx = run_obj.ann_obj.goid2idx[goid]
        # get the row corresponding to the current goids annotations 
        y = run_obj.ann_matrix[idx,:].toarray()[0]

        if run_obj.net_obj.weight_gmw is True:
            start_time = time.process_time()
            # weight the network for each GO term individually
            W,_,_ = run_obj.net_obj.weight_GMW(y, goid)
            L = genemania.setup_laplacian(W)
            params_results['%s_weight_time'%(alg)] += time.process_time() - start_time
        if alg in ['genemaniaplus']:
            # remove the negative examples
            y = (y > 0).astype(int)

        # now actually run the algorithm
        scores, process_time, wall_time, iters = genemania.runGeneMANIA(
                L, y, tol=float(run_obj.params['tol']), Milu=Milu,
                verbose=run_obj.kwargs.get('verbose', False))
        if run_obj.kwargs.get('verbose', False) is True:
            tqdm.write("\t%s converged after %d iterations " % (alg, iters) +
                    "(%0.3f sec, %0.3f wall_time) for %s" % (process_time, wall_time, goid))

        ## if they're different dimensions, then set the others to zeros 
        #if len(scores_arr) < goid_scores.shape[1]:
        #    scores_arr = np.append(scores_arr, [0]*(goid_scores.shape[1] - len(scores_arr)))
        # limit the scores to the target nodes
        if len(run_obj.target_prots) != len(scores):
            mask = np.ones(len(scores), np.bool)
            mask[run_obj.target_prots] = 0
            # everything that's not a target prot will be set to 0
            scores[mask] = 0
        # 0s are not explicitly stored in lil matrix
        goid_scores[idx] = scores

        # also keep track of the time it takes for each of the parameter sets
        alg_name = "%s%s" % (alg, run_obj.params_str)
        params_results["%s_wall_time"%alg_name] += wall_time
        params_results["%s_process_time"%alg_name] += process_time

    run_obj.goid_scores = goid_scores
    run_obj.params_results = params_results
    return
