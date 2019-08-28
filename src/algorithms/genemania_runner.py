
import time
import src.algorithms.genemania as genemania
import src.algorithms.alg_utils as alg_utils
from tqdm import tqdm, trange


def setupInputs(run_obj):
    # setup the output file

    # may need to make sure the inputs match
    ## if there are more annotations than nodes in the network, then trim the extra pos/neg nodes
    #num_nodes = self.P.shape[0] if self.weight_gm2008 is False else self.normalized_nets[0].shape[0]
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
    elif run_obj.net_obj.weight_gm2008:
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


# nothing to do here
def setupOutputs(run_obj):
    return


def run(run_obj):
    """
    Function to run GeneMANIA
    *goids_to_run*: goids for which to run the method. 
        Must be a subset of the goids present in the ann_obj
    """
    params_results = run_obj.params_results 
    goid_scores = run_obj.goid_scores 

    L = run_obj.L
    alg = run_obj.name
    run_obj.params.pop('should_run', None)  # remove the should_run parameter
    print("Running %s with these parameters: %s" % (alg, run_obj.params))

    # run GeneMANIA on each GO term individually
    for goid in tqdm(run_obj.goids_to_run):
        idx = run_obj.ann_obj.goid2idx[goid]
        # get the row corresponding to the current goids annotations 
        y = run_obj.ann_matrix[idx,:].toarray()[0]

        if run_obj.net_obj.weight_gm2008 is True:
            start_time = time.process_time()
            # weight the network for each GO term individually
            W, process_time = run_obj.net_obj.weight_GM2008(y, goid)
            L = genemania.setup_laplacian(W)
            params_results['%s_weight_time'%(alg)] += time.process_time() - start_time

        # now actually run the algorithm
        scores, process_time, wall_time, iters = genemania.runGeneMANIA(L, y, tol=float(run_obj.params['tol']), verbose=run_obj.kwargs.get('verbose', False))
        if run_obj.kwargs.get('verbose', False) is True:
            tqdm.write("\t%s converged after %d iterations " % (alg, iters) +
                    "(%0.3f sec, %0.3f wall_time) for %s" % (process_time, wall_time, goid))

        ## if they're different dimensions, then set the others to zeros 
        #if len(scores_arr) < goid_scores.shape[1]:
        #    scores_arr = np.append(scores_arr, [0]*(goid_scores.shape[1] - len(scores_arr)))
        goid_scores[idx] = scores
        # make sure 0s are removed
        goid_scores.eliminate_zeros()

        # also keep track of the time it takes for each of the parameter sets
        alg_name = "%s%s" % (alg, run_obj.params_str)
        params_results["%s_wall_time"%alg_name] += wall_time
        params_results["%s_process_time"%alg_name] += process_time

    run_obj.goid_scores = goid_scores
    run_obj.params_results = params_results
    return
