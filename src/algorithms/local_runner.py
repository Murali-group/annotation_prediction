
import time
from . import fastsinksource_runner as fss_runner
#from tqdm import tqdm, trange
from scipy import sparse as sp
import numpy as np


def setupInputs(run_obj):
    # setup is the same as for fastsinksource
    fss_runner.setupInputs(run_obj)


# setup the params_str used in the output file
def setup_params_str(weight_str, params, name="localplus"):
    # ss_lambda affects the network that all these methods use
    ss_lambda = params.get('lambda', 0)
    params_str = "%s-l%s" % (weight_str, ss_lambda)

    return params_str


def get_alg_type():
    # doesn't really matter in this case
    return "term-based"


def setupOutputFile(run_obj):
    return


# nothing to do here
def setupOutputs(run_obj, **kwargs):
    return


def run(run_obj):
    """
    Function to run Local and LocalPlus
    *goids_to_run*: goids for which to run the method. 
        Must be a subset of the goids present in the ann_obj
    """
    params_results = run_obj.params_results
    P, alg = run_obj.P, run_obj.name

    #if 'solver' in params:
    # make sure the goid_scores matrix is reset
    # because if it isn't empty, overwriting the stored scores seems to be time consuming
    goid_scores = sp.lil_matrix(run_obj.ann_matrix.shape, dtype=np.float)
    # Local doesn't have any parameters, so no need to print this
    #print("Running %s with these parameters: %s" % (alg, params))

    # get just the postive examples for localplus
    ann_mat = run_obj.ann_obj.ann_matrix
    if alg in ['localplus']:
        ann_mat = (ann_mat > 0).astype(int)

    # Run Local on all terms
    goid_scores, process_time, wall_time = runLocal(
        P, ann_mat)

    if run_obj.kwargs.get('verbose', False) is True:
        print("\t%s finished after %0.4f sec (%0.4f process time)" % (alg, wall_time, process_time))
    alg_name = "%s%s" % (alg, run_obj.params_str)
    params_results["%s_wall_time"%alg_name] += wall_time
    params_results["%s_process_time"%alg_name] += process_time

    run_obj.goid_scores = goid_scores
    run_obj.params_results = params_results
    return


def runLocal(P, ann_mat):
    """
    Baseline method where each node's score is the average score of its neighbors.
        Essentially one iteration of SinkSource.
    *P*: Network as a scipy sparse matrix. Should already be normalized
    *ann_mat*: annotation matrix (pos: 1, unk: 0, neg: -1)
        Rows are terms and columns are genes
    """
    start_wall_time = time.time()
    start_process_time = time.process_time()
    goid_scores = P.dot(ann_mat.T).T
    wall_time = time.time() - start_wall_time
    process_time = time.process_time() - start_process_time

    return goid_scores, process_time, wall_time


def str_(s):
    return str(s).replace('.','_')

