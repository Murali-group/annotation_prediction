
import sys
from collections import defaultdict
#import src.setup_sparse_networks as setup
import src.algorithms.alg_utils as alg_utils
import src.algorithms.fastsinksource_runner as fastsinksource
import src.algorithms.sinksource_bounds_runner as ss_bounds
import src.algorithms.local_runner as local
import src.algorithms.genemania_runner as genemania
import src.algorithms.apt_birg_rank_runner as birgrank
import src.algorithms.async_rw_runner as async_rw
#import src.algorithms.sinksource_bounds
#from src.algorithms.aptrank_birgrank.birgrank import birgRank
#import src.algorithms.aptrank_birgrank.run_birgrank as run_birgrank
import numpy as np
from scipy import sparse as sp


LibMapper = {
    'sinksource': fastsinksource,
    'sinksourceplus': fastsinksource,
    'fastsinksource': fastsinksource,
    'fastsinksourceplus': fastsinksource,
    'sinksource_bounds': ss_bounds,
    'sinksourceplus_bounds': ss_bounds,
    'local': local,
    'localplus': local,
    'genemania': genemania,
    'genemaniaplus': genemania,
    'aptrank': birgrank,
    'birgrank': birgrank,
    'async_rw': async_rw,
}


#AlgorithmMapper = {
#    'fastsinksource': fastsinksource.run,
#    'genemania': genemania.run,
#}


#OutputParser = {
#    'fastsinksource': fastsinksource.setupOutputs,
#    'genemania': genemania.setupOutputs,
#}


class Runner(object):
    '''
    A runnable analysis to be incorporated into the pipeline
    *kwargs*: Checked for out_pref, verbose, and forecealg options.
        Can be used to pass additional variables needed by the runner
    '''
    def __init__(self, name, net_obj, ann_obj,
                 out_dir, params, **kwargs):
        self.name = name
        self.net_obj = net_obj
        self.ann_obj = ann_obj
        self.out_dir = "%s/%s/" % (out_dir, name)
        params.pop('should_run', None)  # remove the should_run parameter
        self.params = params
        self.kwargs = kwargs
        self.verbose = kwargs.get('verbose', False) 
        self.forced = kwargs.get('forcealg', False) 
        # for term-based algorithms, can limit the goids for which they will be run
        self.goids_to_run = kwargs.get('goids_to_run', ann_obj.goids)
        # also can limit the nodes for which scores are stored with this
        self.target_prots = kwargs.get('target_nodes', np.arange(len(ann_obj.prots)))

        # track measures about each run (e.g., running time)
        self.params_results = defaultdict(int) 
        # store the node scores for each GO term in a sparse matrix
        # using lil matrix so 0s are automatically not stored
        self.goid_scores = sp.lil_matrix(ann_obj.ann_matrix.shape, dtype=np.float)

        # keep track of the weighting method for writing to the output file later
        self.setupParamsStr(net_obj.weight_str, params, name)
        self.out_pref = kwargs.get('out_pref', self.out_dir+'pred-scores'+self.params_str)


    # if the algorithm is not inmplemented in Python (e.g., MATLAB, R)
    # use this function to setup files and such
    def setupInputs(self):
        return LibMapper[self.name].setupInputs(self)

    # run the method
    def run(self):
        return LibMapper[self.name].run(self)

    # if the method is not in Python and was called elsewhere (e.g., R), 
    # then parse the outputs of the method
    def setupOutputs(self, **kwargs):
        if self.params.get('debug_scores'):
            out_file = self.params.get('debug_scores_file', 
                    "%sdebug-scores%s%s.txt" % (
                        self.out_dir, self.params_str,
                        self.kwargs.get('postfix','')) )
            #print("Debugging scores. Writing/Appending to %s" % (out_file))
            # use the number of target prots as the default, 
            # since those are the ones for which scores will be stored
            num_to_write = kwargs.get('num_pred_to_write', len(self.target_prots))
            if (num_to_write > 1000 or num_to_write == -1) \
                    and len(self.goids_to_run) > 10:
                print("num_pred_to_write %s for %d terms too much output. Must be < 1000 and < 10, respectively" % (
                    num_to_write, len(self.goids_to_run)))
            else:
                #for goid in self.goids_to_run:
                #    scores = self.goid_scores[self.ann_obj.goid2idx[goid]]
                alg_utils.write_output(
                        self.goid_scores, self.goids_to_run, self.ann_obj.prots,
                        out_file, num_pred_to_write=num_to_write, 
                        goid2idx=self.ann_obj.goid2idx)

        return LibMapper[self.name].setupOutputs(self, **kwargs)

    # setup the params_str used in the output file
    def setupParamsStr(self, weight_str, params, name):
        self.params_str = LibMapper[self.name].setup_params_str(weight_str, params, name)

    def get_alg_type(self):
        return LibMapper[self.name].get_alg_type()


def get_runner_params_str(name, dataset, params):
    """
    Get the params string for a runner without actually creating the runner object
    *dataset*: dictionary of datset settings and file paths. Used to get the 'net_settings': 'weight_method' (e.g., 'swsn' or 'gmw')
    """
    # get the weight str used when writing output files
    unweighted = dataset['net_settings'].get('unweighted', False) if 'net_settings' in dataset else False
    weight_method = ""
    if dataset.get('multi_net',False) is True: 
        weight_method = "-"+dataset['net_settings']['weight_method']
    weight_str = '%s%s' % (
        '-unw' if unweighted else '', weight_method)

    return LibMapper[name].setup_params_str(weight_str, params, name=name)

