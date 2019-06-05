
import sys
from collections import defaultdict
sys.path.append("src/algorithms")
import setup_sparse_networks as setup
import alg_utils
import fastsinksource_runner as fastsinksource
import genemania_runner as genemania
import apt_birg_rank_runner as birgrank
import sinksource_bounds
from aptrank_birgrank.birgrank import birgRank
import aptrank_birgrank.run_birgrank as run_birgrank
import numpy as np
from scipy import sparse


LibMapper = {
    'fastsinksource': fastsinksource,
    'genemania': genemania,
    'aptrank': birgrank,
    'birgrank': birgrank,
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
    '''
    def __init__(self, name, net_obj, ann_obj, out_dir, params, **kwargs):
        self.name = name
        self.net_obj = net_obj
        self.ann_obj = ann_obj
        self.out_dir = "%s/%s/" % (out_dir, name)
        self.params = params
        self.kwargs = kwargs
        self.verbose = kwargs.get('verbose', False) 
        self.forced = kwargs.get('forcealg', False) 

        # track measures about each run (e.g., running time)
        self.params_results = defaultdict(int) 
        # store the node scores for each GO term in a sparse matrix
        self.goid_scores = sparse.lil_matrix(ann_obj.ann_matrix.shape, dtype=np.float)

        # keep track of the weighting method for writing to the output file later
        self.weight_str = '%s%s%s' % (
            'unw-' if net_obj.unweighted else '', 
            'gm2008-' if net_obj.weight_gm2008 else '',
            'swsn-' if net_obj.weight_swsn else '')
        self.setupParamsStr()

    # if the method is not in Python and needs to be called elsewhere, use this
    def setupInputs(self):
        LibMapper[self.name].setupInputs(self)

    # run the method
    def run(self):
        return LibMapper[self.name].run(self)

    # if the method is not in Python and was called elsewhere (e.g., R), 
    # then parse the outputs of the method
    def setupOutputs(self):
        LibMapper[self.name].setupOutputs(self)

    # setup the params_str used in the output file
    def setupParamsStr(self):
        self.params_str = LibMapper[self.name].setup_params_str(self)

