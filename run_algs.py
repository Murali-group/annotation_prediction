
# Script to run the algorithms on a given network and GO terms
#print("Importing libraries")
from optparse import OptionParser,OptionGroup
from collections import defaultdict
import os
import sys
from tqdm import tqdm
import time
sys.path.append("src")
import version_settings as v_settings
import utils.file_utils as utils
sys.path.append("src/algorithms")
import setup_sparse_networks as setup
import alg_utils
import fastsinksource
import genemania
import sinksource_bounds
from aptrank_birgrank.birgrank import birgRank
import aptrank_birgrank.run_birgrank as run_birgrank
# only import aptrank if its being run
# as it uses the cvx solver not installed on all machines
#from aptrank.aptrank import AptRank
from scipy import sparse
import numpy as np
# needed for cross-validation
try:
    from sklearn.model_selection import KFold
except ImportError:
    pass


# These algorithms don't use any negative examples
POSITIVE_ALGS = [
    "sinksourceplus-bounds",
    "fastsinksourceplus",  
    "localplus",  
    ]

class Alg_Runner:
    """
    Base class for running algorithms
    """
    def __init__(
            self, version, exp_name,
            W, prots, ann_matrix, goids,
            algorithms=["fastsinksource"], weight_swsn=False, weight_gm2008=False,
            unweighted=False, ss_lambda=None, 
            alpha=0.8, eps=0.0001, max_iters=1000,
            rank_all=False, rank_pos_neg=False, compare_ranks=False,
            taxon=None, num_pred_to_write=100, 
            tol=1e-05, aptrank_data=None, theta=.5, mu=.5, br_lambda=.5,
            k=8, s=5, t=0.5, diff_type="twoway",
            only_cv=False, cross_validation_folds=None, 
            forcenet=False, forcealg=False, verbose=False, progress_bar=True):
        """
        *eps*: Convergence cutoff for sinksource and sinksourceplus
        *aptrank_data*: tuple containing the dag_matrix, annotation matrix, and the goids of the hierarchy.
        """
        self.version = version
        self.exp_name = exp_name
        #self.W = W
        self.prots = prots
        self.ann_matrix = ann_matrix
        self.goids = goids
        self.algorithms = algorithms
        self.weight_swsn, self.weight_gm2008 = weight_swsn, weight_gm2008
        self.unweighted = unweighted
        self.ss_lambda = ss_lambda
        # parameters for algorithms
        self.alpha, self.eps, self.max_iters = alpha, eps, max_iters
        # sinksource_bounds parameters
        self.rank_all = rank_all
        self.rank_pos_neg = rank_pos_neg
        self.compare_ranks = compare_ranks
        # genemania parameters
        self.tol = tol
        # aptrank parameters
        if aptrank_data is not None:
            self.dag_matrix, self.pos_matrix, self.dag_goids = aptrank_data
        self.theta, self.mu, self.br_lambda = theta, mu, br_lambda
        self.k, self.s, self.t, self.diff_type = k, s, t, diff_type
        # evaluation parameters
        self.taxon = taxon
        self.only_cv = only_cv
        self.cross_validation_folds = cross_validation_folds
        self.num_pred_to_write = num_pred_to_write
        self.forcenet = forcenet
        self.forcealg = forcealg
        self.verbose = verbose
        # option to display progress bar while iterating over GO terms
        self.progress_bar = progress_bar

        # TODO handle the multiple networks more clearly
        if (self.weight_swsn or self.weight_gm2008):
            print("\tkeeping individual STRING networks to be weighted later")
            self.sparse_networks, self.net_names = W
            # TODO more clearly explain the option to allow for both unweighting the STRING networks 
            # and combining them with the GeneMANIA findKernelWeights method
            if self.unweighted:
                print("\tmodifying the 'coexpression' and 'experiments' string networks to be unwieghted. They will be combined later")
                new_sparse_networks = []
                # just unweight the 'coexpression' and 'experiments' string networks for now. 
                # Leave the sequence based networks weighted
                for i in range(len(self.net_names)):
                    net = self.sparse_networks[i]
                    if self.net_names[i] in ['coexpression', 'experiments']: 
                        # convert all of the entries to 1s to "unweight" the network
                        net = (net > 0).astype(int) 
                        new_sparse_networks.append(net)
                    else:
                        new_sparse_networks.append(net)
                self.sparse_networks = new_sparse_networks
            print("\tnormalizing the networks")
            self.normalized_nets = []
            for net in self.sparse_networks:
                self.normalized_nets.append(setup._net_normalize(net))
        else:
            if self.unweighted is True:
                print("\tsetting all edge weights to 1 (unweighted)")
                #and re-normalizing by dividing each edge by the node's degree")
                W = (W > 0).astype(int) 
            self.P = alg_utils.normalizeGraphEdgeWeights(W, ss_lambda=self.ss_lambda)

            if 'genemania' in self.algorithms:
                print("\nCreating Laplacian matrix for GeneMANIA")
                self.L = genemania.setup_laplacian(W)

        # used to map from node/prot to the index and vice versa
        self.node2idx = {n: i for i, n in enumerate(prots)}
        # used to map from index to goid and vice versa
        self.goid2idx = {g: i for i, g in enumerate(goids)}

    def main(self):
        INPUTSPREFIX, RESULTSPREFIX, _, selected_strains = v_settings.set_version(self.version)
        self.INPUTSPREFIX = INPUTSPREFIX
        self.RESULTSPREFIX = RESULTSPREFIX
        for alg in self.algorithms:
            print("starting running algo %s" % alg)
            self.out_dir = "%s/all/%s/%s" % (self.RESULTSPREFIX, alg, self.exp_name)
            utils.checkDir(self.out_dir)

            self.out_pref = "%s/pred-%s%s" % (
                self.out_dir, 'unw-' if self.unweighted else '',
                'l%d-'%int(self.ss_lambda) if self.ss_lambda is not None else '')
            # "prediction mode" is run if the user doesn't specify --only-cv
            if self.only_cv is False:
                if self.num_pred_to_write == 0:
                    out_pref = None
                else:
                    out_pref = self.out_pref

                # now run the algorithm with all combinations of parameters
                goid_scores, curr_params_results = self.run_alg_with_params(
                        alg, out_pref=out_pref)

            # "cross validation mode" is run if the user specifies --cross-validation-folds X
            if self.cross_validation_folds is not None:
                if self.weight_swsn is True or alg == "birgrank" or alg=="aptrank":
                    self.run_cv_all_goterms(alg, folds=self.cross_validation_folds)
                else:
                    out_pref = "%s/cv-%dfolds-%s%sl%d-" % (self.out_dir, self.cross_validation_folds,
                        'unw-' if self.unweighted else '', 
                        'gm2008-' if self.weight_gm2008 else '',
                        0 if self.ss_lambda is None else int(self.ss_lambda))
                    cv_params_results = self.run_goterm_cv(alg,
                            folds=self.cross_validation_folds, out_pref=out_pref)
                    print("Total time taken by algorithms to compute scores for cross validation:")
                    print(", ".join(["%s: %0.3f" % (key, val) for key, val in sorted(cv_params_results.items())]))

        if self.only_cv is False:
            return goid_scores, curr_params_results
        else:
            return

    def run_birgrank_pred(self, alg, out_pref=None):
        if alg == 'birgrank':
            alpha, theta, mu, br_lambda = self.alpha, self.theta, self.mu, self.br_lambda
            if out_pref is not None:
                out_pref += 'a%s-t%s-m%s-l%s-eps%s' % (
                    str_(alpha), str_(theta), str_(mu), str_(br_lambda), str_(self.eps))
        elif alg == 'aptrank':
            br_lambda, k, s, t, diff_type = self.br_lambda, self.k, self.s, self.t, self.diff_type
            if out_pref is not None:
                out_pref += 'l%s-k%s-s%s-t%s-%s' % (
                    str_(br_lambda), str_(k), str_(s), str_(t), diff_type)
        dag_goids2idx = {g: i for i, g in enumerate(self.dag_goids)}
        utils.checkDir(os.path.dirname(out_pref))
        # TODO specify a subset of nodes
        #print("WARNING!!! Running on a subset of the first 1000 nodes")
        #test_nodes = list(range(1000))
        test_nodes = None

        # run each algorithm on these folds
        if alg == 'birgrank' or alg == 'aptrank':
            # the W matrix is already normalized, so I can run
            # birgrank/aptrank from here
            Xh, params_results = self.run_aptrank_with_params(
                self.pos_matrix, self.dag_matrix, alg=alg, nodes=test_nodes, out_pref=out_pref)

            # limit the scores to only the GOIDs for which we have annotations
            goid_scores = sparse.lil_matrix(self.ann_matrix.shape, dtype=np.float)
            for i in range(len(self.goids)):
                goid_scores[i] = Xh[dag_goids2idx[self.goids[i]]]
        return goid_scores

    def run_cv_all_goterms(self, alg, folds=5):
        """
        Split the positives and negatives into folds across all GO terms
        and then run the algorithms on those folds.
        For now, only setup to run BirgRank. TODO allow running each algorithm on the same split of data. 
        """
        # because each fold contains a different set of positives, and combined they contain all positives,
        # store all of the prediction scores from each fold in a matrix
        combined_fold_scores = sparse.lil_matrix(self.ann_matrix.shape, dtype=np.float)
        orig_ann_matrix = self.ann_matrix
        out_pref = "%s/cv-%dfolds-swsn-%sl%d-" % (self.out_dir, self.cross_validation_folds,
                'unw-' if self.unweighted else '', 0 if self.ss_lambda is None else int(self.ss_lambda))
        if alg == 'birgrank':
            alpha, theta, mu, br_lambda = self.alpha, self.theta, self.mu, self.br_lambda
            out_pref += 'a%s-t%s-m%s-l%s' % (
                str_(alpha), str_(theta), str_(mu), str_(br_lambda))
            dag_goids2idx = {g: i for i, g in enumerate(self.dag_goids)}
        elif alg == 'aptrank':
            br_lambda, k, s, t, diff_type = self.br_lambda, self.k, self.s, self.t, self.diff_type
            out_pref += 'l%s-k%s-s%s-t%s-%s' % (
                str_(br_lambda), str_(k), str_(s), str_(t), diff_type)
            dag_goids2idx = {g: i for i, g in enumerate(self.dag_goids)}
        utils.checkDir(os.path.dirname(out_pref))

        ann_matrix_folds = self.split_cv_all_goterms(folds=folds)
        for curr_fold, (train_ann_mat, test_ann_mat) in enumerate(ann_matrix_folds):
            print("*  "*20)
            print("Fold %d" % (curr_fold+1))

            # weight the network if needed
            if self.weight_swsn is True:
                W, process_time = setup.weight_SWSN(train_ann_mat, self.sparse_networks,
                        net_names=self.net_names, nodes=self.prots)
                self.P = alg_utils.normalizeGraphEdgeWeights(W, ss_lambda=self.ss_lambda)
                if alg == 'genemania':
                    self.L = genemania.setup_laplacian(W)

            # run each algorithm on these folds
            if alg == 'birgrank' or alg == 'aptrank':
                # TODO the matrix for BirgRank and the train_matrix do not have the same goids. 
                # I need to build a pos_mat with the train_matrix annotations
                train_pos_mat = sparse.lil_matrix(self.pos_matrix.shape)
                # the birgrank/aptrank annotation (positive) matrix has a different # GO terms,
                # but the same node columns. So I just need to align the rows
                # Also, speed-up birgrank by only getting the scores for the nodes that are a positive or negative for at least 1 GO term
                test_nodes = set()
                for i in range(len(self.goids)):
                    dag_goid_idx = dag_goids2idx[self.goids[i]]
                    train_pos_mat[dag_goid_idx] = train_ann_mat[i]
                    test_nodes.update(set(list(train_ann_mat[i].nonzero()[1])))
                # now set the negatives to 0 as birgrank doesn't use negatives
                train_pos_mat[train_pos_mat < 0] = 0

                # the W matrix is already normalized, so I can run
                # birgrank/aptrank from here
                Xh, params_results = self.run_aptrank_with_params(
                    train_pos_mat, self.dag_matrix, alg=alg, nodes=test_nodes) 
                goid_scores = sparse.lil_matrix(self.ann_matrix.shape, dtype=np.float)
                # limit the scores to only the GOIDs for which we have annotations
                for i in range(len(self.goids)):
                    goid_scores[i] = Xh[dag_goids2idx[self.goids[i]]]
            else:
                # change the annotation matrix to the current fold
                self.ann_matrix = train_ann_mat
                goid_scores, _ = self.run_alg_with_params(alg)

            # store only the scores of the test (left out) positives and negatives
            for i in range(len(self.goids)):
                test_pos, test_neg = alg_utils.get_goid_pos_neg(test_ann_mat, i)
                curr_goid_scores = goid_scores[i].toarray().flatten()
                curr_comb_scores = combined_fold_scores[i].toarray().flatten()
                curr_comb_scores[test_pos] = curr_goid_scores[test_pos]
                curr_comb_scores[test_neg] = curr_goid_scores[test_neg]
                combined_fold_scores[i] = curr_comb_scores 

        #curr_goids = self.dag_goids if alg == 'birgrank' else self.goids
        # now evaluate the results and write to a file
        self.evaluate_ground_truth(
            combined_fold_scores, self.goids, orig_ann_matrix, out_pref,
            #non_pos_as_neg_eval=opts.non_pos_as_neg_eval,
            alg=alg)

    def split_cv_all_goterms(self, folds=5):
        """
        Split the positives and negatives into folds across all GO terms
        Required for running CV with birgrank/aptrank
        *returns*: a list of tuples containing the (train pos, train neg, test pos, test neg)
        """
        print("Splitting all annotations into %d folds by splitting each GO terms annotations into folds, and then combining them" % (folds))
        # TODO there must be a better way to do this than getting the folds in each go term separately
        # list of tuples containing the (train pos, train neg, test pos, test neg) 
        ann_matrix_folds = []
        for i in range(folds):
            train_ann_mat = sparse.lil_matrix(self.ann_matrix.shape, dtype=np.float)
            test_ann_mat = sparse.lil_matrix(self.ann_matrix.shape, dtype=np.float)
            ann_matrix_folds.append((train_ann_mat, test_ann_mat))

        for i in tqdm(range(self.ann_matrix.shape[0]), total=self.ann_matrix.shape[0], disable=not self.progress_bar):
            goid = self.goids[i]
            positives, negatives = alg_utils.get_goid_pos_neg(self.ann_matrix, i)
            if len(positives) < folds or len(negatives) < folds:
                continue
            # print("%d positives, %d negatives for goterm %s" % (len(positives), len(negatives), goid))
            kf = KFold(n_splits=folds, shuffle=True)
            kf_neg = KFold(n_splits=folds, shuffle=True)
            kf.get_n_splits(positives)
            kf_neg.get_n_splits(negatives)
            fold = 0

            for (pos_train_idx, pos_test_idx), (neg_train_idx, neg_test_idx) in zip(kf.split(positives), kf_neg.split(negatives)):
                train_pos, test_pos= positives[pos_train_idx], positives[pos_test_idx]
                train_neg, test_neg = negatives[neg_train_idx], negatives[neg_test_idx]
                train_ann_mat, test_ann_mat = ann_matrix_folds[fold]
                fold += 1
                # build an array of the scores and set it in the goid sparse matrix of scores
                for pos, neg, mat in [(train_pos, train_neg, train_ann_mat), (test_pos, test_neg, test_ann_mat)]:
                    pos_neg_arr = np.zeros(len(self.prots))
                    pos_neg_arr[list(pos)] = 1
                    pos_neg_arr[list(neg)] = -1
                    mat[i] = pos_neg_arr

        return ann_matrix_folds

    def run_aptrank_with_params(self, pos_mat, hierarchy_mat, alg='birgrank', nodes=None, out_pref=None):
        """ Run a protein-based algorithm that uses the hierarchy
        *hierarchy_matrix*: matrix of hierarchy relationships
        *pos_matrix*: matrix of goterm - protein annotations (1 or 0).
            Normally these would be the leaf annotations, not propagated
        *nodes*: set of nodes for which to run RWR and get GO term scores

        *returns*: a matrix of prediction scores with the same
            dimensions as pos_mat
        """
        assert (pos_mat.shape[0] == hierarchy_mat.shape[0]), \
            "Error: annotation and hierarchy matrices " + \
            "do not have the same shape: %d, %d" % (
                pos_mat.shape[0], hierarchy_mat.shape[0])

        # UPDATE 2019-01-04: Include the birgrank lambda parameter which controls the direction of the flow within the hierarchy
        # dH = l * H + (1-l)H^T
        # a value of 1 would be only upwards, while 0 would be downwards
        dH = (self.br_lambda * hierarchy_mat) + ((1-self.br_lambda) * hierarchy_mat.T) 

        # remove the negatives as aptrank doesn't use them
        #pos_mat = self.ann_matrix.copy()
        #pos_mat[pos_mat < 0] = 0
        if alg == 'birgrank':
            start_time = time.process_time()
            Xh = birgRank(self.P, pos_mat.transpose(), dH,
                        alpha=self.alpha, theta=self.theta, mu=self.mu, 
                        eps=self.eps, max_iters=self.max_iters,
                        nodes=nodes, verbose=self.verbose)
            Xh = Xh.T
            process_time = time.process_time() - start_time
        elif alg == 'aptrank':
            # make sure aptrank is imported
            from aptrank.aptrank import AptRank
            # try specifying more cores than there are available
            # to split the problem up into smaller chunks
            #num_cores = 12
            #num_cores = 24
            num_cores = 48
            start_time = time.process_time()
            runner = AptRank(self.P, pos_mat.transpose(), dH,
                        K=self.k, S=self.s, T=self.t, NCores=num_cores, diffusion_type=self.diff_type)
            Xh = runner.algorithm()
            Xh = Xh.T
            process_time = time.process_time() - start_time
        else:
            print("alg %s not yet implemented." % (alg))
            return None, None
        # now write the scores to a file
        if out_pref is not None:
            out_file = "%s.txt" % (out_pref)
            print("\twriting top %d scores to %s" % (self.num_pred_to_write, out_file))

            with open(out_file, 'w') as out:
                for i in range(Xh.shape[0]):
                    scores = Xh[i].toarray().flatten()
                    # convert the nodes back to their names, and make a dictionary out of the scores
                    scores = {self.prots[j]:s for j, s in enumerate(scores)}
                    self.write_scores_to_file(scores, goid=self.goids[i], file_handle=out,
                            num_pred_to_write=self.num_pred_to_write)
        params_results = {'%s_process_time'%alg: process_time}
        return Xh, params_results

    def run_alg_with_params(self, alg, out_pref=None):
        """ Call the appropriate algorithm's function
        """
        params_results = {}

        # TODO run SS and Local separately because they have different parameters
        if alg in ["fastsinksourceplus", "fastsinksource", "localplus", "local"]:
            goid_scores, params_results = self.run_ss_with_params(
                    alg, out_pref=out_pref)
        elif alg in ['sinksource-bounds', 'sinksourceplus-bounds']:
            goid_scores, params_results = self.run_ss_bounds_with_params(
                    alg, out_pref=out_pref)
        elif alg in ["genemania"]:
            out_file = "%sresults.txt" % (out_pref) if out_pref is not None else None
            goid_scores, params_results = self.run_alg_on_goterms(alg,
                    out_file=out_file)
        elif alg in ['birgrank', 'aptrank']:
            goid_scores = self.run_birgrank_pred(alg, out_pref=out_pref)
            params_results = {}

        return goid_scores, params_results

    def run_ss_with_params(self, alg, out_pref=None):
        all_params_results = {} 
        a = self.alpha
        out_file = None
        if out_pref is not None:
            out_file = "%sa%s-eps%s-maxi%d.txt" % (
                    out_pref, str_(a), str_(self.eps), self.max_iters)

        goid_scores, params_results = self.run_alg_on_goterms(
            alg, out_file=out_file, a=a, eps=self.eps)
        all_params_results.update(params_results)

        return goid_scores, all_params_results

    def run_ss_bounds_with_params(self, alg, out_pref=None):
        all_params_results = {}
        a = self.alpha
        out_file = None
        if out_pref is not None:
            out_file = "%s-a%s-maxi%d.txt" % (out_pref, str_(a), self.max_iters)
        # the ranks file will only be written if the according settings are used
        ranks_file = "%scompare-%sranks-a%s.txt" % (self.out_pref,
                'pos-neg-' if sparse.issparse(self.rank_pos_neg) else 'all-', 
                str_(a))
        goid_scores, params_results = self.run_alg_on_goterms(
            alg, out_file=out_file, a=a, ranks_file=ranks_file)
        all_params_results.update(params_results)

        return goid_scores, all_params_results

    def run_alg_on_goterms(
            self, alg, out_file=None, a=0.8, eps='-', ranks_file=None):
        """ Run the specified algorithm with the given parameters for each goterm 
        *returns*: a sparse lil matrix of scores from the algorithm for each goterm
            and a dictionary of summary statistics about the run
        """
        # scores from the algorithm for each goterm
        params_results = defaultdict(int)
        # store the results in a sparse matrix
        goid_scores = sparse.lil_matrix(self.ann_matrix.shape, dtype=np.float)
        try:
            if out_file is not None:
                if self.forcealg is False and os.path.isfile(out_file):
                    print("%s already exists. Use --forcealg to overwrite" % (out_file))
                    print("\tskipping %s" % (alg))
                    return {}, {}

                # saves a lot of time keeping the file handle open
                file_handle = open(out_file, 'w')
                file_handle.write("#goterm\tprot\tscore\n")
            if self.compare_ranks and 'squeeze' in alg:
                # write the ranks to a file
                if self.taxon is None:
                    if self.forcealg is False and os.path.isfile(ranks_file):
                            print("%s already exists. Use --forcealg to overwrite" % (ranks_file))
                            return {}, {}
                    else:
                        print("Writing rank comparison to: %s" % (ranks_file))
                        rank_fh = open(ranks_file, 'w', buffering=100)
                        rank_fh.write("#goterm\tnum_pos\titer\tkendalltau\tnum_unranked\tmax_unr_stretch\tmax_d\tUB\tfmax\tavgp\tauprc\tauroc\n")
                else:
                    # if this is for a specific taxon ID, then include that in the header line
                    if not os.path.isfile(ranks_file):
                        print("Writing rank comparison to: %s" % (ranks_file))
                        rank_fh = open(ranks_file, 'w', buffering=100)
                        rank_fh.write("#goterm\ttaxon\tnum_pos\titer\tkendalltau\tnum_unranked\tmax_unr_stretch\tmax_d\tUB\tfmax\tavgp\tauprc\tauroc\n")
                    else:
                        print("Appending rank comparison to: %s" % (ranks_file))
                        rank_fh = open(ranks_file, 'a', buffering=100)

            print("Running %s for %d goterms. Writing to %s" % (alg, self.ann_matrix.shape[0], out_file))
            for i in tqdm(range(self.ann_matrix.shape[0]), total=self.ann_matrix.shape[0], disable=not self.progress_bar):
                goid = self.goids[i]
                # get the row corresponding to the current goids annotations 
                positives, negatives = alg_utils.get_goid_pos_neg(self.ann_matrix, i)
                # if there are more annotations than nodes in the network, then trim the extra pos/neg nodes
                num_nodes = self.P.shape[0] if self.weight_gm2008 is False else self.normalized_nets[0].shape[0]
                if len(self.prots) > num_nodes: 
                    positives = positives[np.where(positives < num_nodes)]
                    negatives = negatives[np.where(negatives < num_nodes)]

                if self.weight_gm2008 is True:
                    start_time = time.process_time()
                    y = np.zeros(self.normalized_nets[0].shape[0])
                    y[positives] = 1
                    y[negatives] = -1
                    # weight the network for each GO term individually
                    W = setup.weight_GM2008(y, self.normalized_nets, self.net_names, goid)
                    self.P = alg_utils.normalizeGraphEdgeWeights(W, ss_lambda=self.ss_lambda)
                    if alg == 'genemania':
                        self.L = genemania.setup_laplacian(W)
                    params_results['%s_weight_time'%(alg)] += time.process_time() - start_time
                rank_pos_neg = None 
                # if the matrix of left-out positives were passed in, then get the current pos and neg
                if sparse.issparse(self.rank_pos_neg):
                    pos, neg = alg_utils.get_goid_pos_neg(self.rank_pos_neg, i)
                    rank_pos_neg = (set(pos), set(neg))

                # now actually run the algorithm
                scores_arr, curr_params_results, ss_squeeze = self.run_alg(alg, positives, negatives, 
                        a=a, eps=eps, goid=goid, rank_pos_neg=rank_pos_neg)
                # if they're different dimensions, then set the others to zeros 
                if len(scores_arr) < goid_scores.shape[1]:
                    scores_arr = np.append(scores_arr, [0]*(goid_scores.shape[1] - len(scores_arr)))
                goid_scores[i] = scores_arr
                for key in curr_params_results:
                    params_results[key] += curr_params_results[key]

                if self.compare_ranks and 'squeeze' in alg:
                    # compare how long it takes for the ranks to match the previous run
                    tqdm.write("\tRepeating the run, but comparing the ranks from the previous run at each iteration")
                    # keep only the nodes with a non-zero score
                    scores = {n: s for n, s in enumerate(scores_arr) if s > 0}
                    # ranks is a list containing the ranked order of nodes.
                    # The node with the highest score is first, the lowest is last
                    if rank_pos_neg is not None:
                        pos_neg_nodes = rank_pos_neg[0] | rank_pos_neg[1]
                        ranks = [n for n in sorted(set(scores.keys()) & pos_neg_nodes, key=scores.get, reverse=True)]
                    else:
                        ranks = [n for n in sorted(scores, key=scores.get, reverse=True)]
                    # rerun SinkSourceBounds, and compare the ranking at each step
                    _, _, ss_squeeze = self.run_alg(alg, positives, negatives, a=a, eps=eps, goid=goid, 
                            ranks_to_compare=ranks, scores_to_compare=ss_squeeze.scores_to_compare,
                            rank_pos_neg=rank_pos_neg)
                    rank_fh.write(''.join("%s%s\t%d\t%d\t%0.6f\t%d\t%d\t%0.4e\t%0.4e\t%0.4f\t%0.4f\t%0.4f\t%0.4f\n" % (
                        goid, "\t%s"%self.taxon if self.taxon is not None else "", len(positives), i+1, ss_squeeze.kendalltau_list[i],
                        ss_squeeze.num_unranked_list[i], ss_squeeze.max_unranked_stretch_list[i], ss_squeeze.max_d_list[i], ss_squeeze.UB_list[i],
                        ss_squeeze.eval_stats_list[i][0], ss_squeeze.eval_stats_list[i][1], ss_squeeze.eval_stats_list[i][2], ss_squeeze.eval_stats_list[i][3])
                                        for i in range(ss_squeeze.num_iters)))

                if out_file is not None:
                    # convert the nodes back to their names, and make a dictionary out of the scores
                    scores = {self.prots[i]:s for i, s in enumerate(scores_arr)}
                    self.write_scores_to_file(scores, goid=goid, file_handle=file_handle,
                            num_pred_to_write=self.num_pred_to_write)

        except:
            if out_file is not None:
                file_handle.close()
            raise

        if out_file is not None:
            file_handle.close()
            print("Finished running %s for %d goterms. Wrote to %s" % (alg, len(self.goids), out_file))
        if self.compare_ranks and 'squeeze' in alg:
            rank_fh.close()
            print("Finished running rank comparison. Wrote to: %s" % (ranks_file))

        return goid_scores, params_results

    def run_alg(self, alg, positives, negatives, rank_pos_neg=None,
                a=0.8, eps='-', goid='-', ranks_to_compare=None, scores_to_compare=None):
        """ Run the specified algorithm with the given parameters using the given positive and negative examples
        *returns*: a numpy array of scores from the algorithm for each node,
            a dictionary of summary statistics about the run,
            and the alg runner object containing more statistics about the run if alg is 'ss_bounds', 
        """
        num_unk = self.P.shape[0] - len(positives)
        if alg in POSITIVE_ALGS:
            negatives=None
        else:
            num_unk -= len(negatives) 
        ss_obj = None 
        params_results = {} 

        # TODO streamline calling the correct function. They all take the same parameters
        # This uses the same UB as Ripple
        if alg in ['fastsinksourceplus', 'fastsinksource']:
            scores, process_time, wall_time, iters = fastsinksource.runFastSinkSource(
                self.P, positives, negatives=negatives, max_iters=self.max_iters,
                eps=eps, a=a, verbose=self.verbose)
            params_results["%s_wall_time"%alg] = wall_time
        elif alg in ['localplus', 'local']:
            scores, process_time = fastsinksource.runLocal(
                    self.P, positives, negatives=negatives)
            iters = 1
        elif alg == 'genemania':
            y = np.zeros(self.P.shape[0])
            y[positives] = 1
            y[negatives] = -1
            scores, process_time, wall_time, iters = genemania.runGeneMANIA(self.L, y, tol=self.tol, verbose=self.verbose)
            params_results["%s_wall_time"%alg] = wall_time
        elif alg in ['sinksourceplus-bounds', 'sinksource-bounds']:
            ss_obj = sinksource_bounds.SinkSourceBounds(
                    self.P, positives, negatives=negatives, a=a, verbose=self.verbose,
                    rank_all=self.rank_all, rank_pos_neg=rank_pos_neg,
                    ranks_to_compare=ranks_to_compare, scores_to_compare=scores_to_compare, max_iters=self.max_iters)
            scores = ss_obj.runSinkSourceBounds() 
            process_time, update_time, iters, comp = ss_obj.get_stats()
            params_results["%s_update_time"%alg] = update_time

        tqdm.write("\t%s converged after %d iterations " % (alg, iters) +
                "(%0.4f sec) for goterm %s" % (process_time, goid))

        # also keep track of the time it takes for each of the parameter sets
        params_results["%s_process_time"%alg] = process_time

        return scores, params_results, ss_obj

    def write_scores_to_file(self, scores, goid='', out_file=None, file_handle=None,
            num_pred_to_write=100, header="", append=True):
        """
        *scores*: dictionary of node_name: score
        *num_pred_to_write*: number of predictions (node scores) to write to the file 
            (sorted by score in decreasing order). If -1, all will be written
        """

        if num_pred_to_write == -1:
            num_pred_to_write = len(scores) 

        if out_file is not None:
            if append:
                print("Appending %d scores for goterm %s to %s" % (num_pred_to_write, goid, out_file))
                out_type = 'a'
            else:
                print("Writing %d scores for goterm %s to %s" % (num_pred_to_write, goid, out_file))
                out_type = 'w'

            file_handle = open(out_file, out_type)
        elif file_handle is None:
            print("Warning: out_file and file_handle are None. Not writing scores to a file")
            return 

        # write the scores to a file, up to the specified number of nodes (num_pred_to_write)
        file_handle.write(header)
        for n in sorted(scores, key=scores.get, reverse=True)[:num_pred_to_write]:
            file_handle.write("%s\t%s\t%s\n" % (goid, n, str(scores[n])))
        return

    def run_goterm_cv(self, alg, folds=5, out_pref=None):
        """
        For every GO term, split the positives and negatives into the specified number of folds 
            and run and evaluate the given algorithm
        """
        params_results = defaultdict(int)

        print("Running cross-validation for %d goterms using %d folds for %s; a=%s, eps=%s" % (
            self.ann_matrix.shape[0], self.cross_validation_folds, alg, self.alpha, self.eps))

        out_file = None
        if out_pref is not None:
            out_file = "%sa%s-eps%s-maxi%d.txt" % (
                    out_pref, str_(self.alpha), str_(self.eps),
                    self.max_iters)
            print("Writing CV results to %s" % (out_file))
            file_handle = open(out_file, 'w')
            file_handle.write("#goterm\tfmax\n")

        for i in tqdm(range(self.ann_matrix.shape[0]), total=self.ann_matrix.shape[0], disable=not self.progress_bar):
            goid = self.goids[i]
            positives, negatives = alg_utils.get_goid_pos_neg(self.ann_matrix, i)
            print("%d positives, %d negatives for goterm %s" % (len(positives), len(negatives), goid))
            kf = KFold(n_splits=folds, shuffle=True)
            kf_neg = KFold(n_splits=folds, shuffle=True)
            kf.get_n_splits(positives)
            kf_neg.get_n_splits(negatives)
            fold = 0
            # because each fold contains a different set of positives, and combined they contain all positives,
            # store all of the prediction scores from each fold in a list
            combined_fold_scores = np.zeros(self.P.shape[0], dtype='float128')
            if self.weight_gm2008 is True:
                unknowns = set(range(self.normalized_nets[0].shape[0])) - set(positives) 
            else:
                unknowns = set(range(self.P.shape[0])) - set(positives) 
            if 'plus' not in alg:
                unknowns = unknowns - set(negatives)

            for (pos_train_idx, pos_test_idx), (neg_train_idx, neg_test_idx) in zip(kf.split(positives), kf_neg.split(negatives)):
                fold += 1
                pos_train, pos_test = positives[pos_train_idx], positives[pos_test_idx]
                neg_train, neg_test = negatives[neg_train_idx], negatives[neg_test_idx]

                nodes_to_rank = None 
                if self.rank_pos_neg is True:
                    nodes_to_rank = (set(list(pos_test)), set(list(neg_test)))

                if self.weight_gm2008 is True:
                    y = np.zeros(self.normalized_nets[0].shape[0])
                    y[pos_train] = 1
                    y[neg_train] = -1

                    # SWSN weighting is handled in a different function
                    W = setup.weight_GM2008(y, self.normalized_nets, self.net_names, goid)
                    self.P = alg_utils.normalizeGraphEdgeWeights(W, ss_lambda=self.ss_lambda)
                    if alg == 'genemania':
                        self.L = genemania.setup_laplacian(W)

                scores, curr_params_results, _ = self.run_alg(alg, pos_train, neg_train,
                        rank_pos_neg=nodes_to_rank, goid=goid, a=self.alpha, eps=self.eps)
                for key in curr_params_results:
                    params_results[key] += curr_params_results[key]

                # add a check to ensure the scores are actually available
                if len(scores) == 0:
                    print("WARNING: No scores found. Skipping")
                    continue

                nodes_to_track = list(pos_test) + list(neg_test)
                # the test positives and negatives will appear in a single fold
                if nodes_to_rank is not None:
                    fold_pos_ranks = [i for i, x in enumerate(sorted(scores, key=scores.get)) if x in nodes_to_rank[0]]
                    print("Ranks of left out positives:")
                    print(fold_pos_ranks)
                combined_fold_scores[nodes_to_track] = scores[nodes_to_track]

            if len(combined_fold_scores) == 0:
                continue

            # sort the combined scores by the score, and then compute the metrics on the combined scores
            prec, recall, fpr = alg_utils.compute_eval_measures(combined_fold_scores, positives, negatives)
            fmax = alg_utils.compute_fmax(prec, recall)
            tqdm.write("\toverall fmax: %0.3f" % (fmax))
            if out_file is not None:
                file_handle.write("%s\t%0.4f\n" % (goid, fmax))
        if out_file is not None:
            file_handle.close()
            print("Finished running %s for %d goterms. Wrote to %s" % (alg, self.ann_matrix.shape[0], out_file))

        return params_results

    def evaluate_ground_truth(
            self, goid_scores, goids, true_ann_matrix, out_file,
            non_pos_as_neg_eval=False, taxon='-',
            alg='', write_prec_rec=False, append=True):

        score_goids2idx = {g: i for i, g in enumerate(goids)}
        print("Computing fmax from ground truth of %d goterms" % (true_ann_matrix.shape[0]))
        goid_stats = {}
        goid_num_pos = {} 
        goid_prec_rec = {}
        for i in range(true_ann_matrix.shape[0]):
            goid = self.goids[i]
            # make sure the scores are actually available first
            if goid not in self.goid2idx:
                print("WARNING: goid %s not in initial set of %d goids" % (
                                  goid, len(self.goids)))
                continue
            # get the row corresponding to the current goids annotations 
            goid_ann = true_ann_matrix[i,:].toarray().flatten()
            positives = np.where(goid_ann > 0)[0]
            # to get the scores, map the current goid index to the
            # index of the goid in the scores matrix
            scores = goid_scores[score_goids2idx[goid]]
            # this is only needed for aptrank since it does not return a sparse matrix
            if sparse.issparse(scores):
                scores = scores.toarray().flatten()
            goid_num_pos[goid] = len(positives)
            if len(positives) == 0:
                if self.verbose:
                    print("%s has 0 positives after restricting to nodes in the network. Skipping" % (goid))
                continue
            if non_pos_as_neg_eval is True:
                # leave everything not a positive as a negative
                negatives = None
            else:
                # alternatively, use the negatives from that species as the set of negatives
                negatives = np.where(goid_ann < 0)[0]
                if len(negatives) == 0:
                    print("WARNING: 0 negatives for %s - %s. Skipping" % (goid, taxon))
                    continue
            prec, recall, fpr, pos_neg_stats = alg_utils.compute_eval_measures(scores, positives, negatives=negatives, track_pos_neg=True)
            if write_prec_rec:
                goid_prec_rec[goid] = (prec, recall, pos_neg_stats)
            fmax = alg_utils.compute_fmax(prec, recall)
            avgp = alg_utils.compute_avgp(prec, recall)
            if len(prec) == 1:
                auprc = 0
                auroc = 0
            else:
                auprc = alg_utils.compute_auprc(prec, recall)
                auroc = alg_utils.compute_auroc([r for r, f in fpr], [f for r, f in fpr])
            goid_stats[goid] = (fmax, avgp, auprc, auroc)
            if self.verbose:
                print("%s fmax: %0.4f" % (goid, fmax))

        if not write_prec_rec:
            # don't write the header each time
            if not os.path.isfile(out_file) or not append:
                print("Writing results to %s" % (out_file))
                with open(out_file, 'w') as out:
                    if taxon == '-':
                        out.write("#goid\tfmax\tavgp\tauprc\tauroc\t# ann\n")
                    else:
                        out.write("#taxon\tgoid\tfmax\tavgp\tauprc\tauroc\t# test ann\n")
            else:
                print("Appending results to %s" % (out_file))
            with open(out_file, 'a') as out:
                out.write(''.join(["%s%s\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%d\n" % (
                    "%s\t"%taxon if taxon != "-" else "",
                    g, fmax, avgp, auprc, auroc, goid_num_pos[g]
                    ) for g, (fmax, avgp, auprc, auroc) in goid_stats.items()]))

        if write_prec_rec:
            goid = list(goid_prec_rec.keys())[0]
            out_file_pr = out_file.replace('.txt', "prec-rec%s%s.txt" % (
                taxon, '-%s'%(goid) if len(goid_prec_rec) == 1 else ""))
            print("writing prec/rec to %s" % (out_file_pr))
            with open(out_file_pr, 'w') as out:
                out.write("goid\tprec\trec\tnode\tscore\tidxpos/neg\n")
                for goid, (prec, rec, pos_neg_stats) in goid_prec_rec.items():
                    out.write(''.join(["%s\t%0.4f\t%0.4f\t%s\t%0.4f\t%d\t%d\n" % (
                        goid, p, r, self.prots[n], s, idx, pos_neg) for p,r,(n,s,idx,pos_neg) in zip(prec[1:], rec[1:], pos_neg_stats)]))


def parse_args(args):
    ## Parse command line args.
    usage = '%s [options]\n' % (sys.argv[0])
    parser = OptionParser(usage=usage)

    # general parameters
    group = OptionGroup(parser, 'Main Options')
    group.add_option('','--version', type='string', default="2018_06-seq-sim-e0_1",
                     help="Version of the PPI to run. Default: %s" % ("2018_06-seq-sim-e0_1") + \
                     "\nOptions are: %s." % (', '.join(v_settings.ALLOWEDVERSIONS)))
    group.add_option('-N','--net-file', type='string',
                     help="Network file to use. Default is the version's default network")
    group.add_option('-A', '--algorithm', action="append",
                     help="Algorithm for which to get predictions. Default is all of them. Options: '%s'" % ("', '".join(alg_utils.ALGORITHMS)))
    group.add_option('', '--exp-name', type='string',
                     help="Outputs will be under this experiment name.")
    group.add_option('', '--pos-neg-file', type='string', 
                     help="File containing positive and negative examples for each GO term")
    group.add_option('', '--only-functions', type='string',
                     help="Run using only the functions in a specified file (should contain only the ID meaning without the leading GO:00).")
    group.add_option('-G', '--goterm', type='string', action="append",
                     help="Specify the GO terms to use (should be in GO:00XX format)")
    parser.add_option_group(group)

    add_alg_opts(parser)
    add_string_opts(parser)

    # additional parameters
    group = OptionGroup(parser, 'Additional options')
    group.add_option('-W', '--num-pred-to-write', type='int', default=100,
                     help="Number of predictions to write to the file. If 0, none will be written. If -1, all will be written. Default=100")
    group.add_option('', '--only-cv', action="store_true", default=False,
                     help="Perform cross-validation only")
    group.add_option('-C', '--cross-validation-folds', type='int',
                     help="Perform cross validation using the specified # of folds. Usually 5")
    # TODO finish adding this option
    #group.add_option('-T', '--ground-truth-file', type='string',
    #                 help="File containing true annotations with which to evaluate predictions")
    group.add_option('', '--forcealg', action="store_true", default=False,
                     help="Force re-running algorithms if the output files already exist")
    group.add_option('', '--forcenet', action="store_true", default=False,
                     help="Force re-building network matrix from scratch")
    group.add_option('', '--verbose', action="store_true", default=False,
                     help="Print additional info about running times and such")
    parser.add_option_group(group)

    (opts, args) = parser.parse_args(args)

    if opts.exp_name is None or opts.pos_neg_file is None:
        print("--exp-name, --pos-neg-file, required")
        sys.exit(1)

    # if neither are provided, just use all GO terms in the pos/neg file
#    if opts.goterm is None and opts.only_functions is None:
#        print("--goterm or --only_functions required")
#        sys.exit(1)

    if opts.algorithm is None:
        opts.algorithm = alg_utils.ALGORITHMS

    if opts.version not in v_settings.ALLOWEDVERSIONS:
        print("ERROR: '%s' not an allowed version. Options are: %s." % (opts.version, ', '.join(v_settings.ALLOWEDVERSIONS)))
        sys.exit(1)

    #if opts.algorithm not in v_settings.ALGORITHM_OPTIONS:
    #    print "--algorithm %s is not a valid algorithm name" % (opts.algorithm)
    #    sys.exit(1)

    # TODO
    for alg in opts.algorithm:
        if alg not in alg_utils.ALGORITHMS:
            print("ERROR: '%s' not a valid algorithm name. Algorithms are: '%s'." % (alg, ', '.join(alg_utils.ALGORITHMS)))
            sys.exit(1)

    if 'aptrank' in opts.algorithm:
        from aptrank.aptrank import AptRank

    validate_string_opts(opts)

    return opts


def add_alg_opts(parser):
    # parameters for running algorithms
    group = OptionGroup(parser, 'Algorithm options')
    group.add_option('', '--unweighted', action="store_true", default=False,
                     help="Option to ignore edge weights when running algorithms. Default=False (weighted)")
    group.add_option('-l', '--sinksourceplus-lambda', type=float, 
                     help="lambda parameter to specify the weight connecting the unknowns to the negative 'ground' node. Default=None")
    group.add_option('-a', '--alpha', type=float, default=0.8,
                     help="Alpha insulation parameter. Default=0.8")
    group.add_option('', '--eps', type=float, default=0.0001,
                     help="Stopping criteria for SinkSource. Default=0.0001")
    group.add_option('', '--max-iters', type=int, default=1000,
                     help="Maximum # of iterations for FastSinkSource. Default=1000")
    parser.add_option_group(group)

    # bounds parameters
    group = OptionGroup(parser, 'SinkSourceBounds options')
    group.add_option('', '--rank-all', action="store_true", default=False,
                     help="Continue iterating until all nodes ranks are fixed (comparing the UB and LB). Currently only available for SinkSourceBounds")
    group.add_option('', '--rank-pos-neg', action="store_true", default=False,
                     help="During cross-validation, continue iterating until the nodes of the left out positives and negatives are fixed (comparing the UB and LB). Currently only available for SinkSourceBounds")
    group.add_option('', '--compare-ranks', action="store_true", default=False,
                     help="Compare how long it takes (# iters) for the ranks to match the final fixed ranks." +
                     "Currently only implemented with ss_bounds and --rank-all and --rank-pos-neg")
    parser.add_option_group(group)

    # genemania options
    group = OptionGroup(parser, 'GeneMANIA options')
    group.add_option('', '--tol', type=float, default=1e-05,
                     help="Tolerance for convergance for the Scipy Sparse Conjugate Gradient solver.")
    parser.add_option_group(group)

    # birgrank options
    group = OptionGroup(parser, 'BirgRank options')
    group.add_option('-b', '--obo-file', type='string', default=v_settings.GO_FILE,
                     help="GO OBO file which contains the GO DAG. Used if running AptRank/BirgRank. Default: %s" % (v_settings.GO_FILE))
    group.add_option('', '--theta', type=float, default=.5,
                     help="BirgRank parameter: (1-theta) percent of Rtrain used in seeding vectors")
    group.add_option('', '--mu', type=float, default=.5,
                     help="BirgRank parameter: (1-mu) percent of random walkers diffuse from G via Rtrain to H")
    group.add_option('', '--br-lambda', type=float, default=.5,
                     help="BirgRank/AptRank parameter: (1-lambda) percent of random walkers which diffuse downwards within H")
    parser.add_option_group(group)

    # aptrank options
    group = OptionGroup(parser, 'AptRank options')
    group.add_option('', '--apt-k', default=8,
                     help="Markov Chain iterations")
    group.add_option('', '--apt-s', default=5,
                     help="Number of shuffles")
    group.add_option('', '--apt-t', default=0.5,
                     help="Split percentage")
    group.add_option('', '--diff-type', default="twoway",
                     help="Diffusion type: oneway (G to H) or twoway (G to H and H to G)")
    parser.add_option_group(group)


def add_string_opts(parser):
    # parameters for STRING networks
    group = OptionGroup(parser, 'STRING options')
    group.add_option('', '--weight-gm2008', action="store_true", default=False,
                     help="Option to integrate multiple networks using the original GeneMANIA method where networks are weighted individually for each GO term. Only takes effect if multiple networks are present for the specified version. Must specify either this or --unweighted or --weight-swsn.")
    group.add_option('', '--weight-swsn', action="store_true", default=False,
                     help="Option to integrate multiple networks using the SWSN (Simultaneous Weighting with Specific Negatives) method. Only takes effect if multiple networks are present for the specified version. Default=False (integrate the networks for each GO term individually).")
    group.add_option('', '--string-combined', action="store_true", default=False,
            help="Use only the STRING combined network: \n\tcombined_score")
    group.add_option('', '--string-core', action="store_true", default=False,
            help="Use only the 6 core networks: \n\t%s" % (', '.join(setup.CORE_STRING_NETWORKS)))
    group.add_option('', '--string-non-transferred', action="store_true", default=False,
            help="Use all non-transferred networks: \n\t%s" % (', '.join(setup.NON_TRANSFERRED_STRING_NETWORKS)))
    group.add_option('', '--string-all', action="store_true", default=False,
            help="Use all individual 13 STRING networks: \n\t%s" % (', '.join(setup.STRING_NETWORKS)))
    group.add_option('-S', '--string-networks', type='string', 
            help="Comma-separated list of string networks to use. " +
                 "If specified, other STRING options will be ignored." +
                 "Default: %s" % (', '.join(setup.CORE_STRING_NETWORKS)))
    parser.add_option_group(group)


def validate_string_opts(opts):
    # setup the selection of string networks 
    string_networks = []
    # if none are specified, leave the list empty. Useful to keep the same set of nodes (i.e., for validation), but not include the STRING networks
    if opts.string_networks == "":
        pass
    elif opts.string_networks is not None:
        string_networks = opts.string_networks.split(',')
        for net in string_networks:
            if net not in setup.STRING_NETWORKS:
                print("ERROR: STRING network '%s' not one of the" % (net) + 
                      "available choices which are: \n\t%s" % (', '.join(setup.STRING_NETWORKS)))
                sys.exit(1)
    elif opts.string_combined:
        string_networks = ['combined_score']
    elif opts.string_core:
        string_networks = setup.CORE_STRING_NETWORKS
    elif opts.string_non_transferred:
        string_networks = setup.NON_TRANSFERRED_STRING_NETWORKS
    elif opts.string_all:
        string_networks = setup.STRING_NETWORKS
    else:
        string_networks = setup.CORE_STRING_NETWORKS
    opts.string_networks = string_networks

    if 'STRING' in v_settings.NETWORK_VERSION_INPUTS[opts.version]: 
        # if a net_file is specified, then no need to specify the weight type
        if (not opts.net_file or not os.path.isfile(opts.net_file)) and (not opts.weight_gm2008 and not opts.weight_swsn and not opts.unweighted):
            print("ERROR: must specify either --weight-gm2008, --weight-swsn or --unweighted")
            sys.exit(1)
        if opts.weight_gm2008 and opts.weight_swsn:
            print("ERROR: cannot specify both --weight-gm2008 and --weight-swsn")
            sys.exit(1)
        if (opts.net_file and os.path.isfile(opts.net_file)) and (opts.weight_gm2008 or opts.weight_swsn):
            print("ERROR: --net-file is selected. Cannot use --weight-gm2008 or --weight-swsn")
            sys.exit(1)
    else:
        if opts.weight_gm2008 or opts.weight_swsn:
            print("ERROR: version %s doesn't have any multiple networks. " + \
                  "To use the --weight-gm2008 or --weight-swsn options, please choose a version with STRING or update fungcat_settings.py")
            sys.exit(1)


def run():
    opts = parse_args(sys.argv)
    goterms = alg_utils.select_goterms(
            only_functions_file=opts.only_functions, goterms=opts.goterm) 

    # load the network matrix and protein IDs
    net_file = opts.net_file
    if net_file is None:
        INPUTSPREFIX, _, net_file, selected_strains = v_settings.set_version(opts.version) 
    else:
        INPUTSPREFIX, _, _, selected_strains = v_settings.set_version(opts.version) 
    # TODO this should be better organized so that any STRING networks can be used
    if opts.weight_gm2008 or opts.weight_swsn:
        # if net_file is specified, but it doesn't exist, then write it
        if net_file is not None and not os.path.isfile(net_file):
            #print("Option to write net file not yet implemented. Quitting")
            print("Network file %s not found. Quitting." % (net_file))
            sys.exit(1)
        out_pref_net = "%s/sparse-nets/" % (INPUTSPREFIX)
        utils.checkDir(out_pref_net)
        # build the file containing the sparse networks
        sparse_networks, network_names, prots = setup.create_sparse_net_file(
            opts.version, out_pref_net, selected_strains=selected_strains,
            string_nets=opts.string_networks, string_file_cutoff=v_settings.VERSION_STRING_FILE_CUTOFF[opts.version],
            string_cutoff=v_settings.VERSION_STRING_CUTOFF[opts.version],
            forcenet=opts.forcenet)
        # TODO organize this better
        W = (sparse_networks, network_names)
    else:
        W, prots = alg_utils.setup_sparse_network(net_file, forced=opts.forcenet)
    # now build the annotation matrix
    ann_matrix, goids = setup.setup_sparse_annotations(opts.pos_neg_file, goterms, prots)

    # TODO make this more streamlined
    aptrank_data = None
    print("starting to run the algorithm") 
    if 'birgrank' in opts.algorithm or 'aptrank' in opts.algorithm:
        dag_matrix, pos_matrix, dag_goids = run_birgrank.setup_h_ann_matrices(
                prots, opts.obo_file, opts.pos_neg_file, goterms=goids)
        aptrank_data = (dag_matrix, pos_matrix, dag_goids)

    alg_runner = Alg_Runner(
        opts.version, opts.exp_name,
        W, prots, ann_matrix, goids,
        algorithms=opts.algorithm, weight_swsn=opts.weight_swsn,
        weight_gm2008=opts.weight_gm2008,
        unweighted=opts.unweighted, ss_lambda=opts.sinksourceplus_lambda,
        alpha=opts.alpha, eps=opts.eps, max_iters=opts.max_iters, tol=opts.tol,
        rank_all=opts.rank_all, rank_pos_neg=opts.rank_pos_neg, compare_ranks=opts.compare_ranks,
        num_pred_to_write=opts.num_pred_to_write, aptrank_data=aptrank_data,
        only_cv=opts.only_cv, cross_validation_folds=opts.cross_validation_folds,
        k=opts.apt_k, s=opts.apt_s, t=opts.apt_t, diff_type=opts.diff_type,
        forcealg=opts.forcealg, forcenet=opts.forcenet, verbose=opts.verbose)
    print("starting main function")
    alg_runner.main()


# small helper function to replace a period with an underscore
# for example: 0.95 -> 0_95
def str_(s):
    return str(s).replace('.','_')


if __name__ == "__main__":
    run()
