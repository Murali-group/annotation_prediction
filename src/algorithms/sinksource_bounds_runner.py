
import sys, os
import time
from scipy import sparse as sp
import src.algorithms.fastsinksource_runner as fss_runner
import src.algorithms.sinksource_bounds as ss_bounds
import src.algorithms.alg_utils as alg_utils
from tqdm import tqdm, trange
import fcntl


def setupInputs(run_obj):
    # setup is the same as for fastsinksource
    fss_runner.setupInputs(run_obj)
#    # may need to make sure the inputs match
#    ## if there are more annotations than nodes in the network, then trim the extra pos/neg nodes
#    #num_nodes = self.P.shape[0] if self.weight_gmw is False else self.normalized_nets[0].shape[0]
#    #if len(self.prots) > num_nodes: 
#    #    positives = positives[np.where(positives < num_nodes)]
#    #    negatives = negatives[np.where(negatives < num_nodes)]
#
#    # extract the variables out of the annotation object
#    run_obj.ann_matrix = run_obj.ann_obj.ann_matrix
#    run_obj.goids = run_obj.ann_obj.goids
#
#    if run_obj.net_obj.weight_swsn:
#        # TODO if the net obj already has the W_SWSN object, then use that instead
#        W, process_time = run_obj.net_obj.weight_SWSN(run_obj.ann_matrix)
#        run_obj.P = alg_utils.normalizeGraphEdgeWeights(W, ss_lambda=run_obj.params.get('lambda', None))
#        run_obj.params_results['%s_weight_time'%(run_obj.name)] += process_time
#    elif run_obj.net_obj.weight_gmw:
#        # this will be handled on a GO term by GO term basis
#        run_obj.P = None
#    else:
#        run_obj.P = alg_utils.normalizeGraphEdgeWeights(run_obj.net_obj.W, ss_lambda=run_obj.params.get('lambda', None))

    return


# setup the params_str used in the output file
def setup_params_str(
        weight_str, params, name="fastsinksource",
        rank_all=False, rank_pos_neg=None):
    # TODO update with additional parameters
    # ss_lambda affects the network that all these methods use
    ss_lambda = params.get('lambda', 0)
    params_str = "%s-l%s" % (weight_str, ss_lambda)
    if name.lower() not in ["local", "localplus"]:
        a, maxi = params['alpha'], params['max_iters']
        params_str += "-a%s-maxi%s" % ( 
            str_(a), str_(maxi))

    return params_str


def get_alg_type():
    return "term-based"


# write the ranks file
def setupOutputs(run_obj, taxon=None, **kwargs):
    ranks_file = run_obj.out_pref + "-ranks.txt"
    # TODO apply this forced before any of the taxon 
    if not os.path.isfile(ranks_file):
        print("Writing rank stats to %s" % (ranks_file))
        append = False
    else:
        print("Appending rank stats to %s." % (ranks_file))
        append = True
    with open(ranks_file, 'a' if append else 'w') as out:
        # lock it to avoid scripts trying to write at the same time
        fcntl.flock(out, fcntl.LOCK_EX)
        if append is False:
            if taxon is not None:
                out.write("#goterm\ttaxon\tnum_pos\titer\tkendalltau\tnum_unranked\tmax_unr_stretch\tmax_d\tUB\tfmax\tavgp\tauprc\n")
            else:
                out.write("#goterm\tnum_pos\titer\tkendalltau\tnum_unranked\tmax_unr_stretch\tmax_d\tUB\tfmax\tavgp\tauprc\n")
        for goid, rank_stats in run_obj.goid_rank_stats.items():
            if taxon is not None:
                goid += "\t"+taxon
            out.write(''.join("%s\t%s\n" % (goid, stats) for stats in rank_stats))
        fcntl.flock(out, fcntl.LOCK_UN)


def run(run_obj):
    """
    Function to run FastSinkSource, FastSinkSourcePlus, Local and LocalPlus
    *goids_to_run*: goids for which to run the method. 
        Must be a subset of the goids present in the ann_obj
    """
    params_results, goid_scores = run_obj.params_results, run_obj.goid_scores
    goid_rank_stats = {}
    P, alg, params = run_obj.P, run_obj.name, run_obj.params
    print("Running %s with these parameters: %s" % (alg, params))

    # run FastSinkSource on each GO term individually
    #for i in trange(run_obj.ann_matrix.shape[0]):
    #goid = run_obj.goids[i]
    for goid in tqdm(run_obj.goids_to_run):
        idx = run_obj.ann_obj.goid2idx[goid]
        # get the row corresponding to the current goids annotations 
        y = run_obj.ann_matrix[idx,:]
        positives = (y > 0).nonzero()[1]
        negatives = (y < 0).nonzero()[1]
        # if this method uses positive examples only, then remove the negative examples
        if alg in ["fastsinksourceplus", "sinksourceplus", "localplus"]:
            negatives = None

        if run_obj.net_obj.weight_gmw is True:
            start_time = time.process_time()
            # weight the network for each GO term individually
            W, process_time = run_obj.net_obj.weight_GMW(y.toarray()[0], goid)
            P = alg_utils.normalizeGraphEdgeWeights(W, ss_lambda=params.get('lambda', None))
            params_results['%s_weight_time'%(alg)] += time.process_time() - start_time

        a, max_iters = params['alpha'], params['max_iters']
        compare_ranks = params['compare_ranks']
        # rank_all is a T/F option, but 'rank_pos_neg' will be the test/left-out ann matrix 
        # from which we can get the left-out pos/neg for this term
        rank_all, rank_pos_neg = params['rank_all'], params['rank_pos_neg']
        if sp.issparse(rank_pos_neg):
            pos, neg = alg_utils.get_goid_pos_neg(rank_pos_neg, idx)
            rank_pos_neg = (set(pos), set(neg))
        elif rank_pos_neg is True:
            print("ERROR: rank_pos_neg must be the test_ann_mat")
            sys.exit()

        # now actually run the algorithm
        ss_obj = ss_bounds.SinkSourceBounds(
            P, positives, negatives=negatives, max_iters=max_iters,
            a=a, rank_all=rank_all, rank_pos_neg=rank_pos_neg,
            verbose=run_obj.kwargs.get('verbose', False))

        scores_arr = ss_obj.runSinkSourceBounds()
        process_time, update_time, iters, comp = ss_obj.get_stats()

        if run_obj.kwargs.get('verbose', False) is True:
            tqdm.write("\t%s converged after %d iterations " % (alg, iters) +
                    "(%0.4f sec) for %s" % (process_time, goid))

        if compare_ranks:
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

            # left off top-k for now
            #ranks = ranks[:k] if self.rank_topk is True else ranks
            ss_obj = ss_bounds.SinkSourceBounds(
                P, positives, negatives=negatives, max_iters=max_iters,
                a=a, rank_all=rank_all, rank_pos_neg=rank_pos_neg, ranks_to_compare=ranks,
                verbose=run_obj.kwargs.get('verbose', False))
            ss_obj.runSinkSourceBounds()

            rank_stats = ["%d\t%d\t%0.4e\t%d\t%d\t%0.2e\t%0.2e\t%0.4f\t%0.4f\t%0.4f" % (
                len(positives), i+1, ss_obj.kendalltau_list[i], ss_obj.num_unranked_list[i],
                ss_obj.max_unranked_stretch_list[i], ss_obj.max_d_list[i], ss_obj.UB_list[i],
                ss_obj.eval_stats_list[i][0], ss_obj.eval_stats_list[i][1],
                ss_obj.eval_stats_list[i][2])
                                for i in range(ss_obj.num_iters)]
            goid_rank_stats[goid] = rank_stats

            #rank_fh.write(''.join("%s%s\t%d\t%d\t%0.6f\t%d\t%d\t%0.4e\t%0.4e\t%0.4f\t%0.4f\t%0.4f\t%0.4f\n" % (
            #    goid, "\t%s"%self.taxon if self.taxon is not None else "", len(positives), i+1, ss_squeeze.kendalltau_list[i],
            #    ss_squeeze.num_unranked_list[i], ss_squeeze.max_unranked_stretch_list[i], ss_squeeze.max_d_list[i], ss_squeeze.UB_list[i],
            #    ss_squeeze.eval_stats_list[i][0], ss_squeeze.eval_stats_list[i][1], ss_squeeze.eval_stats_list[i][2], ss_squeeze.eval_stats_list[i][3])
            #                    for i in range(ss_squeeze.num_iters)))

        ## if they're different dimensions, then set the others to zeros 
        #if len(scores_arr) < goid_scores.shape[1]:
        #    scores_arr = np.append(scores_arr, [0]*(goid_scores.shape[1] - len(scores_arr)))
        goid_scores[idx] = scores_arr
        # make sure 0s are removed
        goid_scores.eliminate_zeros()

        # also keep track of the time it takes for each of the parameter sets
        alg_name = "%s%s" % (alg, run_obj.params_str)
        #params_results["%s_wall_time"%alg_name] += wall_time
        params_results["%s_process_time"%alg_name] += process_time
        params_results["%s_update_time"%alg_name] += update_time

    run_obj.goid_scores = goid_scores
    run_obj.params_results = params_results
    run_obj.goid_rank_stats = goid_rank_stats
    return


def str_(s):
    return str(s).replace('.','_')
