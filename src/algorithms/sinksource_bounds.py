
# function to efficiently and accurately compute the top-k sinksource scores
# Algorithm and proofs adapted from:
# Zhang et al. Fast Inbound Top-K Query for Random Walk with Restart, KDD, 2015

#import SinkSource
import sys
import time
import operator
from collections import defaultdict
# expects python 3 and networkx 2
#import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
import src.algorithms.alg_utils as alg_utils
import src.evaluate.eval_utils as eval_utils
import random
from scipy.stats import kendalltau  #, spearmanr, weightedtau


class SinkSourceBounds:

    def __init__(self, P, positives, negatives=None, a=0.8, 
                 rank_all=False, rank_pos_neg=None,
                 verbose=False, ranks_to_compare=None,
                 max_iters=1000):
        """
        *P*: Row-normalized sparse-matrix representation of the graph
        *rank_all*: require that the ranks of all nodes be fixed using their UB and LB
        *rank_pos_neg*: tuple of sets of positive and negative nodes. 
            We only require that the LB and UB of positives not overlap with any negative nodes.
            The k parameter will be ignored.
        *ranks_to_compare*: A list of nodes where the index of the node in the list is the rank of that node.
            For example, if node 20 was ranked first and node 50 was ranked second, the list would have [20, 50]
            Used to compare with the current ranking after each iteration.
        *max_iters*: Maximum number of iterations to run power iteration
        *returns*: The set of top-k nodes, and current scores for all nodes
        """
        self.P = P
        self.positives = positives
        self.negatives = negatives
        self.a = a
        self.rank_all = rank_all
        self.rank_pos_neg = rank_pos_neg
        self.ranks_to_compare = ranks_to_compare
        #self.scores_to_compare = scores_to_compare
        self.max_iters = max_iters
        self.verbose = verbose

    def runSinkSourceBounds(self):
        self.num_nodes = self.P.shape[0]
        # f: initial vector f of amount of score received from positive nodes
        self.P, self.f, self.node2idx, self.idx2node = alg_utils.setup_fixed_scores(
            self.P, self.positives, self.negatives, a=self.a, 
            remove_nonreachable=True, verbose=self.verbose)
        if len(self.f) == 0:
            print("WARNING: no unknown nodes were reachable from a positive (P matrix and f vector empty after removing nonreachable nodes).")
            print("Setting all scores to 0")
            return [], defaultdict(int)
        # if rank_nodes is specified, map those node ids to the current indices
        #if self.rank_nodes is not None:
        #    self.rank_nodes = set(self.node2idx[n] for n in self.rank_nodes if n in self.node2idx)
        if self.rank_pos_neg is not None:
            unr_pos_nodes, unr_neg_nodes = self.rank_pos_neg
            # some of them could've been unreachable, so remove those and fix the mapping
            self.unr_pos_nodes = set(self.node2idx[n] \
                    for n in unr_pos_nodes if n in self.node2idx)
            self.unr_neg_nodes = set(self.node2idx[n] \
                    for n in unr_neg_nodes if n in self.node2idx)

        if self.verbose:
            if self.negatives is not None:
                print("\t%d positives, %d negatives, %d unknowns, a=%s"
                        % (len(self.positives), len(self.negatives), self.P.shape[0], str(self.a)))
            else:
                print("\t%d positives, %d unknowns, a=%s"
                        % (len(self.positives), self.P.shape[0], str(self.a)))

        all_LBs = self._SinkSourceBounds()

        # set the default score to 0 for the rank_nodes as some of them may be unreachable from positives
        scores_arr = np.zeros(self.num_nodes)
        indices = [self.idx2node[n] for n in range(len(all_LBs))]
        scores_arr[indices] = all_LBs
        #self.scores_to_compare = all_LBs

        if self.verbose:
            print("SinkSourceBounds finished after %d iterations (%0.3f total sec, %0.3f sec to update)"
                  % (self.num_iters, self.total_time, self.total_update_time))

        return scores_arr

    def _SinkSourceBounds(self):
        """
        *returns*: The current scores for all nodes
        """
        # TODO check to make sure t > 0, s > 0, k > 0, 0 < a < 1 and such
        unranked_nodes, LBs, prev_LBs, UBs = self.initialize_sets()
        # use this variable to indicate if we are ranking a subsample of the nodes first
        initial_unranked_nodes = None

        # the infinity norm is simply the maximum value in the vector
        max_f = self.f.max()
        if self.verbose:
            print("\tmax_f: %0.4f" % (max_f))

        self.num_iters = 0
        # total number of computations performed during the update function
        self.total_comp = 0
        # amount of time taken during the update function
        self.total_update_time = 0
        # TODO use cpu time, not just system time. time.process_time() should work
        start_time = time.process_time()
        # also keep track of the max score change after each iteration
        self.max_d_list = []
        # keep track of the UB after each iteration
        self.UB_list = []
        # keep track of how fast the nodes are ranked
        self.num_unranked_list = []
        self.kendalltau_list = []
        # keep track of fmax, avgp, auprc, auroc at each iteration
        self.eval_stats_list = []
        #self.spearmanr_list = []
        # keep track of the biggest # of nodes with continuously overlapping upper or lower bounds
        max_unranked_stretch = 0 
        self.max_unranked_stretch_list = [] 
        # keep track of the maximum difference of the current scores to the final score
        self.max_d_compare_ranks_list = []
        ## also keep track of how many nodes have a fixed ranking from the top of the list
        #num_ranked_from_top = 0 
        #self.num_ranked_from_top_list = [] 

        # iterate until the top-k are attained
        # R is not updated if either rank_all or rank_pos_neg is True
        while len(unranked_nodes) > 0:
            # also stop for any of these criteria
            if (self.rank_all is True or self.rank_pos_neg is not None) \
                    and len(unranked_nodes) == 0:
                # if the subset of nodes are ranked, but all nodes are not, then keep going
                if initial_unranked_nodes is not None:
                    unranked_nodes = initial_unranked_nodes
                    initial_unranked_nodes = None 
                else:
                    break

            if self.verbose:
                print("\tnum_iters: %d, |R|: %d, |unranked_nodes|: %d, max_unranked_stretch: %d" % (
                    self.num_iters, len(R), len(unranked_nodes), max_unranked_stretch))
            if self.num_iters > self.max_iters:
                if self.verbose:
                    print("\thit the max # iters: %d. Stopping." % (self.max_iters))
                break
            self.num_iters += 1
            # keep track of how long it takes to update the bounds at each iteration
            curr_time = time.process_time()

            # power iteration
            LBs = self.a*csr_matrix.dot(self.P, prev_LBs) + self.f

            update_time = time.process_time() - curr_time
            self.total_update_time += update_time
            max_d = (LBs - prev_LBs).max()
            prev_LBs = LBs.copy()
            UB = self.computeUBs(max_f, self.a, self.num_iters)
            #if self.scores_to_compare is not None:
            #    max_d_compare_ranks = (self.scores_to_compare - LBs).max()

            if self.verbose:
                #if self.scores_to_compare is not None:
                #    print("\t\t%0.4f sec to update scores. max_d: %0.2e, UB: %0.2e, max_d_compare_ranks: %0.2e" % (update_time, max_d, UB, max_d_compare_ranks))
                #else:
                print("\t\t%0.4f sec to update scores. max_d: %0.2e, UB: %0.2e" % (update_time, max_d, UB))

            # check to see if the set of nodes to rank have a fixed ranking
            UBs = LBs + UB
            self.unr_pos_nodes, self.unr_neg_nodes = self.check_fixed_rankings(
                    LBs, UBs, unr_pos_nodes=self.unr_pos_nodes, unr_neg_nodes=self.unr_neg_nodes) 
            # the sets are disjoint, so just combine them
            unranked_nodes = list(self.unr_pos_nodes) + list(self.unr_neg_nodes)

            self.max_unranked_stretch_list.append(max_unranked_stretch)
            self.max_d_list.append(max_d) 
            #if self.scores_to_compare is not None:
            #    self.max_d_compare_ranks_list.append(max_d_compare_ranks) 
            self.UB_list.append(UB) 
            self.num_unranked_list.append(len(unranked_nodes))
            if self.ranks_to_compare is not None:
                # also append a measure of the similarity between the current ranking and the rank to compare with
                # get the current node ranks
                scores = {self.idx2node[n]:LBs[n] for n in range(len(LBs))}
                nodes_with_ranks = set(self.ranks_to_compare)
                nodes_to_rank = set(scores.keys()) & nodes_with_ranks
                # check to make sure we have a rank for all of the nodes
                if len(nodes_to_rank) != len(nodes_with_ranks):
                    print("ERROR: some nodes do not have a ranking")
                    print("\t%d nodes_to_rank, %d ranks_to_compare" % (len(nodes_to_rank), len(nodes_with_ranks)))
                    sys.exit()
                # builds a dictionary of the node as the key and the current rank as the value
                # e.g., {50: 0, 20: 1, ...}
                curr_ranks = {n: i for i, n in enumerate(sorted(nodes_to_rank, key=scores.get, reverse=True))}
                # if I sort using ranks_to_compare directly, then for the first couple iterations when many nodes are tied at 0, 
                # will be left in the order they were in (i.e., matching the correct/fixed ordering)
                #curr_ranks = {n: i for i, n in enumerate(sorted(self.ranks_to_compare, key=scores.get, reverse=True))}
                # get the current rank of the nodes in the order of the ranks_to_compare 
                # for example, if the ranks_to_compare has 20 at 0 and 50 at 1, and the current rank is 50: 0, 20: 1,
                # then compare_ranks will be [1, 0]
                compare_ranks = [curr_ranks[n] for n in self.ranks_to_compare]
                # compare the two rankings
                # for example: curr rank: [1,0], orig rank: [0,1] 
                self.kendalltau_list.append(kendalltau(compare_ranks, range(len(self.ranks_to_compare)))[0])
                # this is no longer needed
                #self.spearmanr_list.append(spearmanr(compare_ranks, range(len(self.ranks_to_compare)))[0])
                if self.rank_pos_neg is not None:
                    # need to include all nodes because otherwise the recall will be higher
                    # from the unreachable positives that were removed
                    scores_arr = np.zeros(self.num_nodes)
                    indices = [self.idx2node[n] for n in range(len(LBs))]
                    scores_arr[indices] = LBs
                    prec, recall, fpr = eval_utils.compute_eval_measures(scores_arr, self.rank_pos_neg[0], self.rank_pos_neg[1])
                    fmax = eval_utils.compute_fmax(prec, recall)
                    avgp = eval_utils.compute_avgp(prec, recall)
                    auprc = eval_utils.compute_auprc(prec, recall)
                    #auroc = alg_utils.compute_auroc([r for r, f in fpr], [f for r, f in fpr])
                    self.eval_stats_list.append((fmax, avgp, auprc))

        self.total_time = time.process_time() - start_time
        self.total_comp += len(self.P.data)*self.num_iters
        return LBs

    def computeUBs(self, max_f, a, i):
        if a == 1:
            return 1
        else:
            additional_score = (a**(i) * max_f) / (1-a)

        return additional_score

    def initialize_sets(self):
        unranked_nodes = set(np.arange(self.P.shape[0]).astype(int))

        # set the initial lower bound (LB) of each node to f or 0
        LBs = self.f.copy()
        # dictionary of LBs at the previous iteration
        prev_LBs = np.zeros(len(LBs))
        # dictionary of Upper Bonds for each node
        UBs = np.ones(self.P.shape[0])

        return unranked_nodes, LBs, prev_LBs, UBs

    def get_stats(self):
        """
        Returns the total time, time to update scores, # of iterations, # of computations (estimated), 
        the max_d at each iteration, and the initial size of the graph.
        """
        return self.total_time, self.total_update_time, self.num_iters, self.total_comp


    def check_fixed_rankings(self, LBs, UBs, unranked_nodes=None, unr_pos_nodes=None, unr_neg_nodes=None):
        """
        *nodes_to_rank*: a set of nodes for which to check which nodes have an overlapping UB/LB.
            In other words, get the nodes that are causing the given set of nodes to not have their ranks fixed
        UPDATE: 
        *unr_pos_nodes*: set of positive nodes that are not fixed. 
            only need to check for overlap with negative nodes 
        *unr_neg_nodes*: set of negative nodes that are not fixed. 
            only need to check for overlap with positive nodes
        """
        # find all of the nodes in the top-k whose rankings are fixed
        # n comparisons
        all_scores = []
        # also keep track of the # of nodes in a row that have overlapping upper or lower bounds
        # for now just keep track of the biggest
        max_unranked_stretch = 0
        i = 0
        if unranked_nodes is not None:
            for n in unranked_nodes:
                all_scores.append((n, LBs[n]))
            all_scores_sorted = sorted(all_scores, key=lambda x: (x[1]), reverse=True)
            # the fixed nodes are the nodes that are not in the 
            # "still not fixed" set
            still_not_fixed_nodes = set()
            # for every node, check if the next node's LB+UB > the curr node's LB.
            # If so, the node is not yet fixed
            while i+1 < len(all_scores_sorted):
                curr_LB = all_scores_sorted[i][1]
                curr_i = i
                while i+1 < len(all_scores_sorted) and \
                      curr_LB < UBs[all_scores_sorted[i+1][0]]:
                    still_not_fixed_nodes.add(all_scores_sorted[i+1][0])
                    #print("i+1: %d not fixed" % (i+1))
                    i += 1
                if curr_i != i:
                    #print("i: %d not fixed" % (curr_i))
                    still_not_fixed_nodes.add(all_scores_sorted[curr_i][0])
                    if i - curr_i > max_unranked_stretch:
                        max_unranked_stretch = i - curr_i
                if curr_i == i:
                    i += 1
                #    fixed_nodes.add(all_scores_sorted[i][0])
            return still_not_fixed_nodes, max_unranked_stretch
        elif unr_pos_nodes is not None and unr_neg_nodes is not None:
            # if there aren't any nodes to check their ranking, then simply return
            if len(unr_pos_nodes) == 0 or len(unr_neg_nodes) == 0:
                return set(), set()
            for n in unr_pos_nodes:
                all_scores.append((n, LBs[n]))
            for n in unr_neg_nodes:
                all_scores.append((n, LBs[n]))
            all_scores_sorted = sorted(all_scores, key=lambda x: (x[1]), reverse=True)
            fixed_nodes = unr_pos_nodes | unr_neg_nodes
            # for every node, check if the next node's LB+UB > the curr node's LB.
            # and if one of the overlapping nodes is opposite 
            # If so, the node is not yet fixed
            curr_node = all_scores_sorted[0][0]
            opp_set_pos = False if curr_node in unr_pos_nodes else True
            opp_set = unr_pos_nodes if opp_set_pos else unr_neg_nodes
            while i+1 < len(all_scores_sorted):
                curr_node = all_scores_sorted[i][0]
                curr_LB = all_scores_sorted[i][1]
                # if this is a positive, just check the negatives
                # and vice versa
                curr_i = i
                opp_overlap = False 
                last_opp_node = None
                while i+1 < len(all_scores_sorted) and \
                      curr_LB < UBs[all_scores_sorted[i+1][0]]:
                    next_node = all_scores_sorted[i+1][0]
                    if next_node in opp_set:
                        opp_overlap = True
                        last_opp_node = i+1 
                    i += 1
                if opp_overlap is True:
                    # if there was an overlap with an opposite node,
                    # all of these are not fixed
                    for j in range(curr_i, i+1):
                        j_node = all_scores_sorted[j][0]
                        fixed_nodes.discard(j_node)
                    i = last_opp_node
                    # flip the opposite set
                    opp_set_pos = False if opp_set_pos else True
                    opp_set = unr_pos_nodes if opp_set_pos else unr_neg_nodes
                else:
                    # only need to increment here if there was no overlap
                    i += 1
                    
            unr_pos_nodes -= fixed_nodes 
            unr_neg_nodes -= fixed_nodes 
            return unr_pos_nodes, unr_neg_nodes
        else:
            print("Error: need to pass either the 'unranked_nodes' set or both 'pos_nodes' and 'neg_nodes'")
            return


