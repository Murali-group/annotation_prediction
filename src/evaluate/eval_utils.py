import os
import numpy as np
from scipy import sparse
from collections import defaultdict
import fcntl
from tqdm import tqdm
import time
# needed for evaluation metrics
try:
    from rpy2.robjects.packages import importr
    from rpy2.robjects import FloatVector
    prroc = importr('PRROC')
    use_sklearn = False
except:
    print("WARNING: unable to import R for AUPRC package. Using sklearn")
    from sklearn import metrics
    use_sklearn = True


def evaluate_ground_truth(
        run_obj, eval_ann_obj, out_file,
        non_pos_as_neg_eval=False, taxon='-',
        early_prec=None, write_prec_rec=False, 
        append=False, **kwargs):
    """
    *early_prec*: A list of recall values for which to get the precision. 
        Each will get its own column in the output file
    *write_prec_rec*: For every term, write a file containing the 
        precision and recall at every positive and negative example
    *term_ic_vec*: information content for each term in a vector
    """
    goid_scores, goids = run_obj.goid_scores, run_obj.goids_to_run 
    eval_ann_matrix, prots = eval_ann_obj.ann_matrix, eval_ann_obj.prots
    score_goid2idx = run_obj.ann_obj.goid2idx
    # only evaluate the terms that are in both the goid_scores matrix and the eval matrix
    goids_to_eval = set(goids) & set(eval_ann_obj.goids)
    if len(goids_to_eval) != len(goids) or len(goids_to_eval) != len(eval_ann_obj.goids):
        print("\nINFO: only %d goids both have scores (/%d goids) and are in the eval matrix (/%d goids)" % (
            len(goids_to_eval), len(goids), len(eval_ann_obj.goids)))

    print("Computing fmax from ground truth for %d goterms" % (len(goids_to_eval)))
    goid_stats = {}
    goid_num_pos = {} 
    goid_prec_rec = {}

    for goid in goids_to_eval:
        score_idx = score_goid2idx[goid]
        eval_idx = eval_ann_obj.goid2idx[goid]
        # get the row corresponding to the current goids annotations 
        goid_ann = eval_ann_matrix[eval_idx,:].toarray().flatten()
        positives = np.where(goid_ann > 0)[0]
        # to get the scores, map the current goid index to the
        # index of the goid in the scores matrix
        scores = goid_scores[score_idx]
        # this is only needed for aptrank since it does not return a sparse matrix
        if sparse.issparse(scores):
            scores = scores.toarray().flatten()
        goid_num_pos[goid] = len(positives)
        if len(positives) == 0:
            if kwargs['verbose']:
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
        prec, recall, fpr, pos_neg_stats = compute_eval_measures(
                scores, positives, negatives=negatives, 
                track_pos=True, track_neg=True)
        # Takes too much RAM to store these values for all terms 
        # so only store them when they will be written to a file
        if write_prec_rec:
            goid_prec_rec[goid] = (prec, recall, pos_neg_stats)
        fmax = compute_fmax(prec, recall)
        avgp = compute_avgp(prec, recall)
        if len(prec) == 1:
            auprc = 0
            auroc = 0
        else:
            if use_sklearn:
                auprc = compute_auprc(prec, recall)
                auroc = compute_auroc([r for r, f in fpr], [f for r, f in fpr])
            else:
                # get the scores of the positive nodes and the scores of the negative nodes
                # and compute the auprc using the R package, which is more accurate than sklearn
                auprc, auroc = compute_auprc_auroc(scores[positives], scores[negatives])
        eprec_vals = []
        if early_prec is not None:
            # compute the early precision at specified values
            eprec_vals = compute_early_prec(
                prec, recall, pos_neg_stats, early_prec, goid_num_pos[goid])
        if kwargs['verbose']:
            print("%s fmax: %0.4f" % (goid, fmax))

        # also compute the smin on a term-by-term basis to see if its comparable to the fmax and auprc
        # make a copy of the matrices with just the single term, and then pass it to the compute_smin function
        idx_vec = np.zeros(goid_scores.shape[0])
        idx_vec[eval_idx] = 1
        diag = sparse.diags(idx_vec, shape=(len(idx_vec), len(idx_vec)))
        # set everything but the row of the curent term to 0
        term_scores = diag.dot(goid_scores).asformat('csr')
        term_eval_mat = diag.dot(eval_ann_matrix).asformat('csr')
        # # make sure to store the scores the same way we would compute the smin over all values
        # term_scores2 = store_pos_neg_scores(term_scores, term_eval_mat, verbose=False)
        ru_vec, mi_vec, smin = compute_smin(term_scores, term_eval_mat, kwargs['term_ic_vec']) 

        goid_stats[goid] = (fmax, avgp, auprc, auroc, smin, eprec_vals)

    # skip writing the output file if there's only one term specified
    if write_prec_rec and len(goid_prec_rec) == 1:
        print("skipping writing %s" % (out_file))
    else:
        out_str = ""
        # Build the table of validation measures, sorted by # ann per term
        for g in sorted(goid_stats, key=goid_num_pos.get, reverse=True):
            fmax, avgp, auprc, auroc, smin, eprec_vals = goid_stats[g]
            # format the values so they'll be ready to be written to the output file
            early_prec_str = '\t'+'\t'.join("%0.4f" % (p) for p in eprec_vals) \
                             if len(eprec_vals) > 0 else ""
            out_str += "%s%s\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%d%s\n" % (
                "%s\t"%taxon if taxon not in ["-", None] else "",
                g, fmax, avgp, auprc, auroc, smin, goid_num_pos[g], early_prec_str)
        # don't re-write the header if this file is being appended to
        if not os.path.isfile(out_file) or not append:
            print("Writing results to %s\n" % (out_file))
            header_line = "#goid\tfmax\tavgp\tauprc\tauroc\tsmin"
            if taxon not in ['-', None]:
                header_line = "#taxon\t%s\t# test ann" % (header_line)
            else:
                header_line += "\t# ann"
            if early_prec is not None:
                header_line += '\t'+'\t'.join(["eprec-rec%s" % (r) for r in early_prec])
            out_str = header_line+"\n" + out_str
        else:
            print("Appending results to %s\n" % (out_file))
        with open(out_file, 'a' if append else 'w') as out:
            # lock it to avoid scripts trying to write at the same time
            fcntl.flock(out, fcntl.LOCK_EX)
            out.write(out_str)
            fcntl.flock(out, fcntl.LOCK_UN)

    if write_prec_rec:
        goid = list(goid_prec_rec.keys())[0]
        out_file_pr = out_file.replace('.txt', "prec-rec%s%s.txt" % (
            taxon if taxon not in ['-', None] else '',
            '-%s'%(goid) if len(goid_prec_rec) == 1 else ""))
        print("writing prec/rec to %s" % (out_file_pr))
        with open(out_file_pr, 'w') as out:
            out.write("#goid\tprec\trec\tnode\tscore\tidx\tpos/neg\n")
            #for goid, (prec, rec, pos_neg_stats) in sorted(goid_prec_rec.items(), key=goid_num_pos.get, reverse=True):
            for goid, (prec, rec, pos_neg_stats) in goid_prec_rec.items():
                out.write(''.join(["%s\t%0.4f\t%0.4f\t%s\t%0.4e\t%d\t%d\n" % (
                    goid, p, r, prots[n], s, idx, pos_neg) for p,r,(n,s,idx,pos_neg,_) in zip(prec, rec, pos_neg_stats)]))


def compute_eval_measures(scores, positives, negatives=None, 
        track_pos=False, track_neg=False):
    """
    Compute the precision and false-positive rate at each change in recall (true-positive rate)
    *scores*: array containing a score for each node
    *positives*: indices of positive nodes
    *negatives*: if negatives are given, then the FP will only be from the set of negatives given
    *track_pos*: if specified, track the score and rank of the positive nodes,
        and return a tuple of the node ids in order of their score, their score, their idx, and 1/-1 for pos/neg
    *track_neg*: also track the score and rank of the negative nodes
    """
    #f1_score = metrics.f1score(positives, 
    #num_unknowns = len(scores) - len(positives) 
    positives = set(positives)
    check_negatives = False
    if negatives is not None:
        check_negatives = True 
        negatives = set(negatives)
    else:
        print("TODO. Treating all non-positives as negatives not yet implemented.")
    # compute the precision and recall at each change in recall
    # use np.argsort
    nodes_sorted_by_scores = np.argsort(scores)[::-1]
    precision = []
    recall = []
    fpr = []
    pos_neg_stats = []  # tuple containing the node, score, idx, pos/neg assign., and the # positives assigned so far
    # TP is the # of correctly predicted positives so far
    TP = 0
    FP = 0
    rec = 0
    for i, n in enumerate(nodes_sorted_by_scores):
        # TODO this could be slow if there are many positives
        if n in positives:
            TP += 1
            # precisions is the # of true positives / # true positives + # of false positives (or the total # of predictions)
            precision.append(TP / float(TP + FP))
            # recall is the # of recovered positives TP / TP + FN (total # of positives)
            rec = TP / float(len(positives))
            recall.append(rec)
            # fpr is the FP / FP + TN
            fpr.append((rec, FP / float(len(negatives))))
            if track_pos:
                pos_neg_stats.append((n, scores[n], i, 1, TP+FP)) 
        elif check_negatives is False or n in negatives:
            FP += 1
            # store the prec and rec even though recall doesn't change since the AUPRC will be affected
            precision.append(TP / float(TP + FP))
            recall.append(TP / float(len(positives)))
            fpr.append((rec, FP / float(len(negatives))))
            if track_neg:
                pos_neg_stats.append((n, scores[n], i, -1, TP+FP)) 
        #else:
        #    continue

    # TODO shouldn't happen
    if len(precision) == 0:
        precision.append(0)
        recall.append(1)

    #print(precision[0], recall[0], fpr[0])

    if track_pos or track_neg:
        return precision, recall, fpr, pos_neg_stats
    else:
        return precision, recall, fpr


def compute_fmax(prec, rec, fmax_idx=False):
    """
    *fmax_idx*: also return the index at which the harmonic mean is the greatest
    """
    f_measures = []
    for i in range(len(prec)):
        p, r = prec[i], rec[i]
        if p+r == 0:
            harmonic_mean = 0
        else:
            # see https://en.wikipedia.org/wiki/Harmonic_mean#Two_numbers
            harmonic_mean = (2*p*r)/(p+r)
        f_measures.append(harmonic_mean)
    if fmax_idx:
        idx = np.argmax(np.asarray(f_measures))
        return max(f_measures), idx
    else:
        return max(f_measures)


def compute_avgp(prec, rec):
    # average precision score
    # see http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score
    avgp = 0
    prev_r = 0 
    for p,r in zip(prec, rec):
        recall_change = r - prev_r
        avgp += (recall_change*p)
        prev_r = r
    #avgp = avgp / float(len(alg_prec_rec))
    return avgp


def compute_auprc(prec, rec):
    auprc = metrics.auc(rec, prec)
    return auprc


def compute_auroc(tpr, fpr):
    auroc = metrics.auc(fpr, tpr)
    return auroc


def compute_auprc_auroc(pos_scores, neg_scores):
    prCurve = prroc.pr_curve(
        scores_class0=FloatVector(pos_scores),
        scores_class1=FloatVector(neg_scores))
    auprc = prCurve[1]
    rocCurve = prroc.roc_curve(
        scores_class0=FloatVector(pos_scores),
        scores_class1=FloatVector(neg_scores))
    auroc = rocCurve[1]
    return float(np.asarray(auprc)[0]), float(np.asarray(auroc)[0])


def compute_early_prec(prec, rec, pos_neg_stats, recall_vals, num_pos):
    early_prec_values = []
    for curr_recall in recall_vals:
        # if a k recall is specified, get the precision at the recall which is k * # ann in the left-out species
        if 'k' in curr_recall:
            k_val = float(curr_recall.replace('k',''))
            num_nodes_to_get = k_val * num_pos 
            # the pos_neg_stats tracks the prec for every node.
            # So find the precision value for first node with an index >= (k * # ann)
            for idx, (_, _, _, _, curr_num_pos) in enumerate(pos_neg_stats): 
                if curr_num_pos >= num_nodes_to_get:
                    break
            # get the precision at the corresponding index
            p = prec[idx]
            early_prec_values.append(p)
        else:
            curr_recall = float(curr_recall)
            # find the first precision value where the recall is >= the specified recall
            for p, r in zip(prec, rec):
                if r >= curr_recall:
                    early_prec_values.append(p)
                    break
    return early_prec_values


def compute_smin(all_pos_neg_scores, eval_mat, term_ic_vec, out_file=None, verbose=False, **kwargs):
    """
    *all_pos_neg_scores*: term x gene matrix with scores for the pos and neg examples to evaluate
    *eval_mat*: term x gene matrix with 1s and -1s for pos and neg examples, respetively
    *term_ic_vec*: information content for each term in a vector
    """
    if len(all_pos_neg_scores.data) == 0:
        print("WARNING: 0 scores with which to compute the smin. Returning np.nan")
        return [], [], np.nan

    # get the annotation matrix for only the goids_to_eval
    pos_mat = (eval_mat > 0).astype(int)
    neg_mat = (eval_mat < 0).astype(int)
    # number of prots is the number of proteins with at least one annotation
    num_prots = len(pos_mat.sum(axis=0).nonzero()[1])
    # At every score cutoff tau compute the remaining uncertainty (ru) and misinformation (mi)
    ru_list = []
    mi_list = []
    # use the score cutoffs rounded to 3 decimal places so that we aren't checking a huge number of score cutoffs (i.e., 1 per node)
    score_cutoffs = sorted(set(
        float("%0.3f"%x) if abs(x) > 0.001 else float("%0.0e"%x) \
        for x in all_pos_neg_scores.data), reverse=True)
    weighted_pos_mat = pos_mat.multiply(term_ic_vec)
    weighted_neg_mat = neg_mat.multiply(term_ic_vec)
    if verbose:
        print("\tpos_mat shape: %s, term_ic_vec shape: %s" % (str(pos_mat.shape), str(term_ic_vec.shape)))
        print("\tnum_pos: %s, num_neg: %s" % (len(pos_mat.data), len(neg_mat.data)))
        print("\tnum_pos*ic: %s, num_neg*ic: %s" % (len(weighted_pos_mat.data), len(weighted_neg_mat.data)))
        print("Computing Smin: ru and mi for %d unique score cutoffs" % (len(score_cutoffs)))
    score_cutoff_num_nonzero = {}
    for tau in tqdm(score_cutoffs, disable=not verbose):
        ru, num_nonzero = compute_remaining_uncertainty(
            all_pos_neg_scores, weighted_pos_mat, num_prots, tau)
        mi, num_nonzero = compute_misinformation(
            all_pos_neg_scores, weighted_neg_mat, num_prots, tau)
        ru_list.append(ru)
        mi_list.append(mi)
        score_cutoff_num_nonzero[tau] = num_nonzero

    ru_vec = np.asarray(ru_list)
    mi_vec = np.asarray(mi_list)
    s_vec = (ru_vec**2 + mi_vec**2) ** (0.5)
    smin_idx = np.argmin(s_vec)
    smin = s_vec[smin_idx]
    smin_tau = score_cutoffs[smin_idx]
    # also compute the fraction of the matrix that have a value at this score cutoff
    frac_with_values = score_cutoff_num_nonzero[smin_tau] / float(len(all_pos_neg_scores.data))

    # 2020-08-11 UPDATE: Also compute prec/rec using all values
    prec_list = []
    rec_list = []
    for tau in tqdm(score_cutoffs, disable=not verbose):
        prec, rec = compute_prec_rec_mat(all_pos_neg_scores, pos_mat, neg_mat, tau)
        prec_list.append(prec) 
        rec_list.append(rec) 

    if verbose:
        print("smin: %0.2f, idx: %s, score_cutoff: %0.6f, frac_with_values: %0.6f (%s/%s), prec: %0.4f, rec: %0.4f" % (
            smin, smin_idx, smin_tau, frac_with_values, score_cutoff_num_nonzero[smin_tau], len(all_pos_neg_scores.data),
            prec_list[smin_idx], rec_list[smin_idx]))

    if out_file is not None:
        print("writing %s" % (out_file))
        with open(out_file, 'w') as out:
            out.write("#score_cutoff\ts\tru\tmi\tprec\trec\n")
            for i, tau in enumerate(score_cutoffs):
                out.write("%0.2e\t%0.4f\t%0.3e\t%0.3e\t%0.3f\t%0.3f\n" % (
                    tau, s_vec[i], ru_vec[i], mi_vec[i], prec_list[i], rec_list[i]))
    return ru_vec, mi_vec, smin


def compute_remaining_uncertainty(goid_scores, weighted_pos_mat, num_prots, tau):
    """
    Will compute the fraction of positive examples that are "missed" (i.e., false negative)
        at the current score cutoff *tau*, weighted by the information content for each term

    *goid_scores*: matrix of term x prot prediction scores. 
    *pos_mat*: matrix of term x prot positive examples
    *tau*: score cutoff
    """
    # Building the neg_pred matrix like this can be expensive since all 0 values would be changed to 1 
    #neg_pred = (goid_scores < tau).astype(int)
    S = goid_scores
    # To get the matrix of scores < tau, we're going to instead remove the
    # parts of the matrix that are > tau.
    # This is an array of the nonzero data indices that pass this filter
    nonzero_mask = np.array(S[S.nonzero()] > tau)[0]
    # get the row and col values of those indices
    rows = S.nonzero()[0][nonzero_mask]
    cols = S.nonzero()[1][nonzero_mask]
    # change all prediction values to 1, then
    # replace the values > tau with 0 to get the < tau matrix
    neg_pred = S.astype(bool)
    # In addition, we must take into account the values < tau that are zero, since those are also false negatives
    # which we can do by adding all positives. The pos preds which are not negatives will be replaced with 0 below.
    if tau >= 0:
        neg_pred += weighted_pos_mat.astype(bool)
    neg_pred = neg_pred.astype(int)
    neg_pred[rows, cols] = 0
    neg_pred.eliminate_zeros()

    # Here is the final ru computation:
    ru = (1/float(num_prots)) * neg_pred.multiply(weighted_pos_mat).sum()
    # also return the number of nonzero entries in the matrix
    num_nonzero = len(neg_pred.data)

    # this could be faster and simpler:
    # # pos_pred = goid_scores.copy()
    # orig_scores = goid_scores.data
    # goid_scores.data = (goid_scores.data < tau).astype(int)
    # ru = (1/float(num_prots)) * goid_scores.multiply(weighted_pos_mat).sum()
    # num_nonzero = np.count_nonzero(goid_scores.data)
    # print("new ru: %0.2f. %2e time. num_nonzero: %s, tau: %s. " % (ru, time.process_time() - start_time2, num_nonzero, tau))
    # goid_scores.data = orig_scores 

    return ru, num_nonzero


def compute_misinformation(goid_scores, weighted_neg_mat, num_prots, tau):
    """
    Will compute the fraction of negative examples that are found (i.e., false positives)
        at the current score cutoff *tau*, weighted by the information content for each term
    """
    # Building the pos_pred matrix like this can be expensive since all 0 values would be changed to 1 if tau < 0
    #pos_pred = (goid_scores > tau).astype(int)
    S = goid_scores
    # To get the matrix of scores > tau, we're going to instead remove the
    # parts of the matrix that are < tau.
    # This is an array of the nonzero data indices that pass this filter
    nonzero_mask = np.array(S[S.nonzero()] < tau)[0]
    # get the row and col values of those indices
    rows = S.nonzero()[0][nonzero_mask]
    cols = S.nonzero()[1][nonzero_mask]
    # change all prediction values to 1, then
    # replace the values < tau with 0 to get the > tau matrix
    pos_pred = S.astype(bool)
    # In addition, we must take into account the values > tau that are zero, since those are also false positives
    # which we can do by adding all negatives. The pos preds which are not negatives will be replaced with 0 below.
    if tau <= 0:
        pos_pred += weighted_neg_mat.astype(bool)
    pos_pred = pos_pred.astype(int)
    pos_pred[rows, cols] = 0
    pos_pred.eliminate_zeros()
    mi = (1/float(num_prots)) * pos_pred.multiply(weighted_neg_mat).sum()
    # also return the number of nonzero entries in the matrix
    num_nonzero = len(pos_pred.data)

    # #pos_pred = goid_scores.copy()
    # orig_scores = goid_scores.data
    # goid_scores.data = (goid_scores.data > tau).astype(int)
    # mi = (1/float(num_prots)) * goid_scores.multiply(weighted_neg_mat).sum()
    # print("new mi: %0.2f. %0.2e time, nnz: %s, tau: %s" % (mi, time.process_time() - start_time2, num_nonzero, tau))
    # num_nonzero = np.count_nonzero(goid_scores.data)
    # goid_scores.data = orig_scores 

    return mi, num_nonzero


def compute_information_content(ann_obj):
    """
    The information content is -log(p(term)), where p(term) is # genes annotated to term 
        divided by # genes with an annotation 
    """
    # number of prots is the number of proteins with at least one annotation
    pos_mat = (ann_obj.ann_matrix > 0).astype(int)
    num_prots = len(pos_mat.sum(axis=0).nonzero()[1])
    # sum the num ann per term, divided by # prots with an annotation
    ic_vec = pos_mat.sum(axis=1) / num_prots
    return -np.log(ic_vec)


def compute_prec_rec_mat(goid_scores, pos_mat, neg_mat, tau):
    """
    Compute the precision and recall for a given score cutoff tau
    """
    # first get all the predicted positives which are nodes with a score > tau
    # This can be expensive since all 0 values would be changed to 1 
    #pos_pred = (goid_scores > tau).astype(int)
    S = goid_scores
    # To get the matrix of scores > tau, we're going to instead remove the
    # parts of the matrix that are < tau.
    # This is an array of the nonzero data indices that pass this filter
    nonzero_mask = np.array(S[S.nonzero()] < tau)[0]
    # get the row and col values of those indices
    rows = S.nonzero()[0][nonzero_mask]
    cols = S.nonzero()[1][nonzero_mask]
    # change all prediction values to 1, then
    # replace the values < tau with 0 to get the > tau matrix
    pos_pred = S.astype(bool)
    # In addition, we must take into account the values > tau that are zero, since those are also false positives
    # which we can do by adding all negatives. The pos preds which are not negatives will be replaced with 0 below.
    if tau <= 0:
        pos_pred += pos_mat.astype(bool)
        pos_pred += neg_mat.astype(bool)
    pos_pred = pos_pred.astype(int)
    pos_pred[rows, cols] = 0
    pos_pred.eliminate_zeros()

    # Figure out which predictions are true and false positives by multiplying with the pos_mat and neg_mat
    TP = pos_pred.multiply(pos_mat).sum()
    FP = pos_pred.multiply(neg_mat).sum()
    # precision: TP / TP + FP
    precision = TP / float(TP + FP)
    # recall: TP / P
    recall = TP / float(pos_mat.sum())
    return precision, recall


def store_pos_neg_scores(goid_scores, test_ann_mat, verbose=True):
    # just store the scores of the positive and negative examples
    test_idx_mat = test_ann_mat.astype(bool).astype(int)
    pos_neg_scores = goid_scores.multiply(test_idx_mat)
    if verbose:
        print("\tmax score: %s, min score: %s, num scores: %s" % (pos_neg_scores.max(), pos_neg_scores.min(), len(pos_neg_scores.data)))
    return pos_neg_scores


def store_terms_eval_mat(run_obj, ann_obj, test_ann_mat, specific_terms=None):
    # just store the scores of the positive and negative examples
    if specific_terms is not None:
        print("\tstoring eval_mat for %d terms" % (len(specific_terms)))
        # also store a version of the test ann mat that has only the most specific terms
        terms_vec = np.zeros(len(ann_obj.goids))
        terms_vec[[ann_obj.goid2idx[t] for t in specific_terms]] = 1
        terms_I = sparse.diags(terms_vec)
        # multply by a diagonal matrix with 1s at the specific terms to limit the test ann mat to those terms
        test_ann_mat_sp_terms = terms_I.dot(test_ann_mat)
        run_obj.eval_mat += test_ann_mat_sp_terms
    else:
        run_obj.eval_mat += test_ann_mat
