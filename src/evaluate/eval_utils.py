import os
import numpy as np
from scipy import sparse
# needed for evaluation metrics
try:
    from sklearn import metrics
except ImportError:
    print("WARNING: Unable to import sklearn")
    pass


def evaluate_ground_truth(
        goid_scores, ann_obj, out_file,
        non_pos_as_neg_eval=False, taxon='-',
        alg='', write_prec_rec=False, append=True, **kwargs):
    true_ann_matrix = ann_obj.ann_matrix
    goids, prots = ann_obj.goids, ann_obj.prots

    score_goids2idx = {g: i for i, g in enumerate(goids)}
    print("Computing fmax from ground truth of %d goterms" % (true_ann_matrix.shape[0]))
    goid_stats = {}
    goid_num_pos = {} 
    goid_prec_rec = {}
    for i in range(true_ann_matrix.shape[0]):
        goid = goids[i]
        # make sure the scores are actually available first
        #if goid not in self.goid2idx:
        #    print("WARNING: goid %s not in initial set of %d goids" % (
        #                        goid, len(self.goids)))
        #    continue
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
        prec, recall, fpr, pos_neg_stats = compute_eval_measures(scores, positives, negatives=negatives, track_pos_neg=True)
        if write_prec_rec:
            goid_prec_rec[goid] = (prec, recall, pos_neg_stats)
        fmax = compute_fmax(prec, recall)
        avgp = compute_avgp(prec, recall)
        if len(prec) == 1:
            auprc = 0
            auroc = 0
        else:
            auprc = compute_auprc(prec, recall)
            auroc = compute_auroc([r for r, f in fpr], [f for r, f in fpr])
        goid_stats[goid] = (fmax, avgp, auprc, auroc)
        if kwargs['verbose']:
            print("%s fmax: %0.4f" % (goid, fmax))

    if not write_prec_rec:
        # don't write the header each time
        if not os.path.isfile(out_file) or not append:
            print("Writing results to %s" % (out_file))
            with open(out_file, 'w') as out:
                if taxon in ['-', None]:
                    out.write("#goid\tfmax\tavgp\tauprc\tauroc\t# ann\n")
                else:
                    out.write("#taxon\tgoid\tfmax\tavgp\tauprc\tauroc\t# test ann\n")
        else:
            print("Appending results to %s" % (out_file))
        with open(out_file, 'a') as out:
            out.write(''.join(["%s%s\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%d\n" % (
                "%s\t"%taxon if taxon not in ["-", None] else "",
                g, fmax, avgp, auprc, auroc, goid_num_pos[g]
                ) for g, (fmax, avgp, auprc, auroc) in goid_stats.items()]))

    if write_prec_rec:
        goid = list(goid_prec_rec.keys())[0]
        out_file_pr = out_file.replace('.txt', "prec-rec%s%s.txt" % (
            taxon if taxon not in ['-', None] else '',
            '-%s'%(goid) if len(goid_prec_rec) == 1 else ""))
        print("writing prec/rec to %s" % (out_file_pr))
        with open(out_file_pr, 'w') as out:
            out.write("goid\tprec\trec\tnode\tscore\tidxpos/neg\n")
            for goid, (prec, rec, pos_neg_stats) in goid_prec_rec.items():
                out.write(''.join(["%s\t%0.4f\t%0.4f\t%s\t%0.4f\t%d\t%d\n" % (
                    goid, p, r, prots[n], s, idx, pos_neg) for p,r,(n,s,idx,pos_neg) in zip(prec[1:], rec[1:], pos_neg_stats)]))


def compute_eval_measures(scores, positives, negatives=None, track_pos_neg=False):
    """
    Compute the precision and false-positive rate at each change in recall (true-positive rate)
    *scores*: dictionary containing a score for each node
    *negatives*: if negatives are given, then the FP will only be from the set of negatives given
    *track_pos_neg*: if specified, track the score and rank of the positive and negative nodes,
        and return a tuple of the node ids in order of their score, their score, their idx, and 1/-1 for pos/neg
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
    # TODO I should call numpy argsort to ensure I'm using the full precision when comparing values
    # use np.argsort
    #nodes_sorted_by_scores = sorted(scores, key=scores.get, reverse=True)
    nodes_sorted_by_scores = np.argsort(scores)[::-1]
    #print("computing the rank of positive nodes")
    # this is really slow...
    #pos_ranks = sorted([nodes_sorted_by_scores.index(p)+1 for p in positives])
    #print("%d positives, %d pos_ranks" % (len(positives), len(pos_ranks)))
    #print(pos_ranks)
    #print([scores[s] for s in nodes_sorted_by_scores[:pos_ranks[0]+1]])
    precision = [1]
    recall = [0]
    fpr = []
    pos_neg_stats = []  # tuple containing the node, score and idx
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
            if track_pos_neg:
                pos_neg_stats.append((n, scores[n], i, 1)) 
        elif check_negatives is False or n in negatives:
            FP += 1
            fpr.append((rec, FP / float(len(negatives))))
        #else:
        #    continue

    # TODO how should I handle this case?
    if len(precision) == 0:
        precision.append(0)
        recall.append(1)

    #print(precision[0], recall[0], fpr[0])

    if track_pos_neg:
        return precision, recall, fpr, pos_neg_stats
    else:
        return precision, recall, fpr


def compute_fmax(prec, rec):
    f_measures = []
    for i in range(len(prec)):
        p, r = prec[i], rec[i]
        if p+r == 0:
            harmonic_mean = 0
        else:
            harmonic_mean = (2*p*r)/(p+r)
        f_measures.append(harmonic_mean)
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
