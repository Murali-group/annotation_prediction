
# script to see how well we can retrieve annotations of another species
# using the annotations of the other 18

from optparse import OptionParser,OptionGroup
from collections import defaultdict
import os
import sys
from tqdm import tqdm
import os
import src.setup_sparse_networks as setup
import src.algorithms.alg_utils as alg_utils
import src.utils.file_utils as utils
import src.evaluate.eval_utils as eval_utils
from tqdm import tqdm, trange
import numpy as np
from scipy import sparse


def eval_loso(
        alg_runners, ann_obj, taxon_file, eval_ann_obj=None, 
        taxons=None, only_taxon_file=None, **kwargs):
    """
    *alg_runners*: 
        Each is expected to have the same ann_obj, prots, and goids
    *eval_ann_obj*: annotation object that will only be used for evaluation
    *taxon_file*: 
    *only_taxon_file*: 
    """

    species_to_uniprot_idx = get_uniprot_species(taxon_file, ann_obj)
    selected_species, taxons = get_selected_species(species_to_uniprot_idx, only_taxon_file, taxons)

    # change the taxons to be all. Then nothing will be left out
    if kwargs['keep_ann']:
        if eval_ann_obj is None:
            print("\nERROR: an evaluation matrix (i.e., pos_neg_file_eval) must be given with the 'keep_ann' option")
            sys.exit(1)
        #print("\nRunning algs with all annotations in the --pos-neg-file and evaluating based on the annotations in the --pos-neg-file-eval.")
        #print("\tindividual species will be evaluated after")
        print("\nEvaluating %d individual species, %d goterms, without leaving out any annotations for predictions." % (len(taxons), len(ann_obj.goids)))
        #taxons = ['all']
    else:
        print("Running the LOSO evaluation for %d species, %d goterms" % (len(taxons), len(ann_obj.goids)))
        # -------------------------------
        alg_taxon_terms_to_skip = get_already_run_terms(alg_runners, **kwargs) 

    # now perform LOSO validation
    params_results = defaultdict(int)
    # for each species, leave out all of its annotations and then attempt to recover them
    for t in tqdm(sorted(taxons)):
        tqdm.write("\n" + "-"*30)
        tqdm.write("Taxon: %s - %s" % (
            t, selected_species[t]))

        # leave out the annotations for this taxon ID
        train_ann_mat, test_ann_mat, sp_goterms = leave_out_taxon(
            t, ann_obj, species_to_uniprot_idx,
            eval_ann_obj=eval_ann_obj, 
            #terms_to_skip=alg_taxon_terms_to_skip[alg][t] if t in alg_taxon_terms_to_skip[alg] else None,
            **kwargs)
        tqdm.write("\t%d/%d goterms with >= %d annotations" % (len(sp_goterms), len(ann_obj.goids), kwargs['num_test_cutoff']))
        if len(sp_goterms) == 0:
            print("\tskipping")
            continue

        taxon_prots = species_to_uniprot_idx[t]
        for run_obj in alg_runners:
            alg = run_obj.name
            print("Alg: %s" % (alg))

            if t in alg_taxon_terms_to_skip[alg]:  
                terms_to_skip = alg_taxon_terms_to_skip[alg][t]
                sp_goterms = [t for t in sp_goterms if t not in terms_to_skip]
                print("\t%d to run that aren't in the output file yet." % (len(sp_goterms)))
                if len(sp_goterms) == 0:
                    continue
            # limit the run_obj to run on the terms for which there are annotations
            run_obj.goids_to_run = sp_goterms
            # limit the scores stored to the current taxon's prots
            run_obj.target_prots = list(taxon_prots)

            run_and_eval_algs(
                run_obj, ann_obj, 
                train_ann_mat, test_ann_mat,
                taxons=taxons, taxon=t, **kwargs)

    # I'm not resetting the params_results in run_obj,
    # so those params_results were already appended to.
    # just built this dict to make sure its not empty
    alg_params_results = defaultdict(int)
    for run_obj in alg_runners:
        for key in run_obj.params_results:
            alg_params_results[key] += run_obj.params_results[key]
    if len(params_results) != 0 or len(alg_params_results) != 0:
        # write the running times to a file
        write_stats_file(alg_runners, params_results)
        alg_params_results.update(params_results)
        print("Final running times: " + ' '.join([
            "%s: %0.4f" % (key, val) for key, val in sorted(alg_params_results.items()) if 'time' in key]))
    print("")
    return params_results


def get_uniprot_species(taxon_file, ann_obj):
    print("Getting species of each prot from %s" % (taxon_file))
    # for each of the 19 species, leave out their annotations 
    # and see how well we can retrieve them 
    uniprot_to_species = utils.readDict(taxon_file, 1,2)
    # also build the reverse, but with the node idx instead of the UniProt ID
    node2idx = {n: i for i, n in enumerate(ann_obj.prots)}
    global species_to_uniprot_idx
    species_to_uniprot_idx = defaultdict(set)
    for p in uniprot_to_species:
        species_to_uniprot_idx[uniprot_to_species[p]].add(node2idx.get(p))
    for t in species_to_uniprot_idx:
        species_to_uniprot_idx[t].discard(None) 
    return species_to_uniprot_idx


def get_selected_species(species_to_uniprot_idx, only_taxon_file=None, taxons=None):
    selected_species = {t: '-' for t in species_to_uniprot_idx}
    if only_taxon_file is not None:
        selected_species = utils.readDict(only_taxon_file, 1, 2)
    # if not taxon IDs were specified, then use the either the only_taxon_file, or all of the taxon  
    if taxons is None:
        taxons = selected_species.keys()
    else:
        found_taxons = []
        for t in taxons:
            if t not in species_to_uniprot_idx:
                print("WARNING: taxon '%s' not found. skipping" % (t))
            else:
                found_taxons.append(t)
        taxons = found_taxons
    if len(taxons) == 0:
        print("No taxon IDs found. Quitting.")
        sys.exit()
    return selected_species, taxons


def get_already_run_terms(alg_runners, **kwargs):
    # for each alg, taxon and go term pair, see which already exist and skip them
    alg_taxon_terms_to_skip = defaultdict(dict)
    for run_obj in alg_runners:
        alg = run_obj.name
        # define the output file path to see if it already exists
        #exp_type="%sloso" % ("all-sp-" if kwargs['keep_ann'] else '')
        exp_type = "loso"
        out_file = "%s/%s%s%s.txt" % (
            run_obj.out_dir, exp_type, run_obj.params_str, kwargs.get("postfix", ""))

        if os.path.isfile(out_file) and kwargs['forcealg']:
            print("Removing %s as results will be appended to it for each taxon" % (out_file))
            os.remove(out_file)
            # the ranks file is for sinksource_bounds
            ranks_file = out_file.replace('.txt','-ranks.txt')
            if '_bounds' in alg and os.path.isfile(ranks_file):
                print("\tAlso removing %s" % (ranks_file))
                os.remove(ranks_file)
            stats_file = out_file.replace('.txt','-stats.txt')
            if os.path.isfile(stats_file):
                print("\tAlso removing %s" % (stats_file))
                os.remove(stats_file)
        # if the output file already exists, skip the terms that are already there
        # unless --write-prec-rec is specified with a single term.
        # then only the full prec_rec file will be written
        elif kwargs['write_prec_rec'] and len(kwargs['goterm']) == 1:
            pass
        elif os.path.isfile(out_file): 
            print("WARNING: %s results file already exists. Appending to it" % (out_file))
            # check which results already exist and append to the rest
            print("Reading results from %s " % (out_file))
            taxon_terms_completed = utils.readColumns(out_file, 1, 2)
            alg_taxon_terms_to_skip[alg] = {taxon: set() for taxon, term in taxon_terms_completed}
            for taxon, term in taxon_terms_completed:
                alg_taxon_terms_to_skip[alg][taxon].add(term)
            print("\t%d taxon - term pairs already finished" % (len(taxon_terms_completed)))
    return alg_taxon_terms_to_skip 


def run_and_eval_algs(
        run_obj, ann_obj,
        train_ann_mat, test_ann_mat,
        taxon=None, **kwargs):
    goids, prots = ann_obj.goids, ann_obj.prots
    params_results = defaultdict(int)

    if kwargs.get('keep_ann', False) is True: 
        print("Keeping all annotations when making predictions")
    elif kwargs.get('non_pos_as_neg_eval', False) is True: 
        print("Evaluating using all non-ground-truth positives for the taxon as false positives")
    else:
        print("Evaluating using only the ground-truth negatives predicted as positives as false positives")

    # change the annotation matrix to the current training positive examples
    curr_ann_obj = setup.Sparse_Annotations(train_ann_mat, goids, prots)
    # make an ann obj with the test ann mat
    test_ann_obj = setup.Sparse_Annotations(test_ann_mat, goids, prots)
    # if this is a gene based method, then run it on only the nodes which have a pos/neg annotation
    # TODO make this an option
    if run_obj.get_alg_type() == 'gene-based':
        run_obj.kwargs['nodes_to_run'] = test_ann_mat.sum(axis=0).nonzero()[1]

    # setup the output file. Could be used by the runners to write temp files or other output files
    exp_type="loso" 
    postfix = kwargs.get("postfix", "")
    if kwargs['keep_ann']:
        exp_type = "eval-per-taxon" 
    out_file = "%s/%s%s%s.txt" % (
        run_obj.out_dir, exp_type, run_obj.params_str, postfix)
    run_obj.out_pref = out_file.replace('.txt','')
    utils.checkDir(os.path.dirname(out_file))

    # for sinksource_bounds, keep track of which nodes are either a left-out pos or left-out neg
    if run_obj.name in ['sinksource_bounds', 'sinksourceplus_bounds']:
        run_obj.params['rank_pos_neg'] = test_ann_mat

    # if predictions were already generated, and taxon is set to 'all', then use those.
    # otherwise, generate the prediction scores
    if kwargs['keep_ann'] and run_obj.goid_scores.getnnz() != 0:
        print("Using already computed scores")
    else:
        # replace the ann_obj in the runner with the current training annotations  
        run_obj.ann_obj = curr_ann_obj
        #alg_runners = run_eval_algs.setup_runners([alg], alg_settings, curr_ann_obj, **kwargs)
        if kwargs.get('verbose'):
            utils.print_memory_usage()
        run_obj.setupInputs()
        if kwargs.get('verbose'):
            utils.print_memory_usage()
        run_obj.run()
        if kwargs.get('verbose'):
            utils.print_memory_usage()
        run_obj.setupOutputs(taxon=taxon)

    # now evaluate 
    # this will write a file containing the fmax and other measures for each goterm 
    # with the taxon name in the name of the file
    eval_utils.evaluate_ground_truth(
        run_obj, test_ann_obj, out_file,
        #non_pos_as_neg_eval=opts.non_pos_as_neg_eval,
        taxon=taxon, append=True, **kwargs)
    for key in run_obj.params_results:
        params_results[key] += run_obj.params_results[key]

    return params_results


def leave_out_taxon(t, ann_obj, species_to_uniprot_idx,
                    eval_ann_obj=None, keep_ann=False, 
                    non_pos_as_neg_eval=False, eval_goterms_with_left_out_only=False,
                    oracle=False, num_test_cutoff=1, **kwargs):
    """
    Training positives are removed from testing positives, and train pos and neg are removed from test neg
        I don't remove training negatives from testing positives, because not all algorithms use negatives
    *t*: species to be left out. If t is None or 'all', then no species will be left out, and keep_ann must be True.
    *eval_ann_obj*: 
    *eval_goterms_with_left_out_only*: if eval_ann_obj is given and keep_ann is False, 
        only evaluate GO terms that have at least 2% of annotations. 
        Useful to speed-up processing for term-based algorithms
    *oracle*: remove train negatives that are actually test positives
    *num_test_cutoff*: minimum number of annotations for each go term in the left-out species 
    """
    if t == "all":
        t = None
    # leave this taxon out by removing its annotations
    # rather than a dictionary, build a matrix
    ann_matrix, goids, prots = ann_obj.ann_matrix, ann_obj.goids, ann_obj.prots
    train_ann_mat = sparse.lil_matrix(ann_matrix.shape, dtype=np.float)
    test_ann_mat = sparse.lil_matrix(ann_matrix.shape, dtype=np.float)
    sp_goterms = []
    #skipped_eval_no_left_out_ann = 0
    for idx, goid in enumerate(goids):
        pos, neg = alg_utils.get_goid_pos_neg(ann_matrix, idx)
        ann_pos = set(list(pos))
        ann_neg = set(list(neg))
        # first setup the training annotations (those used as positives/negatives for the algorithm)
        if keep_ann:
            train_pos = ann_pos 
            train_neg = ann_neg 
        else:
            train_pos = ann_pos - species_to_uniprot_idx[t]
            train_neg = ann_neg - species_to_uniprot_idx[t]
        eval_pos = ann_pos.copy()
        eval_neg = ann_neg.copy()
        # setup the testing annotations (those used when evaluating the performance)
        if eval_ann_obj is not None:
            if goid not in eval_ann_obj.goid2idx:
                eval_pos, eval_neg = set(), set()
            else:
                eval_pos, eval_neg = alg_utils.get_goid_pos_neg(eval_ann_obj.ann_matrix, eval_ann_obj.goid2idx[goid])
                eval_pos = set(list(eval_pos))
                eval_neg = set(list(eval_neg))
            # if this species has little-to-no annotations that are being left-out, then we can skip it
            #if not keep_ann and eval_goterms_with_left_out_only:
                ## If the percentage of left-out ann is less than 2%, then skip it
                #if (len(ann_pos) - len(train_pos)) / float(len(train_pos)) < .02:
                #    skipped_eval_no_left_out_ann += 1 
                #    continue
        if t is None:
            test_pos = eval_pos
            test_neg = eval_neg
            if non_pos_as_neg_eval:
                # everything minus the positives
                test_neg = set(prots) - test_pos
        else:
            # TODO I should limit these to the proteins in the network
            test_pos = eval_pos & species_to_uniprot_idx[t]
            # UPDATE 2018-06-27: Only evaluate the species prots as negatives, not all prots
            if non_pos_as_neg_eval:
                test_neg = species_to_uniprot_idx[t] - eval_pos
                test_neg.discard(None)
                #test_neg = np.asarray(sorted(test_neg)).astype(int)
            else:
                test_neg = eval_neg & species_to_uniprot_idx[t]
        # UPDATE 2018-06-30: Remove test positives/negatives that are part of the training positives/negatives
        # don't remove test positives if its a training negative because not all algorithms use negatives
        test_pos -= train_pos 
        if oracle:
            train_neg -= test_pos
        test_neg -= train_pos | train_neg 
        # build an array of the scores and set it in the goid sparse matrix of scores
        # UPDATE 2019-07: Some algorithms are node-based and could benefit from the extra annotations
        pos_neg_arr = np.zeros(len(prots))
        pos_neg_arr[list(train_pos)] = 1
        pos_neg_arr[list(train_neg)] = -1
        train_ann_mat[idx] = pos_neg_arr
        pos_neg_arr = np.zeros(len(prots))
        pos_neg_arr[list(test_pos)] = 1
        pos_neg_arr[list(test_neg)] = -1
        test_ann_mat[idx] = pos_neg_arr
        # UPDATE 2018-10: Add a cutoff on both the # of training positive and # of test pos
        if len(train_pos) < num_test_cutoff or len(test_pos) < num_test_cutoff or \
           (len(train_neg) == 0 or len(test_neg) == 0):
            continue
        sp_goterms.append(goid) 

    #if eval_ann_matrix is not None and not keep_ann and eval_goterms_with_left_out_only:
    #    print("\t%d goterms skipped_eval_no_left_out_ann (< 0.02 train ann in the left-out species)" % (skipped_eval_no_left_out_ann))

    return train_ann_mat.tocsr(), test_ann_mat.tocsr(), sp_goterms


def write_stats_file(alg_runners, params_results, **kwargs):
    # for each alg, write the params_results
    # if --forcealg was set, then thsi will be overwritten. Otherwise, append to it
    for run_obj in alg_runners:
        if run_obj.out_pref is None:
            continue
        out_file = "%s-stats.txt" % (run_obj.out_pref)
        print("Writing stats to %s" % (out_file))
        # write the stats of this run. 
        with open(out_file, 'a') as out:
            # first write this runner's param_results
            out.write("".join("%s\t%s\n" % (key, val) for key,val in sorted(run_obj.params_results.items())))
            # then write the rest (e.g., SWSN running time)
            out.write("".join("%s\t%s\n" % (key, val) for key,val in sorted(params_results.items())))


