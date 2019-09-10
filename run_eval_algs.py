
import argparse
import yaml
import itertools
from collections import defaultdict
import os
import sys
from tqdm import tqdm
import time
import numpy as np
from scipy import sparse
# packages in this repo
# add this file's directory to the path so these imports work from anywhere
sys.path.insert(0,os.path.dirname(__file__))
#from src import setup_sparse_networks as setup
import src.setup_sparse_networks as setup
import src.algorithms.alg_utils as alg_utils
import src.algorithms.runner as runner
import src.algorithms.aptrank_birgrank.run_birgrank as run_birgrank
import src.evaluate.eval_utils as eval_utils
import src.evaluate.cross_validation as cross_validation
import src.evaluate.eval_leave_one_species_out as eval_loso
import src.utils.file_utils as utils
import src.utils.string_utils as string_utils


#class Alg_Runner:
#    def __init__(self, net_version, exp_name, net_obj, ann_obj, runners, **kwargs):
#    
#        self.net_version = net_version
#        self.exp_name = exp_name
#        self.kwargs = kwargs
#        self.verbose = kwargs.get('verbose', False) 
#        self.forced = kwargs.get('forcealg', False) 


def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)
    with open(args.config, 'r') as conf:
        config_map = yaml.load(conf, Loader=yaml.FullLoader)
    # TODO check to make sure the inputs are correct in config_map

    #if opts.exp_name is None or opts.pos_neg_file is None:
    #    print("--exp-name, --pos-neg-file, required")
    #    sys.exit(1)

    return config_map, kwargs


def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser()  #description='')

    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str, default="config-files/config.yaml",
            help="Configuration file")
    group.add_argument('--goterm', '-G', type=str, action="append",
            help="Specify the GO terms to use. Can use this option multiple times")

    # evaluation parameters
    group = parser.add_argument_group('Evaluation options')
    group.add_argument('--only-eval', action="store_true", default=False,
            help="Perform evaluation only (i.e., skip prediction mode)")
    group.add_argument('--cross-validation-folds', '-C', type=int,
            help="Perform cross validation using the specified # of folds. Usually 5")
    group.add_argument('--num-reps', type=int, default=1,
            help="Number of times to repeat the CV process. Default=1")
    group.add_argument('--cv-seed', type=int, 
            help="Seed to use for the random number generator when splitting the annotations into folds. " + \
            "If --num-reps > 1, the seed will be incremented by 1 each time. Should only be used for testing purposes")
    group.add_argument('--loso', action="store_true", default=False,
            help="Leave-One-Species-Out evaluation. For each species, leave out all of its annotations " +
            "and evaluate how well they can be recovered from the annotations of the other species. " +
            "Must specify the 'taxon_file' in the config file")
    group.add_argument('--taxon', '-T', dest="taxons", type=str, action='append',
            help="Specify the species taxonomy ID for which to evaluate. Multiple may be specified. Otherwise, all species will be used")
    group.add_argument('--write-prec-rec', action="store_true", default=False,
            help="Also write a file containing the precision and recall for every positive example. " + \
            "If a single term is given, only the prec-rec file, with the term in its name, will be written.")
    group.add_argument('--early-prec', '-E', type=str, action="append", default=["k1"],
            help="Report the precision at the specified recall value (between 0 and 1). " + \
            "If prefixed with 'k', for a given term, the precision at (k * # ann) # of nodes is given. Default: k1")

    # additional parameters
    group = parser.add_argument_group('Additional options')
    group.add_argument('--num-pred-to-write', '-W', type=int, default=10,
            help="Number of predictions to write to the file for predictions mode (meaning if --only-eval isn't specified). " + \
            "If 0, none will be written. If -1, all will be written. Default=10")
    group.add_argument('--factor-pred-to-write', '-N', type=float, 
            help="Write the predictions <factor>*num_pos for each term to file. " +
            "For example, if the factor is 2, a term with 5 annotations would get the nodes with the top 10 prediction scores written to file.")
    # TODO finish adding this option
    #group.add_argument('-T', '--ground-truth-file', type=str,
    #                 help="File containing true annotations with which to evaluate predictions")
    group.add_argument('--postfix', type=str, default='',
            help="String to add to the end of the output file name(s)")
    group.add_argument('--forcealg', action="store_true", default=False,
            help="Force re-running algorithms if the output files already exist")
    group.add_argument('--forcenet', action="store_true", default=False,
            help="Force re-building network matrix from scratch")
    group.add_argument('--verbose', action="store_true", default=False,
            help="Print additional info about running times and such")

    return parser


def run(config_map, **kwargs):
    input_settings = config_map['input_settings']
    input_dir = input_settings['input_dir']
    alg_settings = config_map['algs']
    output_settings = config_map['output_settings']
    # combine the evaluation settings in the config file and the kwargs
    kwargs.update(config_map['eval_settings'])

    for dataset in input_settings['datasets']:
        net_obj, ann_obj, eval_ann_obj = setup_dataset(dataset, input_dir, alg_settings, **kwargs) 
        # if there are no annotations, then skip this dataset
        if len(ann_obj.goids) == 0:
            print("No terms found. Skipping this dataset")
            continue
        # the outputs will follow this structure:
        # outputs/<net_version>/<exp_name>/<alg_name>/output_files
        out_dir = "%s/%s/%s/" % (output_settings['output_dir'], dataset['net_version'], dataset['exp_name'])
        alg_runners = setup_runners(alg_settings, net_obj, ann_obj, out_dir, **kwargs)

        # first run prediction mode since it is the fastest
        if kwargs['only_eval'] is False:
            # run algorithms in "prediction" mode 
            run_algs(alg_runners, **kwargs) 
            # if specified, write the SWSN combined network to a file
            save_net = dataset['net_settings'].get('save_net', None) if 'net_settings' in dataset else None
            if net_obj.weight_swsn is True and save_net is not None:
                out_file = "%s/%s/%s" % (input_dir, dataset['net_version'], save_net)
                # the SWSN network is part of the runner object. Need to organize that better
                net_obj.save_net(out_file)

            # if a pos_neg_file_eval was passed in (e.g., for temporal holdout validation),
            # use it to evaluate the predictions
            if eval_ann_obj is not None:
                exp_type = "eval"
                # For LOSO, 'all-sp-loso' was used in the past
                #if kwargs.get('keep_ann') is not None:
                #    exp_type="all-sp-loso" 
                for run_obj in alg_runners:
                    out_file = "%s/%s%s%s.txt" % (
                        run_obj.out_dir, exp_type, run_obj.params_str, kwargs.get("postfix", ""))
                    utils.checkDir(os.path.dirname(out_file))
                    eval_utils.evaluate_ground_truth(
                        run_obj, eval_ann_obj, out_file, **kwargs)

        if kwargs['cross_validation_folds'] is not None:
            # run cross validation
            cross_validation.run_cv_all_goterms(alg_runners, ann_obj, folds=kwargs['cross_validation_folds'], **kwargs)

        if kwargs['loso'] is not None:
            # add the taxon file paths for this dataset to kwargs
            for arg in ['taxon_file', 'only_taxon_file']:
                kwargs[arg] = "%s/%s" % (input_dir, dataset[arg]) 
            # now run the leave-one-species-out eval
            eval_loso.eval_loso(alg_runners, ann_obj, eval_ann_obj=eval_ann_obj, **kwargs)


def setup_net(input_dir, dataset, **kwargs):
    # load the network matrix and protein IDs
    net_files = None
    if 'net_files' in dataset:
        net_files = ["%s/%s/%s" % (input_dir, dataset['net_version'], net_file) for net_file in dataset['net_files']]
    unweighted = dataset['net_settings'].get('unweighted', False) if 'net_settings' in dataset else False
    if dataset.get('multi_net',False) is True: 
        # if multiple file names are passed in, then map each one of them
        if isinstance(net_files, list) or 'string_net_files' in dataset:
            string_net_files = ["%s/%s/%s" % (input_dir, dataset['net_version'], string_net_file) for string_net_file in dataset['string_net_files']]
            string_nets = None 
            if 'string_nets' in dataset['net_settings']:
                string_nets = string_utils.convert_string_naming_scheme(dataset['net_settings']['string_nets'])
            # they all need to have the same rows and columns, which is handled by this function
            # this function also creates the multi net file if it doesn't exist
            string_cutoff = dataset['net_settings'].get('string_cutoff', 150) 
            out_pref = "%s/sparse-nets/c%d-" % (os.path.dirname(string_net_files[0]), string_cutoff)
            utils.checkDir(os.path.dirname(out_pref))
            sparse_nets, net_names, prots = setup.create_sparse_net_file(
                    out_pref, net_files=net_files, string_net_files=string_net_files, 
                    string_nets=string_nets,
                    string_cutoff=string_cutoff,
                    forcenet=kwargs['forcenet'])
        else:
            # if a .mat file with multiple sparse matrix networks inside of it is passed in, read that here
            net_names_file = "%s/%s/%s" % (input_dir, dataset['net_version'], dataset['net_settings']['net_names_file'])
            node_ids_file  = "%s/%s/%s" % (input_dir, dataset['net_version'], dataset['net_settings']['node_ids_file'])
            sparse_nets, net_names, prots = alg_utils.read_multi_net_file(net_file, net_names_file, node_ids_file)

        weight_method = dataset['net_settings']['weight_method'].lower()
        net_obj = setup.Sparse_Networks(
            sparse_nets, prots, net_names=net_names,
            weight_method=weight_method, unweighted=unweighted,
        )
    else:
        if net_files is None:
            print("ERROR: no net files specified in the config file. Must provide either 'net_files', or 'string_net_files'")
            sys.exit()
        W, prots = alg_utils.setup_sparse_network(net_files[0], forced=kwargs['forcenet'])
        net_obj = setup.Sparse_Networks(W, prots, unweighted=unweighted)
    return net_obj


def get_algs_to_run(alg_settings):
    # these are the algs to run
    algs = []
    for alg in alg_settings:
        if alg_settings[alg].get('should_run', [True])[0] is True:
            algs.append(alg)
    return algs


def setup_dataset(dataset, input_dir, alg_settings, **kwargs):
    # setup the network matrix first
    net_obj = setup_net(input_dir, dataset, **kwargs)
    #ann_obj = setup_annotations(input_dir, dataset, **kwargs)

    # limit the terms to whatever is specified either in the only_functions_file,
    # or the --goterm command-line option
    only_functions_file = None
    # if specific goterms are passed in_then ignore the only functions file
    if kwargs['goterm'] is None and 'only_functions_file' in dataset and dataset['only_functions_file'] != '':
        only_functions_file = "%s/%s" % (input_dir, dataset['only_functions_file'])
    selected_goterms = alg_utils.select_goterms(
            only_functions_file=only_functions_file, goterms=kwargs['goterm']) 

    # now build the annotation matrix
    pos_neg_file = "%s/%s" % (input_dir, dataset['pos_neg_file'])
    ann_matrix, goids = setup.setup_sparse_annotations(pos_neg_file, selected_goterms, net_obj.nodes)
    ann_obj = setup.Sparse_Annotations(ann_matrix, goids, net_obj.nodes)

    eval_ann_obj = None
    # also check if a evaluation pos_neg_file was given
    if dataset.get('pos_neg_file_eval', '') != '':
        pos_neg_file_eval = "%s/%s" % (input_dir, dataset['pos_neg_file_eval'])
        ann_matrix, goids = setup.setup_sparse_annotations(pos_neg_file_eval, selected_goterms, net_obj.nodes)
        eval_ann_obj = setup.Sparse_Annotations(ann_matrix, goids, net_obj.nodes)

    algs = get_algs_to_run(alg_settings)
    # this will be handled in the birgrank and aptrank runners(?)
    # read the extra data for birgrank/aptrank if it is specified
    # TODO move to birgrank/aptrank runners
    if 'birgrank' in algs or 'aptrank' in algs:
        obo_file = alg_settings['birgrank']['obo_file'][0] if 'birgrank' in algs else alg_settings['aptrank']['obo_file'][0]
        # TODO get the dag matrix/matrices without using the pos_neg_file
        dag_matrix, _, dag_goids = run_birgrank.setup_h_ann_matrices(
                net_obj.nodes, obo_file, pos_neg_file, goterms=selected_goterms)
        ann_obj.dag_matrix = dag_matrix
        ann_obj.dag_goids = dag_goids

    return net_obj, ann_obj, eval_ann_obj


def setup_runners(alg_settings, net_obj, ann_obj, out_dir, **kwargs):
    # these are the algs with 'should_run' set to [True]
    algs = get_algs_to_run(alg_settings)
    print("Algs to run: %s" % (', '.join(algs)))

    alg_runners = []
    for alg in algs:
        params = alg_settings[alg]
        combos = [dict(zip(params, val))
            for val in itertools.product(
                *(params[param] for param in params))]
        for combo in combos:
            if alg in ['birgrank', 'aptrank']:
                kwargs['dag_matrix'] = ann_obj.dag_matrix
                kwargs['dag_goids'] = ann_obj.dag_goids
            run_obj = runner.Runner(alg, net_obj, ann_obj, out_dir, combo, **kwargs)
            alg_runners.append(run_obj) 

    print("\t%d total runners" % (len(alg_runners)))
    return alg_runners


def run_algs(alg_runners, **kwargs):
    """
    Runs all of the specified algorithms with the given network and annotations.
    Each runner should return the GO term prediction scores for each node in a sparse matrix.
    """
    # first check to see if the algorithms have already been run
    # and if the results should be overwritten
    if kwargs['forcealg'] is True or kwargs['num_pred_to_write'] == 0:
        runners_to_run = alg_runners
    else:
        runners_to_run = []
        for run_obj in alg_runners:
            out_file = "%s/pred%s.txt" % (run_obj.out_dir, run_obj.params_str)
            if os.path.isfile(out_file):
                print("%s already exists. Use --forcealg to overwite" % (out_file))
            else:
                runners_to_run.append(run_obj)

    params_results = {}

    print("Generating inputs")
    # now setup the inputs for the runners
    for run_obj in runners_to_run:
        run_obj.setupInputs()

    print("Running the algorithms")
    # run the algs
    # TODO storing all of the runners scores simultaneously could be costly (too much RAM).
    for run_obj in runners_to_run:
        run_obj.run()
        print(run_obj.params_results)
        params_results.update(run_obj.params_results)

    # parse the outputs. Only needed for the algs that write output files
    for run_obj in runners_to_run:
        run_obj.setupOutputs()

        # write to file if specified
        num_pred_to_write = kwargs['num_pred_to_write']
        if kwargs.get('factor_pred_to_write') is not None:
            # make a dictionary with the # ann*factor for each term
            num_pred_to_write = {} 
            for i in range(run_obj.ann_matrix.shape[0]):
                y = run_obj.ann_matrix[i,:]
                positives = (y > 0).nonzero()[1]
                num_pred_to_write[run_obj.goids[i]] = len(positives) * kwargs['factor_pred_to_write']
        if num_pred_to_write != 0:
            out_file = "%s/pred%s.txt" % (run_obj.out_dir, run_obj.params_str)
            # TODO generate the output file paths in the runner object
            #out_file = run_obj.out_file
            utils.checkDir(os.path.dirname(out_file)) 
            write_output(run_obj.goid_scores, run_obj.ann_obj.goids, run_obj.ann_obj.prots,
                         out_file, num_pred_to_write=num_pred_to_write)

    print(params_results)
    print("Finished")


def write_output(goid_scores, goids, prots, out_file, num_pred_to_write=10):
    """
    *num_pred_to_write* can either be an integer, or a dictionary with a number of predictions to write for each term
    """
    # now write the scores to a file
    if isinstance(num_pred_to_write, dict):
        #print("\twriting top %d*num_pred scores to %s" % (kwargs['factor_pred_to_write'], out_file))
        print("\twriting top <factor>*num_pred scores to %s" % (out_file))
    else:
        print("\twriting top %d scores to %s" % (num_pred_to_write, out_file))

    with open(out_file, 'w') as out:
        out.write("#goterm\tprot\tscore\n")
        for i in range(goid_scores.shape[0]):
            scores = goid_scores[i].toarray().flatten()
            # convert the nodes back to their names, and make a dictionary out of the scores
            scores = {prots[j]:s for j, s in enumerate(scores)}
            num_to_write = num_pred_to_write
            if isinstance(num_to_write, dict):
                num_to_write = num_pred_to_write[goids[i]]
            write_scores_to_file(scores, goid=goids[i], file_handle=out,
                    num_pred_to_write=int(num_to_write))


def write_scores_to_file(scores, goid='', out_file=None, file_handle=None,
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
        file_handle.write("%s\t%s\t%0.6e\n" % (goid, n, scores[n]))
    return


if __name__ == "__main__":
    config_map, kwargs = parse_args()
    run(config_map, **kwargs)
