
from optparse import OptionParser,OptionGroup
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
sys.path.append(os.path.dirname(__file__))
import src.setup_sparse_networks as setup
import src.algorithms.alg_utils as alg_utils
import src.algorithms.runner as runner
import src.algorithms.aptrank_birgrank.run_birgrank as run_birgrank
import src.evaluate.cross_validation as cross_validation
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


def parse_args(args):
    parser = setup_opts()
    (opts, args) = parser.parse_args(args)
    kwargs = vars(opts)
    with open(opts.config, 'r') as conf:
        config_map = yaml.load(conf)
    # TODO check to make sure the inputs are correct in config_map

    #if opts.exp_name is None or opts.pos_neg_file is None:
    #    print("--exp-name, --pos-neg-file, required")
    #    sys.exit(1)

    return config_map, kwargs


def setup_opts():
    ## Parse command line args.
    usage = '%prog [options]\n'
    parser = OptionParser(usage=usage)

    # general parameters
    group = OptionGroup(parser, 'Main Options')
    group.add_option('','--config', type='string', default="config-files/config.yaml",
                     help="Configuration file")
    group.add_option('-G', '--goterm', type='string', action="append",
                     help="Specify the GO terms to use (should be in GO:00XX format)")
    parser.add_option_group(group)

    # additional parameters
    group = OptionGroup(parser, 'Additional options')
    group.add_option('-W', '--num-pred-to-write', type='int', default=10,
                     help="Number of predictions to write to the file. If 0, none will be written. If -1, all will be written. Default=10")
    group.add_option('-N', '--factor-pred-to-write', type='float', 
                     help="Write the predictions <factor>*num_pos for each term to file. For example, if the factor is 2, a term with 5 annotations would get the nodes with the top 10 prediction scores written to file.")
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

    return parser


def run():
    config_map, kwargs = parse_args(sys.argv)
    input_settings = config_map['input_settings']
    input_dir = input_settings['input_dir']
    alg_settings = config_map['algs']
    output_settings = config_map['output_settings']

    for dataset in input_settings['datasets']:
        net_obj, ann_obj = setup_dataset(dataset, input_dir, alg_settings, **kwargs) 
        # the outputs will follow this structure:
        # outputs/<net_version>/<exp_name>/<alg_name>/output_files
        out_dir = "%s/%s/%s/" % (output_settings['output_dir'], dataset['net_version'], dataset['exp_name'])
        alg_runners = setup_runners(alg_settings, net_obj, ann_obj, out_dir, **kwargs)
        if kwargs['cross_validation_folds'] is not None:
            # run cross validation
            cross_validation.run_cv_all_goterms(alg_runners, ann_obj, folds=kwargs['cross_validation_folds'], **kwargs)
        if kwargs['only_cv'] is False:
            # run algorithms in "prediction" mode 
            run_algs(alg_runners, **kwargs) 


def setup_net(input_dir, dataset, **kwargs):
    # load the network matrix and protein IDs
    net_file = None
    if 'net_file' in dataset:
        if not isinstance(net_file, list):
            net_file = "%s/%s/%s" % (input_dir, dataset['net_version'], dataset['net_file'])
        else:
            net_file = dataset['net_file']
    if dataset.get('multi_net',False) is True: 
        # if multiple file names are passed in, then map each one of them
        if isinstance(net_file, list) or 'string_net_files' in dataset:
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
                    out_pref, net_files=net_file, string_net_files=string_net_files, 
                    string_nets=string_nets,
                    string_cutoff=string_cutoff,
                    forcenet=kwargs['forcenet'])
        else:
            net_names_file = "%s/%s/%s" % (input_dir, dataset['net_version'], dataset['net_settings']['net_names_file'])
            node_ids_file  = "%s/%s/%s" % (input_dir, dataset['net_version'], dataset['net_settings']['node_ids_file'])
            sparse_nets, net_names, prots = alg_utils.read_multi_net_file(net_file, net_names_file, node_ids_file)

        weight_method = dataset['net_settings']['weight_method'].lower()
        net_obj = setup.Sparse_Networks(
            sparse_nets, prots, net_names=net_names,
            weight_swsn=True if weight_method == 'swsn' else False,
            weight_gm2008=True if weight_method in ['gmw', 'gm2008'] else False,
            unweighted=True if weight_method == 'unweighted' else False,
        )
    else:
        W, prots = alg_utils.setup_sparse_network(net_file, forced=kwargs['forcenet'])
        weight_method = dataset['net_settings']['weight_method'].lower() if 'net_settings' in dataset else None
        net_obj = setup.Sparse_Networks(W, prots, 
            unweighted=True if weight_method == 'unweighted' else False)
    return net_obj


def get_algs_to_run(alg_settings):
    # these are the algs to run
    algs = []
    for alg in alg_settings:
        if alg_settings[alg].get('should_run', [True])[0] is True:
            algs.append(alg)
    return algs


def setup_dataset(dataset, input_dir, alg_settings, **kwargs):
    only_functions_file = None
    if 'only_functions_file' in dataset:
        only_functions_file = "%s/%s" % (input_dir, dataset['only_functions_file'])
    selected_goterms = alg_utils.select_goterms(
            only_functions_file=only_functions_file, goterms=kwargs['goterm']) 

    net_obj = setup_net(input_dir, dataset, **kwargs)

    # now build the annotation matrix
    pos_neg_file = "%s/%s" % (input_dir, dataset['pos_neg_file'])
    ann_matrix, goids = setup.setup_sparse_annotations(pos_neg_file, selected_goterms, net_obj.nodes)
    ann_obj = setup.Sparse_Annotations(ann_matrix, goids, net_obj.nodes)

    algs = get_algs_to_run(alg_settings)
    # this will be handled in the birgrank and aptrank runners(?)
    # read the extra data for birgrank/aptrank if it is specified
    # TODO move to birgrank/aptrank runners
    if 'birgrank' in algs or 'aptrank' in algs:
        obo_file = alg_settings['birgrank']['obo_file'][0] if 'birgrank' in algs else alg_settings['aptrank']['obo_file'][0]
        dag_matrix, pos_matrix, dag_goids = run_birgrank.setup_h_ann_matrices(
                net_obj.nodes, obo_file, pos_neg_file, goterms=selected_goterms)
        ann_obj.dag_matrix = dag_matrix
        ann_obj.pos_matrix = pos_matrix
        ann_obj.dag_goids = dag_goids

    return net_obj, ann_obj


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
            run_obj = runner.Runner(alg, net_obj, ann_obj, out_dir, combo, **kwargs)
            alg_runners.append(run_obj) 

    print("\t%d total runners" % (len(alg_runners)))
    return alg_runners


def run_algs(alg_runners, **kwargs):
    """
    Runs all of the specified algorithms with the given network and annotations.
    Each runner should return the GO term prediction scores for each node in a sparse matrix.
    """
    print("Generating inputs")
    # now setup the inputs for the runners
    for run_obj in alg_runners:
        run_obj.setupInputs()

    print("Running the algorithms")
    # run the algs
    # TODO storing all of the runners scores simultaneously could be costly (too much RAM).
    for run_obj in alg_runners:
        run_obj.run()

    # parse the outputs. Only needed for the algs that write output files
    for run_obj in alg_runners:
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
            utils.checkDir(os.path.dirname(out_file)) 
            write_output(run_obj.goid_scores, run_obj.ann_obj.goids, run_obj.ann_obj.prots,
                         out_file, num_pred_to_write=num_pred_to_write)

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
    run()
