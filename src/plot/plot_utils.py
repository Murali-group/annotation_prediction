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
import pandas as pd
import matplotlib
matplotlib.use('Agg') # To save files remotely.  Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# make this the default for now
sns.set_style('darkgrid')
import numpy as np
# add two levels up to the path
#from os.path import dirname
#base_path = dirname(dirname(dirname(__file__)))
#sys.path.append(base_path)
#print(sys.path)
#os.chdir(base_path)
#print(base_path)
#import run_eval_algs
#from src.algorithms import runner as runner
#sys.path.append(base_path + "/src/algorithms")
#print(sys.path)
#import runner
import src.algorithms.runner as runner
import src.utils.file_utils as utils
import src.evaluate.eval_utils as eval_utils


ALG_NAMES = {
    'localplus': 'Local+', 'local': 'Local',
    'sinksource': 'SinkSource', 'sinksourceplus': 'SinkSource+',
    'fastsinksource': 'FSS', 'fastsinksourceplus': 'FSS+',
    'genemania': 'GeneMANIA',  
    'birgrank': 'BirgRank', 'aptrank': 'AptRank OneWay',
    }

# TODO set colors for algorithms
# and an ordering


def main(config_map, **kwargs):
    input_settings = config_map['input_settings']
    #input_dir = input_settings['input_dir']
    alg_settings = config_map['algs']
    output_settings = config_map['output_settings']
    if kwargs.get('term_stats') is not None:
        df_stats_all = pd.DataFrame()
        for f in kwargs['term_stats']:
            df_stats = pd.read_csv(f, sep='\t')
            df_stats_all = pd.concat([df_stats_all, df_stats])
        kwargs['term_stats'] = df_stats_all

    utils.checkDir(os.path.dirname(kwargs['out_pref']))
    # plot prec-rec separately from everything else
    if kwargs['prec_rec']:
        # loop through all specified terms, or use an empty string if no terms were specified
        terms = kwargs['goterm'] if kwargs['goterm'] is not None else ['']
        for term in terms:
            term = '-'+term if term != '' else ''
            prec_rec = 'prec-rec' + term
            #kwargs['prec_rec'] = prec_rec
            df_all = load_all_results(input_settings, alg_settings, output_settings, prec_rec_str=prec_rec, **kwargs)
            if len(df_all) == 0:
                print("no terms found. Quitting")
                sys.exit()

            title = '-'.join(df_all['plot_exp_name'].unique())
            plot_curves(df_all, title=title, **kwargs)
    else:
        # get the path to the specified files for each alg
        df_all = load_all_results(input_settings, alg_settings, output_settings, **kwargs)
        if len(df_all) == 0:
            print("no terms found. Quitting")
            sys.exit()
        algs = df_all['Algorithm'].unique()

        print("\t%d algorithms, %d plot_exp_name values\n" % (len(algs), len(df_all['plot_exp_name'].unique())))
        #print(df_all.head())
        results_overview(df_all, measures=kwargs['measures'])

        # TODO currently only handles one dataset
        title = '-'.join(df_all['plot_exp_name'].unique())

        # now attempt to figure out what labels/titles to put in the plot based on the net version, exp_name, and plot_exp_name
        for measure in kwargs['measures']:
            if kwargs['boxplot']:
                plot_boxplot(df_all, measure=measure, title=title, **kwargs)
            if kwargs['scatter']:
                plot_scatter(df_all, measure=measure, title=title, **kwargs) 


def plot_curves(df, out_pref="test", title="", ax=None, **kwargs):
    """
    Plot precision recall curves, or (TODO) ROC curves 
    """
    # make a prec-rec plot per term
    for term in sorted(df["#goid"].unique()):
        curr_df = df[df['#goid'] == term]
        # get only the positive idx to plot prec_rec
        curr_df = curr_df[curr_df['pos/neg'] == 1]
        # also put the fmax on the plot, and add it to the label
        new_alg_names = []
        fmax_points = {}
        for alg in curr_df['Algorithm'].unique():
            df_alg = curr_df[curr_df['Algorithm'] == alg]
            #print(df_alg['prec'], df_alg['rec'])
            fmax, idx = eval_utils.compute_fmax(df_alg['prec'].values, df_alg['rec'].values, fmax_idx=True)
            new_alg_name = "%s (%0.3f)" % (alg, fmax)
            new_alg_names.append(new_alg_name) 
            fmax_points[alg] = (df_alg['prec'].values[idx], df_alg['rec'].values[idx])

        fig, ax = plt.subplots()
        #sns.pointplot(x='rec', y='prec', hue='Algorithm', 
        # TODO get only the positive examples. Then show the standard deviation from the repititions
        sns.lineplot(x='rec', y='prec', hue='Algorithm', data=curr_df,
                ci=None, ax=ax, legend=False,
                )
                #xlim=(0,1), ylim=(0,1), ci=None)

        ax.set_xlim(0,1)
        ax.set_ylim(0,1)

        ax.legend(title="Alg (Fmax)", labels=new_alg_names)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")

        # also add the fmax point to the plot
        for i, alg in enumerate(fmax_points):
            prec, rec = fmax_points[alg]
            print(term, alg, prec, rec)
            print(term, 2.0 / ((1.0/prec) + (1.0/rec)))
            #print((2*prec*rec) / (prec + rec))
            ax.plot([rec], [prec], marker="*", color=sns.color_palette()[i])

        if kwargs.get('term_stats') is not None:
            df_stats = kwargs['term_stats'] 
            curr_df_stats = df_stats[df_stats['#GO term'] == term]
            # TODO what if there are multiple stats lines?
            term_name = curr_df_stats['GO term name'].values[0]
            term_cat = curr_df_stats['GO category'].values[0]
            # map O to Phenotypic abnormality
            cat_map = {"O": "PA"}
            term_cat = cat_map[term_cat] if term_cat in cat_map else term_cat
            term_ann = curr_df_stats['# positive examples'].values[0]
            print(term_name, term_cat, term_ann)
            ax.set_title(title + "\n %s (%s) - %s, %s ann" % (term_name, term, term_cat, term_ann))
        else:
            ax.set_title(title + " %s" % (term))

        if out_pref is not None:
            out_file = "%s%s-prec-rec.pdf" % (out_pref, term)
            print("Writing %s" % (out_file))
            plt.savefig(out_file, bbox_inches='tight')
            plt.savefig(out_file.replace('.pdf','.png'), bbox_inches='tight')
            plt.close()


def plot_scatter(df, measure='fmax', out_pref="test", title="", ax=None, **kwargs):
    algs = df['Algorithm'].unique()
    df2 = df[['fmax', 'Algorithm']]
    df2 = df2.pivot(columns='Algorithm')
    df2.columns = [' '.join(col).strip() for col in df2.columns.values]
    # if there are only two algorithms, make a joint plot
    if len(algs) == 2:
        g = sns.jointplot(df2.columns[0], df2.columns[1], data=df2, xlim=(-0.02,1.02), ylim=(-0.02,1.02))
        # also plot x=y
        g.ax_joint.plot((0,1),(0,1))
    # if there are more, a pairplot
    else:
        g = sns.pairplot(data=df2)
        g.set(xlim=(-0.02,1.02), ylim=(-0.02,1.02))
        # draw the x=y lines
        for i in range(len(algs)):
            for j in range(len(algs)):
                if i != j:
                    g.axes[i][j].plot((0,1),(0,1))

    plt.title(title)

    if out_pref is not None:
        out_file = "%s-%s-scatter.pdf" % (out_pref, measure)
        print("Writing %s" % (out_file))
        plt.savefig(out_file, bbox_inches='tight')


def plot_boxplot(df, measure='fmax', out_pref="test", title="", ax=None, **kwargs):
    sns.boxplot(x=measure, y='Algorithm', data=df, ax=ax,
               fliersize=1.5, #order=[kwargs['alg_names'][a] for a in algorithms],
               #palette=plt_utils.my_palette)
                )

    measure_map = {'fmax': r'F$_{\mathrm{max}}$'}
    xlabel = measure.upper()
    if measure in measure_map:
        xlabel = measure_map[measure]
    plt.xlabel(xlabel)
    plt.ylabel("")
    plt.title(title)

    if out_pref is not None:
        out_file = "%s-%s-boxplot.pdf" % (out_pref, measure)
        print("Writing %s" % (out_file))
        plt.savefig(out_file, bbox_inches='tight')


def results_overview(df, measures=['fmax']):
    """
    Print an overview of the number of values / terms, as well as the median fmax
    """
    for plot_exp_name in sorted(df['plot_exp_name'].unique()):
        df_curr = df[df['plot_exp_name'] == plot_exp_name]
        print(plot_exp_name)
        # limit the goterms to those that are also present for SinkSource(?)
        for measure in measures:
            for alg in sorted(df_curr['Algorithm'].unique()):
                df_alg = df_curr[df_curr['Algorithm'] == alg][measure]
                print("\t%s: %0.3f \t\t(%d terms)" % (alg, df_alg.median(), len(df_alg)))


def load_all_results(input_settings, alg_settings, output_settings, prec_rec_str="", **kwargs):
    """
    Load all of the results for the datasets and algs specified in the config file
    """
    df_all = pd.DataFrame()
    for dataset in input_settings['datasets']:
        if kwargs['algs'] is not None:
            algs = kwargs['algs']
        else:
            algs = alg_settings.keys()
            print(algs)
        for alg in algs:
            # TODO use should_run?
            alg_params = alg_settings[alg]
            # If no algs were passed in, then use the 'should_run' value
            if kwargs['algs'] is None:
                if alg_params.get('should_run', [False])[0] is False:
                    continue
            curr_seed = kwargs.get('cv_seed')
            if 'cv-' in kwargs['exp_type']:
                for rep in range(1,kwargs.get('num_reps',1)+1):
                    if curr_seed is not None:
                        curr_seed += rep-1
                    curr_exp_type = "%s-rep%s%s" % (kwargs['exp_type'], rep, 
                            "-seed%s" % (curr_seed) if curr_seed is not None else "")
                    df = load_alg_results(
                        dataset, alg, alg_params, prec_rec=prec_rec_str,
                        results_dir=output_settings['output_dir'], exp_type=curr_exp_type)
                    add_dataset_settings(dataset, df) 
                    df['rep'] = rep
                    df_all = pd.concat([df_all, df])
            else:
                df = load_alg_results(
                    dataset, alg, alg_params, prec_rec=prec_rec_str, 
                    results_dir=output_settings['output_dir'], exp_type=kwargs['exp_type'])
                add_dataset_settings(dataset, df) 
                df_all = pd.concat([df_all, df])
    return df_all


def add_dataset_settings(dataset, df):
    # also add the net version and exp_name
    df['net_version'] = dataset['net_version']
    df['exp_name'] = dataset['exp_name']
    if 'net_settings' in dataset and 'weight_method' in dataset['net_settings']:
        df['weight_method'] = dataset['net_settings']['weight_method'] 
    # if they specified a name to use in the plot for this experiment, then use that
    plot_exp_name = "%s %s" % (dataset['net_version'], dataset['exp_name'])
    if 'plot_exp_name' in dataset:
        plot_exp_name = dataset['plot_exp_name']
    df['plot_exp_name'] = plot_exp_name
    return df


def load_alg_results(dataset, alg, alg_params, prec_rec="", results_dir='outputs', exp_type='cv-5folds'):
    """
    For a given dataset and algorithm, build the file path and load the results
    *results_dir*: the base output directory
    """
    alg_name = alg
    # if a name is specified to use when plotting, then get that
    if 'plot_name' in alg_params:
        alg_name = alg_params['plot_name'][0]
    elif alg in ALG_NAMES:
        alg_name = ALG_NAMES[alg]

    out_dir = "%s/%s/%s" % (
        results_dir, dataset['net_version'], dataset['exp_name'])
    combos = [dict(zip(alg_params.keys(), val))
        for val in itertools.product(
            *(alg_params[param] for param in alg_params))]
    print("%d combinations for %s" % (len(combos), alg))
    # load the CV file for each parameter combination for this algorithm
    df_all = pd.DataFrame()
    for param_combo in combos:
        # first get the parameter string for this runner
        params_str = runner.get_runner_params_str(alg, dataset, param_combo)
        cv_file = "%s/%s/%s%s%s.txt" % (out_dir, alg, exp_type, params_str, prec_rec)
        if not os.path.isfile(cv_file):
            print("\tnot found %s - skipping" % (cv_file))
            continue
        print("\treading %s" % (cv_file))
        df = pd.read_csv(cv_file, sep='\t')
        if len(combos) == 1: 
            df['Algorithm'] = alg_name
        else:
            df['Algorithm'] = alg_name + params_str
        df_all = pd.concat([df_all, df])
    return df_all


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
    if kwargs['measure'] is None:
        kwargs['measure'] = ['fmax']
    kwargs['measures'] = kwargs['measure']
    del kwargs['measure']
    kwargs['algs'] = kwargs['alg']
    del kwargs['alg']

    return config_map, kwargs


def setup_opts():
    ## Parse command line args.
    usage = '%prog [options]\n'
    parser = OptionParser(usage=usage)

    # general parameters
    group = OptionGroup(parser, 'Main Options')
    # TODO take multiple config files
    group.add_option('','--config', type='string', default="config-files/config.yaml",
                     help="Configuration file")
    group.add_option('-A', '--alg', type='string', action="append",
                     help="Algorithms to plot. Must be in the config file. If specified, will ignore 'should_run' in the config file")
    group.add_option('-o', '--out-pref', type='string', default="test",
                     help="Output prefix for writing plot to file. Default: test")
    group.add_option('-G', '--goterm', type='string', action="append",
                     help="Specify the GO terms to use (should be in GO:00XX format)")
    group.add_option('', '--forceplot', action='store_true', default=False,
                     help="Force overwitting plot files")
    group.add_option('', '--exp-type', type='string', default='cv-5folds',
                     help='Type of experiment (e.g., cv-5fold, temporal-holdout). Default: cv-5folds')
    group.add_option('', '--num-reps', type='int', default=1,
                     help="If --exp-type is <cv-Xfold>, this number of times CV was repeated. Default=1")
    group.add_option('', '--cv-seed', type='int', default=1,
                     help="Seed used when running CV")
    group.add_option('', '--term-stats', type='string', action='append',
                     help="File which contains the term name, # ann and other statistics such as depth. Can specify multiple")
    parser.add_option_group(group)

    # plotting parameters
    group = OptionGroup(parser, 'Plotting Options')
    group.add_option('', '--measure', action="append",
                     help="Evaluation measure to use. May specify multiple. Options: 'fmax', 'avgp', 'auprc', 'auroc'. Default: 'fmax'")
    group.add_option('', '--boxplot', action='store_true', default=False,
                     help="Compare all runners in the config file using a boxplot")
    group.add_option('', '--scatter', action='store_true', default=False,
                     help="Make a joint plot, or pair plot if more than two runners are given.")
    group.add_option('', '--prec-rec', action='store_true', default=False,
                     help="Make a precision recall curve for each specified term")
    parser.add_option_group(group)

    return parser


if __name__ == "__main__":
    config_map, kwargs = parse_args(sys.argv)

    main(config_map, **kwargs)
