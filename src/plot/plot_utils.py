import argparse
import yaml
import itertools
import os
import sys
#from collections import defaultdict
#from tqdm import tqdm
#import time
#import numpy as np
#from scipy import sparse
from scipy.stats import kruskal, mannwhitneyu
# plotting imports
import matplotlib
matplotlib.use('Agg')  # To save files remotely. 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# make this the default for now
sns.set_style('darkgrid')
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
import run_eval_algs
import src.algorithms.runner as runner
import src.utils.file_utils as utils
import src.evaluate.eval_utils as eval_utils


ALG_NAMES = {
    'localplus': 'Local', 'local': 'Local-',
    'sinksource': 'SinkSource', 'sinksourceplus': 'SinkSource+',
    'fastsinksource': 'FSS', 'fastsinksourceplus': 'FSS+',
    'genemania': 'GeneMANIA',  
    'birgrank': 'BirgRank', 'aptrank': 'AptRank OneWay',
    }

measure_map = {'fmax': r'F$_{\mathrm{max}}$'}

# tried to be fancy :P
# colors: https://coolors.co/ef6e4a-0ec9aa-7c9299-5d88d3-96bd33
my_palette = ["#EF6E4A", "#0EC9AA", "#7C9299", "#5D88D3", "#96BD33", "#937860", "#efd2b8"]
# for comparing sinksource with local
#my_palette = ["#EF6E4A", sns.xkcd_rgb["deep sky blue"], "#96BD33", "#937860", "#efd2b8"]
#my_palette = [sns.xkcd_rgb["orange"], sns.xkcd_rgb["azure"], "#96BD33", "#937860", "#efd2b8"]


def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser()

    # general parameters
    group = parser.add_argument_group('Main Options')
    # TODO take multiple config files
    group.add_argument('--config', type=str, required=True,
                     help="Configuration file")
    group.add_argument('--alg', '-A', dest='algs', type=str, action="append",
                     help="Algorithms to plot. Must be in the config file. If specified, will ignore 'should_run' in the config file")
    group.add_argument('--out-pref', '-o', type=str, default="",
                     help="Output prefix for writing plot to file. Default: outputs/viz/<net_version>/<exp_name>/")
    group.add_argument('--goterm', '-G', type=str, action="append",
                     help="Specify the GO terms to use (should be in GO:00XX format)")
    group.add_argument('--exp-type', type=str, default='cv-5folds',
                     help='Type of experiment (e.g., cv-5fold, loso, temporal-holdout). Default: cv-5folds')
    group.add_argument('--num-reps', type=int, default=1,
                     help="If --exp-type is <cv-Xfold>, this number of times CV was repeated. Default=1")
    group.add_argument('--cv-seed', type=int,
                     help="Seed used when running CV")
    group.add_argument('--only-terms-file', type=str, 
                     help="File containing a list of terms (in the first col, tab-delimited) for which to limit the results")
    group.add_argument('--only-terms-name', type=str, default='',
                     help="If --only-terms is specified, use this option to append a name to the file. Default is to use the # of terms")
    group.add_argument('--postfix', type=str, default='',
                     help="Postfix to add to the end of the files")

    # plotting parameters
    group = parser.add_argument_group('Plotting Options')
    group.add_argument('--measure', action="append",
                     help="Evaluation measure to use. May specify multiple. Options: 'fmax', 'avgp', 'auprc', 'auroc'. Default: 'fmax'")
    group.add_argument('--boxplot', action='store_true', default=False,
                     help="Compare all runners in the config file using a boxplot")
    group.add_argument('--line', action='store_true', default=False,
                     help="Compare all runners on all datasets in the config file using a lineplot")
    group.add_argument('--scatter', action='store_true', default=False,
                     help="Make a scatterplot, or pair plot if more than two runners are given." +
                     "If the # ann are given with --term-stats, then plot the fmax by the # ann")
    group.add_argument('--prec-rec', action='store_true', default=False,
                     help="Make a precision recall curve for each specified term")

    group = parser.add_argument_group('Parameter / Statisical Significance Options')
    group.add_argument('--compare-param', type=str,
                       help="name of parameter to compare (e.g., alpha)")
    group.add_argument('--max-val', type=float,
                       help="Maximum value of the parameter against which to compare statistical significance (e.g., 1.0 for alpha")

    # figure parameters
    group = parser.add_argument_group('Figure Options')
    # Moved to the config file
    group.add_argument('--title','-T', 
                     help="Title to give the figure. Default is the exp_name ")
    group.add_argument('--for-paper', action='store_true', default=False,
                     help="Exclude extra information from the title and make the labels big and bold")
    group.add_argument('--horiz','-H', dest="horizontal", action='store_true', default=False,
                     help="Flip the plot so the measure is on the y-axis (horizontal). Default is x-axis (vertical)")
    group.add_argument('--png', action='store_true', default=False,
                     help="Write a png in addition to a pdf")
    group.add_argument('--term-stats', type=str, action='append',
                     help="File which contains the term name, # ann and other statistics such as depth. " +
                     "Useful to add info to title of prec-rec plot. Can specify multiple")
    group.add_argument('--forceplot', action='store_true', default=False,
                     help="Force overwitting plot files if they exist. TODO not yet implemented.")

    return parser


def load_config_file(config_file):
    with open(config_file, 'r') as conf:
        config_map = yaml.load(conf, Loader=yaml.FullLoader)
    return config_map


def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)
    config_map = load_config_file(kwargs['config'])
    # TODO check to make sure the inputs are correct in config_map

    #if opts.exp_name is None or opts.pos_neg_file is None:
    #    print("--exp-name, --pos-neg-file, required")
    #    sys.exit(1)
    if kwargs['measure'] is None:
        kwargs['measure'] = ['fmax']
    kwargs['measures'] = kwargs['measure']
    del kwargs['measure']
    kwargs['alg_names'] = ALG_NAMES

    return config_map, kwargs


def main(config_map, ax=None, out_pref='', **kwargs):

    input_settings, alg_settings, output_settings, out_pref, kwargs = setup_variables(
        config_map, out_pref, **kwargs)
    print(out_pref, kwargs.get('out_pref'))
    kwargs['out_pref'] = out_pref

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
            # limit to the specified terms
            if kwargs['only_terms'] is not None:
                df_all = df_all[df_all['#goid'].isin(kwargs['only_terms'])]

            title = '-'.join(df_all['plot_exp_name'].unique())
            plot_curves(df_all, title=title, **kwargs)
    else:
        # get the path to the specified files for each alg
        df_all = load_all_results(input_settings, alg_settings, output_settings, **kwargs)
        if len(df_all) == 0:
            print("no terms found. Quitting")
            sys.exit()
        # limit to the specified terms
        if kwargs.get('only_terms') is not None:
            df_all = df_all[df_all['#goid'].isin(kwargs['only_terms'])]
        num_terms = df_all['#goid'].nunique()
        if kwargs['exp_type'] == "loso":
            sp_taxon_pairs = df_all['#taxon'].astype(str) + df_all['#goid']
            num_terms = sp_taxon_pairs.nunique()
            #num_terms = df_all.groupby(['#taxon', '#goid']).size()
            print(num_terms)
        algs = df_all['Algorithm'].unique()

        print("\t%d algorithms, %d plot_exp_name values\n" % (len(algs), len(df_all['plot_exp_name'].unique())))
        #print(df_all.head())
        results_overview(df_all, measures=kwargs['measures'])

        if kwargs.get('title'):
            title = kwargs['title']
        else:
            title = '-'.join(df_all['plot_exp_name'].unique())
        if not kwargs.get('for_paper'):
            title += " \n %d%s %s" % (
                    num_terms, ' %s'%kwargs.get('only_terms_name', ''),
                    "sp-term pairs" if kwargs['exp_type'] == 'loso' else 'terms')
        kwargs['title'] = title

        # now attempt to figure out what labels/titles to put in the plot based on the net version, exp_name, and plot_exp_name
        for measure in kwargs['measures']:
            if kwargs['boxplot']:
                ax = plot_boxplot(df_all, measure=measure, ax=ax, **kwargs)
            if kwargs['scatter']:
                ax = plot_scatter(df_all, measure=measure, ax=ax, **kwargs) 
            if kwargs['line']:
                ax = plot_line(df_all, measure=measure, ax=ax, **kwargs)
    return ax


def setup_variables(config_map, out_pref='', **kwargs):
    """
    Function to setup the various args specified in kwargs
    """
    input_settings = config_map['input_settings']
    #input_dir = input_settings['input_dir']
    alg_settings = config_map['algs']
    output_settings = config_map['output_settings']
    if config_map.get('eval_settings'):
        kwargs.update(config_map['eval_settings'])
    if config_map.get('plot_settings'):
        #config_map['plot_settings'].update(kwargs)
        kwargs.update(config_map['plot_settings'])
        # overwrite whatever is in the plot settings with the specified args
        if kwargs.get('out_pref') and out_pref != '':
            del kwargs['out_pref']
            #kwargs['out_pref'] = out_pref
        else:
            out_pref = kwargs['out_pref']
    if kwargs.get('term_stats') is not None:
        df_stats_all = pd.DataFrame()
        for f in kwargs['term_stats']:
            df_stats = pd.read_csv(f, sep='\t')
            df_stats_all = pd.concat([df_stats_all, df_stats])
        kwargs['term_stats'] = df_stats_all

    if out_pref == "":
        out_pref = "%s/viz/%s/%s/" % (
                output_settings['output_dir'], 
                input_settings['datasets'][0]['net_version'], 
                input_settings['datasets'][0]['exp_name'])
    if kwargs['only_terms_file'] is not None:
        only_terms = pd.read_csv(kwargs['only_terms_file'], sep='\t', index_col=None)
        only_terms = only_terms.iloc[:,0].values
        print("limitting to %d terms from %s" % (len(only_terms), kwargs['only_terms_file']))
        kwargs['only_terms'] = only_terms
        # setup the name to add to the output file
        only_terms_postfix = kwargs['only_terms_name'].lower() + str(len(kwargs['only_terms'])) + '-'
        out_pref += only_terms_postfix

    # TODO only create the output dir if plots are will be created
    if out_pref is not None:
        utils.checkDir(os.path.dirname(out_pref))

    return input_settings, alg_settings, output_settings, out_pref, kwargs


def savefig(out_file, **kwargs):
    print("Writing %s" % (out_file))
    plt.savefig(out_file, bbox_inches='tight')
    if kwargs.get('png'):
        plt.savefig(out_file.replace('.pdf','.png'), bbox_inches='tight')
    plt.close()


def set_labels(ax, title, xlabel, ylabel, **kwargs):
    xlabel, ylabel = (ylabel, xlabel) if kwargs.get('horizontal') else (xlabel, ylabel)
    if kwargs.get('for_paper'):
        ax.set_xlabel(xlabel, fontsize=11, weight="bold")
        ax.set_ylabel(ylabel, fontsize=11, weight="bold")
        ax.set_title(title, fontsize=18, weight="bold")
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)


def plot_line(df, measure='fmax', out_pref="test", title="", ax=None, **kwargs):
    # default list of markers
    markers = ['o', 's', 'P', '^', 'X', '*', '+', 'v', 'x',]
    #print(df[[measure, 'plot_exp_name', 'Algorithm']].head())
    x,y = measure, 'plot_exp_name'
    # flip the x and y axis if specified
    x,y = (y,x) if kwargs.get('horizontal') else (x,y)
        
    # doesn't work for categorical data
    #sns.lineplot(x=measure, y='pen-alg', data=df, ax=ax,
    ax = sns.pointplot(x=x, y=y, data=df, ax=ax,
            hue='Algorithm', ci=None,
            markers=markers,
               #order=[kwargs['alg_names'][a] for a in algorithms],
            palette=my_palette,
                )
    plt.setp(ax.lines,linewidth=1)  # set lw for all lines of g axes
    if kwargs['horizontal'] and len(df['plot_exp_name'].unique()[-1]) > 20:
        ax.tick_params(axis='x', rotation=45)

    xlabel = "Median %s" % (measure_map.get(measure, measure.upper()))
    ylabel = kwargs.get('exp_label', '')
    set_labels(ax, title, xlabel, ylabel, **kwargs)

    if out_pref is not None:
        out_file = "%s%s-line.pdf" % (out_pref, measure)
        savefig(out_file, **kwargs)
    return ax


def plot_boxplot(df, measure='fmax', out_pref="test", title="", ax=None, **kwargs):
    df['Algorithm'] = df['Algorithm'].astype(str)
    df = df[['Algorithm', measure]]
    #print(df.head())
    df = df.pivot(columns='Algorithm', values=measure)
    #print(df.head())
    #ax = sns.boxplot(x=measure, y='Algorithm', data=df, ax=ax,
    ax = sns.boxplot(data=df, ax=ax,
                     fliersize=1.5, order=[kwargs['alg_names'][a] for a in kwargs['algs']],
                     orient='v' if not kwargs.get('horizontal') else 'h',
                     palette=my_palette,
                )

    xlabel = kwargs.get('exp_label', '')
    ylabel = measure_map.get(measure, measure.upper())
    if kwargs['share_measure'] is True:
        ylabel = ""
    set_labels(ax, title, xlabel, ylabel, **kwargs)

    if out_pref is not None:
        out_file = "%s%s-boxplot.pdf" % (out_pref, measure)
        savefig(out_file, **kwargs)
    return ax


def plot_curves(df, out_pref="test", title="", ax=None, **kwargs):
    """
    Plot precision recall curves, or (TODO) ROC curves 
    """
    # make a prec-rec plot per term
    for term in sorted(df["#goid"].unique()):
        curr_df = df[df['#goid'] == term]
        # get only the positive examples to plot prec_rec
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
        # TODO show the standard deviation from the repititions
        sns.lineplot(x='rec', y='prec', hue='Algorithm', data=curr_df,
                ci=None, ax=ax, legend=False,
                )
                #xlim=(0,1), ylim=(0,1), ci=None)

        ax.set_xlim(-0.02,1.02)
        ax.set_ylim(-0.02,1.02)

        ax.legend(title="Alg (Fmax)", labels=new_alg_names)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")

        # also add the fmax point to the plot
        for i, alg in enumerate(fmax_points):
            prec, rec = fmax_points[alg]
            ax.plot([rec], [prec], marker="*", color=sns.color_palette()[i])

        if kwargs.get('term_stats') is not None:
            df_stats = kwargs['term_stats'] 
            curr_df_stats = df_stats[df_stats['#GO term'] == term]
            # TODO what if there are multiple stats lines?
            term_name = curr_df_stats['GO term name'].values[0]
            term_cat = curr_df_stats['GO category'].values[0]
            # For HPO, map O to Phenotypic abnormality
            cat_map = {"O": "PA", 'P': 'BP', 'F': 'MF', 'c': 'CC'}
            term_cat = cat_map[term_cat] if term_cat in cat_map else term_cat
            term_ann = curr_df_stats['# positive examples'].values[0]
            print(term_name, term_cat, term_ann)
            ax.set_title(title + "\n %s (%s) - %s, %s ann" % (term_name, term, term_cat, term_ann))
        else:
            ax.set_title(title + " %s" % (term))

        if out_pref is not None:
            out_file = "%s%s-prec-rec.pdf" % (out_pref, term)
            savefig(out_file)


def plot_scatter(df, measure='fmax', out_pref="test", title="", ax=None, **kwargs):
    # change the index to the terms
    df.set_index(df.columns[0], inplace=True)
    algs = df['Algorithm'].unique()
    df2 = df[[measure, 'Algorithm']]
    if kwargs['term_stats'] is not None:
        df_stats = kwargs['term_stats'] 
        # change the index to the terms
        df_stats.set_index(df_stats.columns[0], inplace=True)
        # plot the fmax by the # annotations
        df2['num_ann'] = df_stats['# positive examples']

        # now plot the # annotations on the x axis, and the fmax on the y axis
        ax = sns.scatterplot('num_ann', measure, hue='Algorithm', data=df2, ax=ax,
                             linewidth=0,)
        ax.set_title(title)
    else:
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

        # move the plots down a bit so the title isn't overalpping the subplots
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(title)

    if out_pref is not None:
        out_file = "%s%s-scatter.pdf" % (out_pref, measure)
        savefig(out_file, **kwargs)


def results_overview(df, measures=['fmax']):
    """
    Print an overview of the number of values / terms, as well as the median fmax
    """
    print("net_version\texp_name\tmeasure\talg\tmedian\t# terms")
    #for plot_exp_name in sorted(df['plot_exp_name'].unique()):
    for plot_exp_name in df['plot_exp_name'].unique():
        df_curr = df[df['plot_exp_name'] == plot_exp_name]
        net_version, exp_name = df_curr['net_version'].unique()[0], df_curr['exp_name'].unique()[0]
        # limit the goterms to those that are also present for SinkSource(?)
        for measure in measures:
            for alg in sorted(df_curr['Algorithm'].unique()):
                df_alg = df_curr[df_curr['Algorithm'] == alg][measure]
                #print("%s\t%s\t%s\t%0.3f\t%d" % (plot_exp_name, measure, alg, df_alg.median(), len(df_alg)))
                print("%s\t%s\t%s\t%s\t%0.3f\t%d" % (net_version, exp_name, measure, alg, df_alg.median(), len(df_alg)))


def get_algs_to_run(alg_settings, **kwargs):
    # if there aren't any algs specified by the command line (i.e., kwargs),
    # then use whatever is in the config file
    if kwargs['algs'] is None:
        algs_to_run = run_eval_algs.get_algs_to_run(alg_settings)
        kwargs['algs'] = [a.lower() for a in algs_to_run]
        print("\nNo algs were specified. Using the algorithms in the yaml file:")
        print(str(kwargs['algs']))
        if len(algs_to_run) == 0:
            print("ERROR: Must specify algs with --alg or by setting 'should_run' to [True] in the config file")
            sys.exit("Quitting")
    else:
        # make the alg names lower so capitalization won't make a difference
        kwargs['algs'] = [a.lower() for a in kwargs['algs']]
    return kwargs['algs']


def load_all_results(input_settings, alg_settings, output_settings, prec_rec_str="", **kwargs):
    """
    Load all of the results for the datasets and algs specified in the config file
    """
    df_all = pd.DataFrame()
    algs = get_algs_to_run(alg_settings, **kwargs)
    for dataset in input_settings['datasets']:
        for alg in algs:
            alg_params = alg_settings[alg]
            curr_seed = kwargs.get('cv_seed')
            if 'cv-' in kwargs['exp_type']:
                for rep in range(1,kwargs.get('num_reps',1)+1):
                    if curr_seed is not None:
                        curr_seed += rep-1
                    curr_exp_type = "%s-rep%s%s" % (kwargs['exp_type'], rep, 
                            "-seed%s" % (curr_seed) if curr_seed is not None else "")
                    df = load_alg_results(
                        dataset, alg, alg_params, prec_rec_str=prec_rec_str,
                        results_dir=output_settings['output_dir'], **kwargs,  #exp_type=curr_exp_type,
                        #only_terms=kwargs.get('only_terms'), postfix=kwargs.get('postfix',''),
                    )
                    add_dataset_settings(dataset, df) 
                    df['rep'] = rep
                    df_all = pd.concat([df_all, df])
            else:
                df = load_alg_results(
                    dataset, alg, alg_params, prec_rec_str=prec_rec_str, 
                    results_dir=output_settings['output_dir'], **kwargs,  #exp_type=kwargs['exp_type'],
                    #only_terms=kwargs.get('only_terms'), postfix=kwargs.get('postfix',''),
                )
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


def load_alg_results(
        dataset, alg, alg_params, prec_rec_str="", 
        results_dir='outputs', exp_type='cv-5folds', 
        only_terms=None, postfix='', **kwargs):
    """
    For a given dataset and algorithm, build the file path and load the results
    *prec_rec_str*: postfix to change file name. Usually 'prec-rec' if loading precision recal values
    *results_dir*: the base output directory
    *exp_type*: The string specifying the evaluation type. For example: 'cv-5folds' or 'th' for temporal holdout
    *terms*: a set of terms for which to limit the output
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
        cv_file = "%s/%s/%s%s%s%s.txt" % (out_dir, alg, exp_type, params_str, postfix, prec_rec_str)
        if not os.path.isfile(cv_file):
            print("\tnot found %s - skipping" % (cv_file))
            continue
        print("\treading %s" % (cv_file))
        df = pd.read_csv(cv_file, sep='\t')
        # hack to get the script to plot just the parameter value
        if kwargs.get('compare_param') is not None:
            df['Algorithm'] = str(param_combo[kwargs['compare_param']])
            df['alg_name'] = alg_name
        elif len(combos) == 1: 
            df['Algorithm'] = alg_name
        else:
            df['Algorithm'] = alg_name + params_str
        df_all = pd.concat([df_all, df])
    return df_all


if __name__ == "__main__":
    config_map, kwargs = parse_args()

    main(config_map, **kwargs)
