# Compare the given algorithms and network combinations using a heatmap
# this script, along with some manual effort in inkscape, was used to make Figure 4 in the main paper

from collections import defaultdict
import itertools
import argparse
import os, sys
from tqdm import tqdm
import numpy as np
#import utils.file_utils as utils
# also compute the significance of sinksource vs local
from scipy.stats import kruskal, mannwhitneyu, wilcoxon
# plotting imports
import matplotlib
matplotlib.use('Agg')  # To save files remotely.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# make this the default for now
sns.set_style('darkgrid')
# my local imports
fss_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0,fss_dir)
from src.plot import plot_utils
from src.utils import stats_utils
#from src.evaluate import eval_leave_one_species_out as eval_loso
#from src import setup_sparse_networks as setup
#import run_eval_algs


measure_map = plot_utils.measure_map


def main(config_map, ax=None, out_pref='', measures=['fmax'], **kwargs):
    # copy the alg names over from plot utils
    kwargs['alg_names'] = plot_utils.ALG_NAMES
    input_settings, alg_settings, output_settings, out_pref, kwargs = plot_utils.setup_variables(
        config_map, out_pref, **kwargs)

    # load the results using plot_utils
    df_all = plot_utils.load_all_results(
        input_settings, alg_settings, output_settings, **kwargs)

    # since we want to compare to localplus for many of the network combinations,
    # its easier to set them all to the baseline than check for the right plot_exp_name later on
    if 'Local' in df_all['Algorithm'].unique() \
       or 'localplus' in df_all['Algorithm'].unique():
        print("Replacing all localplus values with those for the baseline (ssnT)")
        df_temp = df_all[df_all['Algorithm'] != 'Local']
        # get only the values for ssnT
        df_local = df_all[(df_all['plot_exp_name'] == 'ssnT') & (df_all['Algorithm'] == 'Local')]
        if len(df_local) == 0:
            print("\tWARNING: 'ssnT' not found in results. Skipping")
        else:
            # and append those values for every other plot_exp_name
            for plot_exp_name in df_all['plot_exp_name'].unique():
                df_local['plot_exp_name'] = plot_exp_name
                df_temp = pd.concat([df_temp, df_local])
            print("\t%d total rows became %d" % (len(df_all), len(df_temp)))
            df_all = df_temp

    # limit to the specified datasets
    # TODO run once per dataset
    # if only one is specified, use that
    if len(input_settings['datasets']) == 1:
        kwargs['plot_exp_name'] = list(df_all['plot_exp_name'].unique())[0]
        print("setting --plot-exp-name to %s" % (kwargs['plot_exp_name'])) 
        dataset = input_settings['datasets'][0]
        df = df_all
    else:
        if not kwargs.get('plot_exp_name'):
            kwargs['plot_exp_name'] = list(df_all['plot_exp_name'].unique())[1]
            print("setting --plot-exp-name to %s" % (kwargs['plot_exp_name'])) 
        plot_exp_names = [kwargs['plot_exp_name']] 
        if kwargs.get('plot_exp_name2'):
            plot_exp_names.append(kwargs['plot_exp_name2'])

        print("limitting results to these exp_names: '%s'" % ("', '".join(plot_exp_names)))
        df = df_all[df_all['plot_exp_name'].isin(plot_exp_names)]

        # get the settings of the dataset for the specified plot_exp_name
        for curr_dataset in input_settings['datasets']:
            if curr_dataset['plot_exp_name'] == kwargs['plot_exp_name']:
                dataset = curr_dataset
                break

    if len(df) == 0:
        sys.exit("No results match the specified settings. Quitting")

    print("\t%d algorithms, %d plot_exp_name values\n" % (df_all['Algorithm'].nunique(), len(df_all['plot_exp_name'].unique())))
    plot_utils.results_overview(df_all, measures=measures)

    out_dir = "outputs/viz/loso/%s-%s/" % (
            dataset['net_version'].split('/')[-1], dataset['exp_name'].split('/')[-1])
    os.makedirs(out_dir, exist_ok=True)

    for measure in measures:
        #print("Creating plots for '%s'" % (measure))
        generate_plots(df, out_dir=out_dir, measure=measure, **kwargs)
    #compare_algs()
    #results_overview(ev_code_results, measures=measures, **kwargs) 
    #generate_plots(df, **kwargs) 


def generate_plots(df, out_dir=None, **kwargs):

    plot_exp_name1 = kwargs['plot_exp_name']
    if kwargs.get('plot_exp_name2'):
        plot_exp_name2 = kwargs['plot_exp_name2']
        plot_exp_name_combinations = [
            (plot_exp_name1, plot_exp_name2, "Reds"),
            (plot_exp_name2, plot_exp_name1, "Blues")]
    else:
        plot_exp_name2 = plot_exp_name1
        plot_exp_name_combinations = [
            (plot_exp_name1, plot_exp_name2, "Reds")]
            #(plot_exp_name1, plot_exp_name2, "Blues")]

    for exp1, exp2, cmap in plot_exp_name_combinations:
        df_diffs, df_sig_sp = eval_stat_sig(df, exp1, exp2, **kwargs)
        # store this as a heatmap
        if out_dir is not None:
            out_file = "%s/%s-%s-%dalgs%s.txt" % (
                out_dir,exp1,exp2,df['Algorithm'].nunique(),
                kwargs.get('plot_postfix',''))
            print("writing %s" % (out_file))
            df_diffs.to_csv(out_file)

            ax = sns.heatmap(df_diffs.round(2)*100, cmap=cmap, vmin=0, annot=True)
            ax.set_ylabel(exp1)
            ax.set_xlabel(exp2)
            out_file = out_file.replace('.txt','.pdf')
            print("writing %s" % (out_file))
            plt.tight_layout()
            plt.savefig(out_file, bbox_inches='tight')
            plt.close()


def eval_stat_sig(
        df_orig, exp1, exp2, measure='fmax', **kwargs):
    """
    *exp1*: plot_exp_name1 is the first network combination to compare
    *exp2*: plot_exp_name2 is the second network combination to compare
    """
    df = df_orig.copy()
    # reset the index to the sp-term pairs to make sure we're comparing the same thing across algorithms and experiments
    df['taxon-goid'] = df['#taxon'].astype(int).astype(str) + '-' + df['#goid']
    df.set_index('taxon-goid', inplace=True)

    # subset the dataframe to the specified alg and plot exp name
    diffs = defaultdict(dict)
    #pval_correction = kwargs.get('pval_correction','BH') 
    num_sig_sp = defaultdict(dict)
    #alg1 = kwargs['alg_names'].get(alg1, alg1)
    algs = df['Algorithm'].unique()
    for i, alg1 in enumerate(algs):
        for j, alg2 in enumerate(algs):
            if i == j and exp1 == exp2:
                continue
            df1 = df[(df['plot_exp_name'] == exp1) & (df['Algorithm'] == alg1)]
            df2 = df[(df['plot_exp_name'] == exp2) & (df['Algorithm'] == alg2)]
            # make sure they line up
            df1 = df1.loc[df1.index.isin(df2.index)]
            df2 = df2.loc[df2.index.isin(df1.index)]
            # now compute the percentage of sp-term pairs for which one method is better than the other and vice versa
            diff = df1[measure] - df2[measure]
            perc_improvement = sum(diff > 0) / float(len(diff))
            #eq = sum(diff == 0) / float(len(diff))
            key1 = alg1
            key2 = alg2
            diffs[key1][key2] = perc_improvement

            # also evaluate the # sig species
            sp_pval, sp_qval = calc_sig_species(
                df1, df2, measure=measure, **kwargs)
            # now check how many are significant
            sig_species = set(s for s, qval in sp_qval.items() if qval < 0.05)
            #print("\t%d/%d species with qval < 0.05" % (len(sig_species), len(sp_qval)))
            num_sig_sp[key1][key2] = len(sig_species)
            #df1_fmax = df1[measure].values
            #df2_fmax = df2[measure].values
            #test_statistic, pval = wilcoxon(df1_fmax, df2_fmax, alternative='greater') 
            #print("alg1\texp1\talg2\texp2\t# sp-term pairs\tpval")
            #print("%s\t%s\t%s\t%s\t%s\t%0.3e" % (
            #    alg1, exp1, alg2, exp2, len(df1), pval))
    df_diffs = pd.DataFrame(diffs).T.round(3)
    # make sure the rows and columns are aligned
    df_diffs = df_diffs[df_diffs.index]
    print("\n%s (rows) vs %s (cols)" % (exp1, exp2))
    print("Percentage improvement on the rows:")
    print(df_diffs)

    df_sig_sp = pd.DataFrame(num_sig_sp).T
    # make sure the rows and columns are aligned
    df_sig_sp = df_sig_sp[df_sig_sp.index]
    print("\nOut of %d total species, the # with a qval < 0.05 are:" % (len(sp_qval)))
    print(df_sig_sp)

    return df_diffs, df_sig_sp


def calc_sig_species(
        df1, df2, measure='fmax',
        sp_term_cutoff=5, pval_correction='BH', **kwargs):
    """
    Compute the number of species for which there is a statistically significant improvement
        (measured by the one-sided Wilcoxon signed-rank test)
        of the (fmax) values in df1 vs df2.

    *sp_term_cutoff*: Minimum number of terms for a species to be considered
    *correction*: Multiple hypothesis testing correction to use.
        Options: BH (Benjamini-Hochberg), BF (Bonferroni). Default: BH
    
    *returns*: two dictionaries with the taxon ID as the key, and as the value:
        1. the pval 
        2. the correctd pval/qval
    """
    sp_pval = {}
    ## each taxon gets its own line
    #taxon_table = defaultdict(list)
    # compare individual species
    curr_species = df1['#taxon'].unique()
    # limit the species for which we run the test to those that have at least 5 terms
    species_with_terms = set([s for s, df_s in df1.groupby('#taxon') \
                            if df_s['#goid'].nunique() >= sp_term_cutoff])
    #print("%d species with at least %d terms" % (
    #    len(species_with_terms), sp_term_cutoff))
    for s in curr_species:
        #name = kwargs['sp_names'].get(str(s),'-') if 'sp_names' in kwargs else '-'
        a1_fmax = df1[df1['#taxon'] == s][measure]
        a2_fmax = df2[df2['#taxon'] == s][measure]
        pval = None
        if s in species_with_terms:
            try:
                #test_statistic, pval = mannwhitneyu(a1_fmax, a2_fmax, alternative='greater') 
                # TODO requires an updated versino of scipy
                test_statistic, pval = wilcoxon(
                    a1_fmax, a2_fmax, alternative='greater') 
                #taxon_table[s] += [
                #    name, alg1, alg2,
                #    "%0.3f"%a1_fmax.median(), "%0.3f"%a2_fmax.median(),
                #    "%0.2e"%pval]
            except ValueError:
                pass
        if pval is None:
        #    taxon_table[s] += [
        #        name, alg1, alg2,
        #        "%0.3f"%a1_fmax.median(), "%0.3f"%a2_fmax.median(), '-']
            pval = 1
        sp_pval[s] = pval

    # now correct the p-values
    corrected_sp_pval = {}
    if pval_correction == 'BH':
        sp_sorted_by_pval = sorted(sp_pval, key=sp_pval.get)
        pvals = [sp_pval[s] for s in sp_sorted_by_pval]
        qvals = list(stats_utils.calc_benjamini_hochberg_corrections(pvals))
        corrected_sp_pval = {s: qvals[i] for i, s in enumerate(sp_sorted_by_pval)}
    else:
        # use a regular bonferroni correction
        corrected_sp_pval = {s: pval*len(sp_pval) for s,pval in sp_pval.items()}
    return sp_pval, corrected_sp_pval


def setup_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description='Script for setting up various string experiments')

        parser.add_argument('--config', required=True,
            help='Configuration file')
    group = parser.add_argument_group('Additional options')
    # group.add_argument('--alg1', default="fastsinksource",
    #         help="First algorithm for the scatterplot")
    # group.add_argument('--alg2', default="localplus",
    #         help="Second alg for the scatterplot")
    # group.add_argument('--compare-taxon-median', action='store_true', default=False,
    #         help="Summarize the fmax scatterplot by taking the median for each taxon.")
    group.add_argument('--plot-exp-name',
            help="Limit the results to the specified plot-exp-name. Not needed if only one is in the config file.")
    group.add_argument('--plot-exp-name2',
            help="If specified, compare the algorithms in the first experiment name to those in the second.")
    group.add_argument('--pval-correction', default="BH",
           help="Type of multiple-hypothesis correction when computing # sig species. " + \
           "Options: BH (Benjamini Hochberg), BF (Bonferroni). Default=BH")

    return parser


def parse_args():
    parser = plot_utils.setup_opts()
    parser = setup_parser(parser)
    args = parser.parse_args()
    kwargs = vars(args)
    config_map = plot_utils.load_config_file(kwargs['config'])
    # TODO check to make sure the inputs are correct in config_map

    #if opts.exp_name is None or opts.pos_neg_file is None:
    #    print("--exp-name, --pos-neg-file, required")
    #    sys.exit(1)
    if kwargs['measure'] is None:
        kwargs['measure'] = ['fmax']
    kwargs['measures'] = kwargs['measure']
    del kwargs['measure']

    return config_map, kwargs


if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map, **kwargs)
