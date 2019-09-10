
# This is a utility script to combine multiple subplots using a config file for each subplot

import os
import sys
#import argparse
#import yaml
#import itertools
#from collections import defaultdict
#from tqdm import tqdm
#import time
#import numpy as np
#from scipy import sparse
import matplotlib
matplotlib.use('Agg')  # To save files remotely.  Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
#import pandas as pd
import seaborn as sns
# make this the default for now
sns.set_style('darkgrid')
#from src.algorithms import runner
#from src.utils import file_utils
#from src.evaluate import eval_utils
#from src.plot import plot_utils
import src.plot.plot_utils as plot_utils


def main(config_map, **kwargs):
    """
    For every config file specified, create the corresponding plot, 
    and put it in the specified grid location
    """
    plt_kwargs = config_map.get('plt_kwargs', {})
    plot_settings = config_map['plot_settings']
    kwargs.update(plot_settings)
    subplot_kwargs = kwargs.copy()
    # don't create figures for the individual plots
    del subplot_kwargs['out_pref']
    for measure in kwargs['measures']:
        subplot_kwargs['measures'] = [measure]

        # setup the grid
        grid_size = (plt_kwargs['nRows'], plt_kwargs['nCols'])
        #gs = plt.gridspec.GridSpec()
        grid_locs = plot_settings['grid_loc']
        figsize = tuple(plt_kwargs['figsize'])
        plt.figure(figsize=figsize)
        sharey=None
        axes = []

        # create the subplots and add them
        for i, config_file in enumerate(plot_settings['config_files']):
            grid_loc = tuple(grid_locs[i])
            print(grid_loc, config_file)
            #continue
            ax = plt.subplot2grid(grid_size, grid_loc, sharey=sharey)
            curr_config_map = plot_utils.load_config_file(config_file)
            ax = plot_utils.main(curr_config_map, ax=ax, out_pref=None, **subplot_kwargs)
            if i == 0:
                sharey = ax
            axes.append(ax)

        plt.tight_layout()
        # TODO add a letter to the top-left of the ax
        # now write to a file
        out_file = "%s%s.pdf" % (kwargs['out_pref'], measure)
        plot_utils.savefig(out_file, **kwargs)


if __name__ == "__main__":
    config_map, kwargs = plot_utils.parse_args()

    main(config_map, **kwargs)
