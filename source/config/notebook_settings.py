# flake8: noqa
import logging
import logging.config
import pandas as pd
import numpy as np
import math
import re
import os
import sys
import json
import random
import pickle as pkl
import plotly.express as px
from pathlib import Path
import helpsk as hlp
import helpsk.plot as hlpp

import plotly.io as pio
from helpsk.logging import Timer

import glob
import pprint as pp
import textwrap
import wordcloud
#import spacy
import nltk

pio.renderers.default='notebook'

os.chdir('/code')

logging.config.fileConfig(
    "source/config/logging_to_file.conf",
    defaults={'logfilename': 'output/log.log'},
    disable_existing_loggers=False
)

# https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html#available-options
pd.options.display.max_columns = 30 # default 20
pd.options.display.max_rows = 60 # default 60
pd.options.display.float_format = '{:.2f}'.format
# pd.options.display.precision = 2
pd.options.display.max_colwidth = 200 # default 50; -1 = all
# otherwise text between $ signs will be interpreted as formula and printed in italic
pd.set_option('display.html.use_mathjax', False)

# np.set_printoptions(edgeitems=3) # default 3

import matplotlib
from matplotlib import pyplot as plt

figure_size = (hlpp.STANDARD_WIDTH / 1.25, hlpp.STANDARD_HEIGHT / 1.25)

plot_params = {
    'figure.figsize': figure_size, 
    'axes.labelsize': 'medium',
    'axes.titlesize': 'medium',
    'xtick.labelsize': 'medium',
    'ytick.labelsize':'medium',
    'figure.dpi': 100,
}
# adjust matplotlib defaults
matplotlib.rcParams.update(plot_params)

import seaborn as sns
sns.set_style("darkgrid")


from typing import Callable, Optional
import numpy as np
from helpsk.string import format_number
import matplotlib.pyplot as plt
import arviz as az

def plot_hdi(
        samples,
        title: Optional[str] = None,
        transformation: Optional[Callable] = None,
        vertical_factor_66: int = 1,
        vertical_factor_95: int = 1,
        ticks: Optional[list] = None,
        decimals: int = 1):
    """
    Plot the 66% and 95% Hight Density Interval of `samples` passed in. The min, median, and max
    is also plotted.

    Args:
        samples:
            np.array of integers or floats
        title:
            title of the plot
        transformation:
            transformation to do for displaying the min, median, max, and 66% and 95% low and high
            values for the HDI.
    """
    sim_min, sim_median, sim_max = np.quantile(samples, q=[0, 0.5, 1])
    sim_95_hdi_prob_low, sim_95_hdi_prob_hi = az.hdi(samples, hdi_prob=0.95)
    sim_66_hdi_prob_low, sim_66_hdi_prob_hi = az.hdi(samples, hdi_prob=0.66)

    def plot_text(x, label, above=True, factor=1):
        y = 0.018 + (0.025 * (factor - 1))
        if not above:
            y *= -1

        if transformation:
            x_formatted = format_number(transformation(x), places=decimals)
        else:
            x_formatted = format_number(x, places=decimals)

        return plt.text(
            x=x, y=y,
            s=f"{label}:\n{x_formatted}",
            ha='center', va='center', fontsize=9,
        )

    fig, ax = plt.subplots(1)
    plt.plot([sim_min, sim_max], [0, 0], color='gray')
    plt.plot([sim_95_hdi_prob_low, sim_95_hdi_prob_hi], [0, 0], color='black', linewidth=3)
    plt.plot([sim_66_hdi_prob_low, sim_66_hdi_prob_hi], [0, 0], color='black', linewidth=7)
    plt.plot(sim_median, 0, 'o', markersize=15, color='black')
    ax.set_yticklabels([])

    plot_text(x=sim_min, label='min')
    plot_text(x=sim_max, label="max")
    plot_text(x=sim_median, label="median", above=False)
    plot_text(x=sim_66_hdi_prob_low, label="HDI 66", factor=vertical_factor_66)
    plot_text(x=sim_66_hdi_prob_hi, label="HDI 66", factor=vertical_factor_66)
    plot_text(x=sim_95_hdi_prob_low, label="HDI 95", above=False, factor=vertical_factor_95)
    plot_text(x=sim_95_hdi_prob_hi, label="HDI 95", above=False, factor=vertical_factor_95)

    if ticks:
        plt.xticks(ticks)

    if transformation:
        labels = [item.get_text() for item in ax.get_xticklabels()]
        for index in range(len(labels)):
            labels[index] = format_number(
                transformation(float(labels[index].replace('−', '-'))),
                places=decimals
            )
        ax.set_xticklabels(labels)

    fig.set_size_inches(w=7, h=2.2)
    # plt.xlim((115, 195))
    if not title:
        title = "HDI"
    plt.suptitle(title)


import pandas as pd
def percentiles(values, p=[0, 0.01, 0.05, 0.5, 0.95, 0.99, 1]) -> pd.DataFrame:
    return pd.DataFrame(dict(
        percentile=p,
        value=np.quantile(values, q=p)
    ))
