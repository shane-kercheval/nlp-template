import os
import logging
import logging.config
import pandas as pd
import matplotlib
import seaborn as sns
import helpsk.plot as hlpp
import plotly.io as pio


pio.renderers.default = 'notebook'
os.chdir('/code')

logging.config.fileConfig(
    "source/config/logging_to_file.conf",
    defaults={'logfilename': 'output/log.log'},
    disable_existing_loggers=False
)

# https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html#available-options
pd.options.display.max_columns = 30  # default 20
pd.options.display.max_rows = 60  # default 60
pd.options.display.float_format = '{:.2f}'.format
# pd.options.display.precision = 2
pd.options.display.max_colwidth = 200  # default 50; -1 = all
# otherwise text between $ signs will be interpreted as formula and printed in italic
pd.set_option('display.html.use_mathjax', False)

# np.set_printoptions(edgeitems=3) # default 3

figure_size = (hlpp.STANDARD_WIDTH / 1.25, hlpp.STANDARD_HEIGHT / 1.25)

plot_params = {
    'figure.figsize': figure_size,
    'axes.labelsize': 'medium',
    'axes.titlesize': 'medium',
    'xtick.labelsize': 'medium',
    'ytick.labelsize': 'medium',
    'figure.dpi': 100,
}
# adjust matplotlib defaults
matplotlib.rcParams.update(plot_params)

sns.set_style("darkgrid")
