# Beautify plots
# Reference:
# - https://matplotlib.org/api/matplotlib_configuration_api.html#matplotlib.rc
# - https://matplotlib.org/tutorials/introductory/customizing.html#customizing-with-matplotlibrc-files

import matplotlib as mpl

mpl.rc('lines', linewidth=3)

axes = { 'titleweight': 'bold',
         'titlesize': 14,
         'labelsize': 14,
         'linewidth': 2.5
       }


mpl.rc('axes',   **axes) # pass in the axes dict as kwargs

font = {#'family' : 'monospace',
        'weight' : 'bold',
        'size'   : 12 }

mpl.rc('font', **font)  # pass in the font dict as kwargs

