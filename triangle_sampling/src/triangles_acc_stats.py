#!/usr/bin/env python

# Mabel Zhang
# 12 Sep 2015
#
# Plot accuracy vs sampling density
#


# Python
import os
import argparse

# Numpy
import numpy as np

# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# My packages
from tactile_collect import tactile_config
from util.ansi_colors import ansi_colors


def main ():

  arg_parser = argparse.ArgumentParser ()

  # Set to True to upload to ICRA. (You can't view the plot in OS X)
  # Set to False if want to see the plot for debugging.
  arg_parser.add_argument ('--truetype', action='store_true', default=False,
    help='Tell matplotlib to generate TrueType 42 font, instead of rasterized Type 3 font. Specify this flag for uploading to ICRA.')
  arg_parser.add_argument ('--notitle', action='store_true', default=False,
    help='Do not plot titles, for paper figures, description should all be in caption.')

  args = arg_parser.parse_args ()


  # For ICRA PDF font compliance. No Type 3 font (rasterized) allowed
  #   Ref: http://phyletica.org/matplotlib-fonts/
  # You can do this in code, or edit matplotlibrc. But problem with matplotlibrc
  #   is that it's permanent. When you export EPS using TrueType (42), Mac OS X
  #   cannot convert to PDF. So you won't be able to view the file you
  #   outputted! Better to do it in code therefore.
  #   >>> import matplotlib
  #   >>> print matplotlib.matplotlib_fname()
  #   Ref: http://matplotlib.1069221.n5.nabble.com/Location-matplotlibrc-file-on-my-Mac-td24960.html
  if args.truetype:
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

  draw_title = not args.notitle


  #####
  # Define values. Get these from Google Spreadsheet
  #####

  nSamples = [300, 250, 200, 150, 100, 50, 25, 10]
  nSamplesRatio = 0.95

  '''
  out_of = 96

  nn_correct = np.array ([78, 75, 74, 77, 74, 71, 71, 67])
  nn_acc = nn_correct / float (out_of)

  svm_correct = np.array ([80, 82, 82, 83, 81, 73, 78, 68])
  svm_acc = svm_correct / float (out_of)
  '''

  # 100 random split average
  nn_acc = np.array ([0.781667, 0.777396, 0.764583, 0.774583, 0.739896, 0.728333, 0.719375, 0.635625])

  svm_acc = np.array ([0.836354, 0.841458, 0.850938, 0.847812, 0.841458, 0.822604, 0.796771, 0.700625])

  # For plot title
  bins = [10, 10, 10]

  acc_path = tactile_config.config_paths ('custom',
    'triangle_sampling/imgs/acc/')


  #####
  # Init plotting colors
  #####

  # Can set higher numbers when have more things to plot!
  n_colors = 2

  # colormap_name: string from
  #   http://matplotlib.org/examples/color/colormaps_reference.html
  colormap_name = 'PiYG'

  # 11 for max # sensors on any segment of hand (palm has 11)
  colormap = get_cmap (colormap_name, n_colors)

  # Argument here needs to be in range [0, luc], where luc is the 2nd arg
  #   specified to get_cmap().
  #color = colormap (color_idx)



  #####
  # Plot
  #####

  plt.plot (nSamples, nn_acc, 'o', markersize=5,
    markeredgewidth=0, color=colormap(0))
  nn_hdl, = plt.plot (nSamples, nn_acc, '--', color=colormap(0), linewidth=2,
    label='Euclidean 5NN')

  plt.plot (nSamples, svm_acc, 'o', markersize=5,
    markeredgewidth=0, color=colormap(1))
  svm_hdl, = plt.plot (nSamples, svm_acc, color=colormap(1), linewidth=2,
    label='Linear SVM')


  # Ref: http://matplotlib.org/users/legend_guide.html
  plt.legend (handles=[nn_hdl, svm_hdl], loc=4)

  plt.grid (True)

  if draw_title:
    plt.title ('Accuracy vs. Samples per 20 cm Local Neighborhood, \nfor 3D Histograms with Bins %d %d %d' % (bins[0], bins[1], bins[2]))

  # 20 cm is the sampling sphere diameter. Radius is defined in
  #   sample_pcl.cpp. Mult 2 to get diameter.
  plt.xlabel ('Samples per 20 cm Local Neighborhood')
  plt.ylabel ('Accuracy')


  # Save to file
  out_name = os.path.join (acc_path, 'acc.eps')
  plt.savefig (out_name, bbox_inches='tight')
  print ('Plot saved to %s' % out_name)

  plt.show ()


if __name__ == '__main__':
  main ()


