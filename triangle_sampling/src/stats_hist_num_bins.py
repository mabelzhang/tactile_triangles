#!/usr/bin/env python

# Mabel Zhang
# 23 Jan 2016
#
# Number of bins in histogram vs. classification accuracy.
#
# Only implemented for Gazebo trained data.
#
# Calls hist_conf_writer.py, triangles_reader.py, triangles_svm.py,
#   in that order, using the various histogram bin and triangle parameters
#   defined in variables.
#   If those files' argparse interface change, just change the args here
#   accordingly, on the lines that call them using subprocess.call().
#
# Usage:
#   $ rosrun triangle_sampling stats_hist_num_bins.py --gazebo [--use_existing_hists]
#

# Python
import argparse
import subprocess
import csv
import os

import numpy as np

# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# My packages
from tactile_collect import tactile_config
from util.ansi_colors import ansi_colors
from util.matplotlib_util import black_background, black_legend, \
  custom_colormap_neon
from triangle_sampling.config_paths import get_sampling_subpath, \
  get_nbins_acc_stats_name, get_img_path
from triangle_sampling.config_hist_params import TriangleHistogramParams as \
  HistP


# Modified from triangles_acc_stats.py, which has simple plotting code
def plot_line (xdata, ydata, title, xlbl, ylbl, out_name, color, lbl,
  stdev=None, do_save=True, black_bg=False):

  #####
  # Plot
  #####

  plt.plot (xdata, ydata, 'o', markersize=5,
    markeredgewidth=0, color=color)
  #hdl, = 
  plt.plot (xdata, ydata, '-', linewidth=2, color=color, label=lbl)

  # Plot error bars
  #   http://matplotlib.org/1.2.1/examples/pylab_examples/errorbar_demo.html
  #   Plot error bar without line, fmt='':
  #   http://stackoverflow.com/questions/18498742/how-do-you-make-an-errorbar-plot-in-matplotlib-using-linestyle-none-in-rcparams
  if stdev:
    plt.errorbar (xdata, ydata, fmt='', capthick=2, yerr=stdev, color=color)

  plt.grid (True, color='gray')
  #plt.grid (False)

  if title:
    plt.title (title)
  plt.xlabel (xlbl)
  plt.ylabel (ylbl)

  if black_bg:
    black_background ()


  # Save to file
  if do_save:
    plt.savefig (out_name, bbox_inches='tight')
    print ('Plot saved to %s' % out_name)

  #plt.show ()

  #return hdl



def main ():

  #####
  # User adjust parameter
  #####

  # Set to a high number when don't want to prune anything
  prune_thresh_l = 0.5

  # Bin configurations to test and get stats on
  # For quick testing
  #nbins_list = [10]  # ICRA configuration
  #nbins_list = [10, 12]
  #nbins_list = [8, 10, 12, 14]
  #nbins_list = [8, 10]
  #nbins_list = [6, 8, 10, 12, 14, 16]
  #nbins_list = [18, 20, 22, 24, 26]
  # IROS
  nbins_list = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]

  # Pick triangle parameters
  # For quick testing
  #prs_list = [(HistP.L0, HistP.L1, HistP.A0)]  # ICRA configuration
  # IROS
  prs_list = [ \
    (HistP.A0, HistP.A1, HistP.L0),
    (HistP.A1, HistP.A2, HistP.L2),
    (HistP.L0, HistP.L1, HistP.A0),
    (HistP.L1, HistP.L2, HistP.A2),
    (HistP.L0, HistP.L1, HistP.L2)
  ]

  # Thorough test, all possibilities
  # 5 possible combinations for solving a triangle:
  #   1. Given 2 angles, and a side that's NOT btw the 2 angles
  #   2. Given 2 angles, and the side that IS btw the 2 angles
  #   3. Given 2 sides, and the angle btw the 2 sides
  #   4. Given 2 sides, and an angle NOT in btw the 2 sides
  #   5. Given 3 sides
  # 1 and 2 can be combined to this case: a# a# l#, because we don't record
  #   which side is btw which angles in training. That requires extra
  #   book-keeping, you'd have to say, always fix on the largest angle, side
  #   CCW is side 1, angle CCW is angle 2, etc. But in 3D, CW and CCW are
  #   arbitrary, it's not easy to define.
  #   We can alternate the parameters, e.g. a0 a1 l0, a0 a2 l0... that's a lot.
  # 3 and 4 can be combined to this case: l# l# a#.
  # 5 is its own case, l0 l1 l2.
  # So we have 3 general cases. For first two, rotate parameters.
  #   Order doesn't matter, because which one you use on which histogram
  #   dimension is arbitrary, and SVM doesn't care. So no permutation needed,
  #   only iterate on combination.
  # Ref https://www.mathsisfun.com/algebra/trig-solving-triangles.html
  '''
  prs_list = [ \
    # Case 1: a# a# l#
    (HistP.A0, HistP.A1, HistP.L0),
    (HistP.A0, HistP.A2, HistP.L0),
    (HistP.A1, HistP.A2, HistP.L0),
    #
    (HistP.A0, HistP.A1, HistP.L1),
    (HistP.A0, HistP.A2, HistP.L1),
    (HistP.A1, HistP.A2, HistP.L1),
    #
    (HistP.A0, HistP.A1, HistP.L2),
    (HistP.A0, HistP.A2, HistP.L2),
    (HistP.A1, HistP.A2, HistP.L2),
    # Case 2: l# l# a#
    (HistP.L0, HistP.L1, HistP.A0),  # ICRA paper case
    (HistP.L0, HistP.L2, HistP.A0),
    (HistP.L1, HistP.L2, HistP.A0),
    #
    (HistP.L0, HistP.L1, HistP.A1),
    (HistP.L0, HistP.L2, HistP.A1),
    (HistP.L1, HistP.L2, HistP.A1),
    #
    (HistP.L0, HistP.L1, HistP.A2),
    (HistP.L0, HistP.L2, HistP.A2),
    (HistP.L1, HistP.L2, HistP.A2),
    # Case 3: l0 l1 l2
    (HistP.L0, HistP.L1, HistP.L2)
  ]
  '''


  #####
  # Parse command line args
  #   Ref: Tutorial https://docs.python.org/2/howto/argparse.html
  #        Full API https://docs.python.org/dev/library/argparse.html
  #####

  arg_parser = argparse.ArgumentParser ()

  arg_parser.add_argument ('--pcd', action='store_true', default=False,
    help='Run on PCL files')
  arg_parser.add_argument ('--gazebo', action='store_true', default=False,
    help='Run on Gazebo files')

  arg_parser.add_argument ('histSubdirParam1', type=str,
    help='Used to create directory name to read from.\n' + \
      'Only used for point cloud (--pcd mode). nSamples used when triangles were sampled.\n')
  arg_parser.add_argument ('histSubdirParam2', type=str,
    help='Used to create directory name to read from.\n' + \
      'Only used for point cloud (--pcd mode). nSamplesRatio used when triangles were sampled.\n')

  arg_parser.add_argument ('--meta', type=str, default='models_gazebo_csv.txt',
    help='String. Base name of meta list file in triangle_sampling/config directory')

  arg_parser.add_argument ('--use_existing_hists', action='store_true',
    default=False,
    help='Specify this to skip generating files, useful if you already generated all pruned triangles and histograms in a previous run, and just want to test SVM on different files that are already on disk.')
  arg_parser.add_argument ('--plot_existing_accs', action='store_true',
    default=False,
    help='Specify this to plot directly the accuracies already stored in nbins_vs_acc_<triParams>.csv files. Nothing will be run, just plotting from a chart.')
  arg_parser.add_argument ('--append_to_existing_accs', action='store_true',
    default=False, help='Specify this to append accuracy results from SVM to existing nbins_vs_acc_*.csv files, instead of overwriting them. Choose this so that you can experiment with different parameters individually, then simply concatenate them together, instead of having to rerun every single one again.')

  # Set to True to upload to ICRA. (You can't view the plot in OS X Preview)
  # Set to False if want to see the plot for debugging.
  arg_parser.add_argument ('--truetype', action='store_true', default=False,
    help='Tell matplotlib to generate TrueType 42 font, instead of rasterized Type 3 font. Specify this flag for uploading to ICRA.')
  arg_parser.add_argument ('--notitle', action='store_true', default=False,
    help='Do not plot titles, for paper figures, description should all be in caption.')

  arg_parser.add_argument ('--black_bg', action='store_true', default=False,
    help='Boolean flag. Plot with black background, useful for black presentation slides.')

  args = arg_parser.parse_args ()

  if args.gazebo:
    hist_subpath = 'csv_gz_hists'

    sampling_subpath = ''

    hist_path = tactile_config.config_paths ('custom',
      os.path.join ('triangle_sampling', hist_subpath))

    mode_suff = '_gz'

  elif args.pcd:

    hist_subpath = 'csv_hists'

    if not args.histSubdirParam1 or not args.histSubdirParam2:
      print ('%sIn --pcd mode, must specify histSubdirParam1 and histSubdirParam2. Check your args and retry. Terminating...%s' % ( \
      ansi_colors.FAIL, ansi_colors.ENDC))
      return

    nSamples = int (args.histSubdirParam1)
    nSamplesRatio = float (args.histSubdirParam2)
    sampling_subpath = get_sampling_subpath (nSamples, nSamplesRatio, 
      endSlash=False)

    hist_path = tactile_config.config_paths ('custom',
      os.path.join ('triangle_sampling', hist_subpath, sampling_subpath))

    mode_suff = '_pcd'


  use_existing_hists = args.use_existing_hists
  if args.use_existing_hists:
    print ('%suse_existing_hists flag set to %s, will skip generating files and just run SVM.%s' % ( \
      ansi_colors.OKCYAN, use_existing_hists, ansi_colors.ENDC))

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
  # Run SVM for different histogram bin sizes
  #####

  if not args.plot_existing_accs:

    # Overwrite file in first iter, append to file in subsequent iters
    write_new_file = ''
    if not args.append_to_existing_accs:
      write_new_file = '--overwrite_stats'
 
    # Loop through each triangle parameters choice
    for p_i in range (0, len (prs_list)):
 
      prs_str = '%s,%s,%s' % prs_list[p_i]
 
      # Loop through each bin size choice
      for b_i in range (0, len (nbins_list)):
 
        nbins_str = '%d,%d,%d' % (nbins_list [b_i], nbins_list [b_i],
          nbins_list [b_i])
 
 
        if not use_existing_hists:
 
          # subprocess.call() prints stuff to rospy.loginfo, AND returns.
          # subprocess.call() waits till process finishes.
          # subprocess.Popen() doesn't wait.
          # Ref: https://docs.python.org/2/library/subprocess.html
         
          #####
          # rosrun triangle_sampling hist_conf_writer.py 0 0 --nbins #
          #   This prunes triangles above a length threshold, saves the triangle
          #     data after pruning, and saves hist_conf.csv file with min/max
          #     ranges of triangles data for calculating histograms later.
          #####
         
          # Don't need to run for PCL data, all triangles should already be
          #   generated - I currently do that manually by running
          #   sample_pcl_calc_hist.py.
          if args.gazebo:
            #   Pass in --nbins, to write to hist_conf.csv file
            prune_args = ['rosrun', 'triangle_sampling',
              'hist_conf_writer.py', '0', '0',
              '--no_prune',
              '--long_csv_path', '--meta', args.meta,
              '--nbins', nbins_str,
              '--prs', prs_str]
           
            if args.gazebo:
              prune_args.append ('--gazebo')
          
            # Not using this for prune.py right now. Not sure how to best
            #   implement this feature. Prune would need to check every
            #   triangle file exists, which is okay, but what about histogram
            #   ranges? It'd have to take current hist_conf.csv's range, and
            #   compare to any new triangle files that it does load! That's a
            #   a mess! I rather just run the program! It only takes a few
            #   seconds.
            #if use_existing_hists:
            #  prune_args.append ('--use_existing_hists')
          
            p = subprocess.call (prune_args)
       
          #####
          # rosrun triangle_sampling triangles_reader.py 0 0 --gazebo
          #   This reads pruned triangles data, calculates histograms, and saves
          #     the histograms to file.
          #####
         
          # Don't need to run for PCL data, all triangles should already be
          #   generated - I currently do that manually by running
          #   sample_pcl_calc_hist.py.
          if args.gazebo:
            tri_reader_args = ['rosrun', 'triangle_sampling', 'triangles_reader.py',
              prs_str, nbins_str]
            tri_reader_args.append ('--gazebo')
         
            # Generate histograms from triangle files
            p = subprocess.call (tri_reader_args)
 
        else:
          print ('%sSkipping prune_triangle_outliers.py and triangles_reader.py, because --use_existing_hists was specified. Make sure this is what you want! Any errors of missing files can be due to this flag, if you do not have existing files yet! Currently we do not automatically check for that, as this is a convenience feature up to user to decide to use.%s' % (ansi_colors.OKCYAN, ansi_colors.ENDC))
  
        # end if not use_existing_hists
 
  
        #####
        # rosrun triangle_sampling triangles_svm.py 0 0 --rand_splits --meta
        #   models_active_test.txt --write_stats [--overwrite_stats] --gazebo
        #
        #   This loads the histograms data, which are object descriptors, and runs
        #     SVM classification. It saves nbins vs. accuracy data to stats.
        #####
  
        svm_args = ['rosrun', 'triangle_sampling', 'triangles_svm.py',
          '--rand_splits', '--truetype', '--notitle',
          '--meta', args.meta,
          '--write_stats']
  
        # After first call to triangles_svm.py, which wrote a new file, next
        #   ones should append to the same file.
        if b_i == 0:
          if write_new_file:
            svm_args.append (write_new_file)

        if args.pcd:
          svm_args.append ('--pcd')
          # These 4 args need to be together in this order, to be parsed
          #   correctly
          svm_args.append (args.histSubdirParam1)
          svm_args.append (args.histSubdirParam2)
          svm_args.append (prs_str)
          svm_args.append (nbins_str)
        elif args.gazebo:
          svm_args.append ('--gazebo')
          svm_args.append (prs_str)
          svm_args.append (nbins_str)
  
        # Run SVM, write stats to a csv file
        p = subprocess.call (svm_args)

  # end if not args.plot_existing_accs


  #####
  # Load ALL accuracy vs nbins csv data, for all triangle parameter choices,
  #   plot each triangle parameter choice as a curve.
  #####

  # Init plotting colors

  # Can set higher numbers when have more things to plot!
  n_colors = len (prs_list)

  # colormap_name: string from
  #   http://matplotlib.org/examples/color/colormaps_reference.html
  colormap_name = 'jet'
  # This colormap pops out more brilliantly on black background
  if args.black_bg:
    #colormap_name = 'Set1'
    colormap_name = custom_colormap_neon ()

  colormap = get_cmap (colormap_name, n_colors)

  # Generate image name. Only need one image, plot all
  #   triangle parameter choices in one graph, in different colors.
  stats_plot_path = get_img_path ('nbins_triparams_acc')
  # Discard directory name, just want the base name from this
  stats_plot_base = get_nbins_acc_stats_name ('',
    mode_suff, '')
  # Get base name, replace .csv extension with .eps
  stats_plot_base = os.path.splitext (os.path.basename (stats_plot_base)) [0] + '.eps'
  stats_plot_name = os.path.join (stats_plot_path, stats_plot_base)

  #hdls = []

  # Loop through each triangle param
  for p_i in range (0, len (prs_list)):

    # Make sure arg after % is a tuple, not list. Can use tuple() to cast.
    prs_str = '%s,%s,%s' % prs_list[p_i]

    # Argument here needs to be in range [0, luc], where luc is the 2nd arg
    #   specified to get_cmap().
    color = colormap (p_i)


    #####
    # Load ONE accuracy vs nbins csv data, plot it.
    #####

    # file name e.g. <hist_path>/_stats/nbins_vs_acc_l0l1a0.csv
    b_stats_name = get_nbins_acc_stats_name (hist_path, mode_suff,
      prs_str)

    if not os.path.exists (b_stats_name):
      print ('%sFile not found: %s. Has it been created?%s' % (
        ansi_colors.FAIL, b_stats_name, ansi_colors.ENDC))
      return

    # Dictionary of (nbins, accuracy) key-value pair
    nbins_accs = {}

    print ('Loading average accuracy from one of the files generated: %s' % ( \
      b_stats_name))

    # Read csv stats
    with open (b_stats_name, 'rb') as b_stats_file:
      b_stats_reader = csv.DictReader (b_stats_file)

      for row in b_stats_reader:

        nbins0 = float (row [HistP.BINS3D_TITLES [0]])
        nbins1 = float (row [HistP.BINS3D_TITLES [1]])
        nbins2 = float (row [HistP.BINS3D_TITLES [2]])

        # Assumption: Use same number of bins for all dimensions, for generality
        assert ((nbins0 == nbins1) and (nbins1 == nbins2) and (nbins2 == nbins0))

        if nbins0 not in nbins_accs.keys ():
          nbins_accs [nbins0] = np.zeros ((0, ))
        nbins_accs [nbins0] = np.append (nbins_accs [nbins0],
          float (row ['acc']))

    # List of floats. Average accuracies
    avg_accs = []
    stdevs = []

    # Sort bin counts, so can use them as x-axis values, in order
    nbins_loaded = np.sort (nbins_accs.keys ())
    print ('nbins  avg_acc  stdev')
    # Append accuracies, in the order of sorted bin counts
    for nbins in nbins_loaded:
      # Calc avg accuracy
      avg_accs.append (np.mean (nbins_accs [nbins]))
      # Calc stdev
      stdevs.append (np.std (nbins_accs [nbins]))

      print (' %3d   %.4f   %.4f' % (nbins, avg_accs[len(avg_accs)-1],
        stdevs[len(stdevs)-1]))

    print ('Average standard deviation across all data: %.4f' % (
      np.mean (stdevs)))

    # Plot
    title = ''
    if draw_title:
      title = 'Accuracy vs. Number of Histogram Bins',
    plot_line (nbins_loaded, avg_accs, title,
      'Number of bins per 3D histogram dimension',
      'Average accuracy over 100 random train-test splits',
      stats_plot_name, color, prs_str, stdev=stdevs, do_save=False,
      black_bg=args.black_bg)
    #hdls.append (hdl)

  # Ref: http://matplotlib.org/users/legend_guide.html
  legend = plt.legend (loc=4)
  # Set color to black bg, white text
  if args.black_bg:
    black_legend (legend)

  # Ref savefig() black background: https://stackoverflow.com/questions/4804005/matplotlib-figure-facecolor-background-color
  fig = plt.gcf ()
  plt.savefig (stats_plot_name, bbox_inches='tight',
    facecolor=fig.get_facecolor (), edgecolor='none', transparent=True)
  print ('Plot saved to %s' % stats_plot_name)
  plt.show ()


if __name__ == '__main__':
  main ()

