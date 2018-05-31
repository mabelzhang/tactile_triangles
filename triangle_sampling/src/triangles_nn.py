#!/usr/bin/env python

# Mabel Zhang
# 4 Sep 2015
#
# Python version of triangles_nn.m.
#
# Loads all histograms and compute nearest neighbors using scikit-learn.
#
# This script uses argparse to handle command line arguments.
#   usage: triangles_nn.py [-h] [-k K] [--meta META] [--exhaustive EXHAUSTIVE]
# Example:
#   To get per-class confusion matrix (usually you want this):
#     rosrun triangle_sampling triangles_nn.py
#   To get per-object confusion matrix:
#     rosrun triangle_sampling triangles_nn.py --exhaustive 1
#
#
# Ref:
#   Python nearest neighbors:
#     Tutorial and theory http://scikit-learn.org/stable/modules/neighbors.html
#     Class API http://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors
#     Fn API http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors
#   Python confusion matrix:
#     Tutorial http://scikit-learn.org/stable/auto_examples/plot_confusion_matrix.html
#


# ROS
import rospkg

# Python
import argparse
import os

# Numpy
import numpy as np
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier  # for knn
from sklearn.cross_validation import train_test_split
from sklearn.metrics.pairwise import chi2_kernel  # For chisquare dist for hists

import matplotlib

# My packages
from tactile_collect import tactile_config
from util.ansi_colors import ansi_colors
from triangle_sampling.load_hists import load_hists
from util.classification_tools import draw_confusion_matrix, calc_accuracy, \
  draw_confusion_matrix_per_sample
from triangle_sampling.metrics import hist_inter_dist, l2_dist, chisqr_dist, \
  inner_prod_dist
from triangle_sampling.config_paths import get_sampling_subpath, \
  parse_args_for_svm, config_hist_paths_from_args
from util.matplotlib_util import custom_colormap_neon


# Parameters:
#   hists: nSamples x nDims NumPy 2D array
#   lbls: Labels. Used only for printing results and drawing plot
def calc_knn_dists (hists, lbls, catnames, knn_k, img_path, img_suff,
  draw_title, metric=None,
  print_nn=True, exhaustive=False, exhaustive_ticks=False, bg_color='white'):

  nSamples = np.shape (hists) [0]
 
  # Calc knn distances. This is good for nSamples x nSamples confusion matrix
  #   plots. Temperature color indicates distance. Hot = short distance,
  #   cold = long distance.
  if exhaustive:
    knn_k = nSamples
    print ('%sknn_k set to %d, for exhaustive pairwise distances%s' %\
      (ansi_colors.OKCYAN, knn_k, ansi_colors.ENDC))
  [knn_idx, knn_dists] = calc_knn (hists, knn_k, metric)

  # Print results
  if print_nn:
    for i in range (0, np.shape (knn_idx) [0]):
 
      print ('%d (truth %s):' % (i, catnames[lbls[i]])),
 
      # 1st NN is itself, so start with 2nd NN.
      for j in range (1, knn_k):
 
        curr_nn_idx = knn_idx [i] [j]
 
        # Make sure you don't have custom_catids, if you do, then thsi will need
        #   to change. The index will not simply be lbls[i] and lbls[curr_nn_idx],
        #   need to map from custom_catids back to the 0:n non-custom catids
        #   somehow..
        print ('%dNN %d %s,' % (j + 1, curr_nn_idx,
          catnames [lbls [curr_nn_idx]])),
      print ('')


  # If not exhaustive, then did not get all nSamples distances from calc_knn,
  #   cannot draw object-vs-object confusion matrix.
  if exhaustive:

    # Fill in the tick strings, with each object's class
    per_sample_ticks = []

    if exhaustive_ticks:

      # User adjust this to plot whatever labels you need

      # For IROS camera-ready paper, only put ticks on x axis, font 40
      #draw_xticks = True
      #draw_yticks = False
      #fontsize = 40

      # For IROS 2016 camera-ready video, put ticks on x and y axes, font 20
      draw_xticks = True
      draw_yticks = True
      fontsize = 25

      for i in range (0, len (lbls)):
        per_sample_ticks.append (catnames [lbls [i]])

    else:
      draw_xticks = False
      draw_yticks = False
      fontsize = 25

    # Use jet_r. Reverse colors, so that small dists are hot, far dists are cold
    #   http://stackoverflow.com/questions/3279560/invert-colormap-in-matplotlib
    #cmap_name = 'jet_r'
    # Use my custom neon colormap. Uncomment the (4, 3, 66) dark indigo to make
    #   diagonal a darker color, looks nicer
    cmap_name = custom_colormap_neon () + '_r'

    img_name_obj = os.path.join (img_path, 'nn_per_sample' + img_suff + '.eps')
    draw_confusion_matrix_per_sample (knn_idx, knn_dists, per_sample_ticks,
      img_name=img_name_obj, title_prefix='NN ', draw_title=draw_title,
      draw_xylbls=False, draw_xticks=draw_xticks, draw_yticks=draw_yticks,
      fontsize=fontsize, bg_color=bg_color, cmap_name=cmap_name)


# Parameters:
#   hists: nSamples by d. d = size of histogram
def calc_knn (hists, k, metric=None):

  # Tutorial http://scikit-learn.org/stable/modules/neighbors.html
  # Class API http://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors
  # Function API http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors
  if metric:
    nbrs = NearestNeighbors (n_neighbors=k, algorithm='auto',
       metric=metric).fit (hists)
  else:
    nbrs = NearestNeighbors (n_neighbors=k, algorithm='auto').fit (hists)

  knn_dists, knn_idx = nbrs.kneighbors (hists)

  #print (knn_idx)
  #print (knn_dists)

  return (knn_idx, knn_dists)


# Parameters:
#   samples: numpy array, n x d
#   lbls: list or numpy array, n x 1
#   idx_start_hold_as_test: Row index in samples matrix. This row and all rows
#     after are to be held as test data. This is useful for testing on real
#     robot data, while training on synthetic data. You'll need to know which
#     index the robot data start, remember Python index is 0 based.
#     If you have 192 synthetic objects, then real objects start at [192].
# Ref:
#   2D NN classifier example (not useful for high dimensional data, other than 3 lines): http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#example-neighbors-plot-classification-py
#   Class API: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
def knn_classify (samples, lbls, k, idx_start_hold_as_test=None,
  random_state=0):

  #####
  # Split into train and test sets
  #####

  # If not specified, just pass all samples to train test split
  if not idx_start_hold_as_test:
    # n
    idx_start_hold_as_test = np.shape (samples) [0]

  # Default test ratio is 0.25
  test_size = 0.5

  if not idx_start_hold_as_test:
    samples_tr, samples_te, lbls_tr, lbls_te = train_test_split (samples, lbls,
      test_size=test_size, random_state=random_state)

  # Hold out real-robot data
  else:
    samples_tr, samples_te, lbls_tr, lbls_te = train_test_split (
      samples[0:idx_start_hold_as_test, :], lbls[0:idx_start_hold_as_test],
      test_size=test_size, random_state=random_state)

    # Append the held out data rows at the end of test split
    samples_te = np.append (samples_te, samples[idx_start_hold_as_test:, :], axis=0)
    lbls_te = np.append (lbls_te, lbls[idx_start_hold_as_test:], axis=0)
    lbls_te = lbls_te.astype (int)


  # Find the original indices of test samples, so caller can see what objects
  #   are misclassified, by tracing back to the input object.
  idx_te = []
  for i in range (0, len (lbls_te)):
    # http://stackoverflow.com/questions/25823608/find-matching-rows-in-2-dimensional-numpy-array
    # axis=1 specifies look for rows that are same
    idx_te.append (np.where ((samples == samples_te [i, :]).all (axis=1)) \
      [0] [0])


  #####
  # Classify
  #####

  # Default metric='minowski' and p=2, which gives Euclidean distance
  clf = KNeighborsClassifier (k, 'distance')

  # Failed metrics:
  # Everything is predicted as "tool" or "hammer". Not sure why, so just use
  #   default.
  #clf = KNeighborsClassifier (k, 'distance', metric=chi2_kernel)
  # Everything is predicted as first class. Probably didn't implement this
  #   correctly... But if I don't use the negative, then most things are 
  #   predicted as teapots, with a few other categories lit up in confusion
  #   matrix too, but accuracy is 0/96....
  #clf = KNeighborsClassifier (k, 'distance', metric=neg_hist_inter_1d)
  # Almost everything is classified as class 0, accuracy 5/96
  #clf = KNeighborsClassifier (k, 'distance', metric=neg_chi2_kernel)

  # Train the classifier
  clf.fit (samples_tr, lbls_tr)

  # Test the classifier (run classification)
  lbls_pred = clf.predict (samples_te)

  return (idx_te, lbls_te, lbls_pred)


def main ():

  #####
  # User adjust param
  #####

  # Index at which real robot data start. All data including and after this
  #   row will be held as test data. 0 based.
  idx_start_real_robot = 192
  #idx_start_real_robot = 7
 

  #####
  # Parse command line args
  #   Ref: Tutorial https://docs.python.org/2/howto/argparse.html
  #        Full API https://docs.python.org/dev/library/argparse.html
  #####

  arg_parser = argparse.ArgumentParser ()

  #arg_parser.add_argument ('nSamples', type=int,
  #  help='nSamples used when triangles were sampled, used to create directory name to read from.')
  #arg_parser.add_argument ('nSamplesRatio', type=float,
  #  help='nSamplesRatio used when triangles were sampled, used to create directory name to read from.')

  arg_parser.add_argument ('-k', type=int, default=5,
    help='Integer. k number of neighbors in kNN')

  # store_true is for boolean flags with no arguments
  arg_parser.add_argument ('--exhaustive', action='store_true', default=False,
    help='Boolean flag. Do kNN on ALL samples, so you get (n-1) x (n-1) distances printout. If True, overrides -k.')
  arg_parser.add_argument ('--exhaustive_ticks', action='store_true', default=False,
    help='Boolean flag. Draw confusion matrix with class names. Do not use this if you have many objects! Suitable for debugging small number of objects.')

  # I moved iros 2016 files into their own subdirectory, to keep them apart
  #   and safe from later work and wiping!
  arg_parser.add_argument ('--iros2016', action='store_true', default=False,
    help='Boolean flag. Load files in iros2016 subdirectory, to generate Fig 8 in IROS 2016 paper.')
  arg_parser.add_argument ('--black_bg', action='store_true', default=False,
    help='Boolean flag. Plot with black background, useful for black presentation slides.')

  args, valid = parse_args_for_svm (arg_parser)
  if not valid:
    return


  #####
  # Parse args for each mode
  #####

  sampling_subpath = ''

  if args.pcd:

    sampling_subpath, tri_nbins_subpath, hist_parent_path, hist_path, \
      img_mode_suffix, tri_suffix, _ = \
      config_hist_paths_from_args (args)

  elif args.real:

    _, tri_nbins_subpath, hist_parent_path, hist_path, \
      img_mode_suffix, tri_suffix, _ = \
        config_hist_paths_from_args (args)


  metafile_name = args.meta

  exhaustive = args.exhaustive
  if exhaustive:
    # Fill in later
    knn_k = -1
  else:
    knn_k = args.k

  exhaustive_ticks = args.exhaustive_ticks

  doKDE = args.kde
  img_kde_suffix = ''
  if doKDE:
    img_kde_suffix = '_kde'

  isSynthetic = not args.real

  rand_splits = args.rand_splits

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

  if args.black_bg:
    bg_color = 'black'
  else:
    bg_color = 'white'


  print ('%sSettings: knn_k = %d (-1 if exhaustive, will be set later). Exhaustive = %s. Meta file = %s. Do KDE = %s. isSynthetic = %s. rand_splits = %s. TrueType font = %s. Plot titles = %s %s' % \
    (ansi_colors.OKCYAN, knn_k, exhaustive, metafile_name, doKDE, isSynthetic,
     rand_splits, args.truetype, draw_title, ansi_colors.ENDC))


  #####
  # Init params
  #####

  # These are same as sample_pcl.cpp
  rospack = rospkg.RosPack ()
  pkg_path = rospack.get_path ('triangle_sampling')
  model_metafile_path = os.path.join (pkg_path, 'config/', metafile_name)

  if doKDE:
    hist_subpath = 'csv_kde'
  else:
    if args.pcd:
      hist_subpath = 'csv_hists'
    elif args.real:
      hist_subpath = 'csv_bx_hists'
    # Haven't tested any other cases
    else:
      return

  img_path = tactile_config.config_paths ('custom',
    'triangle_sampling/imgs/conf_mats/')



  #####
  # Read inputs
  #####

  print ('Loading descriptor data from %s' % hist_path)
  [samples, lbls, catnames, _, catids, _] = load_hists ( \
    model_metafile_path, hist_path, tri_suffix=tri_suffix,
    mixed_paths=False, sampling_subpath=sampling_subpath,
    tri_nbins_subpath=tri_nbins_subpath)

  print ('Dimensions of data (d, in n x d):')
  print (np.shape (samples))

  nSamples = np.shape (samples) [0]
  nBins = np.shape (samples) [1]

  print ('%d samples, %d bins' % (nSamples, nBins))
  print ('Truths in all data (train and test)')

  # Print out the class name for each object, for seeing what the category of
  #   a closest neighor is.
  for i in range (0, len (lbls)):
    # Make sure you don't have custom_catids, if you do, then thsi will need
    #   to change. The index will not simply be lbls[i],
    #   need to map from custom_catids back to the 0:n non-custom catids
    #   somehow..
    print ('%d: %s' % (i, catnames [lbls [i]]))

  if isSynthetic:
    # For synthetic data, no data is held out from splitting
    idx_start_real_robot = nSamples
  # As of Jul 2016, we don't do train-PCL test-real-robot anymore, so no 
  #   holding out data. Run on real.txt, not models_and_real.txt.
  else:
    idx_start_real_robot = 0


  #####
  # Run classification
  #####

  img_suff = img_mode_suffix + img_kde_suffix
  if sampling_subpath:
    img_suff += ('_' + sampling_subpath)
  if tri_nbins_subpath: 
    img_suff += ('_' + tri_nbins_subpath)


  calc_knn_dists (samples, lbls, catnames, knn_k, img_path, img_suff, draw_title,
    metric=chisqr_dist, #inner_prod_dist, #l2_dist, #hist_inter_dist, #None,
    print_nn=True, exhaustive=exhaustive, exhaustive_ticks=exhaustive_ticks,
    bg_color=bg_color)


  # Classify using knn. This is good for nClasses x nClasses confusion matrix
  #   plots. Temperature color indicates number of samples classified correct.
  #   Hot = more correct, cold = fewer correct.

  # Run once, no random
  if not rand_splits:
    n_random_splits = 1
    # Specify a fixed number, so no randomness
    random_state = 0

  # Run 10 times, random splits, get average accuracy
  else:
    n_random_splits = 100
    # Specify None, so random each call
    # Ref: http://stackoverflow.com/questions/28064634/random-state-pseudo-random-numberin-scikit-learn
    random_state = None

  accs = []


  # Don't run classification with exhaustive... might kill my computer
  if not exhaustive:

    for i in range (0, n_random_splits):

      idx_test, lbls_test, lbls_pred = knn_classify (samples, lbls, knn_k,
        idx_start_real_robot, random_state=random_state)
    
      print ('Truths and Predictions for test data:')
      for i in range (0, len (idx_test)):
        print ('%d (truth %s): predicted %s' % (idx_test [i],
          catnames [lbls_test [i]], catnames [lbls_pred [i]]))
  
      # Calculate accuracy
      accs.append (calc_accuracy (lbls_test, lbls_pred, len(catids)))
  
    # Draw confusion matrix (if there are multiple random splits, this is for
    #   the very last run)
    img_name_cl = os.path.join (img_path, 'nn_per_class' + img_kde_suffix + '_' + \
      sampling_subpath + '.eps')
    draw_confusion_matrix (lbls_test, lbls_pred, catnames, img_name=img_name_cl,
      title_prefix='NN ', draw_title=draw_title, bg_color=bg_color)

    if rand_splits:
      print ('%sAverage accuracy over %d runs of random splits: %f %s' % \
        (ansi_colors.OKCYAN, n_random_splits, np.mean (np.asarray (accs)),
        ansi_colors.ENDC))


if __name__ == '__main__':
  main ()

