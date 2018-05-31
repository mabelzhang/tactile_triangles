#!/usr/bin/env python

# Mabel Zhang
# 4 Sep 2015
#
# Scikit-Learn version of split_and_train.py (uses liblinear. Data split is
#   manually coded by me) in tactile_collect package.
#


# ROS
import rospkg

# Python
import os
import csv
import time
import pickle
import argparse

# Numpy
import numpy as np
from sklearn import svm
from sklearn.metrics.pairwise import chi2_kernel  # For chisquare dist for hists
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib  # For saving SVM model to file

import matplotlib

# My packages
from triangle_sampling.load_hists import load_hists, read_hist_config, \
  scale_bin_range_to_decimeter
from tactile_collect import tactile_config
from util.classification_tools import my_train_test_split, \
  draw_confusion_matrix, calc_accuracy
from triangle_sampling.metrics import hist_inter_kernel, hist_inter_dist, \
  hist_minus_hist_inter_dist, \
  kl_divergence_kernel, kl_divergence_dist, inner_prod_kernel
from util.ansi_colors import ansi_colors
from triangle_sampling.config_paths import \
  get_sampling_subpath, get_nbins_acc_stats_name, \
  get_triparams_nbins_subpath, get_svm_model_path, \
  parse_args_for_svm, config_hist_paths_from_args, \
  get_recog_meta_path, get_recog_confmat_path
from triangle_sampling.calc_hists import find_bin_edges_and_volume
from triangle_sampling.plot_hist_rviz import interact_plot_hist
from triangle_sampling.config_hist_params import TriangleHistogramParams as \
  HistP

# Local
from plot_hist_dd import create_subplot, plot_hist_dd_given, show_plot, \
  flatten_hist
from triangles_nn import calc_knn_dists


# Parameters:
#   hist_parent_path: Directory in which triParams_* folders are found.
#   hist_path: Directory in which hist_conf.csv can be found
#   overwrite_stats: If True, will overwrite existing file. Else append to file.
#   avg_acc: Average accuracy over many random splits
#   prs_str: String of three chosen triangle parameters, e.g. 'l0,l1,a0'
def write_stat (hist_parent_path, hist_path, overwrite_stats, accs, #avg_acc,
  mode_suffix):

  # Read histogram config file, for number of bins
  hist_conf_name = os.path.join (hist_path, 'hist_conf.csv')
  ((pr1, pr2, pr3), nbins, bin_range3D) = read_hist_config (hist_conf_name)
  # If decimeters mode, scale up bin range
  #   Don't think you want this anymore. New hist_conf.csv files already have
  #   decimeter bin ranges.
  #if HistP.decimeter:
  #  _, bin_range3D = scale_bin_range_to_decimeter (None, bin_range3D)

  prs_str = '%s,%s,%s' % (pr1, pr2, pr3)

  # If file doesn't exist, then can't append, just (over)write
  #   e.g. .../csv_*hists/_stats/nbins_vs_acc_l0l1a0.csv
  stats_name = get_nbins_acc_stats_name (hist_parent_path, mode_suffix, prs_str)
  if not os.path.exists (stats_name):
    overwrite_stats = True
  print ('Average accuracy will be outputted to %s' % (stats_name))


  # Default to append to file
  open_mode = 'a'
  if overwrite_stats:
    open_mode = 'wb'
    print ('%sOverwriting (or creating if non-existent) nbins vs. accuracy file%s' % ( \
      ansi_colors.OKCYAN, ansi_colors.ENDC))


  # csv header
  column_titles = []
  column_titles.extend (HistP.BINS3D_TITLES)
  #column_titles.append ('avg_acc')
  column_titles.append ('acc')

  stats_file = open (stats_name, open_mode)
  stats_writer = csv.DictWriter (stats_file, fieldnames=column_titles,
    restval='-1')
  # If not overwriting, then header has already been written, don't write it.
  if overwrite_stats:
    stats_writer.writeheader ()

  # Write nbins from hist_conf.csv file. nbins will be on x-axis when plotted
  #row.update (zip (column_titles, [nbins[0], nbins[1], nbins[2], avg_acc]))

  for i in range (0, len (accs)):
    # Each row is a dictionary. Keys are column title, values are floats.
    row = dict ()

    row.update (zip (column_titles, [nbins[0], nbins[1], nbins[2], accs[i]]))
    stats_writer.writerow (row)

  stats_file.close ()


# EXPERIMENTAL. experimental
# Parameters:
#   samples_3d: n x d NumPy matrix. n is the length from reshaping a 3D
#     histogram to row vector, e.g. 10 x 10 x 10.
def convert_samples_to_1dhists (samples_3d, hist_path):

  # Read histogram config file, for number of bins
  hist_conf_name = os.path.join (hist_path, 'hist_conf.csv')
  ((pr1, pr2, pr3), nbins, bin_range3D) = read_hist_config (hist_conf_name)

  # For sanity check only. Checks if the 3D histograms were normalized
  #   correctly by np.histogramdd(). Result was they are correct.
  CHECK_SUM_ONE = False
  edgesdd, _, bin_volume = find_bin_edges_and_volume (nbins, bin_range3D)

  nSamples = np.shape (samples_3d) [0]

  # Concatenation of 3 1D histograms.
  #   Number of dimensions (columns) is the sum of bins in each dimension
  # Pre-allocate, faster than re-allocating and moving memory each time use
  #   np.append().
  samples_1d = np.zeros ((nSamples, np.sum (nbins)))

  for s_i in range (0, nSamples):

    # Sanity check: check that the histograms were normalized correctly by
    #   NumPy histogramdd(), i.e. bin counts (height) * bin width = 1.
    if CHECK_SUM_ONE:
      height = np.sum (samples_3d [s_i, :])
      print ('This should be 1, is it? %.1f' % (height * bin_volume))


    hist_linear = samples_3d [s_i, :]

    # Pre-allocate, faster than re-allocating and moving memory each time use
    #   np.append().
    hist1d = np.zeros ((np.sum (nbins), ))

    for d in range (0, 3):

      # This does the right thing, as long as write_hist_3d.py
      #   write_hist_3d_csv() reshapes the 3D histogram to 1D by
      #   np.reshape(histdd, (histdd.size, )), without any other fancy
      #   parameters. That will give the default reshaping, and to reshape
      #   it back, it's just
      #   np.reshape(hist_linear, histdd.shape), without any params.
      #   histdd.shape is (10,10,10), or nbins.
      histdd = np.reshape (hist_linear, nbins)

      hist1d_d = flatten_hist (histdd, d)

      # Normalize. Copied from plot_hist_dd.py
      #real_width = edgesdd [d] [1] - edgesdd [d] [0]
      #heights = sum (hist1d_d)
      #area = real_width * heights
      #hist1d_d /= area

      if d == 0:
        hist1d [0 : nbins [0]] = hist1d_d
      else:
        hist1d [nbins [d-1] : nbins [d-1] + nbins [d]] = hist1d_d

    samples_1d [s_i, :] = hist1d


  return samples_1d



def main ():

  #####
  # User adjust param
  #####

  # Used for train_sim_test_real=True mode only
  #   Otherwise, this is reset to the number of objects in meta file.
  # Index at which real robot data start. All data including and after this
  #   row will be held as test data. 0 based.
  idx_start_real_robot = 192

  # Show confusion matrix at the end. Disable this if you are running prune.py
  #   to generate accuracy plots from many different histogram bin and
  #   triangle param choices.
  show_plot_at_end = True #False


  # Debug tools flags

  # Show confusion matrix after every random split, for each of the 100 splits!
  #   Useful for detailed debugging.
  show_plot_every_split = False

  # Enable to enter test sample index to see 1d histogram intersectin plots
  interactive_debug = False

  experimental_1dhists = True
  debug_print_all_labels = False

  # Print all histogram intersection distances
  debug_draw_nn_dists = False

  # To find the index of a specific object you are interested in seeing the
  #   histogram of. If it is a test object, it will print in cyan. Then you
  #   can use the interactive_debug=True mode to enter the index and see the
  #   3 flattened 1D histograms - they are saved to file too.
  debug_find_specific_obj = False
  specific_obj_startswith = '6ed884'

  # Set true_cat and pred_cat below, to find objects of a specific
  #   category that are predicted as another specific category. It will print
  #   in cyan.
  debug_find_specific_misclass = False
  specific_lbl = 'hammer'
  specific_wrong_lbl = 'bottle'

  # Only used for active sensing monte carlo tree search, in active_touch
  #   package. Only set to True when you are working with active_touch and are
  #   sure you want to retrain the model.
  # Set this flag to False at other times, so you don't mistakenly overwrite
  #   model file!! Model file is not committed to repo!
  # This will train SVM with probability=True, in order to output probs at
  #   test time.
  save_svm_to_file = True

 
  #####
  # Parse command line args
  #####

  arg_parser = argparse.ArgumentParser ()

  arg_parser.add_argument ('--black_bg', action='store_true', default=False,
    help='Boolean flag. Plot with black background, useful for black presentation slides.')

  args, valid = parse_args_for_svm (arg_parser)
  if not valid:
    return

  if args.black_bg:
    bg_color = 'black'
  else:
    bg_color = 'white'


  #####
  # Parse args for each mode
  #####

  img_mode_suffix = ''

  doKDE = args.kde
  img_kde_suffix = ''
  if doKDE:
    img_kde_suffix = '_kde'

  # Sampling density for PCL
  sampling_subpath = ''
  # Triangle parameters choice, number of histogram bins choice
  tri_nbins_subpath = ''

  meta_train_test_base = ''
  meta_train_base = ''
  meta_test_base = ''

  if args.pcd:

    # Sanity check
    if len (args.histSubdirParam2) != 3:
      print ('%sERROR: Expect histSubdirParam2 to have three elements, for --pcd mode. Check your args and retry.%s' % ( \
        ansi_colors.FAIL, ansi_colors.ENDC))
      return

    sampling_subpath, tri_nbins_subpath, hist_parent_path, hist_path, \
      img_mode_suffix, tri_suffix, _ = \
        config_hist_paths_from_args (args)

    mixed_paths = False

    # One meta file. Let scikit-learn split into train and test randomly
    one_meta_train_test = True
    meta_train_test_base = args.meta

  elif args.gazebo or args.real:

    # Sanity check
    if len (args.histSubdirParam2) != 1:
      print ('%sERROR: Expect histSubdirParam2 to only have one element, for --gazebo mode. Check your args and retry.%s' % ( \
        ansi_colors.FAIL, ansi_colors.ENDC))
      return

    _, tri_nbins_subpath, hist_parent_path, hist_path, \
      img_mode_suffix, tri_suffix, _ = \
        config_hist_paths_from_args (args)

    mixed_paths = False

    # One meta file. Let scikit-learn split into train and test randomly
    one_meta_train_test = True
    meta_train_test_base = args.meta

  # For mixed data, need to specify both kinds of histSubdirParams, a pair for
  #   PCL data's sampling_subpath, a pair for Gazebo data's tri_nbins_subpath.
  elif args.mixed:

    # Sanity check
    if len (args.histSubdirParam2) != 3:
      print ('%sERROR: Expect histSubdirParam2 to have three elements, for --mixed mode. Check your args and retry.%s' % ( \
        ansi_colors.FAIL, ansi_colors.ENDC))
      return

    sampling_subpath, tri_nbins_subpath, hist_parent_path, hist_path, \
      img_mode_suffix, tri_suffix, _ = \
        config_hist_paths_from_args (args)
    #print ('hist_path: %s' % hist_path)
    #print ('  sampling_subpath: %s' % sampling_subpath)

    mixed_paths = True
    train_path = tactile_config.config_paths ('custom', '')

    # Two meta files, for fixed train and test sets. Don't let scikit-learn
    #   split the sets.
    one_meta_train_test = False
    meta_train_base = args.meta_train
    meta_test_base = args.meta_test

  # end if args.pcd or args.real


  train_sim_test_real = False

  # In mixed mode, train on PCL data, test on Gazebo data
  if args.mixed:
    rand_splits = False
    print ('%sIn mixed mode. rand_splits automatically set to FALSE.%s' % (\
      ansi_colors.OKCYAN, ansi_colors.ENDC))
  else:
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


  # For writing average accuracy at the end to a stats file
  #   Used by stats_hist_num_bins.py
  write_stats = args.write_stats
  overwrite_stats = args.overwrite_stats
  # If not even writing stats, overwrite doesn't make sense. Don't overwrite
  if overwrite_stats and not write_stats:
    print ('%s--overwrite_stats was specified, without specifying write_stats. That does not make sense, automatically setting overwrite_stats to False.%s' % (\
    ansi_colors.WARNING, ansi_colors.ENDC))

    overwrite_stats = False

  print ('%sSettings: Do KDE = %s. train_sim_test_real = %s. rand_splits = %s. TrueType font = %s. Plot titles = %s %s' % \
    (ansi_colors.OKCYAN, doKDE, train_sim_test_real, rand_splits,
     args.truetype, draw_title, ansi_colors.ENDC))
  if one_meta_train_test:
    print ('One meta file %s, to split into train and test sets' % meta_train_test_base)
  else:
    print ('%sTwo meta files, train on %s, test on %s%s' % ( \
      ansi_colors.OKCYAN, meta_train_base, meta_test_base, ansi_colors.ENDC))


  # Path to save confusion matrix image to
  img_path = get_recog_confmat_path ()

  img_suff = img_mode_suffix + img_kde_suffix
  if sampling_subpath:
    img_suff += ('_' + sampling_subpath)
  if tri_nbins_subpath: 
    img_suff += ('_' + tri_nbins_subpath)

  img_name = os.path.join (img_path, 'svm' + img_suff + '.eps')


  #####
  # Load objects data
  #####

  print ('Loading descriptor data from %s' % hist_path)

  # One meta file
  if one_meta_train_test:

    meta_train_test_name = os.path.join (get_recog_meta_path (),
      meta_train_test_base)
 
    # Each row of samples is the histogram for one objects
    # lbls is numSamples size list with category integer label for each sample
    [samples, lbls, catnames, catcounts, catids, sample_names] = load_hists ( \
      meta_train_test_name, hist_path, tri_suffix=tri_suffix,
      mixed_paths=False, sampling_subpath=sampling_subpath,
      tri_nbins_subpath=tri_nbins_subpath)
 
    numSamples = np.shape (samples) [0]
    nBins = np.shape (samples) [1]
    print ('%d samples, %d bins' % (numSamples, nBins))
    print ('%d labels' % len(lbls))
 
    if not train_sim_test_real:
      # For training and testing on same domain data, all data go into fair
      #   train-test split, no data is held out.
      idx_start_real_robot = numSamples

  # Load train and test sets from separate files
  # Output: samples_tr, samples_te, lbls_tr, lbls_te, catnames, catids
  else:

    meta_train_name = os.path.join (get_recog_meta_path (),
      meta_train_base)

    # Each row of samples is the histogram for one objects
    # lbls is numSamples size list with category integer label for each sample
    # For now, just assuming this is PCL data only, so pass in
    #   mixed_paths=False, to do the default replace pcd with csv thing.
    [samples_tr, lbls_tr, catnames, catcounts_tr, catids, sample_names_tr] = \
      load_hists ( \
        meta_train_name, train_path, tri_suffix=tri_suffix,
        mixed_paths=mixed_paths, sampling_subpath=sampling_subpath,
        tri_nbins_subpath=tri_nbins_subpath)

    meta_test_name = os.path.join (get_recog_meta_path (),
      meta_test_base)

    # Each row of samples is the histogram for one objects
    # lbls is numSamples size list with category integer label for each sample
    [samples_te, lbls_te_tmp, catnames_te_tmp, catcounts_te_tmp, \
      catids_te_tmp, sample_names_te] = load_hists ( \
        meta_test_name, train_path, tri_suffix=tri_suffix,
        mixed_paths=mixed_paths, sampling_subpath=sampling_subpath,
        tri_nbins_subpath=tri_nbins_subpath)

    print ('%d train samples, %d test samples, %d bins' % ( \
      np.shape (samples_tr) [0], np.shape (samples_te) [0],
      np.shape (samples_tr) [1]))


    # Renumber test set's catids, so they match training set's catids, in case
    #   categories in the two meta files don't come in same order!

    fix_test_catids = False
    for i in range (0, len (catnames_te_tmp)):
      # If catname from test meta doesn't match catname from train meta (i.e.
      #   they're ordered differently, then catids in test meta won't match
      #   catids in train meta, need to rematch)
      if catnames_te_tmp [i] != catnames [i]:
        fix_test_catids = True
        break

    # Don't need to renumber
    if not fix_test_catids:
      lbls_te = lbls_te_tmp

    # Renumber test catids to training set's, by matching catname strings
    # NOT TESTED! Because this never happened in my meta files. Can
    #   deliberately swap the order of categories in a meta file to test
    #   this, when I have more time.
    else:
      print ('%sOrder of categories in test meta file is different from order in train meta file. Re-numbering labels in test samples. This code is NOT TESTED, WATCH CLOSELY for mistakes.%s' % ( \
        ansi_colors.WARNING, ansi_colors.ENDC))

      lbls_te = np.zeros (len (lbls_te_tmp), dtype=int)

      # array [test_idx] = train_idx. Note the value is the INDEX in catnames_*
      #   and catids_*, not the actualy category ID value!
      test_to_train_idx = []

      # Find the index of the categories in training catnames
      for test_idx in range (0, len (catnames_te_tmp)):
        # Index of this test category in training catnames list
        train_idx = catnames.index (catnames_te_tmp [test_idx])
        # array [test_idx] = train_idx
        test_to_train_idx.append (train_idx)

      # Renumber old test labels with the new indices
      # Loop through each old category index
      for test_idx in range (0, len (catids_te_tmp)):

        train_idx = test_to_train_idx [test_idx]

        # Find old test labels that match this old category ID (value, not idx)
        samples_idx = np.where (np.array (lbls_te_tmp) == \
          catids_te_tmp [test_idx])

        # Renumber all these old test labels with the new label ID in training
        lbls_te [samples_idx] = catids [train_idx]

      # Convert NumPy array to list
      lbls_te = lbls_te.tolist ()

    # end if not fix_test_catids

    samples = np.append (samples_tr, samples_te, axis=0)
    lbls = lbls_tr + lbls_te
    sample_names = sample_names_tr + sample_names_te



  #####
  # Testing if 1d hists concatenated work better for --mixed mode, because I
  #   can't figure out why 1d hists look good and classification of train-PCL
  #   test-Gazebo predicts everything as a skinny object - banana, hammer, tool.
  #####

  # EXPERIMENTAL
  if args.mixed and experimental_1dhists:

    print ('%sEXPERIMENTAL run: converting all 3d histograms to concatenation of 3 1d histograms, to see what happens. Disable this for real run!!!%s' % (ansi_colors.OKCYAN, ansi_colors.ENDC))

    samples = convert_samples_to_1dhists ( \
      np.append (samples_tr, samples_te, axis=0), hist_path)

    # First len(lbls_tr) are training samples
    samples_tr = samples [0 : len (lbls_tr), :]
    # Last ones are test samples
    samples_te = samples [len (lbls_tr) : np.shape (samples) [0], :]


    print ('%d training samples, dimension %d' % (np.shape (samples_tr) [0],
      np.shape (samples_tr) [1]))
    print ('%d test samples, dimension %d' % (np.shape (samples_te) [0],
      np.shape (samples_te) [1]))
    print ('')


  #####
  # Draw big confusion matrix of pairwise distances
  #####

  if debug_draw_nn_dists:

    # Sometimes regrouping helps you see it better, sometimes it doesn't.
    # Set it based on your use case
    REGROUP = False

    print ('')
    print ('Plotting confusion matrix of all samples')

    if args.mixed:
      samples = np.append (samples_tr, samples_te, axis=0)
      lbls = lbls_tr + lbls_te

    if REGROUP:
      # Group samples of the same classes together, so conf mat clear diagonal
      #   means good.
      # Preallocate to save time.
      samples_grouped = np.zeros (samples.shape)
      lbls_grouped = np.zeros ((samples.shape [0], ))
     
      lbls_np = np.asarray (lbls)
      nSamples_grouped = 0
     
      nSamples_perCat = []
     
      print ('Regrouped indices just for confusion matrix plot:')
      for i in range (0, len (catids)):
     
        # Find labels that match current catid
        samples_idx = np.where (lbls_np == catids [i])
        nSamples_currCat = samples_idx [0].size
     
        # Print BEFORE updating nSamples_grouped!
        print ('%s: %d to %d' % (catnames [i], nSamples_grouped,
          nSamples_grouped + nSamples_currCat - 1))
     
        # Append to the end of filled rows in lbls_grouped and samples_grouped
        lbls_grouped [nSamples_grouped : \
          nSamples_grouped + nSamples_currCat] = catids [i]
        samples_grouped [nSamples_grouped : \
          nSamples_grouped + nSamples_currCat, :] = samples [samples_idx, :]
     
        nSamples_grouped += nSamples_currCat
        nSamples_perCat.append (nSamples_currCat)
     
      lbls_grouped = lbls_grouped.tolist ()

    else:
      samples_grouped = samples
      lbls_grouped = lbls

    calc_knn_dists (samples_grouped, lbls_grouped, catnames, len(lbls),
      img_path, img_suff, draw_title=draw_title,
      metric=None,
      #metric=hist_inter_dist,
      #metric=hist_minus_hist_inter_dist,
      #metric=kl_divergence_dist,
      print_nn=False, exhaustive=True, exhaustive_ticks=False)


  #####
  # Main loop
  #####

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


  # Print out all the input path and labels, to manually inspect they're
  #   correct
  if debug_print_all_labels:
    lbls = lbls_tr + lbls_te
    sample_names = sample_names_tr + sample_names_te
    for i in range (0, len (lbls)):
 
      last_dir = os.path.split (os.path.dirname (sample_names [i])) [1]
      base = os.path.basename (sample_names [i])
 
      print ('%s: %s' % (catnames [lbls [i]], last_dir + '/' + base))


  # Seconds
  start_time = time.time ()

  for sp_i in range (0, n_random_splits):

    print ('%sRandom split %d%s' % (ansi_colors.OKCYAN, sp_i, ansi_colors.ENDC))

    if one_meta_train_test:

      #####
      # Split into train and test sets
      #   Ref: http://scikit-learn.org/stable/auto_examples/plot_confusion_matrix.html
      #####
     
      # Default test ratio is 0.25
      # lbls are Python lists
      # Use my_train_test_split(), to split satisfsying INDIVIDUAL-CATEGORY
      #   ratio. Split such that 0.5 per category is used for train, 0.5 per
      #   category is used for test. This is better if you have very few data,
      #   when sklearn's train_test_split() can end up giving you 0 items for
      #   test data, or 1 for train and 4 for test!
      # Otherwise, you can also call sklearn's train_test_split() many times,
      #   once for every category, and then sort out the indices yourself...
      samples_tr, samples_te, lbls_tr, lbls_te = my_train_test_split (
      # Use sklearn's train_test_split(), to split satisfying OVERALL ratio,
      #   no regard to how many are train and test in individual categories.
      #samples_tr, samples_te, lbls_tr, lbls_te = train_test_split (
        samples[0:idx_start_real_robot, :], lbls[0:idx_start_real_robot],
        test_size=0.5, random_state=random_state)
     
      # Append the held out data rows at the end of test split
      #   Use NumPy arrays for easier manipulation
      samples_te = np.append (samples_te, samples[idx_start_real_robot:, :], axis=0)
      lbls_te = np.append (lbls_te, lbls[idx_start_real_robot:], axis=0)
      lbls_te = lbls_te.astype (int)


      # Find the original indices of test samples, so caller can see what
      #   objects are misclassified, by tracing back to the input object.
      idx_te = []
      for l_i in range (0, len (lbls_te)):
        # http://stackoverflow.com/questions/25823608/find-matching-rows-in-2-dimensional-numpy-array
        # axis=1 specifies look for rows that are same
        idx_te.append (np.where ((samples == samples_te [l_i, :]).all (axis=1)) \
          [0] [0])

      #print ('lbls_tr:')
      #print (lbls_tr)
      #print (type (lbls_tr))
      #print ('lbls_te:')
      #print (lbls_te)
      #print (type (lbls_te))

      # Debug output
      print ('Total train %d instances, test %d instances' % ( \
        np.size (lbls_tr), np.size (lbls_te)))
      for c_i in range (0, len (catnames)):
        #print (np.where (np.array (lbls_tr) == catids [c_i]))
        #print (np.where (np.array (lbls_te) == catids [c_i]))
        n_tr = np.size (np.where (np.array (lbls_tr) == catids [c_i]))
        n_te = np.size (np.where (np.array (lbls_te) == catids [c_i]))

        print ('%s: train %d instances, test %d instances' % (catnames [c_i],
          n_tr, n_te))

        if n_te == 0:
          print ('Test set for category %s has size 0' % ( \
            catnames [c_i]))
 

    # samples_tr, samples_te, lbls_tr, lbls_te already loaded before the loop.
    else:
      # Treat samples as (samples_tr, samples_te). Then test indices are just
      #   (0 : num test samples), plus offset of (num train samples).
      #idx_te = range (0, len (lbls_te))
      idx_te = range (len (lbls_tr), len (lbls_tr) + len (lbls_te))

    # end if one_meta_train_test

 
    #####
    # Run classification and draw confusion matrix
    #   Ref: http://scikit-learn.org/stable/auto_examples/plot_confusion_matrix.html
    #####

    # Run classifier
    # The confusion matrix example had kernel='linear'. Keeping the default
    #   ('rbf') didn't work, all classes get classified as class 5! For some
    #   reason. So will use linear.
    # 'ovr': One-vs-rest, this is needed to get correct probabilities at test
    #   time. Default option is 'ovo' one-vs-one, which is deprecated, and
    #   outputs (n_samples, n_classes * (n_classes-1) / 2), which requires
    #   voting to predict result. 'ovr' does what SVM base its results on, no
    #   voting.
    # Ref: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    classifier = svm.SVC (kernel='linear', decision_function_shape='ovr')

    # This works fast and similar to 'linear' kernel, but only tested on a
    #   sphere and a cube.
    # 20 Jun 2016. For active touch, inner product kernel is better, `.`
    #   distance is [0, 1], makes more sense for matching an incremental
    #   histogram to a known one. For 3cm-radius cube and sphere, inner prod
    #   distance NN produced better results than SVM with 'linear' kernel.
    #classifier = svm.SVC (kernel=inner_prod_kernel)

    # This works too
    # My custom kernel from metrics.py. Only gets 80/96, when 'linear' gets
    #   81/96. So might as well use linear.
    #classifier = svm.SVC (kernel=hist_inter_kernel)

    # This doesn't work at all. Terrible results
    # My custom kernel. This is slow because I can't figure out how to do it
    #   without nested for-loops, because ai and bi need to be different for
    #   each of bj and aj, to take care of division by 0!
    #classifier = svm.SVC (kernel=kl_divergence_kernel)
 
    # Failed kernels:
    # All classes get classified as class 5 too!
    #classifier = svm.SVC (kernel=chi2_kernel)
    # All classes get classified as class 5 too!
    #classifier = svm.SVC (kernel=neg_chi2_kernel)


    # Only used for active sensing monte carlo tree search, in active_touch
    #   package. At other times, set this flag to False so you don't mistakenly
    #   overwrite model file!! Model file is not committed to repo!
    if save_svm_to_file:

      # Train SVM with probabilities
      #   http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
      classifier.probability = True
      # Refit with probabilities
      classifier_probs_model = classifier.fit (samples_tr, lbls_tr)
      # nSamples x nClasses
      probs_pred = classifier_probs_model.predict_proba (samples_te)

      svm_name, svm_lbls_name = get_svm_model_path (hist_path, img_mode_suffix)

      print ('%sOutputting SVM model to %s%s' % (
        ansi_colors.OKCYAN, svm_name, ansi_colors.ENDC))
      print ('%sOutputting SVM labels to %s%s' % (
        ansi_colors.OKCYAN, svm_lbls_name, ansi_colors.ENDC))

      # Save trained model to file
      # Ref joblib: http://stackoverflow.com/questions/10592605/save-classifier-to-disk-in-scikit-learn
      joblib.dump (classifier_probs_model, svm_name, compress=9)

      # Save correspondence of {integer_label: class_name} to file, so
      #   reader of SVM model can interpret what classes do the predicted
      #   labels represent!
      svm_lbls = {catids[c_i] : catnames[c_i] for c_i in range (0, len (catids))}
      with open (svm_lbls_name, 'wb') as f:
        # HIGHEST_PROTOCOL is binary, good performance. 0 is text format,
        #   can use for debugging.
        pickle.dump (svm_lbls, f, pickle.HIGHEST_PROTOCOL)

      #print ('Probabilities predicted by SVM:')
      #print ('class1 class2 ...')
      #print (probs_pred)

    # end if save_svm_to_file

 
    classifier_model = classifier.fit (samples_tr, lbls_tr)
    lbls_pred = classifier_model.predict (samples_te)
    # I think this works too, predicting from the model trained with
    #   probabilities
    #lbls_pred = classifier_probs_model.predict (samples_te)

    print ('Truths and Predictions for test data:')
    print (' id name     (truth cat): predicted cat')
    for i_i in range (0, len (idx_te)):

      sample_base = os.path.basename (sample_names [idx_te [i_i]])
      short_sample_name = sample_base [0 : min (len (sample_base), 8)]

      true_cat = catnames [lbls_te [i_i]]
      pred_cat = catnames [lbls_pred [i_i]]

      # To help me find a specific object to plot for paper figure
      if debug_find_specific_obj and specific_obj_startswith:
        if sample_base.startswith (specific_obj_startswith):
           #or sample_base.startswith ('109d55'):
          print ('%s%s is idx %d%s' % (ansi_colors.OKCYAN,
            sample_base, idx_te [i_i], ansi_colors.ENDC))

      if debug_find_specific_misclass:
        if true_cat == specific_lbl and pred_cat == specific_wrong_lbl:
          print ('%sObject idx %d is %s predicted as %s%s' % ( \
            ansi_colors.OKCYAN, idx_te [i_i], true_cat, pred_cat, ansi_colors.ENDC))

      # If want to print NN distance
      dist_str = ''

      # Most helpful debug info. TEMPORARILY commented out to generate
      #   nbins vs acc plots. Faster if don't print to screen!
      #print ('%3d %s (truth %s): predicted %s%s' % (idx_te [i_i],
      #  short_sample_name, true_cat, pred_cat, dist_str))

 
    # Calculate accuracy
    accs.append (calc_accuracy (lbls_te, lbls_pred, len(catids)))


    # Do this for every random split, for debugging, user can see conf
    #   mats for every test, not just last one.
    # Draw confusion matrix and save to file
    if show_plot_every_split:
      draw_confusion_matrix (lbls_te, lbls_pred, catnames, img_name=img_name,
        title_prefix='SVM ', draw_title=draw_title, bg_color=bg_color)

    # Show 1d histogram for the sample that user chooses
    if interactive_debug:
      if args.mixed:
        interact_plot_hist (hist_path, sample_names_tr + sample_names_te,
          lbls_tr + lbls_te, catnames, plot_opt='1d')
      else:
        interact_plot_hist (hist_path, sample_names, lbls, catnames,
          plot_opt='1d')

    print ('')

  # end for sp_i = 0 : n_random_splits


  # Copied from triangles_reader.py
  # Print out running time
  # Seconds
  end_time = time.time ()
  print ('Total time for %d random splits: %f seconds.' % \
    (n_random_splits, end_time - start_time))
  if n_random_splits != 0:
    print ('Average %f seconds per split.\n' % ( \
      (end_time - start_time) / n_random_splits))


  # Draw confusion matrix and save to file (if there are multiple random
  #   splits, this is for the very last run)
  if show_plot_at_end:
    draw_confusion_matrix (lbls_te, lbls_pred, catnames, img_name=img_name,
      title_prefix='SVM ', draw_title=draw_title, bg_color=bg_color)

  avg_acc = np.mean (np.asarray (accs))

  if rand_splits:
    print ('%sAverage accuracy over %d runs of random splits: %f %s' % \
      (ansi_colors.OKCYAN, n_random_splits, avg_acc, ansi_colors.ENDC))


  # Write to stats file, for plotting graph for paper
  if write_stats:
    write_stat (hist_parent_path, hist_path, overwrite_stats, accs, #avg_acc,
      img_mode_suffix)


if __name__ == '__main__':
  main ()

