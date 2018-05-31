#!/usr/bin/env python

# Mabel Zhang
# 24 Mar 2016
#
# Refactored from triangles_svm.py
#
# Visualize 3D histogram in three 1D hists in matplotlib, or in a 3D hist
#   (4th dimension, the bin count, is indicated by color) in RViz.
#
# Usage:
#   Gazebo:
#   $ rosrun triangle_sampling plot_hist_rviz.py --gazebo l0,l1,a0 10,10,10 --meta models_active_simple.txt --col_shift 0.12,0,0
#
#   PCD:
#   $ rosrun triangle_sampling plot_hist_rviz.py --pcd 300 0.95 l0,l1,a0 10,10,10 --meta models_active_simple.txt --col_shift 0.12,0,0 --row_shift 0,0.12,0
#

# ROS
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# Python
import argparse
import os
import time
import numpy as np

# Matplotlib, just for colormaps
import matplotlib

# My packages
from triangle_sampling.load_hists import load_hists, read_hist_config, \
  scale_bin_range_to_decimeter, load_one_hist
from triangle_sampling.calc_hists import find_bin_edges_and_volume
from triangle_sampling.config_paths import parse_args_for_svm, \
  get_robot_hists_path, config_hist_paths_from_args, \
  get_floats_from_comma_string, get_ints_from_comma_string, \
  get_recog_meta_path
  #parse_subpath_params, 
from tactile_map.create_marker import create_marker
from tactile_collect import tactile_config

# Local
from plot_hist_dd import create_subplot, plot_hist_dd_given, show_plot, \
  flatten_hist


# Interactive debugging. Lets user enter a sample index, plots and displays
#   the 3D histogram as three 1D plots, saves the plots to files.
# Parameters:
#   hist_path: Full path to folder that contains hist_conf.csv
#   sample_names: List of full paths to histogram .csv files
#   lbls, catnames: Used only to construct image file name to save.
#   tri_params: List of 3 strings, e.g. ['l0', 'l1', 'a0']
#   plot_opt: '1d' or '3d'
#   row_shift: If plot_opt=='3d', this tells how much to shift each 
#     object's 3D histogram. If you are plotting multiple objects, this ensures
#     objects don't overlap each other.
#   max_per_row: If exceed this many in a row, then will shift z up and plot
#     on a new row. It's like an auto row_shift!
def interact_plot_hist (hist_path, sample_names, lbls, catnames, tri_params,
  plot_opt='1d', col_shift=(0.12,0,0), row_shift=(0,0,0.12),
  start_cell_idx=0, max_per_row=1, ns='hist'):

  # Read histogram config file, for number of bins and bin range
  hist_conf_name = os.path.join (hist_path, 'hist_conf.csv')
  # nbins is 3-tuple e.g. (10, 10, 10)
  # bin_range3D is ((min,max), (min,max), (min,max))
  ((pr1, pr2, pr3), nbins, bin_range3D) = read_hist_config (hist_conf_name)

  #_, _, bin_volume = find_bin_edges_and_volume ( \
  #  nbins, bin_range3D)

  # Don't need to plot intersection for now. Just plot the raw histograms
  #   by themselves. I suspect even those look wrong.
  # Make histogram intersection confusion matrix plots
  #figs, success = plot_conf_mat_hist_inter (tri_params, nbins, bin_range3D,
  #  nDims, nObjs, plotMinus, suptitles, obj_names, xlbl_suffix=xlbl_suffix)
  # Actually... don't load the raw triangles. Load the actual histograms.
  #   The histogram csv files are what actually go into SVM.

  # Max number of rows possible, if user plots every single sample in the list
  max_n_rows = np.ceil (len (sample_names) / float (max_per_row))

  sp_idx = -1
  while True:
    uinput = raw_input ('Pick a sample to plot, in range 0 to %d, or q to quit: ' % ( \
      len (sample_names) - 1))

    if uinput.lower () == 'q':
      break

    try:
      sp_idx = int (uinput)
    except ValueError:
      print ('Invalid input (non-integer). Try again.')
      continue

    if sp_idx < 0 or sp_idx >= len (sample_names):
      print ('Invalid input outside of range. Try again.')
      continue

    # Valid input
    print ('Picked a %s' % catnames [lbls [sp_idx]])


    # Get full path of histogram file
    hist_name = sample_names [sp_idx]
 
    histdd, edgesdd = load_one_hist (hist_name, nbins, bin_range3D)


    # Plot 1D histograms
    if plot_opt == '1d':
      interact_plot_1d (sample_names [sp_idx], catnames [lbls [sp_idx]],
        pr1, pr2, pr3, histdd, edgesdd)
    elif plot_opt == '3d':

      plot_3d_opt = {}
      #plot_3d_opt ['bin_volume'] = bin_volume
      # If user wants multiple 3D hists to show up at same time, then they
      #   should be shifted and be put into different namespaces so they are
      #   all visible at same time.
      #if not np.all (np.array (col_shift) == 0):
      sample_base = os.path.basename (sample_names [sp_idx])
      short_sample_name = sample_base [0 : min (len (sample_base), 9)]

      # These dictionary keys must match arg names of interact_plot_3d()
      plot_3d_opt ['ns_suffix'] = short_sample_name
      plot_3d_opt ['ns'] = ns
      plot_3d_opt ['tri_params'] = tri_params
      # For ICRA 2017, don't need specific obj name (ns), just use obj cat
      plot_3d_opt ['title'] = catnames [lbls [sp_idx]]

      # Simple shift by multiplying it by the index. Each subsequent one
      #   gets shifted farther. So only recommend shift_amt to be passed in
      #   as non-zero in one dimension, e.g. 1 0 0, not 1 1 1. Latter would
      #   shift all your 3d hist cubes diagonally, harder to see all of them!
      # Subtract row_idx from max_n_rows, so can reverse on y-axis, such that
      #   low indices are plotted above high indices (i.e. low indices have
      #   higher y-coordinate), the intuitive order.
      row_idx = max_n_rows - 1 - \
        np.floor ((sp_idx + start_cell_idx) / max_per_row)
      col_idx = (sp_idx + start_cell_idx) % max_per_row
      print ('Plotting at (%d, %d)' % (row_idx, col_idx))
      plot_3d_opt ['shift'] = np.array (col_shift) * col_idx + \
        np.array (row_shift) * row_idx

      # Ref pass dictionary as keyword argument: http://stackoverflow.com/questions/2932648/how-do-i-use-a-string-as-a-keyword-argument
      interact_plot_3d (histdd, **plot_3d_opt)


# Used by triangles_svm.py
# Simple helper code to call routines to plot and save three 1D histograms to
#   files.
def interact_plot_1d (sample_name, catname, pr1, pr2, pr3, histdd, edgesdd):

  figs = [None, None, None]
  axes = [None, None, None]
  for d in range (0, 3):
    figs [d], axes [d] = create_subplot ((1, 1))
  subplot_idx = 0
  ylbls = [pr + ' in ' + os.path.basename (sample_name) \
    for pr in (pr1, pr2, pr3)]

  # Don't normalize the 1d hists, in case the magnitudes are drastically
  #   different, then normalizing will hide the differences. Debug info
  #   should be as true as possible.
  plot_hist_dd_given (histdd, edgesdd, figs, axes, subplot_idx, ylbls,
    normalize_1d=False, strict_lims=True)
 
  img_suff = [ \
    catname + \
    '_' + os.path.splitext (os.path.basename (sample_name)) [0] + \
    '_' + pr \
    for pr in (pr1, pr2, pr3)]
  show_plot (figs, img_suff, ndims=3, show=True, save=True)


# Visualize 3D histogram in RViz, `.` need 4D to visualize a 3D histogram.
#   The 4th dimension, bin count, is indicated by color.
# Parameters:
#   histdd: NumPy array, l x m x n, each dimension's size is the number of bins
#     in that dimension.
#   tri_params: List of 3 strings, e.g. ['l0', 'l1', 'a0']
#   shift: Amount to shift the 3D histogram cube in RViz. Useful when you want
#     to plot 3D hist for multiple objects for comparison, so you wouldn't
#     want to plot all at 0 0 0, they'd overlap each other!
#   ns_suffix: Suffix for Marker namespace. Useful when you want to plot for
#     multiple objects and compare, then they can be in different namespaces so
#     the markers don't get replaced!
#   bin_volume: Product of all bin widths (not including the bin counts! So
#     just width1 * width2 * ... * width_n for a n-dimensional histogram).
#     If you have it, pass it in, it'll make the colors closer to
#     the actual histogram. Otherwise a default value is used. This is the 
#     denominator used to put bin counts into a [0, 1] range to determine the
#     bin color!
def interact_plot_3d (histdd, tri_params, shift=(0,0,0), ns='hist',
  ns_suffix='', title=''):
  #, bin_volume=1000):

  if not title:
    title = ns

  # Cube bin width in RViz visualization. This is not the actual bin width in
  #   the histogram.
  # TODO: Might want to scale cube using actual bin width... that requires
  #   passing in edgesdd too.
  vis_bin_w = .01

  cmap = matplotlib.cm.get_cmap ('jet')

  #ttl_bin_counts = float (np.sum (histdd)) / bin_volume  # all blue, too big
  #ttl_bin_counts = float (np.sum (histdd)) * bin_volume  # product is 1, only red and blue boxes, no yellow or orange. `.` Same as not having this var!
  #ttl_bin_counts = float (np.sum (histdd))  # all blue, too big
  # Is the max better, for denominator of color?
  ttl_bin_counts = float (np.max (histdd))  # faint light blue, yellow red. I think this is the most correct color scale. Biggest bin is red, others are cooler colors
  #ttl_bin_counts = float (np.sum (histdd)) / 1000.0  # This produces best result from full histograms... but once turn off zero-bins, these look too red!
  #ttl_bin_counts = 1
  #print ('Total bin counts: %f, bin_volume %f' % (ttl_bin_counts, bin_volume))

  if ttl_bin_counts == 0:
    print ('interact_plot_3d(): Total bin count is 0. Not plotting anything.')
    return

  vis_arr_pub = rospy.Publisher ('/visualization_marker_array',
    MarkerArray, queue_size=2)
  marker_arr = MarkerArray ()
  marker_id = 0

  ns = ns
  if ns_suffix:
    ns = ns + ' ' + ns_suffix

  for i in range (0, np.shape (histdd) [0]):

    # Center (x, y, z) of where the 3D bin is
    # Assuming axis origins at 0 0 0,
    #   1st bin on x-axis is at 1*bin_width/2,
    #   2nd is at 3*bin_width/2,
    #   3rd is at 5*bin_width/2, etc.
    bx = vis_bin_w * (2*i+1) * 0.5

    for j in range (0, np.shape (histdd) [1]):

      by = vis_bin_w * (2*j+1) * 0.5

      for k in range (0, np.shape (histdd) [2]):

        bz = vis_bin_w * (2*k+1) * 0.5

        bin_count = histdd [i, j, k] / ttl_bin_counts
        #if bin_count > 0.0:
        #  print (bin_count)
        rgba = cmap (bin_count)

        # Visualize non-zero bins (zero bins are too many dark blue cubes,
        #   blocks out the actual non-zero bins, even when translucent)
        if bin_count > 0:
          marker_bin = Marker ()
          create_marker (Marker.CUBE, ns, '/base', marker_id,
            bx+shift[0], by+shift[1], bz+shift[2],
            rgba[0], rgba[1], rgba[2], 0.2, vis_bin_w, vis_bin_w, vis_bin_w,
            marker_bin, 0)  # Use 0 duration for forever
          marker_arr.markers.append (marker_bin)
        
          marker_id += 1

  # Draw 3 axes
  axis_len = 0.1
  marker_ax = Marker ()
  create_marker (Marker.LINE_LIST, ns, '/base', marker_id,
    # scale.x is width of line
    0, 0, 0, 1, 1, 1, 0.5, 0.001, 0, 0,
    marker_ax, 0)  # Use 0 duration for forever
  marker_ax.points.append (Point (shift[0], shift[1], shift[2]))
  marker_ax.points.append (Point (shift[0], shift[1], shift[2]+axis_len))
  marker_ax.points.append (Point (shift[0], shift[1], shift[2]))
  marker_ax.points.append (Point (shift[0], shift[1]+axis_len, shift[2]))
  marker_ax.points.append (Point (shift[0], shift[1], shift[2]))
  marker_ax.points.append (Point (shift[0]+axis_len, shift[1], shift[2]))
  marker_arr.markers.append (marker_ax)
  marker_id += 1

  # Draw axis labels
  fontsize = 0.01
  lbl_pos = [[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]]
  for i in range (0, 3):
    marker_lbl = Marker ()
    create_marker (Marker.TEXT_VIEW_FACING, ns, '/base', marker_id,
      shift[0]+lbl_pos[i][0], shift[1]+lbl_pos[i][1], shift[2]+lbl_pos[i][2],
      1, 1, 1, 1, 0, 0, fontsize,
      marker_lbl, 0)
    marker_lbl.text = tri_params [i]
    marker_arr.markers.append (marker_lbl)
    marker_id += 1

  # Draw title of plot
  marker_text = Marker ()
  create_marker (Marker.TEXT_VIEW_FACING, ns, '/base', marker_id,
    #shift[0], shift[1]-0.03, shift[2],
    shift[0]+axis_len*0.5, shift[1]-0.03+axis_len*0.5, shift[2]+axis_len,
    # scale.z specifies height of an uppercase A
    1, 1, 1, 1, 0, 0, fontsize,
    marker_text, 0)
  marker_text.text = title
  marker_arr.markers.append (marker_text)
  marker_id += 1

  for i in range (0, 10):
    vis_arr_pub.publish (marker_arr)
    time.sleep (0.05)

  #print (histdd)


def main ():

  rospy.init_node ('plot_hist_rviz', anonymous=True)


  #####
  # Parse command line args
  #####

  # Add a few custom arguments before calling parse_args_for_svm() for the
  #   generic args shared among programs.
  arg_parser = argparse.ArgumentParser ()
  arg_parser.add_argument ('--col_shift', type=str, default=None,
    help='How much to space from start of one hist to another. One hist occupies 0.12 meters. Recommended value is 0.12,0,0')
  arg_parser.add_argument ('--row_shift', type=str, default=None,
    help='Useful if you want to plot things in two rows, by running this script twice, passing in different row_shift')
  arg_parser.add_argument ('--start_cell', type=str, default=None,
    help='Linear index of the cell where the first plot should start at, e.g. 1. Useful when you want to plot a hist from active_predict.py in 0th cell, then plot all the ground truth after, for paper figure.')
  arg_parser.add_argument ('--max_per_row', type=str, default=None,
    help='Max number of hists to plot per row. When exceed, will shift z up and plot on another row')

  args, valid = parse_args_for_svm (arg_parser)
  if not valid:
    return

  # Amount to shift histogram plotted in RViz
  col_shift = [0.12, 0, 0]
  if args.col_shift:
    col_shift = get_floats_from_comma_string (args.col_shift)
  row_shift = [0, 0, 0.12]
  if args.row_shift:
    row_shift = get_floats_from_comma_string (args.row_shift)

  start_cell = 1
  if args.start_cell:
    start_cell = int (args.start_cell)

  max_per_row = 1
  if args.max_per_row:
    max_per_row = int (args.max_per_row)


  sampling_subpath, tri_nbins_subpath, hist_parent_path, hist_path, \
    mode_suff, tri_suffix, tri_paramStr = \
      config_hist_paths_from_args (args)

  tri_params = tri_paramStr.split (',')

  # Not tested
  if args.real:
    mode_ns = 'real_robo'

  elif args.pcd:
    mode_ns = 'pt_cloud'

  elif args.gazebo:
    mode_ns = 'phys_sim'


  # Config histogram path

  # e.g. csv_hists
  #hist_subpath = get_robot_hists_path (mode_suff)
  # Directory in which triParams_* folders are found
  #hist_parent_path = os.path.join (hist_subpath, sampling_subpath)
  # Directory in which hist_conf.csv can be found
  hist_path = os.path.join (hist_parent_path, tri_nbins_subpath)


  meta_name = os.path.join (get_recog_meta_path (), args.meta)

  # Each row of samples is the histogram for one object
  # lbls is numSamples size list with category integer label for each sample
  [samples, lbls, catnames, catcounts, catids, sample_names] = load_hists ( \
    meta_name, hist_path, tri_suffix=tri_suffix,
    mixed_paths=False, sampling_subpath=sampling_subpath,
    tri_nbins_subpath=tri_nbins_subpath)

  # Print all indices, so user can see an index and what object it is
  for i in range (0, len (sample_names)):
    sample_base = os.path.basename (sample_names [i])
    short_sample_name = sample_base [0 : min (len (sample_base), 8)]
    print ('%3d %s: category %s' % (i,
      short_sample_name, catnames [lbls [i]]))


  # Plot 3D histogram
  interact_plot_hist (hist_path, sample_names, lbls, catnames, tri_params,
    plot_opt='3d', col_shift=col_shift, row_shift=row_shift,
    start_cell_idx=start_cell, max_per_row=max_per_row, ns=mode_ns)


if __name__ == '__main__':
  main ()

