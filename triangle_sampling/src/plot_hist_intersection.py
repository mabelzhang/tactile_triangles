#!/usr/bin/env python

# Mabel Zhang
# 25 Jan 2016
#
# Plots histogram minus histogram intersection, confusion matrix style plot,
#   for Gazebo-trained data.
#   (Point cloud data is already plotted by
#     sample_pcl_calc_hist.py, so doesn't need this file to plot.)
#
# Usage:
#   $ rosrun triangle_sampling plot_hist_intersection.py 0 0 --gazebo
#


# ROS
import rospy
import rospkg

# Python
import os
import argparse

# NumPy
import numpy as np

# My packages
from tactile_collect import tactile_config
#from triangle_sampling.load_hists import load_hists
from triangle_sampling.config_paths import get_sampling_subpath
from triangle_sampling.config_hist_params import TriangleHistogramParams as \
  HistP

# Local
from plot_hist_dd import create_subplot, plot_hist, show_plot


# Plot confusion matrix style plots for histogram intersections
# Parameters:
#   tri_params: Triangles data. Python list of 6 NumPy 2D arrays.
#     tri_params [i] [:, dim] gives data for object i, dimension dim.
#   bins: Number of bins for 1D flattened plots
#   nDims: 6. Size of tri_params list.
#   plotMinus: True to plot histogram minus histogram intersection in addition
#     to histograms-only.
#   suptitles: List of triangle parameter names. Size of list is nDims.
#   obj_names: Object file name
#   xlbl_suffix: ' (Meters)' or ' (Decimeters)', used for x label of lengths
#     only. (Angles will use radians.)
# Returns figure objects.
#   Also return False if see Ctrl+C, True if completed.
def plot_conf_mat_hist_inter (tri_params, bins, bin_range, nDims, nObjs,
  plotMinus, suptitles, obj_names, xlbl_suffix='',
  bg_color='white', fg_color='g', tick_rot=0): #, bins3D=HistP.bins3D):

  # Make plot containers
  #   fig is 1 per window, axes is also 1 per window. 1 axis in each fig.
  figs = [None] * nDims
  axes = [None] * nDims
  xlbl_suffs = [' (Radians)'] * nDims
  # For each triangle parameter, make a nObjs x nObjs subplot grid
  for i in range (0, len (figs)):
    figs[i], axes[i] = create_subplot ([nObjs, nObjs])

    # Set titles on all figures
    figs [i].suptitle ('Histogram Intersections for %s' % \
      suptitles[i])

    # Use lengths suffix
    if suptitles [i] == HistP.L0 or suptitles [i] == HistP.L1 or \
      suptitles [i] == HistP.L2:
      xlbl_suffs [i] = xlbl_suffix

  # Plot histogram minus histogram intersection in addition
  if plotMinus:
    figs.extend ([None] * nDims)
    axes.extend ([None] * nDims)
    # Make a copy of itself, then don't need to do same thing twice
    xlbl_suffs.extend (xlbl_suffs)

    # For each triangle parameter, make a nObjs x nObjs subplot grid
    for i in range (nDims, len (figs)):
      figs[i], axes[i] = create_subplot ([nObjs, nObjs])

      # Set titles on all figures
      figs [i].suptitle ('Histogram Minus Histogram Intersections for %s' % \
        suptitles[i])


  # Store x and y limits to crop graphs at.
  #   OS X doesn't need this, I don't know why Ubuntu does.
  xmins = [None] * len (axes)
  xmaxs = [None] * len (axes)
  ymins = [None] * len (axes)
  ymaxs = [None] * len (axes)


  #####
  # Plot histogram intersection
  #####

  # Rows
  for i in range (0, nObjs):
 
    print ('Plotting row %d' % i)
 
    # Columns
    for j in range (0, nObjs):
 
      print ('Plotting column %d' % j)
 
      subplot_idx = i * nObjs + j
 
      # Each figure is one dimension
      for dim in range (0, nDims):
 
        try:
 
          # Only supply title for top row and leftmost column.
          #   Labels are object names, confusion-matrix style titles.
          subplot_title = ''
          y_lbl = ''
          x_lbl = ''
          # If on first row, draw column titles on top, as titles
          if i == 0:
            subplot_title = shorten_name (obj_names [j])
          # Show x-axis label on bottom row only
          if i == nObjs - 1:
            x_lbl = 'Bins' + xlbl_suffs [dim]

          # If on first column, draw row titles in left margin, as y lbls
          if j == 0:
            y_lbl = shorten_name (obj_names [i])

          axes [dim] [subplot_idx].set_title (subplot_title, fontsize=6)
          # Ref axis label and title font size:
          #   http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.set_xlabel
          #   http://matplotlib.org/api/text_api.html#matplotlib.text.Text
          axes [dim] [subplot_idx].set_ylabel (y_lbl, fontsize=6)
          axes [dim] [subplot_idx].set_xlabel (x_lbl, fontsize=6)

          # Make axes ticks' font size smaller
          # Ref: http://stackoverflow.com/questions/6390393/matplotlib-make-tick-labels-font-size-smaller/11386056#11386056
          axes [dim] [subplot_idx].tick_params ( \
            axis='both', which='major', labelsize=6)
 
 
          # This takes forever to run, I think it gets stuck???
          # This is 6D histogram. Do not plot this via plot_hist_dd, this is not 3D.
          # Don't do D-d histograms anymore, because I only want to do 3D, not 6D. Then this complicates things when I calculate histogram intersection. I know my 3D is correct, from sample_pcl_plotter.py, when flattened 3D and 1D look the same. Don't need sanity check anymore. Just use 3D for outputting descriptor, use 1D for plotting.
          # TODO later: it'd be nice to plot both flattened 3D and 6 individual 1D's, if I have time. Not worth it right now - I'd have to flatten them in order to get 1D histograms to calculate histogram intersection, which means refactoring plot_hist_dd.py. Should probably do it for sanity checks that my 3D hists are correct, if we decide to use this.
          #histidd, edgesi = np.histogramdd (tri_params [i], bins=bins3D, range=bin_range3D, normed=True)
          #histjdd, edgesj = np.histogramdd (tri_params [j], bins=bins3D, range=bin_range3D, normed=True)
          # This should give you equivalent to 1D histograms, if flatten is done correctly
          #histi = flatten_hist (histidd, dim)
          #histj = flatten_hist (histjdd, dim)
          
          
          # Equivalent as above, if above is done correctly. This is a good sanity check - substituting these two lines and the four lines above should give you same results.
          # tri_params [i] [:, dim] gives a (nTriangles x 1) Numpy array

          histi, edgesi = np.histogram (tri_params [i] [:, dim], bins=bins[dim], range=bin_range[dim], normed=True)
          histj, edgesj = np.histogram (tri_params [j] [:, dim], bins=bins[dim], range=bin_range[dim], normed=True)
          
          
          # Calculate histogram intersection, which is just the min
          # Assumption: bins and bin ranges for the histogram ensure that the edges of all objects' histograms are the same, so that we can do min() directly. If this assumption is violated, you'd first have to make sure x-axis edges of the two histograms are identical, before taking min! Otherwise the min you take means nothing!
          hist_inter = np.minimum (histi, histj)

           
          # Histogram intersection btw (i, j) and (j, i) are identical. So
          #   don't need to plot lower triangle of confusion matrix.
          if j >= i:
 
            # Plot in axes [subplot_idx]
            plot_hist (hist_inter, edgesi, axes [dim] [subplot_idx], y_lbl,
              bg_color=bg_color, fg_color=fg_color, tick_rot=tick_rot)
              #xlbl_suffix=xlbl_suffs [dim])

            # Update limits of graph, if curr limits > recorded limits
            # Assumption: First elt of edgesi is == np.min(edgesi), and last
            #   elt of edgesi is == np.max(edgesi). Indexing directly to save
            #   time. Otherwise you have to use np.min(edgesi), np.max(edgesi)
            #   explicitly, which wastes some cycles if you already know first
            #   and last elts are min and max.
            xmins [dim] = min (xmins [dim], edgesi [0])
            ymins [dim] = min (ymins [dim], np.min (hist_inter))
            xmaxs [dim] = max (xmaxs [dim], edgesi [len (edgesi) - 1])
            ymaxs [dim] = max (ymaxs [dim], np.max (hist_inter))
 
 
          if plotMinus:
 
            axes [dim + nDims] [subplot_idx].set_title (subplot_title, fontsize=6)
            axes [dim + nDims] [subplot_idx].set_ylabel (y_lbl, fontsize=6)
            axes [dim + nDims] [subplot_idx].set_xlabel (x_lbl, fontsize=6)
            axes [dim + nDims] [subplot_idx].tick_params ( \
              axis='both', which='major', labelsize=6)


            # Also plot original histogram minus histogram intersection, for
            #   more noticeable comparison.
            # First term must be either row (i) or col (j), not both! Just
            #   pick one! Else you get symmetric plot, which is incorrect!
            minus_hist_inter = abs (histi - hist_inter)
 
            plot_hist (minus_hist_inter, edgesi, 
              axes [dim + nDims] [subplot_idx], y_lbl,
              bg_color=bg_color, fg_color=fg_color, tick_rot=tick_rot)
              #xlbl_suffix=xlbl_suffs[dim])

            # Update limits of graph, if curr limits > recorded limits
            xmins [dim + nDims] = min (xmins [dim + nDims], edgesi [0])
            ymins [dim + nDims] = min (ymins [dim + nDims],
              np.min (minus_hist_inter))
            xmaxs [dim + nDims] = max (xmaxs [dim + nDims],
              edgesi [len (edgesi) - 1])
            ymaxs [dim + nDims] = max (ymaxs [dim + nDims],
              np.max (minus_hist_inter))

 
        except rospy.exceptions.ROSInterruptException, err:
          # Return whatever figs saved so far
          return figs, False


  # Testing
  # OS X Yosemite doesn't need this. Ubuntu 14.04 needs it, else histograms
  #   aren't displayed fully.
  # http://stackoverflow.com/questions/3777861/setting-y-axis-limit-in-matplotlib
  for dim in range (0, len (axes)):

    for subplot_idx in range (0, nObjs * nObjs):

      # Set all subplot limits to be same within each figure, because y and x
      #   ticks need to be common across all plots.
      axes [dim] [subplot_idx].set_xlim (xmins [dim], xmaxs [dim])
      axes [dim] [subplot_idx].set_ylim (ymins [dim], ymaxs [dim])

  return figs, True 


# Parameters:
#   name: string
def shorten_name (name):

  MAX_LEN = 15
  MAX_BASE = 5

  # Take full name of the last directory before base name, take partial base
  #   name. `.` last directory tells object category
  if len (name) > MAX_LEN:

    # Split dir name into a tuple of individual directory names
    dir_last = os.path.split (os.path.dirname (name))
    # Take the last dir
    dir_last = dir_last [len (dir_last) - 1]

    # Get base name without extension
    base_short = os.path.splitext (os.path.basename (name)) [0]
    # If already shorter than MAX_BASE, take it as is. Else, shorten it
    base_short_len = min (MAX_BASE, len (base_short))
    base_short = base_short [0 : base_short_len]

    short = os.path.join (dir_last, base_short)
    short += '...'

  else:
    short = name

  return short

