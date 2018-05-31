#!/usr/bin/env python

# Mabel Zhang
# 28 Jul 2015
#
# Utility function. Plots 3D histogram in matplotlib.
#
# Some code copied from ../../tactile_collect/src/weights_reader.py
#


# ROS
import rospy

# Python
import sys
import csv
import os
import re

# Numpy
import numpy as np

# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
#from matplotlib.colors import Colormap
from matplotlib.cm import get_cmap

# My packages
from triangle_sampling.config_paths import get_img_path
from util.matplotlib_util import black_background


# Create an array of subplots that share x and y axes!
# Parameters:
#   dims: 2-element tuple for subplot dimensions
# Returns one figure object, and an array of axes objects within the figure.
#   Number of axes is the number of subplots, dims[0] * dims[1] elements.
def create_subplot (dims, figsize=None):

  # Create plot
  # Ref sharex sharey: http://matplotlib.org/examples/pylab_examples/subplots_demo.html
  fig, axes = plt.subplots (dims[0], dims[1], sharex=True, sharey=True)

  if dims[0] == 1 and dims[1] == 1:
    axes = np.asarray ([axes], dtype='object')
  else:
    axes = axes.flatten ()

  if figsize:
    # Ref: http://matplotlib.org/api/figure_api.html
    fig.set_size_inches (figsize)

  return fig, axes


# Every parameter is 3 by something, other than obj_idx, which is a scalar,
#   indicating which subplot to use.
# Parameters:
#   data: n by D array, D is number of features (i.e. dimension of data),
#     n is number of samples. Histogram to be plotted will be dimension D.
#   figs: A list of D matplotlib figure windows. One window per dimension,
#     so that the D-d histogram can be visualized in 2D. Each window contains
#     plots of (Dth feature on y-axis) vs (histogram bins on x-axis).
#   axes: list of D Numpy arrays. Each element in list is a set of 
#     matplotlib axes (plotting objects).
#   subplot_idx: scalar to access axes's 2nd dimension. Specifies which subplot
#     to draw on.
#   ylbls: list of strings, has D elements
# Returns:
#   hist, edges: D-dimensional histogram plotted from data.
#     Ret val from np.histogramdd().
def plot_hist_dd (data, bins, figs, axes, subplot_idx, ylbls, bg_color='white',
  fg_color='g', tick_rot=0):

  #####
  # Compute histogram using Numpy
  #####

  # Normalize the histogram, because different objects have different number of
  #   points. Then histogram will have higher numbers for big object model.
  #   Bias is not good.
  hist, edges = np.histogramdd (data, bins=bins, normed=True)
 
  #print ('Shape of 3D histogram:')
  #print (np.shape (hist))
  #print (np.shape (hist[0]))
  #print (hist)
 
  #print ('Shape of edges:')
  #print (np.shape (edges))
  #print (edges)
 
  plot_hist_dd_given (hist, edges, figs, axes, subplot_idx, ylbls,
    bg_color=bg_color, fg_color=fg_color, tick_rot=tick_rot)

  return (hist, edges)


# Same as plot_hist_dd(), but pass in hist and edges directly, instead of data
#   and nbins.
# Parameters:
#   normalize_1d: Specify False if you want to see true values in original
#     histogram after flattening (summing).
def plot_hist_dd_given (hist, edges, figs, axes, subplot_idx, ylbls,
  normalize_1d=True, strict_lims=True, bg_color='white', fg_color='g',
  tick_rot=0):
 
  #####
  # Plot the histogram in matplotlib
  #####

  # Number of dimensions of the histogram
  ndims = np.ndim (hist)

  # For each dimension of the histogram
  for i in range (0, ndims):

    ax = axes [i] [subplot_idx]

    # Make axes ticks' font size smaller
    # Ref: http://stackoverflow.com/questions/6390393/matplotlib-make-tick-labels-font-size-smaller/11386056#11386056
    ax.tick_params (axis='both', which='major', labelsize=6)

    hist_flat = flatten_hist (hist, i)

    # Normalize the histogram, `.` after flattening, not normalized anymore
    #   Be careful that plots after normalizing may be misleading. You may want
    #   to not normalize, if you want to see the true values in the orig
    #   histogram!
    # In matlab, the correct way is divided by trapz(), not by sum().
    #   Ref: http://stackoverflow.com/questions/5320677/how-to-normalize-a-histogram-in-matlab
    #     http://www.mathworks.com/help/matlab/ref/trapz.html
    if normalize_1d:
      real_width = edges [i] [1] - edges [i] [0]
      heights = sum (hist_flat)
      area = real_width * heights
      hist_flat /= area

    plot_hist (hist_flat, edges [i], ax, ylbls [i], strict_lims, bg_color,
      fg_color, tick_rot)

    figs[i].suptitle ('%dD Histogram, Flattened' % (ndims))


# Flatten a D-dimensional histogram to 1D.
# Parameters:
#   currDim: dimension to get. This means flatten along all axes EXCEPT this
#     one.
# Returns NumPy array
def flatten_hist (hist, currDim):

  # Number of dimensions of the histogram
  ndims = np.ndim (hist)

  # Flatten along this dimension, by summing all other dimensions along this
  #   dimension.
  # Test in bare python shell:
  '''
  a = np.array ([[[0,1,2], [3,4,5], [6,7,8], [9,10,11]], [[12,13,14], [15,16,17], [18,19,20], [21,22,23]]])
  np.shape(a)  # (2, 4, 3)
  # To sum such that you get an axis's dimension, you sum everything else BUT
  #   this axis.
  np.sum (a, axis=(1,2))  # array([ 66, 210])  2 elts
  np.sum (a, axis=(2,0))  # array([42, 60, 78, 96])  4 elts
  np.sum (a, axis=(1,0))  # array([ 84,  92, 100])  3 elts
  '''
  # Size of this is the dimension of d-D histogram along ith dim
  # For 1D, don't sum the only dimension! Just leave it.
  if ndims > 1:
    all_but_this = range (0, ndims)
    all_but_this.remove (currDim)
    all_but_this = tuple (all_but_this)
    hist_flat = np.sum (hist, axis=all_but_this)
  else:
    hist_flat = hist

  #print ('hist_flat:')
  #print (hist_flat)

  return hist_flat


# Plots 1D histogram as bar graph
# Parameters:
#   ylbl: string
#   strict_lims: Whether to limit the axis limits to edgesdd values. Setting
#     to True makes sure all your individual histogram plots are neat and 
#     comparable when you lay them side by side
def plot_hist (hist, edges, ax, ylbl, strict_lims=True, #, xlbl_suffix=''):
  bg_color='white', fg_color='g', tick_rot=0):

  # MATLAB "hold on"
  ax.hold (True)
  # MATLAB "grid on"
  ax.grid (True, color='gray')

  # Ref font size of labels: http://stackoverflow.com/questions/12444716/how-do-i-set-figure-title-and-axes-labels-font-size-in-matplotlib
  #ax.set_xlabel ('Bins' + xlbl_suffix, fontsize=6)
  ax.set_ylabel (ylbl)

  if strict_lims:
    ax.set_xlim (edges [0], edges [len (edges)-1])

  # Get histogram centers from edges. n+1 edges means there are n centers.
  centers = (edges[0:len(edges)-1] + edges[1:len(edges)]) * 0.5

  width = (edges[1] - edges[0]) * .5

  #print (edges)
  #print (centers)
  #print (hist)
  ax.bar (centers, hist, width=width, color=fg_color, edgecolor='none')

  # Rotate tick orientation
  # Diagonal 45-degree slanted tick labels
  # Ref: https://stackoverflow.com/questions/14852821/aligning-rotated-xticklabels-with-their-respective-xticks/14854007#14854007
  # axes API: https://matplotlib.org/api/axes_api.html
  if tick_rot != 0:
    ax.set_xticklabels (ax.get_xticks (), rotation=tick_rot)

  if bg_color == 'black':
    black_background (ax=ax)


# Save plot to file, and show it
def show_plot (figs, ylbls, ndims=3, show=True, save=True, bg_color='white'):

  for i in range (0, len (ylbls)):

    if save:
      imgpath = get_img_path ('hists')
      imgname = os.path.join (imgpath,
        'triangle_' + str(ndims) + 'Dhist_' + ylbls[i] + '.eps')

      # Ref savefig() black background: https://stackoverflow.com/questions/4804005/matplotlib-figure-facecolor-background-color
      figs [i].savefig (imgname, bbox_inches='tight',
        facecolor=bg_color, edgecolor='none',
        transparent=True)


      print ('Plot saved to %s' % imgname)

    if show:
      figs[i].show ()


# Save all individual axes in a figure to separate files
# Parameters:
#   ylbls: Dimension name for each figure
#   ylbl_to_save: The dimension name you want to save individual subplots of
#   nRowsCols: Number of columns and rows, should be the same, `.` figure is
#     n x n.
#   tick_rot: used to determine how much room to leave for displaying ticks
def save_individual_subplots (figs, ylbls, ylbl_to_save, nRows, nCols,
  scale_xmin=1.1, scale_x=1.2, scale_y=1.15, ndims=3, show=True, bg_color='white', tick_rot=0):

  x_expand_idx = []
  y_expand_idx = []
  # Expand horizontal space for plots on left column
  for i in range (0, nRows):
    # Subplots on left have indices 0*n, 1*n, 2*n, ..., (n-1)*n. Add margin on
    #   x on left.
    x_expand_idx.append (i * nRows)

  # Expand vertical space for plots on botom row
  for i in range (0, nCols):
    # Subplots at bottom have indices (n-1)*n+0, (n-1)*n+1, (n-1)*n+2, ...,
    #   (n-1)*n+(n-1). Add margin on y at bottom.
    y_expand_idx.append ((nRows - 1) * nCols + i)

  # Constants for scaling bbox, tuned for ticks and labels with fontsize 6
  SCALE_XMIN = scale_xmin
  SCALE_YMIN = 1.1
  SCALE_X = scale_x
  SCALE_Y = scale_y


  # Find the figure that matches the desired ylbl
  for f_i in range (0, len (ylbls)):

    if ylbls [f_i] != ylbl_to_save:
      continue

    # Found the figure with desired ylbl
    else:
      imgpath = get_img_path ('hists')
      imgname = os.path.join (imgpath,
        'triangle_' + str(ndims) + 'Dhist_' + ylbls[f_i])
     
      axes = figs [f_i].get_axes ()

      for sp_i in range (0, len (axes)):
        curr_imgname = imgname + ('_%02d' % sp_i) + '.eps'

        # Ref save a subplot (axis) in a figure to file:
        #   http://stackoverflow.com/questions/4325733/save-a-subplot-in-matplotlib
        extent = axes [sp_i].get_window_extent ().transformed (figs [f_i].dpi_scale_trans.inverted ())

        # Default. Need 1.05 to show entire plot, 1 doesn't show whole thing!
        #   1.3 shows axes labels (font size 6) perfectly, for x or y
        bbox_inches = extent.expanded (SCALE_XMIN, SCALE_YMIN)
        # Record original x y position of box, before expanding
        #   http://matplotlib.org/devel/transformations.html
        xmax_orig = bbox_inches.xmax
        ymax_orig = bbox_inches.ymax

        # expanded() works outwards from center only, can't make box fixed
        #   in one corner. So have to manually move box back after expand.
        # extent is a matplotlib.transforms.BboxBase type
        #   Ref http://matplotlib.org/devel/transformations.html
        # Subplot on lower-left needs to expand in both x and y
        if sp_i in x_expand_idx and sp_i in y_expand_idx:
          bbox_inches = extent.expanded (SCALE_X, SCALE_Y)
        # Only expand x, on left
        elif sp_i in x_expand_idx:
          # x should be same as above, so last row is same width as prev rows
          bbox_inches = extent.expanded (SCALE_X, SCALE_YMIN)
        # Only expand y, on bottom
        elif sp_i in y_expand_idx:
          # Need extra y to show bottom x label, need extra x to show left-most
          #   tick.
          bbox_inches = extent.expanded (SCALE_XMIN, SCALE_Y)

        # Shift content in bbox to sit at upper-right of the expanded bbox
        xmax_new = bbox_inches.xmax
        ymax_new = bbox_inches.ymax
        bbox_inches = bbox_inches.translated (xmax_orig - xmax_new,
          ymax_orig - ymax_new)

        if bg_color == 'black':
          black_background (axes [sp_i])

        # Ref savefig() black background: https://stackoverflow.com/questions/4804005/matplotlib-figure-facecolor-background-color
        figs [f_i].savefig (curr_imgname, bbox_inches=bbox_inches,
          facecolor=bg_color, edgecolor='none',
          transparent=True)

        print ('Individual axes %d in dimension %s saved to %s' % ( \
          sp_i, ylbl_to_save, curr_imgname))

      break


# Parameters:
#   data: Python list of NumPy 2D arrays that are n x d, d is the dimension of
#     the histogram you want.
#   bins: Like numpy.histogramdd, either number of bins, or bin edges
#   xlbls: Python list of nHists lists of nDims strings. Can be a title for
#     each subplot.
#   figsize: Optional size in inches, 2-element tuple. Size for entire figure.
#     If you have more than one subplots, you should calculate width and height
#     yourself. If None, will use default.
#   bg_color: 'white' default, 'black' for powerpoint slides with black bg.
#   fg_color: whatever that matplotlib takes as color, can be a char, or a
#     3-tuple RGB.
#   tick_rot: Rotation of tick label. By deafult, 0 horizontal. Can set to 45
#     for diagonal, for ex.
def plot_hists_side_by_side (data, bins, suptitles, xlbls=[], figsize=None,
  bg_color='white', fg_color='g', tick_rot=0):

  nHists = len (data)
  nDims = data [0].shape [1]

  # Default figure size in inches
  if not figsize:
    row_height = 3
    row_width = 4

    # figsize unit: inches. You can adjust dpi=80 or whatever to get different
    #   number of pixels per inch. dpi=80 creates 80 x 80 pixels per inch
    # Ref: http://stackoverflow.com/questions/332289/how-do-you-change-the-size-of-figures-drawn-with-matplotlib
    figsize = (row_width * nHists, row_height)


  figs = [None] * nDims
  # Each element is a NumPy array of nHists items. i.e. 1 x 4 subplot grid has
  #   4 axes
  axes = [None] * nDims

  # For each triangle parameter, make a 1 x nHists subplot grid
  for i in range (0, nDims):

    # I don't know how to set size using existing figure
    # Create an array of subplots that share x and y axes!
    figs[i], axes[i] = create_subplot ([1, nHists], figsize=figsize)

    # Set titles on all figures
    figs [i].suptitle ('Histogram Intersections for %s' % \
      suptitles[i])


  # For each object, plot all 3 dimensions of its histograms, in 3 separate
  #   figures.
  for subplot_idx in range (0, nHists):
    plot_hist_dd (data [subplot_idx], bins, figs, axes, subplot_idx,
      [''] * nDims, bg_color, fg_color, tick_rot)

    if xlbls:
      for d in range (0, nDims):
        axes [d] [subplot_idx].set_xlabel (xlbls [subplot_idx] [d])

  return figs, True


