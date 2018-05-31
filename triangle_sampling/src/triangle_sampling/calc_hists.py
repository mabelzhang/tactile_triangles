#!/usr/bin/env python

# Mabel Zhang
# 5 Apr 2016
#
# Refactored from triangles_svm.py
#
# An updated version of functions for chords project are in
#   chord_recognition hist_util.py.
#

import numpy as np

# My packages
from util.ansi_colors import ansi_colors

# Works for 3D hists only. Not genearlized to work for other dimensions yet -
#   though that shouldn't be hard.
def find_bin_edges_and_volume (nbins, bin_range3D):

  ndims = len (nbins)

  edgesdd = []
  edgesdd.append (np.linspace (bin_range3D[0][0], bin_range3D[0][1],
    nbins[0] + 1, endpoint=True))
  edgesdd.append (np.linspace (bin_range3D[1][0], bin_range3D[1][1],
    nbins[1] + 1, endpoint=True))
  edgesdd.append (np.linspace (bin_range3D[2][0], bin_range3D[2][1],
    nbins[2] + 1, endpoint=True))
  # TODO Generalizes 3D to d-D. Test after 15 Jan 2016 when return to project.
  #   Remove above hardwired 3D when works.
  #for d in range (0, ndims):
  #  edgesdd.append (np.linspace (bin_ranges[d][0], bin_ranges[d][1],
  #    nbins[d] + 1, endpoint=True))

  # 3-elt list. Each ith element is a list of bin centers in ith dimension
  #   e.g. list of 3 10-elt NumPy arrays.
  centersdd = []
  centersdd.append (np.zeros ([nbins[0], ]))
  centersdd.append (np.zeros ([nbins[1], ]))
  centersdd.append (np.zeros ([nbins[2], ]))
  # TODO Generalizes 3D to d-D. Test after 15 Jan 2016 when return to project.
  #   Remove above hardwired 3D when works.
  #for d in range (0, ndims):
  #  centersdd.append (np.zeros ([nbins[d], ]))
  # Get histogram centers from edges. n+1 edges means there are n centers.
  #   Copied from write_hist_3d.py.
  for i in range (0, ndims):
    centersdd[i][:] = ( \
      edgesdd[i][0:len(edgesdd[i])-1] + edgesdd[i][1:len(edgesdd[i])]) * 0.5

  widths = []
  widths.append (edgesdd [0] [1] - edgesdd [0] [0])
  widths.append (edgesdd [1] [1] - edgesdd [1] [0])
  widths.append (edgesdd [2] [1] - edgesdd [2] [0])
  # TODO Generalizes 3D to d-D. Test after 15 Jan 2016 when return to project.
  #   Remove above hardwired 3D when works.
  #for d in range (0, ndims):
  #  widths.append (edgesdd [d] [1] - edgesdd [d] [0])
  bin_volume = np.product (widths)
  
  # Every bin in the histogram has this volume
  # edgesdd: list of 3 1D numpy vector
  # centersdd: list of 3 1D numpy vector
  # bin_volume: float
  return edgesdd, centersdd, bin_volume


# Slightly faster than normalize_3d_hist_given_bins(), because this skips
#   calling find_bin_edges_and_volume().
# Useful if caller has already called find_bin_edges_and_volume(), and need
#   to repeatedly normalize 3d hists.
# Used by active_touch prob_of_observs.py.
def normalize_3d_hist_given_edges (hist, edgesdd, bin_volume):

  # Sum all bins in histogram to a scalar
  height = np.sum (hist)
  total_volume = height * bin_volume

  # If histogram is empty, no need to normalize, return original
  if total_volume == 0:
    print ('%sWARNING: normalize_3d_hist_given_edges(): Histogram total volume was 0! Is your histogram filled?%s' % (
      ansi_colors.WARNING, ansi_colors.ENDC))
    return hist

  # Point-wise divide histogram bin counts by total volume of histogram, to
  #   make total volume 1
  hist_norm = hist / total_volume

  return hist_norm


def normalize_3d_hist_given_bins (hist, nbins, bin_ranges3D):

  edgesdd, _, bin_volume = find_bin_edges_and_volume (nbins, bin_range3D)

  return normalize_3d_hist_given_edges (hist, edgesdd, bin_volume)


# This simply does the opposite of normalize_3d_hist_given_edges(). To
#   normalize, you divide by sum of bin counts * widths of histogram.
#   To unnormalize, you just multiply by the same thing. The precondition in
#   unnormalizing is that you stored the integer sum of bin counts from the
#   unnormalized histogram somewhere, so you still have access to it!
# Parameters:
#   integer_height_sum: Sum of integer heights of the histogram.
def unnormalize_3d_hist (hist, integer_height_sum):

  # Find original unnormalized volume of histogram
  _, _, bin_volume = find_bin_edges_and_volume (nbins, bin_range3D)
  total_volume = integer_height_sum * bin_volume

  hist_unnorm = hist * total_volume

  return hist_unnorm


# The official way in this system to linearize 3D histograms. Must do this
#   consistently for all files that use the histogram, otherwise you'd have
#   inconsistent data - e.g. some are row-major, some are colume-major,
#   then you get garbage data!
# Current files using this assumption without using this function:
#   triangle_sampling write_hist_dd.py (3D to 1D to write to file, uses exact
#     same line as here)
#   triangle_sampling load_hists.py (loads 1D back to 3D, assumes 3D to 1D
#     at time of writing uses a line exactly same as here)
# Current files calling this function:
#   active_touch prob_of_observs.py
def linearize_hist (histdd):

  hist_linear = np.reshape (histdd, [histdd.size, ]);
  return hist_linear


# Parameters:
#   h1, h2: Two histograms btw which to compute the chi-squared distance
def get_chisqr_dist (h1, h2):

  # Ref formula: http://stats.stackexchange.com/questions/184101/comparing-two-histograms-using-chi-square-distance
  # Add a small offset, so don't get division by 0.
  return np.sum ((h2 - h1) ** 2 / (h2 + h1 + 1e-6))


def get_l2_dist (h1, h2):

  # sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  return np.sqrt (np.sum ((h2 - h1) ** 2))


# Returns a distance btw 0 and 1
def get_inner_prod_dist (h1, h2):

  # Dot product is cos. If h1 == h2 or h1 == -h2, then cos is 1 or -1, resp.
  #   If h1 _|_ h2, then cos is 0. So bigger the cosine magnitude, the closer
  #   the two vectors. This is opposite to definition of "distance", which is
  #   smaller the closer. So do 1 minus.
  # dot product = |A||B| cos (theta)
  #   cos(theta) = dot product / (|A| |B|)
  return 1 - np.abs (np.dot (linearize_hist (h1), linearize_hist (h2)) /
    (np.linalg.norm (h1) * np.linalg.norm (h2)))

