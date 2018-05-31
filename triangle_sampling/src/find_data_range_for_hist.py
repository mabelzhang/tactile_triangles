#!/usr/bin/env python

# Mabel Zhang
# 22 Jan 2016
#
# Refactored from sample_pcl_calc_hist.py, to be used for Gazebo-
#   trained data.
#


import numpy as np

from triangle_sampling.load_hists import scale_bin_range_to_decimeter
# Shorten a name http://stackoverflow.com/questions/840969/how-do-you-alias-a-python-class-to-have-another-name-without-using-inheritance
from triangle_sampling.config_hist_params import TriangleHistogramParams as \
  HistP


# Find min and max in triangles data for one object.
# Parameters:
#   tri_params_one_obj: NumPy matrix of nTriangles x nDims
#   min_vals, max_vals: Current min and max so far.
# Returns updated min_vals, max_vals. If running on multiple objects, caller
#   should overwrite their min_vals and max_vals, then pass in with next
#   object, so the min and max are calculated across all objects.
def find_data_range_in_one_obj (tri_params_one_obj, min_vals, max_vals):

  nDims = np.shape (tri_params_one_obj) [1]

  for dim in range (0, nDims):
    # Minimum and maximum value across all triangles in this object, in this
    #   dimension.
    curr_min = np.min (tri_params_one_obj [:, dim])
    curr_max = np.max (tri_params_one_obj [:, dim])

    # If min in current object's current dimension is smaller than stored
    #   min, update the min value for this dimension.
    if curr_min < min_vals [dim]:
      min_vals [dim] = curr_min
    if curr_max > max_vals [dim]:
      max_vals [dim] = curr_max

  return min_vals, max_vals


# Find min and max in triangles data for ALL objects, to define histogram min
#   and max edges.
# Parameters:
#   tri_params: Triangles data for all objects (`.` need ALL objects, in order
#     to know a histogram min/max range that would work for all objects'
#     descriptors. Else histogram won't be comparable across different objects!
#     Format:
#       A Python list of nObjects NumPy 2D arrays, arrays sized nTriangles x
#       nDims. tri_params [i] [:, dim] gives data for object i, dimension dim.
#       6 dimensions max possible, for 6 params of triangle.
#   decimeter: True if want triangle lengths to be in decimeters unit, instead
#     of meters. This is to balance out the lengths and triangle measures, so
#     they are on the same magnitude 1.0. Otherwise angles are on 1.0
#     magnitude, lengths are on 0.1 magnitude, because most things are a few
#     centimeters or teens, never more than a meter. Then angles' magnitude
#     would dominate histogram and can give you numerical problems e.g. for
#     KDE. e.g. Lengths' bins would be 0.01 in width, angles bins would be
#     0.1 width. (I forgot the exact problem we had, check Google Docs notes.)
def find_data_range_for_hist (tri_params, decimeter,
  bins3D=HistP.bins3D, prs_idx=(HistP.PR1_IDX, HistP.PR2_IDX, HistP.PR3_IDX)):

  #####
  # Find range for histogram bins, so that the same range can be used for
  #   ALL objects. This is a requirement for computing histogram
  #   intersections btw each possible pair of objects.
  # Default range by np.histogram() is simply (min(data), max(data)). So we
  #   just need to find min and max of data from ALL objects.
  #####

  nObjs = len (tri_params)
  if nObjs < 1:
    print ('find_data_range_for_hist(): No objects received. Returning empty lists.')
    return [], []

  # tri_params [i] is nTriangles x nDims
  nDims = np.shape (tri_params [0]) [1]

  min_vals = [1000] * nDims
  max_vals = [-1000] * nDims

  for i in range (0, nObjs):
    min_vals, max_vals = find_data_range_in_one_obj (tri_params [i],
      min_vals, max_vals)

  bin_range, bin_range3D, header, row = make_bin_ranges_from_min_max ( \
    min_vals, max_vals, decimeter, bins3D, prs_idx)

  return bin_range, bin_range3D, header, row


# Parameters:
#   min_vals, max_vals: ret vals from find_data_range_in_one_obj() above.
#   decimeter: True if want to use decimeters as unit for lengths, instead of
#     meters, so that histogram axes balance out at magnitude of 1.0 for
#     lengths and angles.
#   bins3D: Python list of 3 integers, specifying number of bins for 3D
#     histogram. e.g. [10, 10, 10].
def make_bin_ranges_from_min_max (min_vals, max_vals, decimeter,
  bins3D=HistP.bins3D, prs_idx=(HistP.PR1_IDX, HistP.PR2_IDX, HistP.PR3_IDX)):

  pr1_idx, pr2_idx, pr3_idx = prs_idx

  # Get corresponding strings
  pr1 = HistP.TRI_PARAMS [pr1_idx]
  pr2 = HistP.TRI_PARAMS [pr2_idx]
  pr3 = HistP.TRI_PARAMS [pr3_idx]

  # Min and max values in each dimension, i.e. bin EDGES at two ends, not
  #   bin centers.
  # 6 2-tuples
  bin_range = [(min_vals[i], max_vals[i]) for i in range (0, len(min_vals))]
  # 3 2-tuples
  bin_range3D = [(min_vals[i], max_vals[i]) for i in [pr1_idx,
    pr2_idx, pr3_idx]]

  # Only scale L# lengths, not A# angles!
  if decimeter:
    bin_range, bin_range3D = scale_bin_range_to_decimeter (bin_range, 
      bin_range3D)

  print ('3D bin mins (meters and radians):')
  print ('%f %f %f' % (bin_range3D[0][0], bin_range3D[1][0], bin_range3D[2][0]))
  print ('3D bin maxes:')
  print ('%f %f %f' % (bin_range3D[0][1], bin_range3D[1][1], bin_range3D[2][1]))


  #####
  # Prepare the row to write to histogram config file hist_conf.csv.
  #####

  BINS3D_TITLES = [pr1 + '_nbins', pr2 + '_nbins', pr3 + '_nbins']

  # Labels for histogram config file. These correspond to bin_range3D
  # e.g. ['l0_min', 'l0_max', 'l1_min', 'l1_max', 'a0_min', 'a0_max']
  BIN_RANGE_3D_TITLES = [ \
    pr1 + '_min', pr1 + '_max',
    pr2 + '_min', pr2 + '_max',
    pr3 + '_min', pr3 + '_max']

  # Need just one row, with headers to say what each column is
  # Example of hist_conf.csv, which writes header and row to csv:
  #   l0_nbins,l1_nbins,a0_nbins,l0_min,l0_max,l1_min,l1_max,a0_min,a0_max
  #   10,10,10,0.0079200007021427155,0.1477300707226977,0.0056117032604381598,0.13882817046324594,1.0911397185895479,3.1408521787938208

  # List of 9 strings
  header = []
  # 3 strings
  header.extend (BINS3D_TITLES)
  # 6 strings
  header.extend (BIN_RANGE_3D_TITLES)
 
  # List of 9 numbers
  row = []
  # 3-element list
  row.extend (bins3D)
  print ('Number of 3D bins: %d %d %d' % (bins3D[0], bins3D[1], bins3D[2]))
 
  # 3 2-tuples, [(, ), (, ), (, )]
  for i in range (0, len (bin_range3D)):
    for j in range (0, len (bin_range3D [0])):
      row.append (bin_range3D [i] [j])


  return bin_range, bin_range3D, header, row

