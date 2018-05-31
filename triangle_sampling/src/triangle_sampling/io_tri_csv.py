#!/usr/bin/env python

# Mabel Zhang
# 9 Apr 2016
#
# Refactored from triangles_on_robot_to_hists.py
#
# Used by triangles_collect.py and sample_gazebo_playback.py.
#

import os
import csv

import numpy as np

# My packages
from util.ansi_colors import ansi_colors
from triangle_sampling.config_hist_params import TriangleHistogramParams as \
  HistP


# Parameters:
#   tri_name: File full path, a .csv file in csv*_tri directory
#   read_all_params: True for reading all 6 triangle parameters, not just
#     the chosen 3.
#   params: List of 3 strings, e.g. ['l0', 'l1', 'a0']
#   decimeter: Let caller decide whether to use decimeters. e.g.
#     hist_conf_writer.py would NOT want to be in decimeters mode.
# Returns NumPy 2D array of 3 x nTriangles size (or 6 x if read_all_params).
#   Each of the 3 rows stores one parameter for all n triangles in the file.
#   Now converted to a NumPy array of 3 x nTriangles.
# Also returns a tuple of 3 (or 6 if read_all_params) strings for the
#   triangle parameters read.
def read_tri_csv_file (tri_name, read_all_params=False, params=None):

  if not os.path.exists (tri_name):
    print ('%sFile does not exist: %s. Check input and try again.%s' % (\
      ansi_colors.FAIL, tri_name, ansi_colors.ENDC))
    return (None, None)

  L0_IDX = HistP.L0_IDX
  L1_IDX = HistP.L1_IDX
  L2_IDX = HistP.L2_IDX

  A0_IDX = HistP.A0_IDX
  A1_IDX = HistP.A1_IDX
  A2_IDX = HistP.A2_IDX


  L0 = HistP.L0
  L1 = HistP.L1
  L2 = HistP.L2

  A0 = HistP.A0
  A1 = HistP.A1
  A2 = HistP.A2

  if read_all_params:
    params = (L0, L1, L2, A0, A1, A2)


  # To store triangles read from file. Input to histogramdd(), so must
  #   be (nSamples x dimension) array.
  tris = []
  for i in range (0, len (params)):
    tris.append ([])


  # Read the 3 triangle params into a Python list
  with open (tri_name, 'rb') as tri_file:

    tri_reader = csv.DictReader (tri_file)

    # Row is a dictionary. Keys are headers of csv file
    # Each row is 1 triangle.
    for row in tri_reader:

      # Read just pr1, pr2, pr3 params
      if not read_all_params:
        # Get the three triangle params of interest
        tris [0].append (float (row [params [0]]))
        tris [1].append (float (row [params [1]]))
        tris [2].append (float (row [params [2]]))

      # Read all 6 params in file. This is only used by
      #   hist_conf_writer.py
      else:

        tris [L0_IDX].append (float (row [L0]))
        tris [L1_IDX].append (float (row [L1]))
        tris [L2_IDX].append (float (row [L2]))

        tris [A0_IDX].append (float (row [A0]))
        tris [A1_IDX].append (float (row [A1]))
        tris [A2_IDX].append (float (row [A2]))


  # Convert to NumPy array of size 3 x nTriangles (or 6 x nTriangles if
  #   read_all_params)
  tris = np.asarray (tris)

  return (tris, params)


