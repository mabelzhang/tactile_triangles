#!/usr/bin/env python

# Mabel Zhang
# 21 Jan 2016
#
# Configure parameters for triangle histograms.
# Used by multiple files:
#   triangles_on_robot_to_hists.py
#   triangle_sampling/load_hists.py
#   sample_pcl_calc_hist.py
#   write_hist_3d.py


class TriangleHistogramParams:

  # TODO: Only used in hist_conf_writer.py. When works, use in
  #   sample_pcl_calc_hist.py and triangles_on_robot_to_hists.py
  #   as well.

  #####
  # Constants. Do not change these
  #####

  L0 = 'l0'
  L1 = 'l1'
  L2 = 'l2'

  A0 = 'a0'
  A1 = 'a1'
  A2 = 'a2'


  # Indexes triangle .csv files' columns
  L0_IDX = 0
  L1_IDX = 1
  L2_IDX = 2
  A0_IDX = 3
  A1_IDX = 4
  A2_IDX = 5
  TRI_PARAMS_IDX = [L0_IDX, L1_IDX, L2_IDX, A0_IDX, A1_IDX, A2_IDX]

  TRI_PARAMS = [''] * 6
  TRI_PARAMS [TRI_PARAMS_IDX [L0_IDX]] = L0
  TRI_PARAMS [TRI_PARAMS_IDX [L1_IDX]] = L1
  TRI_PARAMS [TRI_PARAMS_IDX [L2_IDX]] = L2
  TRI_PARAMS [TRI_PARAMS_IDX [A0_IDX]] = A0
  TRI_PARAMS [TRI_PARAMS_IDX [A1_IDX]] = A1
  TRI_PARAMS [TRI_PARAMS_IDX [A2_IDX]] = A2


  #####
  # User adjust
  #   Do not set these in a different file! Set them here.
  #####

  # Pick a set of traingle parameters
  # Do NOT read these directly, unless you are in an entry files that
  #   WRITES these numbers TO hist_conf.csv (like hist_conf_writer.py) or
  #   writing probs .pkl files (like sample_gazebo.py).
  #   Other files should read hist_conf.csv to get the configuration for the
  #   files they're processing, not get from here!
  # In your code, do NOT use uppercase. Use lowercase to distinguish btw
  #   constants in this file, and parameters in your local function.
  PR1_IDX = L0_IDX
  PR2_IDX = L1_IDX
  PR3_IDX = A0_IDX

  # For 3D histograms
  # Do NOT read these directly, unless you are in an entry files that
  #   WRITES these numbers TO hist_conf.csv. Other files should read
  #   hist_conf.csv to get the configuration for the files they're processing,
  #   not get from here!
  #   This is because when statting from stats_hist_num_bins.py, you should
  #   not use numbers here, `.` stats_hist_num_bins passes in different numbers
  #   to use to run everything, and stat the outcome for different parameters;
  #   accessing these would always run the same thing!
  bins3D = [10, 10, 10] #[15, 15, 15] #[8, 8, 9] #[30, 30, 36]

  decimeter = False


  #####
  # Constants
  #####

  # Get corresponding strings
  PR1 = TRI_PARAMS [PR1_IDX]
  PR2 = TRI_PARAMS [PR2_IDX]
  PR3 = TRI_PARAMS [PR3_IDX]

  # List of strings
  PRS = (PR1, PR2, PR3)


  # For hist_conf.csv
  BINS3D_TITLES = [PR1 + '_nbins', PR2 + '_nbins', PR3 + '_nbins']


