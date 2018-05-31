#!/usr/bin/env python

# Mabel Zhang
# 8 Apr 2016
#
# Reader and writer functions for *_per_move.csv log files from Gazebo training.
#   Refactored from triangles_collect.py.
#
# Used by triangles_collect.py to write, and sample_gazebo_playback.py to read.
#

# Python
import os
import csv

# NumPy
import numpy as np

# My packages
from triangle_sampling.config_paths import get_robot_obj_params_path


class PerMoveConsts:

  per_move_column_titles = [
    # Won't record collect_idx, `.` it makes concatenating btw runs harder,
    #   would have to increment all indices in current run by the number of
    #   moves in the previous file being picked up!
    # This is easy to infer anyway, the number of lines in file is the total
    #   number of moves, the line number is the current number of moves. No
    #   need to record to file! Save some disk space.
    #'collect_idx',
    'n_new_pts', 'n_new_tris_hand', 'n_new_tris_robot']


# Parameters:
#   per_move_prefix: Permanent object name. For Gazebo data, this may be
#     timestamp (during training in sample_gazebo.py), or permanent object
#     name + timestamp (after training has done, reading for playback in
#     sample_gazeboy_playback.py).
#   mode_suffix: e.g. 'gz_' for Gazebo. Set by config_paths.py
#     config_hist_paths_from_args(..., mode_append=False)
#   obj_cat: If per_move_prefix is a timestring, set this to empty string ''.
#     File will be stored to root of pcd_gz_collected_params.
#     If per_move_prefix is permanent object name + timestring, set this to
#     object category, `.` permanent file would have been stored to
#     pcd_gz_collected_params/<obj_cat>/ .
def get_per_move_name (per_move_prefix, mode_suffix, obj_cat=''):

  per_move_path = get_robot_obj_params_path (mode_suffix)
  if obj_cat:
    per_move_path = os.path.join (per_move_path, obj_cat)

  per_move_name = os.path.join (per_move_path,
    per_move_prefix + '_per_move.csv')

  return per_move_path, per_move_name


# There are more than 1 contact at each move. You can't correspond a point
#   with a number of triangles, because you won't know which point is in
#   which triangle! A point is also in multiple triangles! So the best you
#   can do, is to record the move number, the number of points seen in that
#   move, and the number of triangles seen in that move. That data is still
#   helpful `.` it tells you how many lines in pcd and _hand.csv file to
#   read, to play back the training.
# Record for each point in PCD, how many triangles were obtained.
#   This allows the user to later read the PCD to infer how many triangles in
#   _hand.csv are obtained from each point, thereby "re-playing" the training,
#   without having to rerun training!!! Valuable data to have, to save hours
#   rerunning Gazebo training!!
# Parameters:
#   collect_idx: 1-based. Last index in file should = number of collects
# Used by triangles_collect.py
def record_per_move_n_pts_tris (writer, n_pts, n_tris_hand,
  n_tris_robot):

  row = {
    PerMoveConsts.per_move_column_titles [0] : n_pts,
    PerMoveConsts.per_move_column_titles [1] : n_tris_hand,
    PerMoveConsts.per_move_column_titles [2] : n_tris_robot}

  writer.writerow (row)


# Reads the content of a *_per_move.csv file into a NumPy matrix
# Parameters:
#   per_move_name: Full path to a *_per_move.csv file
def read_per_move_file (per_move_name):

  # Init to 0 rows, number of columns same as that in the _per_move.csv file
  per_move_data = np.empty ((0, len (PerMoveConsts.per_move_column_titles)),
    dtype=int)

  with open (per_move_name, 'rb') as per_move_file:

    per_move_reader = csv.DictReader (per_move_file)

    # Read all rows. Each row is 1 x 6 triangle params
    for row in per_move_reader:

      # 1 x 6 NumPy 2D array. Must be 2D to append correctly to n x 6 mat
      row_np = np.array (
        [[int (row [PerMoveConsts.per_move_column_titles [0]]),
          int (row [PerMoveConsts.per_move_column_titles [1]]),
          int (row [PerMoveConsts.per_move_column_titles [2]])]])

      # n x 6 NumPy 2D array
      per_move_data = np.append (per_move_data, row_np, axis=0)

  # These two columns are: n_pts, n_tris_hand
  return per_move_data [:, 0], per_move_data [:, 1]


