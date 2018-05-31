#!/usr/bin/env python

# Mabel Zhang
# 29 Jan 2017
#
# Utility functions for I/O operations on feasible poses saved to file.
# Used by plan_poses.py (write), execute_poses.py (read).
#

# Python
import os
import pickle

import numpy as np

# My packages
from tactile_collect import tactile_config


# Returns obj_cat/obj_name.pkl
def format_poses_base (obj_name, obj_center, deg_step):

  obj_base = ('%s_x%.4f_y%.4f_z%.4f_%gdeg.pkl' % (obj_name,
    obj_center[0], obj_center[1], obj_center[2], deg_step))

  return obj_base

# poses: n x 7 NumPy array. Each row is (tx ty tz qx qy qz qw)
def write_feasible_poses (poses, obj_cat, poses_base):

  poses_path = tactile_config.config_paths ('custom',
    'triangle_sampling/poses_bx/' + obj_cat)
  poses_name = os.path.join (poses_path, poses_base)

  with open (poses_name, 'wb') as f:
    # HIGHEST_PROTOCOL is binary, good performance. 0 is text format,
    #   can use for debugging.
    pickle.dump (poses, f, pickle.HIGHEST_PROTOCOL)

  print ('%d feasible poses written to %s' % (poses.shape [0], poses_name))


# poses_bases: List of strings. Each string is base
#   name of poses file saved by write_feasible_poses().
def read_training_poses (poses_bases, obj_cats):

  poses_path = tactile_config.config_paths ('custom',
    'triangle_sampling/poses_bx/')

  # Create an empty array without knowing its size
  # http://stackoverflow.com/questions/19646726/unsuccessful-append-to-an-empty-numpy-array
  poses = np.array ([])

  for i in range (len (poses_bases)):

    poses_name = os.path.join (poses_path, obj_cats [i], poses_bases [i])
    print ('Loading poses from %s...' % (poses_name))

    with open (poses_name, 'rb') as f:
      curr_poses = pickle.load (f)

      # Init to the right number of columns
      if poses.size == 0:
        poses = poses.reshape (0, curr_poses.shape [1])
      # Concatenate to larger array
      poses = np.vstack ((poses, curr_poses))
 
  print ('Total %d poses loaded' % (poses.shape [0]))

  return poses


