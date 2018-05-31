#!/usr/bin/env python

# Mabel Zhang
# 5 Feb 2016
#
# Define constants for .pkl files that store active touch training.
#
# Used by io_probs.py, which is used by sample_gazebo.py.
#

import numpy as np

class ActivePKLConsts:

  # Indices in obs_probs [catid] [src_obs] [action] [tgt_obs] list, a 3-elt
  #   list
  ABS_SRC_POSE = 1
  ABS_TGT_POSE = 2


  # Dictionary key constants

  # Meta info
  DAE_NAME_K = 'dae_name'
  OBJ_CENTER_K = 'obj_center'
  TRI_PARAMS_K = 'tri_params'
  META_KEYS = (DAE_NAME_K, OBJ_CENTER_K, TRI_PARAMS_K)

  # n-storage data structure
  ABS_POSES_K = 'abs_poses'
  OBSERVS_K = 'observs'
  OBS_PROBS_K = 'obs_probs_n'

 
# Check if the data passed in saved absolute poses. If so, can recover
#   abs_poses and observs (which maps from an abs_pose to observation(s))
#   at those poses.
# Returns (abs_poses, observs) if absolute poses were saved in obs_probs.
#   Otherwise return ([], {}).
def get_active_abs_poses_nxn (obs_probs):

  abs_poses = []
  observs = {}

  # TODO: Eventually need a unified function to read this dictionary...
  #   Now I have code here, code in active_touch read_movement_probs.py,
  #   if I change structure of dictionary, it's not fun... I'd have to
  #   change reader and writers everywhere... Should make a file called
  #   active_probs_io.py to read and write to this format.
  for catid in obs_probs.keys ():

    for src_obs in obs_probs [catid].keys ():

      for action in obs_probs [catid] [src_obs].keys ():

        for tgt_obs in obs_probs [catid] [src_obs] [action].keys ():

          # No debug absolute poses saved. Then can't get absolute poses
          if len (obs_probs [catid] [src_obs] [action] [tgt_obs]) == 1:
            print ('%sNo absolute poses were saved in the specified active probabilities file. Cannot retrieve any absolute pose information.%s' % ( \
              ansi_colors.WARNING, ansi_colors.ENDC))
            return [], {}

          # Add absolute poses to ret val
          src_pose = obs_probs [catid] [src_obs] [action] [tgt_obs] [ \
            ActivePKLConsts.ABS_SRC_POSE]
          if src_pose not in abs_poses:
            abs_poses.append (src_pose)
            observs [src_pose] = []
          observs [src_pose].append (src_obs)

          tgt_pose = obs_probs [catid] [src_obs] [action] [tgt_obs] [ \
            ActivePKLConsts.ABS_TGT_POSE]
          if tgt_pose not in abs_poses:
            abs_poses.append (tgt_pose)
            observs [tgt_pose] = []
          observs [tgt_pose].append (tgt_obs)

  # Convert to nPoses x 7 NumPy array
  if abs_poses:
    abs_poses = np.asarray (abs_poses)

  return abs_poses, observs


