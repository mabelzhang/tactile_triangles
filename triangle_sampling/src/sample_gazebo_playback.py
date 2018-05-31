#!/usr/bin/env python

# Mabel Zhang
# 8 Apr 2016
#
# Reads *_per_move.csv for an object, to get the number of points and triangles
#   that were recorded at each collection time by triangles_collect.py.
#   Then reads the corresponding number of points from *.pcd, and number of
#   triangles from *_hand.csv.
# * is the permanent object name, specified in the meta file in ../config/ .
#

# ROS
import rospy
import rospkg

# Python
import os
import argparse
import csv

# My packages
from triangle_sampling.parse_models_list import parse_meta_one_line
from triangle_sampling.config_paths import parse_args_for_svm, \
  config_hist_paths_from_args, get_pcd_path, get_robot_tri_path
from util.pcd_write_header import pcd_read_pts_to_array
from triangle_sampling.config_hist_params import TriangleHistogramParams as \
  HistP

# Local
from io_per_move import get_per_move_name, read_per_move_file
from io_tri_csv import read_tri_csv_file


def main ():

  # Copied from prob_of_observs.py
  args, valid = parse_args_for_svm ()
  if not valid:
    return

  if args.gazebo:
    _, tri_nbins_subpath, hist_parent_path, hist_path, \
      img_mode_suffix, tri_suffix, tri_paramStr = \
        config_hist_paths_from_args (args, mode_append=False)
  # So far only ran on Gazebo. Can copy PCL and mixed modes if-stmts
  #   from triangles_svm.py


  # Get a list of the three triangle param names, from the comma-separated
  #   string
  tri_params = tri_paramStr.split (',')
  # Strip any spaces
  tri_params = [p.strip() for p in tri_params]


  # Consts

  #RECORD_PROBS = True
  #io_probs_node = IOProbs (obj_center, pickup_leftoff=False, pickup_from_file='')


  # Config some paths

  meta_base = args.meta

  # Copied from sample_gazebo.py
  rospack = rospkg.RosPack ()
  pkg_path = rospack.get_path ('triangle_sampling')
  meta_name = os.path.join (pkg_path, 'config/', meta_base)

  pcd_path = get_pcd_path (img_mode_suffix)
  tri_path = get_robot_tri_path (img_mode_suffix)


  # Size: number of categories
  catnames = []
  catcounts = []
  # Size: number of object files
  catids = []

  # Read meta list file line by line
  with open (meta_name, 'rb') as meta_file:

    for line in meta_file:

      line = line.strip ()

      # Parse line in file, for base name and category info
      #   Ret val is [basename, cat_idx], cat_idx for indexing catnames.
      parse_result = parse_meta_one_line (line, catnames, catcounts, catids)
      if not parse_result:
        continue

      obj_name = parse_result [0]
      cat_idx = parse_result [1]

      obj_base = os.path.basename (obj_name)
      obj_base = os.path.splitext (obj_name) [0]

      obj_cat = catnames [cat_idx]

      # Load per_move log
      per_move_path, per_move_name = get_per_move_name (obj_base,
        img_mode_suffix, obj_cat=obj_cat)
      n_pts_mat, n_tris_hand_mat = read_per_move_file (per_move_name)

      # Load .pcd point cloud file
      pcd_name = os.path.join (pcd_path, obj_cat, obj_base + '.pcd')
      # nPts x 6
      pcd_mat = pcd_read_pts_to_array (pcd_name)

      # Load _hand.csv triangle file
      tri_name = os.path.join (tri_path, obj_cat, obj_base + '_hand.csv')
      #tri_file = open (tri_name, 'rb')
      #tri_reader = csv.DictReader (tri_file)
      # 3 x nTriangles
      tri_mat, _ = read_tri_csv_file (tri_name, read_all_params=False,
        params=tri_params)

      print (pcd_mat.shape)
      print (tri_mat.shape)

      n_pts_cached = 0

      n_pts_seen = 0
      n_tris_seen = 0

      # Reset vars for each object
      #if RECORD_PROBS:
      #  io_probs_node.reset_vars ()

      # For each collection time recorded in _per_move log
      for c_i in range (0, len (n_pts_mat)):

        # This happens sometimes. Should add an if-stmt in triangles_collect.py
        #   to not write to _per_move file if there are 0 contacts... TODO
        if n_pts_mat [c_i] == 0 and n_tris_hand_mat [c_i] == 0:
          continue

        # No triangles are calculated. Points were being cached
        if n_tris_hand_mat [c_i] == 0:
          # Increment number of points seen before next collection time
          n_pts_cached += (n_pts_mat [c_i])
          continue

        # Increment number of points seen before next collection time
        n_pts_cached += (n_pts_mat [c_i])

        # Grab n_pts_mat [c_i] points from pcd file
        pts = pcd_mat [n_pts_seen : n_pts_seen + n_pts_cached, :]
        n_pts_seen += n_pts_cached
        # Reset after use
        n_pts_cached = 0

        # Replaced this with read_tri_csv_file ()
        # Load n_tris_hand_mat [c_i] points from triangle file
        #for t_i in range (0, n_tris_hand_mat [c_i]):
        #  l = tri_reader.next ()
        #  print (l)

        # Grab n_tris_hand_mat [c_i] points from triangle file
        tris = tri_mat [:, n_tris_seen : n_tris_seen + n_tris_hand_mat [c_i]]
        n_tris_seen += (n_tris_hand_mat [c_i])

        print (pts.shape)
        print (tris.shape)


        # TODO: Record probabilities data, simulated from per_move logs, if
        #   it can't be exactly duplicated as training time. Look at
        #   sample_gazebo.py to remember how I collected the probs data...
        #if RECORD_PROBS:
  
        # 21 Apr 2016
        #
        # Wait... I just realized this is impossible to do!
        # `.` even if I don't store the abs_pose, I need to calculate the
        #   relative movement action. That requires knowing the hand pose,
        #   which I do not save during training!! The pcd only saves the
        #   contact points! You could try to compute using geo_ellipsoid, but
        #   that's risky, because sample_gazebo.py can change the parameters
        #   of the ellipsoid, and here we don't know what params to set to get
        #   the right ellipsoid. So... there is no way to recover probs and
        #   costs data even with per_move files saved!!!
        # I'd have to also save all the poses during training...
        #   Maybe it's easier to just simulate probability things on PCD stuff
        #   first...
        # Or, can I just simulate it for now, before I add code to save all
        #   poses (adding that code is annoying too, how to map btw hand poses
        #   and pcd points and triangles? It may not be straight forward).
        #   Might as well simulate on Gazebo data, if I'm going to do it on
        #   PCL data. Gazebo data is closer to real world.
        #
        #   How to simulate hand pose? Use pcd point position as hand
        #   position. What about hand orientation? Can I use the pcd normals
        #   for that? Are they nonzero in the .pcd files? Yes I checked they
        #   are non-zero. That is one point. But one point != one hand pose,
        #   you can observe many points at each hand pose!
        #
        #   Oh wait there IS data for me to know how many contacts per hand
        #   pose, that's what the per_move.csv file was for! THAT was why I
        #   was willing to store lines with #, 0, 0!! Even if no triangles
        #   are computed, I still stored the number of points, `.` I wanted
        #   to store how many points are touched in each move!!!
        #
        #   Then say n_new_pts in per_move.csv file is 5, to get the simulated
        #   hand position, just take the average center of the 5 points. It
        #   doesn't make sense in practice but I need data now, so make do.
        #   To get the simulated orientation, take.... avg of the 5 normals???
        #   That seems random. Oh well if it gives me data then okay.
        #
        #   That gives me one abs hand pose per line in per_move. Finding the
        #   diff btw the hand pose in two consecutive lines in per_move.csv
        #   will give me the relative movement action a.
        #
        #     observs[abs_pose] = latest_tris
        #
        #   abs_pose is obtained as above.
        #   latest_tris is a list of n_new_tris_hand triangles loaded from
        #     tri_csv file. It will only be once per collection time, not once
        #     per move... Oh... then I STILL have the problem from before, in
        #     sample_gazebo.py!!
        #
        # Let's be clear the objective of active touch training. We really do
        #   need triangles PER MOVE and points PER MOVE, not the accumulated
        #   data, `.` then we can correspond the triangles and points observed
        #   at each hand pose!!! Should restructure per_move.csv file so that
        #   it doesn't use 0s to indicate no triangles computed......
        #   
        #   Never mind, there's just no way around recomputing triangles while
        #   playing back. `.` in training, there's simply no triangles being
        #   computed at each hand pose! So you have nothing to store for those
        #   other lines!!!
        #
        #   So the ONLY way to get triangles for each hand pose, when play
        #   back, is to call the function in triangle_sampling/sample_reflex.py
        #   to calculate triangles from points. I already have points for each
        #   hand pose.
        #
        #   Okay... let's do that.
        #
        # These tasks:
        #   1. Call sample_reflex.py to calculate triangles from points at each
        #      hand pose (each line in per_move.csv file). This gives triangle
        #      observations (obs_tgt) at each line in per_move.csv. obs_src is
        #      simply the obs_tgt from the previous line in per_move.csv.
        #   2. Simulate hand pose (position + rot) as described above. This
        #      gives action a.
        #   This should give us enough simulated data to pass to
        #     add_abs_pose_and_obs() for the active touch data!

        #   TODO NOW HERE implement tasks 1 & 2 above.
        # 
        #
        #   X|
        #   Wait... Don't I have the training parameters saved to 
        #   pcd_gz_collected_params/*.csv??? That'll tell me the deg_step for
        #   the ellipsoid, and object center right??? Then I CAN regenerate the correct ellipsoid! (except I've fixed some bugs in it, so now it'll no longer be the same ellipsoid... So I'd say still should save the actual hand poses, instead of parameters to compute from, `.` code can change from the time of training, ten you won't generate the same poses, and data played back will be wrong...



      # end for each line loaded from _per_move file


      #tri_file.close ()

      print (n_pts_seen)
      print (n_tris_seen)


if __name__ == '__main__':
  main ()

