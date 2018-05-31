#!/usr/bin/env python

# Mabel Zhang
# 15 Mar 2016
#
# Do recording I used to do manually, to make sampling easier on me.
#
# Loads output csv file from sample_gazebo.py log_this_run(), generates a
#   human-readable text file, like ones I used to manually log.
#

# ROS
import rospkg

# Python
import argparse
import subprocess
import os
import csv
#from __future__ import print_function

import numpy as np

# My packages
from util.ansi_colors import ansi_colors
from triangle_sampling.config_paths import get_pcd_path, \
  get_robot_obj_params_path


class SampleGazeboConfigs:

  ELL_GRID_TYPE = 'ell_grid'
  CYL_GRID_TYPE = 'cyl_grid'

  hand_name = 'reflex'


  #####
  # Saving files from sampling
  #####

  # Whether to use timestamps as filename (good for testing, won't overwrite
  #   old files), or use object model file prefix name (good for real
  #   training, keeping the formal files. Will overwrite existing ones!)
  USE_TIMESTAMP_FILENAMES = True

  # Used for real training that goes into SVM. Set to False if just testing!
  #   Records obj, pickup file name, stats, etc to ../out/gz/per_run.csv.
  do_log_this_run = False

  RECORD_PROBS = True


  #####
  # Basic sampling parameters
  #####

  # At bottom center of object
  #   Don't set it too low, too close to ground, otherwise hand has no
  #     clearance and will collide with ground and fly to outerspace.
  # 6 Dec 2016: Setting to farther from /base, to test the bug fix of movement
  #   cost being computed on relative action, instead of absolute action. Test
  #   succeeded, but just leaving this here, this farther distance from /base
  #   can help ensure future such distance-dependent things are robust.
  obj_x = -1 #0
  obj_y = -1 #0
  obj_z = 0.3

  # Specify False if you only want to sample hand frame.
  #   Our focus is hand frame. You can sample robot frame as well, for
  #   comparison. For one, robot frame is probably more accurate, so it's
  #   good for debugging. But if you are running in simulation and tight on
  #   time, then don't do robot frame.
  sample_robot_frame = False

  CYL_PALM_THICKNESS = 0.10
  # IROS 2016 config for gazebo teleport is 0.10 for most objs. 0.08 for teapot
  #ELL_PALM_THICKNESS = 0.10 #0.105
  # IROS 2017 config on real robot
  # Train
  #ELL_PALM_THICKNESS = 0.08  # canister38 on 4 cm pedestal
  #ELL_PALM_THICKNESS = 0.06  # mug27 on 4 cm pedestal
  #ELL_PALM_THICKNESS = 0.08  # sqr_bowl_gray on 4 cm pedestal
  #ELL_PALM_THICKNESS = 0.06  # bowl39 on 4 cm pedestal
  #ELL_PALM_THICKNESS = 0.04  # bottle43 on 4 cm pedestal
  #ELL_PALM_THICKNESS = 0.04  # glass31 on 4 cm pedestal
  #ELL_PALM_THICKNESS = 0.02  # bottle44 on 4 cm pedestal
  #ELL_PALM_THICKNESS = 0.04  # mug37 on 4 cm pedestal
  #ELL_PALM_THICKNESS = 0.02  # jar35 on 4 cm pedestal
  # all objects. found that plan_poses.py sets this for all objects anyway!!!
  ELL_PALM_THICKNESS = 0.08

  # New constants 31 Jan 2017, used in plan_poses.py and execute_poses.py
  RADIUS_ELLIPSOID_CLEARANCE = 0.04
  # Clearance for realistically executable poses, accounting for wood mount
  #   on table. The lower poses, arm can't go that low on table to point
  #   upwards anyway, so don't bother wasting time.
  # Just setting to mount height. I don't think the arm can go lower than
  #   mount and face back up anyway, it'd have to go very low.
  #   This basically eliminates all poses with z below object. Reasonable.
  #ABOVE_TABLE_CLEARANCE = 0.14  # mug37 on tallest pedestal
  #ABOVE_TABLE_CLEARANCE = 0.18  # canister38 on 4 cm pedestal
  #ABOVE_TABLE_CLEARANCE = 0.12  # sqr_bowl_gray on 4 cm pedestal
  #ABOVE_TABLE_CLEARANCE = 0.13  # bowl39 on 4 cm pedestal
  #ABOVE_TABLE_CLEARANCE = 0.20  # bottle43 on 4 cm pedestal
  #ABOVE_TABLE_CLEARANCE = 0.18  # glass31 on 4 cm pedestal
  #ABOVE_TABLE_CLEARANCE = 0.20  # bottle44 on 4 cm pedestal
  #ABOVE_TABLE_CLEARANCE = 0.18  # mug37 on 4 cm pedestal
  ABOVE_TABLE_CLEARANCE = 0.13  # jar35 on 4 cm pedestal


  # Number of fore-finger preshapes to do at each wrist location
  # Official reflex-ros-pkg constants: cylinder 0.0, spherical 0.8, pinch 1.57
  # Calculated: 0.524 rads ~ 30 degs, 1.048 rads ~ 60 degs
  #   Or use 22.5 degs increments: 0.393 rads ~ 22.5 degs, 1.178 r ~ 67.5 degs
  #   Or just use 0.4 rads increments: 0.4, 0.8, 1.2
  # Pinch preshape makes finger break really easy on a cube.
  #preshapes = [0, 0.524, 0.8, 1.048]
  #preshapes = [0, 0.393, 0.8, 1.178]
  #preshapes = [0, 0.4, 0.8, 1.2]  # 4 preshapes, evenly spaced
  preshapes = [0, 0.8]  # 2 preshapes, cylinder, spherical. IROS config
  #preshapes = [0]  # 1 preshape, cylinder
  N_PRESHAPES = len (preshapes)

  # Rotations wrt hand frame, in radians
  #wrist_rots = [(0, 0, 0)]  # No rotation
  # z rots
  #wrist_rots = [(0, 0, 0), (0, 0, 45.0 / 180.0 * np.pi)]
  #wrist_rots = [(0, 0, 0), (0, 0, 45.0 / 180.0 * np.pi), (0, 0, -45.0 / 180.0 * np.pi)]
  # IROS config.
  #   -10 to -20 y-rot is really good in cylinder grid. Good in ellipsoid too.
  wrist_rots = [(0, 0, 0),
    (0, -10 / 180.0 * np.pi, 0)]
  # -20 wrt x is completely useless, whether above or below object. It just
  #   rotates hand away from object. Don't need this rotation.
  #wrist_rots = [(0, 0, 0),
  #  (0, -10 / 180.0 * np.pi, 0),
  #  (-20 / 180.0 * np.pi, 0, 0)]

  # Flag to cache contact points btw consecutive wrist positions. This includes
  #   the wrist_rots at each position too. i.e. accumulating across 2 positions
  #   means accumulating across the 2 rotations in the 1st position and the
  #   2 rotations in the 2nd position, 4 wrist poses total. i.e. wrist_rots is
  #   treated as a nested relationship inside the positions, not a separate
  #   quantity from position.
  # ACCUM_CONTACTS_BTW_POS overwrites ACCUM_CONTACTS_BTW_PRESHAPES.
  ACCUM_CONTACTS_BTW_POS = False
  # If 2 preshapes, use 5. Shouldn't use too many, as goal is allow obj to move.
  # If 4 preshapes, don't use 5, would get 20000 triangles per n choose 3
  #   round! Use 3 or 4, tried 3, might be too few. Try 4 next time.
  # IROS config: N_ACCUM_POS = 2
  N_ACCUM_POS = 2
  ACCUM_CONTACTS_BTW_PRESHAPES = False

  # If using cylinder, set this
  cyl_density = 0.03
  #cyl_density = 0.015
  # For debugging quickly
  #cyl_density = 0.1

  # If using ellipsoid, set this
  # 20 is very good.
  #   30 is not enough to produce good 1d hist inter for 8 cm sphere.
  ell_deg_step = 25 #30 #25 # 20


  # A place away from object and any obstacles, for hand to safely open
  SAFE_PLACE = [[1, 0, 0], [0, 0, 0, 1]]


  PROFILER = False
  STRICT_POSE = True



  #####
  # For per-run log file
  #####

  # Dictionary keys
  OBJ_CAT_K           = 'obj_cat'
  OBJ_BASE_K          = 'obj_base'
  GRID_TYPE_K         = 'grid_type'
  GRID_RES_K          = 'grid_res'
  PALM_THICKNESS_K    = 'palm_thickness'
  TIMESTRING_K        = 'timestring'
  PICKUP_TIMESTRING_K = 'pickup_timestring'
  PICKUP_ELL_IDX_K    = 'pickup_ell_idx'
  PICKUP_Z_K          = 'pickup_z'
  PICKUP_THETA_K      = 'pickup_theta'
  N_CUMU_PTS_K        = 'n_cumu_pts'
  N_CUMU_TRIS_K       = 'n_cumu_tris'
  ELAPSED_SECS_K      = 'elapsed_secs'

  per_run_column_titles = [OBJ_CAT_K, OBJ_BASE_K, GRID_TYPE_K, GRID_RES_K,
    PALM_THICKNESS_K, TIMESTRING_K, PICKUP_TIMESTRING_K, PICKUP_ELL_IDX_K,
    PICKUP_Z_K, PICKUP_THETA_K, N_CUMU_PTS_K, N_CUMU_TRIS_K, ELAPSED_SECS_K]

  # Folder in which per_run.csv and per_run.txt are to be written and read
  #per_run_path = get_robot_obj_params_path ('gz_')
  # Put in repo.
  rospack = rospkg.RosPack ()
  pkg_path = rospack.get_path ('triangle_sampling')
  per_run_path = os.path.join (pkg_path, 'out', 'gz')


  def __init__ (self, USE_CYL_GRID=False, pickup_leftoff=False,
    pickup_from_file='', pickup_z=0, pickup_theta=0, pickup_ell_idx=0):

    self.USE_CYL_GRID = USE_CYL_GRID

    # grid_res: ell_deg_step or cyl_density
    if USE_CYL_GRID:
      self.grid_type = self.CYL_GRID_TYPE
      self.grid_res = self.cyl_density
      self.PALM_THICKNESS = self.CYL_PALM_THICKNESS
    else:
      self.grid_type = self.ELL_GRID_TYPE
      self.grid_res = self.ell_deg_step
      self.PALM_THICKNESS = self.ELL_PALM_THICKNESS

      # For passing to geo_ellipsoid.py
      # Base vector to rotate by
      # Rotate wrt z, by ell_deg_step. This will keep z still at 0 0 1,
      #   just change x and y. This makes point at bottom center of
      #   ellipsoid grid, which is at 0 0 -1 after normalization, same as
      #   -quat_wrt, able to have multiple orientations. Otherwise, all
      #   poses at bottom center will have 0 0 0 1 identity orientation,
      #   because the rotating from -quat_wrt to the point on ellipsoid
      #   surface, 0 0 -1, is identity!
      self.ell_extra_rot_base = (0, 0, 1)


    # Instance var, not class var, `. each instance can have a different value
    self.pickup_leftoff = pickup_leftoff
    self.pickup_from_file = pickup_from_file
    # Cylinder grid
    self.pickup_z = pickup_z
    self.pickup_theta = pickup_theta
    # Ellipsoid grid
    self.pickup_ell_idx = pickup_ell_idx


  # Do recording I used to do manually, to make sampling easier on me.
  # Record a log of the run, including file name, file picked up from, n pts,
  #   n tris, time elapsed, etc.
  # Only tested for ellipsoid grid, because no longer using cylinder grid. I'm
  #   tired of making things backwards compatible and creating a lot of code
  #   that's not being used. Too much clutter.
  # Parameters:
  #   grid_type: 'ell_grid' or 'cyl_grid'
  #   palm_thickness: ELL_PALM_THICKNESS or CYL_PALM_THICKNESS
  #   pickup_ell_idx: Specify if 'ell_grid'
  #   pickup_z, pickup_theta: Specify if 'cyl_grid'
  #   elapsed_time: seconds, from Python time module
  # Ref static method http://stackoverflow.com/questions/735975/static-methods-in-python
  #   static members http://stackoverflow.com/questions/68645/static-class-variables-in-python
  # Not sure why I used classmethod, should use staticmethod
  #   > Because I need access to the object!!
  #   http://stackoverflow.com/questions/136097/what-is-the-difference-between-staticmethod-and-classmethod-in-python
  #@classmethod
  def log_this_run (self,
    obj_cat, obj_basename, palm_thickness,
    timestring, pickup_timestring, pickup_ell_idx, pickup_z, pickup_theta,
    n_cumu_pts, n_cumu_tris, elapsed_time):

    per_run_name = os.path.join (self.per_run_path,
      #timestring + '_per_run.csv')
      # Just do one big file for all objects
      'per_run.csv')
    print ('%sPer-run log will be outputted to %s%s' % \
      (ansi_colors.OKCYAN, per_run_name, ansi_colors.ENDC))
 
    per_run_column_titles = self.per_run_column_titles
 
    # If file exists, just append rows, without header line
    if os.path.exists (per_run_name):
      per_run_file = open (per_run_name, 'a')
      per_run_writer = csv.DictWriter (per_run_file,
        fieldnames=per_run_column_titles, restval='-1')
 
    # If file is new, write header line
    else:
      per_run_file = open (per_run_name, 'wb')
      per_run_writer = csv.DictWriter (per_run_file,
        fieldnames=per_run_column_titles, restval='-1')
      per_run_writer.writeheader ()
 
    row = {
      self.OBJ_CAT_K           : obj_cat,
      self.OBJ_BASE_K          : obj_basename,
      self.GRID_TYPE_K         : self.grid_type,
      self.GRID_RES_K          : self.grid_res,
      self.PALM_THICKNESS_K    : palm_thickness,
      self.TIMESTRING_K        : timestring,
      self.PICKUP_TIMESTRING_K : pickup_timestring,
      self.PICKUP_ELL_IDX_K    : pickup_ell_idx,
      self.PICKUP_Z_K          : pickup_z,
      self.PICKUP_THETA_K      : pickup_theta,
      self.N_CUMU_PTS_K        : n_cumu_pts,
      self.N_CUMU_TRIS_K       : n_cumu_tris,
      self.ELAPSED_SECS_K      : elapsed_time
    }
 
    per_run_writer.writerow (row)


  # Parameters:
  #   csv_name: Full path to a csv file outputted by sample_gazebo.py
  #     log_this_run().
  # Ref static method http://stackoverflow.com/questions/735975/static-methods-in-python
  #   static members http://stackoverflow.com/questions/68645/static-class-variables-in-python
  @classmethod
  def convert_per_run_csv_to_readable (cls):

    csv_name = os.path.join (cls.per_run_path, 'per_run.csv')
    csv_file = open (csv_name, 'rb')
    csv_reader = csv.DictReader (csv_file)
 
    # Replace .csv extension with .txt
    txt_name = os.path.splitext (csv_name) [0] + '.txt'
    txt_file = open (txt_name, 'wb')
    print ('Human-readable per-run log will be outputted to %s' % txt_name)
 
    # Init to invalid number, to indicate no writing of summary stats before 1st
    #   obj
    part_num = -1
    elapsed_secs = 0


    def write_end_of_obj_stats (row, elapsed_secs):

      # Write summary stats of the previous object
      txt_file.write ('%d pts\n' % int (row[cls.N_CUMU_PTS_K]))
      txt_file.write ('%d tris\n\n' % int (row[cls.N_CUMU_TRIS_K]))
     
      txt_file.write ('%g mins\n' % (elapsed_secs / 60.0))
     
      # Two empty lines
      txt_file.write ('\n\n')

 
    prev_row = None

    for row in csv_reader:
 
      # Empty string indicates start of a new object
      if not row[cls.PICKUP_TIMESTRING_K]:
 
        # If this is not the very first object, i.e. a previous object exists
        if prev_row:
          write_end_of_obj_stats (prev_row, elapsed_secs)

        # Reset part number to start of object, 1
        part_num = 1
        elapsed_secs = 0
 
      else:
        # Sanity check: check this row is picking up from prev row. Otherwise
        #   we don't want to count them together!
        if row[cls.PICKUP_TIMESTRING_K] != prev_row[cls.TIMESTRING_K]:
          # Skip this row. I haven't decided what to do if rows are out of
          #   order. Preferably maintain a dictionary of all timestrings in
          #   the csv, and map to elapsed time and the last part number... but
          #   hard to link multiple together if not in order! Need multiple
          #   values with each key: parent, next, elapsed time, part_num. Too
          #   over kill for this simple task. So will just assume rows are in
          #   order!
          print ('%sWARN: This row is not a continuation of previous row! Violation of assumptions. You might want to manually fix the row. Skipping it now:%s' % (
            ansi_colors.WARNING, ansi_colors.ENDC))
          print ('  %s' % row)
          continue

        part_num += 1


      # If new object, output obj name and configurations
      if part_num == 1: 
        txt_file.write ('-- %s %s\n' % (
          row[cls.OBJ_CAT_K], os.path.splitext (row[cls.OBJ_BASE_K])[0]))
      
        if row[cls.GRID_TYPE_K] == cls.ELL_GRID_TYPE:
          txt_file.write ('ell_deg_step %g\n' % float (row[cls.GRID_RES_K]))
        elif row[cls.GRID_TYPE_K] == cls.CYL_GRID_TYPE:
          txt_file.write ('cyl_density %g\n' % float (row[cls.GRID_RES_K]))
        else:
          txt_file.write ('grid_res %g\n' % (float (row[cls.GRID_RES_K])))
      
        txt_file.write ('PALM_THICKNESS: %g\n' % float (row[cls.PALM_THICKNESS_K]))
 
      # Output the timestring file name for each part (each run of
      #   sample_gazebo.py)
      txt_file.write ('pt %d %s' % (part_num, row[cls.TIMESTRING_K]))

      if part_num > 1:
        if row[cls.GRID_TYPE_K] == cls.ELL_GRID_TYPE:
          txt_file.write (' ell_idx %d\n' % int (row[cls.PICKUP_ELL_IDX_K]))
        elif row[cls.GRID_TYPE_K] == cls.CYL_GRID_TYPE:
          txt_file.write (' theta_idx %d z_idx %d\n' % (
            int (row[cls.PICKUP_Z_K]), int (row[cls.PICKUP_THETA_K])))
      else:
        txt_file.write ('\n')

      elapsed_secs += float(row[cls.ELAPSED_SECS_K])
 
      prev_row = row

    # end for row in csv_reader

    # Write last object's end stats
    write_end_of_obj_stats (prev_row, elapsed_secs)
 
    csv_file.close ()
    txt_file.close ()


# Parameters:
#   pcd_name: Base name of a pcd file in the pcd_*_collected directory, without
#     extension.
def view_pcd (pcd_base_noext, cat_subdir=''):

  pcd_path = get_pcd_path ('gz_')
  pcd_name = os.path.join (pcd_path, cat_subdir, pcd_base_noext + '.pcd')

  print ('pcl_viewer %s' % pcd_name)

  subprocess.call (['pcl_viewer', pcd_name])


def main ():

  #####
  # Parse command line args
  #   Ref: Tutorial https://docs.python.org/2/howto/argparse.html
  #        Full API https://docs.python.org/dev/library/argparse.html
  #####

  arg_parser = argparse.ArgumentParser ()

  arg_parser.add_argument ('--pcd_base', type=str, default='',
    help='Base name of a pcd file in the pcd_*_collected/ directory, without extension')

  arg_parser.add_argument ('--cat', type=str, default='',
    help='Object category, the subdirectory in pcd_*_collected/ directory.')

  args = arg_parser.parse_args ()


  if args.pcd_base:
    view_pcd (args.pcd_base, args.cat)

  SampleGazeboConfigs.convert_per_run_csv_to_readable ()


if __name__ == '__main__':
  main ()

