#!/usr/bin/env python

# Mabel Zhang
# 19 Jan 2017
#
# Plan poses on Baxter real robot, store feasible ones to file, which is later
#   used in execute_poses.py to train triangles on real robot.
#
# $ rosrun baxter_reflex moveit_planner
# $ rosrun triangle_sampling plan_poses.py
#
# If you need argparse later, then the parse_args_for_svm() will require
#   histSubdirParam[1|2], so you'd have to run like this:
# $ rosrun triangle_sampling plan_poses.py --real l0,l1,a0 10,10,10
#
# To run, define these params at each run:
#   obj_const: Select an object, its dimension is used
#   grid_step: Step in meters for object centers in a grid to test in batch
#   obj_center_grid_upperleft: Define where grid starts, upperleft corner from
#     robot's perspective
#   tiles_per_dim: Number of tiles per dimension, in x and y. A square grid
#     is tested.
#     E.g. This grid is tiles_per_dim = 3:
#       * * *
#       * * *
#       * * *
#   move_to_top_point() call should be uncommented at first run each seating,
#     or after you move the object. This is so that you know where to place
#     object!
#
#   cfg.ABOVE_TABLE_CLEARANCE: Clearance above table, also how many ellipsoid
#     rings you'd like to avoid, set in
#     ../sample_gazebo_utils.py
#   These may be different for each object, for maximum contact:
#   cfg.ELL_PALM_THICKNESS
#   cfg.RADIUS_ELLIPSOID_CLEARANCE
#
#   n_rots_per_pt: Number of orientations per ellipsoid point. Increasing this
#     lets us maximize the amount of data we get at each feasible pose!
#
#   RECORD_PROBS: True to record probabilities in .pkl files, False to skip.
#     Saving these files to disk can take a long I/O time, so set to False if
#     you aren't training probabilities data!


# ROS
import rospy
from visualization_msgs.msg import MarkerArray
import tf

# Python
import argparse
import os
import time
import pickle
import csv

# NumPy
import numpy as np

# My packages
from util.ansi_colors import ansi_colors
from tactile_collect import tactile_config
from triangle_sampling.sample_gazebo_utils import SampleGazeboConfigs
from triangle_sampling.config_paths import parse_args_for_svm, \
  config_hist_paths_from_args, get_recog_meta_path
from triangle_sampling.load_hists import read_hist_config
from triangle_sampling.geo_ellipsoid import EllipsoidSurfacePoints, \
  find_ellipsoid_seam
from triangle_sampling.io_probs import IOProbs
from baxter_reflex.srv import MoveItPlannerSrv, MoveItPlannerSrvRequest
from baxter_reflex.srv import MoveItAddCollisionSrv, MoveItAddCollisionSrvRequest
from active_touch.execute_actions import ExecuteActionsBaxter
from util.quat_utils import calc_axis_angle_btw_vecs

# Local
from io_poses import format_poses_base, write_feasible_poses
from object_specs import ObjectSpecs


def main ():

  rospy.init_node ('plan_poses', anonymous=True, disable_signals=True)


  #####
  # Parse args to get histogram configs
  # Referenced triangles_reader.py, which writes histograms from collected
  #   triangles. hist_conf.csv doesn't exist yet. The params to write are
  #   specified in cmd line args.

  # Don't need these extra params like l0,l1,a0, 10,10,10. Just don't parse any
  #arg_parser = argparse.ArgumentParser ()
  #args, valid = parse_args_for_svm (arg_parser)
  #if not valid:
  #  return

  # Real robot
  #if args.real:
  #  _, tri_nbins_subpath, hist_parent_path, hist_path, img_mode_suffix, \
  #    _, tri_paramStr = config_hist_paths_from_args (args)
  #  mode_suffix = '_bx'
  #else:
  #  print ('ERROR: --real was not specified. This script only runs with --real.')
  #  return


  #####
  # Init params




  #####
  # User define params

  # gsk_sqr_1000. GSK square media bottle 1000 ml
  # These consts are defined in object_specs.py
  obj_const = ObjectSpecs.GLASS46 #JAR35 #MUG37 #BOTTLE44 #GLASS31 #BOTTLE43 #BOWL39 #SQR_BOWL_GRAY #CANISTER38 #BUCKET48 #MUG37 #GSK_SQR_1000

  obj_name = ObjectSpecs.names [obj_const]
  obj_cat = ObjectSpecs.cats [obj_const]
  obj_dims = ObjectSpecs.dims [obj_const]

  print ('%sTraining object %s (check this is what you want!! Else move_to_top_point() might move to point lower than object and collide with it!)%s' % (
    ansi_colors.OKCYAN, obj_name, ansi_colors.ENDC))


  # Meters
  grid_step = 0.03

  # Rosparam published by moveit_planner.cpp. Run that first.
  #   If table height changes, change it in there!
  TABLETOP_Z = get_tabletop_z_rosparam ()

  # Rosparam published by moveit_planner.cpp. Run that first.
  # Wood mount that Bernd built :) 2 x 4, and a cubic wood block. In future,
  #   cubic block might vary according to object height.
  MOUNT_HEIGHT = get_mount_height_rosparam ()

  # Put a known good pose at center of the 3 x 3 grid. Test around it.
  # "Upperleft" of grid is in robot's perspective.
  # Best center I found
  obj_center_grid_upperleft = np.array ([0.547763, 0.489033,
    calc_object_z (TABLETOP_Z, MOUNT_HEIGHT, obj_dims [2])])
  # Test on a 3x3 grid, including the best center
  #obj_center_grid_upperleft = np.array ([0.547763+grid_step, 0.489033+grid_step,
  #  TABLETOP_Z + MOUNT_HEIGHT + obj_dims [2] * 0.5])

  # Increments to lowerright on tabletop, in robot's perspective, are
  #   -x (lower), -y (right).
  # An area on tabletop is 2D. So tiles_per_dim ** 2 total locations.
  #tiles_per_dim = 3
  # TEMP: Testing a single ellipsoid
  tiles_per_dim = 1

  # TEMP TEST functionality
  deg_step = 45


  #####
  # Define object center candidates for batch testing
  #   Constrained in a square grid space in front of robot
  #
  # Test for the best center that results in the most feasible poses for all
  #   objects.
  # Constrain all objects to be in the 30 x 30 area, with the SAME CENTER, then
  #   this will make sure all poses feasible in training are feasible in test.
  # Repeat for ellipsoids of several different sizes and dimensions, to fit
  #   different objects.

  # nCenters x 3. 3 is for x y z coords
  grid_increments = np.zeros ((tiles_per_dim ** 2, 3))

  for x_i in range (0, tiles_per_dim):
    for y_i in range (0, tiles_per_dim):
      grid_increments [x_i * tiles_per_dim + y_i] = \
        np.array ([-x_i*grid_step, -y_i*grid_step, 0])

  # Test these 5 poses on a 3x3 grid, [1, 3, 4, 5, 7]:
  #    *
  #  *   *
  #    *
  #grid_increments = grid_increments [[1,3,5,7], :]

  obj_centers = obj_center_grid_upperleft + grid_increments

  print ('Object centers that will be tested:')
  print (obj_centers)


  #####
  # Plan poses on ellipsoids at each object center

  # List of 1D NumPy arrays
  success_plans_idx_l = []
  n_plans_l = []

  # Plan each pose on ellipsoid
  for c_i in range (0, obj_centers.shape [0]):

    obj_center = obj_centers [c_i, :]
    print ('\n\n%sNow testing ellipsoid %d out of %d, centered at %f %f %f%s' % (
      ansi_colors.OKCYAN, c_i+1, obj_centers.shape [0],
      obj_center[0], obj_center[1], obj_center[2],
      ansi_colors.ENDC))

    # Name the file to save feasible poses to
    poses_base = format_poses_base (obj_name, obj_center,
      deg_step)

    success_plans_idx, n_plans = plan_an_ellipsoid (obj_dims, obj_center,
      obj_cat, obj_name, deg_step, TABLETOP_Z, poses_base)
    if success_plans_idx is None:
      print ('Terminating on user request')
      return
    success_plans_idx_l.append (success_plans_idx)
    n_plans_l.append (n_plans)

    print ('\n')

  # Print all results
  print ('%sResults of all ellipsoids ran:%s' % (ansi_colors.OKCYAN,
    ansi_colors.ENDC))
  for c_i in range (0, obj_centers.shape [0]):

    print ('Object center %f %f %f:' % (
      obj_centers[c_i, 0], obj_centers[c_i, 1], obj_centers[c_i, 2]))
    print ('Successful plans indices (%d out of %d):' % (
      success_plans_idx_l [c_i].size, n_plans_l [c_i]))
    print (success_plans_idx_l [c_i])



#   poses_base: Base name of file to write feasible poses to
# Returns (None, None) to indicate user indicated program termination.
def plan_an_ellipsoid (obj_dims, obj_center, obj_cat, obj_name, deg_step,
  TABLETOP_Z, poses_base):

  #####
  # Init objects

  cfg = SampleGazeboConfigs ()

  # Tested good for canister38
  cfg.ELL_PALM_THICKNESS = 0.08


  # Load per-object configuration constants ELL_PALM_THICKNESS and
  #   ABOVE_TABLE_CLEARANCE
  config_name = os.path.join (get_recog_meta_path (),
    'obj_consts_real_bx_train_iros2017.csv')

  with open (config_name, 'rb') as config_file:
    config_reader = csv.DictReader (config_file)

    for row in config_reader:

      if row ['obj_name'] == obj_name:

        # ... I didn't realize plan_poses has been setting it to 0.08 for all
        #   objects!!!!!!!!!
        #cfg.ELL_PALM_THICKNESS = float (row ['ELL_PALM_THICKNESS'])

        cfg.ABOVE_TABLE_CLEARANCE = float (row ['ABOVE_TABLE_CLEARANCE'])
        print ('%sLoaded ABOVE_TABLE_CLEARANCE = %f from config file%s' % (
          ansi_colors.OKCYAN, cfg.ABOVE_TABLE_CLEARANCE, ansi_colors.ENDC))
        break


  # User adjust param
  cfg.RECORD_PROBS = True

  # Copied from sample_gazebo.py
  # For writing probabilities data to disk
  # NOTE if use the same instance for multiple objects, need to call
  #   reset_var() between each object. `.` each file should only be for 1 obj!
  if cfg.RECORD_PROBS:
    io_probs_node = IOProbs ('_bx', obj_center,
      save_debug_abs_poses=False, sequential_costs=False,
      pickup_leftoff=False, discretize_m_q_tri=(0.06, 0.05, 0.08))
  else:
    io_probs_node = None

  bx_exec_node = ExecuteActionsBaxter (None, None, None, cfg, io_probs_node)

  vis_arr_pub = rospy.Publisher ('/visualization_marker_array',
    MarkerArray, queue_size=2)


  #####
  # Let user adjust object center and dimensions

  # Call my MoveIt rosservice to let user refine object center and dimensions
  #   using InteractiveMarkers. Add object collision box to planning scene.
  # obj_cat is just for logging, doesn't matter what it is.
  # Set ask_user_input=False, to take the specified center and dims as final,
  #   to enable auto-pilot without human by computer to click the box in RViz.
  obj_dims, obj_center, model_marker = bx_exec_node.set_object_bbox (
    obj_dims, obj_center, obj_cat=obj_cat, ask_user_input=False)


  #####
  # Create an ellipsoid centered at object center

  ell_node = EllipsoidSurfacePoints ()

  ell_node.initialize_ellipsoid ( \
    obj_center,
    # Don't need ELL_PALM_THICKNESS anymore!!! Because I attach
    #   the hand properly now!! This just adds an extra clearance that I don't
    #   need, `.` /left_gripper is now at end of palm thickness. This would
    #   make hand too far from object. Maybe this is why real robot plans
    #   fewer feasible than sim.
    #obj_dims + cfg.ELL_PALM_THICKNESS,
    # 31 Jan 2017: Now I realize it actually wants the radius!! Maybe passing
    #   in diameter worked well in simulation???
    # Still add a clearance, `.` if use radius exactly, then the hand
    #   would be colliding object surface! Plus object bbox is not an
    #   ellipsoid, it's a box, so it's larger than ellipsoid.
    #obj_dims * 0.5 + cfg.RADIUS_ELLIPSOID_CLEARANCE,
    # 5 feb 2017. Above aren't feasible. only under 20 are feasible.
    #   Orig init with diameter is too far, even without EVEN_CLEARANCE.
    obj_dims * 0.5 + cfg.ELL_PALM_THICKNESS,
    deg_step=deg_step,
    alternate_order=False, rings_along_dir='h')
    #alternate_order=True, rings_along_dir='v')

  quat_wrt = [0,0,1]

  # Get all poses on the ellipsoid. These will have z pointing toward object
  #   center, suitable as either wrist pose or hand pose.
  # n x 3, n x 4 NumPy arrays
  #ell_pts, ell_vecs, ell_qs, ell_mats = \
  ell_pts, _, ell_qs, _ = \
    ell_node.get_all_points (quat_wrt=quat_wrt,
    extra_rot_if_iden=cfg.ell_extra_rot_base)


  #####
  # Cut off bottom part of ellipsoid that is too low on table, so that robot
  #   hand does not run into table.

  # Clearance for wrist above table, so that hand doesn't run into table
  #PALM_UPPER_WIDTH = 0.06
  
  # TODO: Actually could tilt the lower ones to face horizontally into obj,
  #   instead of facing the center!!! Just like adjusting the seams, should
  #   adjust the lower poses too... OR JUST ELIMINATE THEM FOR NOW. I have
  #   the center-height ones feasible, so don't need lower ones.

  above_table_idx = np.where (ell_pts [:, 2] > \
    TABLETOP_Z + cfg.ABOVE_TABLE_CLEARANCE) [0]
    #TABLETOP_Z + PALM_UPPER_WIDTH) [0]

  # Keep only the poses above table and clearance
  ell_pts = ell_pts [above_table_idx, :]
  #ell_vecs = ell_vecs [above_table_idx, :]
  ell_qs = ell_qs [above_table_idx, :]
  #ell_mats = ell_mats [above_table_idx, :, :]


  #####
  # Add a clearance constant all around the ellipsoid

  '''
  # Even all around ellipsoid, 4 cm outwards.
  EVEN_CLEARANCE = -0.04

  # 4x4 matrix, with 3rd row, 4th column, being the z-translation
  cl_mat = np.eye (4)
  cl_mat [2, 3] = EVEN_CLEARANCE

  for i in range (ell_pts.shape [0]):

    # Convert a 3-tuple position and 4-tuple quaternion to a 4x4 matrix
    T_mat = tf.transformations.quaternion_matrix (ell_qs [i, :])
    T_mat [0:3, 3] = ell_pts [i, :]

    # Multiply cl_mat on the right, `.` cl_mat applies to the intermediate
    #   frame AFTER the ellipsoid pose transform. If you mult before, it'd be
    #   applying to world frame, not what we want.
    new_T_mat = np.dot (T_mat, cl_mat)

    ell_pts [i, :] = tf.transformations.translation_from_matrix (new_T_mat)
    ell_qs [i, :] = tf.transformations.quaternion_from_matrix (new_T_mat)
  '''


  #####
  # Rotate poses in quadrants I and II (in robot's perspective) on 2D tabletop
  #   wrt z axis in world frame, to make approach direction (z axis in hand
  #   frame) more feasible for where arm is coming from.

  RANG_MIN = 90 / 180.0 * np.pi
  RANG_MAX = 110 / 180.0 * np.pi

  # Pass in the ellipsoid's radii, not the object radii!!! `.` the ellipsoid's
  #   radii is larger, I added extra clearance. if you pass in object radii,
  #   then ((point - center) / radius) will be greater than 1 in some cases,
  #   and arcsin will give invalid value error, `.` domain is [-1, 1].
  seam_idx = find_ellipsoid_seam (obj_center, ell_node.get_radii (), ell_pts)
  # Difference btw horizontal rings' points' orientation is their rotations wrt
  #   robot z axis
  seam_rot_vec = [0, 0, 1]

  # vs
  latitude_angles = ell_node.get_latitude_angles ()

  #print ('RANG_MIN: %f, RANG_MAX: %f' % (RANG_MIN, RANG_MAX))

  # The range of angles within which an extra rotation should be added
  rad_step = deg_step / 180.0 * np.pi
  rang = np.arange (RANG_MIN, RANG_MAX, rad_step)
  # If RANG_MAX should come next, as defined by deg_step, then append it to
  #   the end.
  if np.abs (RANG_MAX - rang [len (rang) - 1] - rad_step) < 0.01:
    rang = np.append (rang, RANG_MAX)
  #print ('rang:')
  #print (rang)

  max_to_rotate = rang [len (rang) - 1] - rang [0]

  # For each orientation, if it is in quadrants I or II, add an extra rotation
  for i in range (ell_qs.shape [0]):

    # Assumption: seam is at latitude_angles[0]. This assumption is made in
    #   find_ellipsoid_seam().
    angle = latitude_angles [i - seam_idx [i]] - latitude_angles [0]
    # Change angle range from [0 360] to [-180 180]
    if angle > np.pi:
      angle -= (2 * np.pi)

    #print ('angle: %f' % angle)


    # if (-135 < angle and angle < -90) or (90 < angle and angle < 135):
    if (-RANG_MAX <= angle and angle <= -RANG_MIN) or \
      (RANG_MIN <= angle and angle <= RANG_MAX):

      # `.` angle should conform to deg_step, in geo_ellipsoid.py. So it has to
      #   be equal to one of the values in rang.
      step_idx = np.where (np.abs (angle) - rang < 0.01) [0]
      if step_idx.size == 0:
        print ('%sThis angle (%f) does not conform to deg_step! Find out what is wrong. Cannot decide how much extra rotation to add to it to make it more feasible for arm%s' % (
          ansi_colors.FAIL, angle, ansi_colors.ENDC))
        return None, None

      # There should only be 1 elt
      step_idx = step_idx [0]

      # deg_step: largest number of degrees to rotate. This makes the pose at
      #   90 + deg_step wrt z (axis sticking out of table) rotate to 90 degs.
      # rang.size is the number of angles in [0, pi] range that need an extra
      #   rotation
      # step_idx: smaller difference from RANG_MIN, need to rotate less (see
      #   hand drawn diagrams on paper 30 Jan 2017). e.g. for deg_step=45, at
      #   90 degs, rotate 22.5; at 135 degs, rotate 45.
      extra_rot_amt = max_to_rotate / rang.size * (step_idx + 1)

      # -135 < angle < -90
      if -RANG_MAX <= angle and angle <= -RANG_MIN:
        extra_rot_amt = + extra_rot_amt
      # 90 < angle < 135
      if RANG_MIN <= angle and angle <= RANG_MAX:
        extra_rot_amt = - extra_rot_amt
      #print ('extra_rot_amt: %f' % extra_rot_amt)

      # This is wrt world z, seam_rot_vec is world frame z vector
      extra_rot_q = tf.transformations.quaternion_about_axis (extra_rot_amt,
        seam_rot_vec)

      # Update quaternion in matrix to the new one
      # Order matters!! First rotate by extra rotation when you're still in
      #   world frame, BEFORE the orig ellipsoid rotation. This will rotate
      #   the entire wrist pose wrt world frame z, which is what we want. if
      #   you swap the order, then you would be rotating by intermediate axis z
      #   wrt pose, and the pose z is the approach vector, so you'll just be
      #   rotating wrt approach vector, which doesn't change z at all (the
      #   objective is to change the approach vector!).
      # TODO: Still seems a little fishy. In deg_step=45, 3rd ring from top,
      #   the 135 degree poses (upper left and upper right) should rotate to
      #   90 degree (z axis pointing horizontally left and right, in top view),
      #   but they don't! They go past a little from 90 degs.
      #   But in deg_step=30, they look fine. Not sure if there's a problem.
      ell_qs [i, :] = tf.transformations.quaternion_multiply (extra_rot_q,
        ell_qs [i, :])


  #####
  # For each position on ellipsoid, rotate additionally multiple orientations
  #   wrt palm z, to maximize finding a feasible orientation at that position.
  #   Palm z points at object, rotating several times wrt z keeps z the same
  #   and gives other xy orientations, some of which may be easier for robot
  #   to reach.
  # Input: ell_pts, ell_vecs, ell_qs, ell_mats
  # Output: ell_pts_ext, ell_vecs_ext, ell_qs_ext, ell_mats_ext

  # Tip: Increase this to maximize feasible poses. Once a pose is feasible,
  #   wrist can be rotated 360 degrees, so this can be set to as large as you
  #   want, to rotate wrist many times at the same z approach pose.
  n_rots_per_pt = 6
  # TEMP to test io_probs.py. TODO remove after 15 Feb 2017!!!
  #n_rots_per_pt = 1

  # 45 deg intervals
  wrist_rots_z = np.linspace (0, np.pi * 2, n_rots_per_pt, endpoint=False)

  extended_size = ell_qs.shape[0] * wrist_rots_z.size

  # Replicate each row, horizontally. Then reshape. Result is each row
  #   replicated multiple times, in order of row 1 1 1, row 2 2 2, 3 3 3, etc.
  # Test with:
  # b = np.array ([[1,2,3], [4,5,6]])
  # np.tile (b, (1, 2)).reshape (4, 3)
  # Result is 4 x 3: array([[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6]])
  ell_pts_ext = np.tile (ell_pts, (1, wrist_rots_z.size)) \
    .reshape (extended_size, ell_pts.shape [1])
  #ell_vecs_ext = np.tile (ell_vecs, (1, wrist_rots_z.size)) \
  #  .reshape (extended_size, ell_vecs.shape [1])

  # Init rotation arrays to extended sizes
  ell_qs_ext = np.zeros ((extended_size, ell_qs.shape [1]))
  #ell_mats_ext = np.zeros ((extended_size, ell_mats.shape[1],
  #  ell_mats.shape[2]))

  # For each point, make additional orientations
  for i in range (0, ell_pts.shape [0]):

    # Apply additional rotations wrt z
    for z_i in range (0, wrist_rots_z.size):

      q_rot = tf.transformations.quaternion_from_euler (0, 0,
        wrist_rots_z [z_i])

      new_q = tf.transformations.quaternion_multiply (ell_qs [i, :], q_rot)
      #new_mat = tf.transformations.quaternion_matrix (new_q)

      # Populate extended arrays
      ell_qs_ext [i * wrist_rots_z.size + z_i, :] = new_q
      #ell_mats_ext [i * wrist_rots_z.size + z_i, :, :] = new_mat


  #####
  # Visualize the modified points
  ell_marker_arr = ell_node.visualize_custom_points (ell_pts_ext, None, #ell_vecs_ext,
    ell_qs_ext, #ell_mats_ext,
    '/base', vis_quat=False, vis_idx=True, vis_frames=True,
    extra_rot_if_iden=cfg.ell_extra_rot_base)
  for i in range (0, 5):
    vis_arr_pub.publish (ell_marker_arr)
    time.sleep (0.5)


  ######
  # Set up physical workspace, with user's help.
  # Move robot hand directly above obj_center, and prompt user to put physical
  #   object directly below. Orientation dosen't matter, just need to know
  #   (x,y) location on tabletop.

  n_pts = ell_pts_ext.shape [0]
  top_pt = np.hstack ((ell_pts_ext [n_pts - 1, :], ell_qs_ext [n_pts - 1, :]))

  # Assumption: Very last point in the ellipsoid is the top point.
  #   There are n_rots_per_pt (4) poses at the top point. We'll just take the
  #   very last pose, to be safe.
  # Assumption: The pose is feasible. Top point is usually feasible, in
  #   whatever orientation. If it turns out unfeasible, then you'll need to
  #   change code, to choose from one of the last n_rots_per_pt (4) points.
  # Comment out if you want to skip top pose, e.g. restarting script when
  #   training same object, like when you just want to change a constant.
  #if not move_to_top_point (top_pt, obj_dims, obj_center, obj_cat):
  #  return None, None


  #####
  # Plan each pose on the ellipsoid, by calling MoveIt

  # n x 7
  ell_poses = np.append (ell_pts_ext, ell_qs_ext, axis=1)

  # Call my MoveIt rosservice to plan the poses, don't execute them.
  #   If a pose is feasible, record it.
  # This is basically a brute force search in the physical space around the
  #   object, to find which poses are feasible!!! Space is discretized for
  #   practicality.
  _, _, success_plans_idx, _ = bx_exec_node.execute_actions (
    ell_poses, exec_plan=True, close_reflex=True)  # Collect triangles
    #ell_poses, exec_plan=False)  # Just plan, no movement

  # Print summary
  print ('Object center %f %f %f:' % (
    obj_center[0], obj_center[1], obj_center[2]))
  print ('Successful plans indices (%d out of %d):' % (success_plans_idx.size,
    ell_pts_ext.shape [0]))
  print (success_plans_idx)


  #####
  # Save poses that were feasible (planned successfully)

  if success_plans_idx.size > 0:
    write_feasible_poses (ell_poses [success_plans_idx, :], obj_cat,
      poses_base)
  else:
    print ('No poses successful. No files written.')


  #####
  # Save probabilities file

  # Copied from sample_gazebo.py
  if cfg.RECORD_PROBS:
    if bx_exec_node.get_recorder_node ().timestring:
      io_probs_node.compute_costs_probs (obj_cat, obj_name, obj_center)
      io_probs_node.write_costs_probs ()
    else:
      print ('%sError: No recorderNode.timestring. No costs and probs file saved. Did you run rosrun instead of roslaunch for sample_gazebo.py?%s' % (ansi_colors.FAIL, ansi_colors.ENDC))


  bx_exec_node.close_output_files ()

  return success_plans_idx, ell_pts_ext.shape [0]



# pose: 1 x 7 1D or 2D NumPy array.
# Return boolean to indicate whether to quit program
def move_to_top_point (pose, obj_dims, obj_center, obj_cat):

  # If pose is 1D, reshape it into 2D, to pass to execute_actions()
  if pose.ndim == 1:
    pose = pose.reshape (1, pose.size)

  cfg = SampleGazeboConfigs ()
  bx_exec_node = ExecuteActionsBaxter (None, None, None, cfg)

  obj_dims, obj_center, model_marker = bx_exec_node.set_object_bbox (
    obj_dims, obj_center, obj_cat=obj_cat, ask_user_input=False)

  print ('Moving robot end-effector to top point, palm facing down:')
  print (pose)

  uinput = raw_input ('%sCLEAR OFF TABLE OF ALL OBSTACLES! Moving robot hand to top point, to indicate where to place object underneath. CLEAR OFF TABLE OF ANY OBSTACLES, then press enter: %s' % (
    ansi_colors.OKCYAN, ansi_colors.ENDC))

  # Set exec_plan=True, to execute the plan
  _, _, success_plans_idx, _ = bx_exec_node.execute_actions (
    pose, exec_plan=True, close_reflex=False)

  bx_exec_node.close_output_files ()


  uinput = raw_input ('%sNow PLACE training object DIRECTLY UNDER robot hand, centered at hand z vector. Press enter when you are done, and poses planning will start. Or enter q to quit: %s' % (
    ansi_colors.OKCYAN, ansi_colors.ENDC))
  if uinput.lower () == 'q':
    print ('Terminating at user request...')
    return False
  else:
    return True


def get_rosparam (param_name):

  param = None

  # Ref parameter server: http://wiki.ros.org/rospy/Overview/Parameter%20Server
  if rospy.has_param (param_name):
    param = rospy.get_param (param_name)
  return param


# Used by execute_actions.py too
def get_tabletop_z_rosparam ():

  param_name = '/moveit_planner/tabletop_z'

  TABLETOP_Z = get_rosparam (param_name)
  if TABLETOP_Z is None:
    # Use some default
    TABLETOP_Z = -0.15
    print ('%sWARN: plan_poses.py does not find rosparam %s!! Using a default (%f)%s' % (
      ansi_colors.WARNING, param_name, TABLETOP_Z, ansi_colors.ENDC))

  return TABLETOP_Z


# Used by execute_actions.py too
def get_mount_height_rosparam ():

  param_name = '/moveit_planner/mount_height'

  MOUNT_HEIGHT = get_rosparam (param_name)
  if MOUNT_HEIGHT is None:
    # Use some default. This is height from the tallest block, to be safe.
    MOUNT_HEIGHT = -0.14
    print ('%sWARN: plan_poses.py does not find rosparam %s!! Using a default (%f)%s' % (
      ansi_colors.WARNING, param_name, MOUNT_HEIGHT, ansi_colors.ENDC))

  return MOUNT_HEIGHT


# Provides a unified way to calculate object center z component.
# Used by plan_poses.py and execute_poses.py
def calc_object_z (TABLETOP_Z, MOUNT_HEIGHT, obj_dim_z):

  return TABLETOP_Z + MOUNT_HEIGHT + obj_dim_z * 0.5


if __name__ == '__main__':
  main ()

