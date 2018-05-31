#!/usr/bin/env python

# Mabel Zhang
# 29 Jan 2017
#
# Train triangles on Baxter real robot.
# Load *feasible* poses saved from plan_poses.py (it should only save feasible
#   poses wrt robot frame /base, although feasibility can be affected by
#   object size). Plan and execute poses, collect triangles, save triangles to
#   file.
#
# $ rosrun baxter_reflex moveit_planner
# $ rosrun triangle_sampling execute_poses.py
#

# ROS
import rospy
from visualization_msgs.msg import MarkerArray

# Python
import shutil
import time

# NumPy
import numpy as np

# My packages
from util.ansi_colors import ansi_colors
from triangle_sampling.sample_gazebo_utils import SampleGazeboConfigs
from active_touch.execute_actions import ExecuteActionsBaxter
from triangle_sampling.geo_ellipsoid import EllipsoidSurfacePoints

# Local
from io_poses import format_poses_base, read_training_poses
from object_specs import ObjectSpecs
from plan_poses import get_tabletop_z_rosparam, get_mount_height_rosparam, \
  calc_object_z


class ExecuteFeasiblePoses:

  def __init__ (self, obj_consts, deg_steps):

    self.obj_consts = obj_consts
    self.deg_steps = deg_steps

    self.cfg = SampleGazeboConfigs ()

    self.training_poses = None
    self.load_poses ()


  # Load multiple poses files, empirically selected by plan_poses.py
  def load_poses (self):

    # Base names for pose files
    poses_bases = []
    # Parent directory name for pose files
    obj_cats = []

    for f_i in range (len (self.obj_consts)):
      # (obj_name, obj_center, deg_step) define a unique file (as defined in
      #   file name in io_poses.py format_poses_base()).
      poses_bases.append (format_poses_base ( \
        ObjectSpecs.names [self.obj_consts [f_i]],
        ObjectSpecs.centers [self.obj_consts [f_i]],
        self.deg_steps [f_i]))
      obj_cats.append (ObjectSpecs.cats [self.obj_consts [f_i]]) 

    self.training_poses = read_training_poses (poses_bases, obj_cats)


  # Visualize the poses loaded, via RViz Markers
  def visualize_poses (self, obj_center, obj_dims):

    if self.training_poses is None:
      self.load_poses ()


    vis_arr_pub = rospy.Publisher ('/visualization_marker_array',
      MarkerArray, queue_size=2)


    ell_node = EllipsoidSurfacePoints ()

    # Arg 3: deg_step doesn't matter. We'll pass in custom points. Won't
    #   use ellipsoid at all. Just using this class for visualization.
    ell_node.initialize_ellipsoid (obj_center,
      obj_dims * 0.5 + self.cfg.RADIUS_ELLIPSOID_CLEARANCE, deg_step=360)

    # Arg vs: Only used if vis_quat=True. I don't need it, so just set
    #   to empty array. If need it, it's just the point minus obj_center:
    #   self.training_poses [:, 0:3] - obj_center.reshape (1, obj_center.size)
    ell_marker_arr = ell_node.visualize_custom_points (
      self.training_poses [:, 0:3],
      np.zeros ((self.training_poses.shape [0], 0)),
      self.training_poses [:, 3:7],
      '/base', vis_quat=False, vis_idx=True, vis_frames=True,
      extra_rot_if_iden=self.cfg.ell_extra_rot_base)

    for i in range (0, 5):
      vis_arr_pub.publish (ell_marker_arr)
      time.sleep (0.5)


  # obj_center, obj_dims: Training object properties. This is for the live
  #   object we are trying to run on now to collect triangles.
  def execute_poses (self, obj_center, obj_dims, obj_cat):

    if self.training_poses is None:
      self.load_poses ()

    bx_exec_node = ExecuteActionsBaxter (None, None, None, self.cfg)

    # Call my MoveIt rosservice to let user refine object center and dimensions
    #   using InteractiveMarkers. Add object collision box to planning scene.
    print ('Do NOT change position of object in RViz. Change object position on table MANUALLY when the next prompt is waiting, to match the position in program! Because position must remain fixed, for the max feasible poses previously selected in robot frame!!!')
    obj_dims, obj_center, model_marker = bx_exec_node.set_object_bbox (
      obj_dims, obj_center, obj_cat=obj_cat, ask_user_input=True)

    # TODO: Check that probs files are stored. cfg RECORD_PROBS=True should do it
    # Execute poses, collect contacts, save triangles file
    _, _, success_plans_idx, tri_filename = bx_exec_node.execute_actions (
      # TODO: Uncomment when on real robot. Mount ReFlex hand and test if
      #   triangles file is saved!!
      self.training_poses, exec_plan=True, close_reflex=True)
      #self.training_poses, exec_plan=False, close_reflex=True)
 
    bx_exec_node.close_output_files ()

    return tri_filename


def main ():

  rospy.init_node ('execute_poses', anonymous=True)


  # Poses to load. These are multiple objects, feasible poses trained in
  #   plan_poses.py. Multiple objects are loaded to maximize feasibility, in
  #   case some are infeasible because of object size or center changed.

  # User define params
  # Define a list of files to load poses from
  # TODO
  obj_consts = [ObjectSpecs.MUG37, ObjectSpecs.BUCKET48]
  # TODO: Uncomment when on real robot!
  deg_steps = [45, 45]
  #deg_steps = [180, 180]

  this_node = ExecuteFeasiblePoses (obj_consts, deg_steps)


  # Live object to collect triangles from

  # Inverse dictionary lookup
  # Assumption: Values in the dictionary are unique!
  names_to_const = dict ((v, k) for (k, v) in ObjectSpecs.names.items ())

  # The live object in front of robot we are collecting training triangles on
  #   now
  while True:
    obj_name = raw_input ('Enter this object''s identifying name, for triangle file name: ')

    try:
      obj_const = names_to_const [obj_name]
    except ValueError:
      print ('%sInvalid object name, object not in object_specs.py. Define it in Python file, or enter an existing name. Need the object name to retrieve best dimension and center empirically selected.%s' % (
        ansi_colors.FAIL, ansi_colors.ENDC))
      continue
    # If didn't hit "continue" in exception above, then name was valid
    print ('Getting dimensions of %s from object_specs.py' % (ObjectSpecs.names [obj_const]))
    break

  #obj_cat = raw_input ('Enter this object''s class name, for file directory to save triangle file to: ')
  obj_cat = ObjectSpecs.cats [obj_const]

  obj_dims = ObjectSpecs.dims [obj_const]

  obj_center = ObjectSpecs.centers [obj_const]
  TABLETOP_Z = get_tabletop_z_rosparam ()
  MOUNT_HEIGHT = get_mount_height_rosparam ()
  obj_center [2] = calc_object_z (TABLETOP_Z, MOUNT_HEIGHT, obj_dims [2])

  # Execute poses and collect triangles
  this_node.visualize_poses (obj_center, obj_dims)
  tri_filename = this_node.execute_poses (obj_center, obj_dims, obj_cat)

  # TODO: Uncomment when on real robot!
  # Rename file
  tri_base = os.path.splitext (os.path.basename (tri_filename)) [0]
  new_tri_filename = os.path.join (os.path.dirname (tri_filename),
    obj_cat, obj_name + '_' + tri_base + '.csv')
  shutil.move (tri_filename, new_tri_filename)


if __name__ == '__main__':
  main ()

