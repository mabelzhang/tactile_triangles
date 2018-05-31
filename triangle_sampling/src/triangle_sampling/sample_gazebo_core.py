#!/usr/bin/env python

# Mabel Zhang
# 16 Aug 2016
#
# Refactored from sample_gazebo.py. The core functionalities of it.
#
# Used by sample_gazebo.py and active_touch pkg execute_actions_gazebo.py.
#


# New print() to allow writing to file http://stackoverflow.com/questions/6159900/correct-way-to-write-line-to-file-in-python
#   Usage: print ('hello', file=f)
# Used for writing sdf file
# Allows comma after print() to not print a new line? Without this, I get error
from __future__ import print_function

# ROS
import rospy
from std_srvs.srv import Empty
import tf
from geometry_msgs.msg import Pose, Point, Quaternion
from visualization_msgs.msg import Marker

# Gazebo
from gazebo_msgs.srv import GetModelState, GetModelStateRequest, \
  SetModelState, SetModelStateRequest

# Python
import os
import csv
import time
import signal
import subprocess
from copy import deepcopy

# NumPy
import numpy as np

# My packages
from util.ansi_colors import ansi_colors
from tactile_map.create_marker import create_marker
from util_msgs.msg import Transform
from baxter_reflex.reflex_control import call_smart_commands, \
  call_move_preshape
from baxter_reflex.wrist_hand_transforms import *
from reflex_gazebo_msgs.srv import SetModelPose, SetModelPoseRequest
from reflex_gazebo_msgs.srv import SetFixedHand, SetFixedHandRequest
from reflex_gazebo_msgs.srv import GetModelSize, GetModelSizeRequest
from reflex_gazebo_msgs.srv import RemoveModel, RemoveModelRequest
from triangle_sampling import sample_reflex


################################################ Loading object into Gazebo ##

def load_rescale_config (pkg_path):

  # Path to read config file to resize 3DNet models
  rescale_base = 'resize_3dnet_cat10_train.csv'
  rescale_name = os.path.join (pkg_path, 'config/', rescale_base)

  # Read the 3 triangle params into a Python list
  with open (rescale_name, 'rb') as rescale_file:
 
    rescale_reader = csv.DictReader (rescale_file)
 
    # Row is a dictionary. Keys are headers of csv file
    # There's only 1 row in this file
    row = next (rescale_reader)

  rescale_map = dict ()
  for k in row.keys ():
    # Convert string to float
    rescale_map [k] = float (row [k])

  return rescale_map


def load_object_into_scene (in_dae_name, obj_cat, rescale_map, cfg, vis_pub,
  gazebo=True):

  ################################################# Set up path configs ##

  # Shorthands for object info, for using to output to files, etc.
  # Using formal object name now, not timestamp anymore. Timestamp is for
  #   testing.
  obj_basename = os.path.basename (in_dae_name)
  obj_prefix = os.path.splitext (obj_basename) [0]


  ########################################################### Constants ##

  _3DNET_SIG = '3DNet'

  # Rotate 90 wrt X, `.` Blender export added 90 degs
  # I've now corrected all the archive3d models to be correct orientation
  #   (Y Forward Z Up when exporting). Shouldn't need this anymore!
  model_R = 0  #np.pi / 2.0
  model_P = 0
  model_Y = 0


  ########################################### Generate SDF object model ##

  # See if need to rescale object 3D model
  if _3DNET_SIG in in_dae_name:
    rescale = rescale_map [obj_cat]
    #print ('Rescaling model by %g' % (rescale))
  # No rescaling
  else:
    rescale = 1


  model_name = ''
  if gazebo:

    # Generate .sdf file, which points to an object .dae file
    print ('%sGenerating SDF file to import DAE mesh%s' % ( \
      ansi_colors.OKCYAN, ansi_colors.ENDC))
    out_sdf_name = generate_sdf (in_dae_name, R=model_R,
      P=model_P, Y=model_Y, scale=rescale)
    print ('')
 
    # Load the generated .sdf file
    print ('%sLoading generated SDF file%s' % (ansi_colors.OKCYAN,
      ansi_colors.ENDC))
    # Generate a random suffix for object each time, so when reloading,
    #   model doesn't have teh "doesn't remove cleanly" problem, which causes
    #   it to be invisible, even when issue transport:: messages 
    #   set_visible(true)!! The transport message works for first load of
    #   model, but not after it's been deleted and reloaded! Has to be a 
    #   clean removal problem.
    model_name = 'object' + str (np.random.randint (0, 10000))
    model_name = spawn_sdf (out_sdf_name, model_name,
      x=cfg.obj_x, y=cfg.obj_y, z=cfg.obj_z)
    # Doesn't seem to do anything. Model still invisible if loaded with same
    #   name as a deleted model
    #call_setbool_rosservice ('/reflex_gazebo/set_model_visible', True)

  # end if gazebo


  # Visualize object .dae model
  # This should be visualized at obj_x, obj_y, obj_z, NOT model_center,
  #   because obj_* is where I place the mesh! model_center is modified
  #   by the object's bbox and no longer at mesh center. I need to draw
  #   the mesh at mesh center, where I originally put it!
  marker_obj = Marker ()
  create_marker (Marker.MESH_RESOURCE, 'obj_mesh', '/base', 0,
    cfg.obj_x, cfg.obj_y, cfg.obj_z, 1, 1, 1, 0.5, rescale, rescale, rescale,
    marker_obj, 0)  # Use 0 duration for forever
  marker_obj.mesh_resource = 'file://' + in_dae_name
  vis_pub.publish (marker_obj)
  # Wait for RViz Marker msg to propagate
  time.sleep (0.3)
  #rospy.sleep (0.5)

  print ('')

  return model_name, marker_obj


# Parameters:
#   in_dae_name: Input file .dae full path
#   model_name: You shouldn't need to change this. For the problem of objects
#     not being removed cleanly by Gazebo, causing the next model of same
#     name not being loaded correctly, change the model_name in spawn_sdf,
#     not here.
def generate_sdf (in_dae_name, model_name="object", R=0, P=0, Y=0,
  scale=1):

  #####
  # Generate output file name
  #####

  # Drop base name
  dir_prefix, _ = os.path.split (in_dae_name)
  # Grab last dir name. This won't contain a slash, guaranteed by split()
  dir_prefix, catname = os.path.split (dir_prefix)

  dir_prefix, _ = os.path.split (dir_prefix)

  # Replace extension with sdf
  dae_prefix = os.path.splitext (os.path.basename (in_dae_name)) [0]
  out_dir_path = os.path.join (dir_prefix, 'sdf', catname)
  out_sdf_name = os.path.join (out_dir_path, dae_prefix + '.sdf')

  print ('Input COLLADA: %s' % in_dae_name)
  print ('Output SDF:    %s' % out_sdf_name)


  #####
  # Generate the SDF file
  #####

  # http://stackoverflow.com/questions/273192/in-python-check-if-a-directory-exists-and-create-it-if-necessary
  if not os.path.exists (out_dir_path):
    os.makedirs (out_dir_path)

  f = open (out_sdf_name, 'w')


  # Result looks like this, copied from a working file cup_2b46c83c.sdf:
  '''
<?xml version="1.0"?>
<sdf version="1.4">

  <model name="object">
    <pose>0.2 -0.3 0  1.57 0 0</pose>
    <static>true</static>
    <link name="object_link">
      <visual name="visual">
        <geometry>
          <mesh><uri>file:///Users/master/graspingRepo/train/triangle_sampling/models/archive3d/dae/cup/cup_2b46c83c.dae</uri></mesh>
        </geometry>
      </visual>

      <collision name="object_collision">
        <geometry>
          <mesh><uri>file:///Users/master/graspingRepo/train/triangle_sampling/models/archive3d/dae/cup/cup_2b46c83c.dae</uri></mesh>
        </geometry>
      </collision>

    </link>
  </model>
</sdf>
  '''

  x = 0  #0.2
  y = 0  #-0.3
  z = 0

  # New print() to allow writing to file http://stackoverflow.com/questions/6159900/correct-way-to-write-line-to-file-in-python
  print ('<?xml version="1.0"?>', file=f)
  print ('<sdf version="1.4">', file=f)

  print ('  <model name="%s">' % model_name, file=f)
  print ('    <pose>%g %g %g  %g %g %g</pose>' % (x, y, z, R, P, Y), file=f)
  # This keeps cup fixed in space
  print ('    <static>true</static>', file=f)
  # This doesn't seem to be needed, as long as static is true
  #print ('    <gravity>0</gravity>', file=f)
  print ('    <link name="%s_link">' % model_name, file=f)
  print ('      <visual name="%s_visual">' % model_name, file=f)
  # Ref transparency: https://bitbucket.org/osrf/gazebo/pull-requests/498/fix-for-668-visual-properties-not-loaded/diff
  print ('        <transparency>0.0</transparency>', file=f)
  print ('        <geometry>', file=f)
  print ('          <mesh>', file=f)
  print ('            <uri>file://%s</uri>' % in_dae_name, file=f)
  print ('            <scale>%g %g %g</scale>' % (scale, scale, scale), file=f)
  print ('          </mesh>', file=f)
  print ('        </geometry>', file=f)
  # Ref material: https://bitbucket.org/osrf/gazebo/pull-requests/498/fix-for-668-visual-properties-not-loaded/diff
  print ('        <material><script>', file=f)
  print ('        <uri>file://media/materials/scripts/gazebo.material</uri>', file=f)
  print ('        <name>Gazebo/Grey</name>', file=f)
  print ('        </script></material>', file=f)
  print ('      </visual>', file=f)
  print ('', file=f)

  print ('      <collision name="%s_collision">' % model_name, file=f)
  print ('        <geometry>', file=f)
  print ('          <mesh>', file=f)
  print ('            <uri>file://%s</uri>' % in_dae_name, file=f)
  print ('            <scale>%g %g %g</scale>' % (scale, scale, scale), file=f)
  print ('          </mesh>', file=f)
  print ('        </geometry>', file=f)
  print ('      </collision>', file=f)
  print ('', file=f)

  print ('    </link>', file=f)
  print ('  </model>', file=f)
  print ('</sdf>', file=f)
  print ('', file=f)


  f.close ()
  print ('Written SDF to file %s' % out_sdf_name)

  return out_sdf_name


# Equivalent of: rosrun gazebo_ros spawn_model -file /Users/master/graspingRepo/reFlexHand/catkin_ws/src/triangle_sampling/src/cup_2b46c83c.sdf -sdf -model object -y 0.2 -x -0.3 -R 1.57
def spawn_sdf (sdf_name, model_name='object', x=0, y=0, z=0, R=0, P=0,
  Y=0):

  rospy.loginfo ('Calling rosrun to spawn object...')

  # subprocess.Popen() spawns a new process and doesn't wait till it finishes.
  #   os.system() blocks and waits till the child process finishes.

  # This doesn't return even after printing error!
  #os.system ('rosrun gazebo_ros spawn_model -file %s -sdf -model %s' % \
  #  (sdf_name, model_name))

  # Each arg must be in a separate string in list, can't do '-sdf -model'
  #   with space in btw, must do ['-sdf', '-model'] as two separate elts.
  # This doesn't return either. stdout=subprocess.PIPE makes it return, but
  #   it doesn't print anything to rospy.loginfo. Without that, it printed.
  #p = subprocess.Popen (['rosrun', 'gazebo_ros', 'spawn_model', '-file',
  #  sdf_name, '-sdf', '-model', model_name])

  # This prints stuff to rospy.loginfo, AND returns. Best so far
  p = subprocess.call (['rosrun', 'gazebo_ros', 'spawn_model', '-file',
    sdf_name, '-sdf', '-model', model_name,
    '-x', str(x), '-y', str(y), '-z', str(z),
    '-R', str(R), '-P', str(P), '-Y', str(Y)])

  # This doesn't work on MacBook Air for some reason, wth??
  #rospy.sleep (1)
  # This doesn't work on MacBook Air for some reason, wth??
  #pause_rate = rospy.Rate (1)
  #pause_rate.sleep ()

  # Seconds
  # rospy.sleep(1) and rospy.Rate.sleep() don't work on OS X for some reason
  time.sleep (0.2)
  #rospy.sleep (0.2)

  rospy.loginfo ('rosrun spawn_model returned')

  return model_name


# Remove a model from world
# Service declaration is in gazebo_msgs/srv/DeleteModel.srv
# rospy service call tutorial:
#   http://wiki.ros.org/ROS/Tutorials/WritingServiceClient(python)
def remove_model (model_name, timeout=5,
  srv_name='/reflex_gazebo/remove_model'):

  # This is my custom rosservice in reflex_gazebo world_plugin.cpp

  #rospy.loginfo ('Waiting for rosservice %s...' % srv_name)
  rospy.wait_for_service (srv_name)

  rospy.loginfo ('Calling rosservice %s on %s with timeout %d s...' % ( \
    srv_name, model_name, timeout))

  try:
    srv = rospy.ServiceProxy (srv_name, RemoveModel)
    req = RemoveModelRequest ()
    req.model_name = model_name

    # This service call really shouldn't time out. If it times out, that means
    #   Gazebo is not responding anymore. It has many bugs removing models.
    #   Restart it.
    # Gazebo physics::Model::GetBoundingBox(), used in the service, sometimes
    #   doesn't return. So need to add a guard.
    # Ref: http://stackoverflow.com/questions/492519/timeout-on-a-python-function-call
    signal.signal (signal.SIGALRM, timeout_handler)
    signal.alarm (timeout)

    try:
      resp = srv (req)
    except Exception, ex:
      print (ex)
      return False

    # Cancel the timer if rosservice returned before timeout
    signal.alarm (0)

  except rospy.ServiceException, e:
    rospy.logerr ('sample_gazebo remove_model(): Service call to %s failed: %s' %(srv_name, e))
    return False

  if resp.success:
    # Too much printouts. Comment out if you want to see other debug info.
    #rospy.loginfo ('rosservice call succeeded')
    return True
  else:
    rospy.logwarn ('rosservice call failed')
    return False


  # The /gazebo/delete_model way seems to cause weird errors in Gazebo, like
  #   the later loaded model with the same name would be invisible!!!! This
  #   is like the weird bug that when use gazebo_ros rosservice to set gravity
  #   to 0, robot's twists become NaN! Completely unreasonable and stupid.
  #   Avoid using gazebo_ros's rosservices!!!

  '''
  # Equivalent of: rosservice call gazebo/delete_model '{model_name: object}'

  srv_name = '/gazebo/delete_model'

  rospy.wait_for_service (srv_name)

  rospy.loginfo ('Calling rosservice %s on %s...' % (srv_name, model_name))
  try:
    srv = rospy.ServiceProxy (srv_name, DeleteModel)
    # Ret vals: string parse_response, int32 result, duration elapsed_time
    req = DeleteModelRequest ()
    req.model_name = model_name
    resp = srv (req)

  except rospy.ServiceException, e:
    rospy.logerr ('sample_gazebo remove_model(): Service call to %s failed: %s' %(srv_name, e))
    return False

  if resp.success:
    # Too much printouts. Comment out if you want to see other debug info.
    #rospy.loginfo ('rosservice call succeeded')
    return True
  else:
    rospy.logwarn ('rosservice call failed: %s' % resp.status_message)
    return False
  '''


######################################################## Gazebo environment ##

def pause_physics (pause=False):

  # Equivalent of: rosservice call /gazebo/set_physics_properties '{gravity: {x: 0.0, y: 0.0, z: 0.0}}'

  if pause:
    srv_name = '/gazebo/pause_physics'
  else:
    srv_name = '/gazebo/unpause_physics'

  return call_empty_rosservice (srv_name)


def call_empty_rosservice (srv_name, wait_time=1):

  #rospy.loginfo ('Waiting for rosservice %s...' % srv_name)
  rospy.wait_for_service (srv_name)

  rospy.loginfo ('Calling rosservice %s...' % srv_name)
  try:
    srv = rospy.ServiceProxy (srv_name, Empty)
    resp = srv ()
    time.sleep (wait_time)
    #rospy.sleep (wait_time)

  except rospy.ServiceException, e:
    rospy.logerr ('sample_gazebo call_empty_rosservice(): Service call to %s failed: %s' %(srv_name, e))
    return False

  return True


# Returns (True, geometry_msgs.Vector3, geometry_msgs.Point) on successful
#   rosservice call.
#   Else returns (False, None, None).
# model center is wrt Gazebo world frame. So make sure you publish a reference
#   tf frame at 0 0 0 0 0 0 1 of Gazebo origin, so that you can use this
#   model center wrt that tf frame!
def get_model_size (model_name, timeout=2):

  # My custom rosservice in world_plugin.cpp. The official gazebo_ros service
  #   sometimes doesn't return. It's really annoying!!! I have to restart this
  #   script, and then it'd return. WTF. So I wrote my own.
  srv_name = '/reflex_gazebo/get_model_size'

  #rospy.loginfo ('Waiting for rosservice %s...' % srv_name)
  rospy.wait_for_service (srv_name)

  rospy.loginfo ('Calling rosservice %s with timeout %d s...' % (srv_name,
    timeout))

  try:
    srv = rospy.ServiceProxy (srv_name, GetModelSize)

    req = GetModelSizeRequest ()
    req.model_name = model_name


    # Gazebo physics::Model::GetBoundingBox(), used in the service, sometimes
    #   doesn't return. So need to add a guard.
    # Ref: http://stackoverflow.com/questions/492519/timeout-on-a-python-function-call
    signal.signal (signal.SIGALRM, timeout_handler)
    signal.alarm (timeout)

    try:
      resp = srv (req)
    except Exception, ex:
      print (ex)
      return (False, None, None)

    # Cancel the timer if rosservice returned before timeout
    signal.alarm (0)


    # Don't need to sleep. My srv returns when it's actually done
    #time.sleep (1)

  except rospy.ServiceException, e:
    rospy.logerr ('sample_gazebo get_model_size(): Service call to %s failed: %s' %(srv_name, e))
    return (False, None, None)

  if resp.success:
    # Too much printouts. Comment out if you want to see other debug info.
    #rospy.loginfo ('rosservice call succeeded')
    return (True, resp.model_size, resp.model_center)
  else:
    rospy.logwarn ('rosservice call failed')
    return (False, None, None)


def get_model_state (model_name):

  srv_name = '/gazebo/get_model_state'

  #rospy.loginfo ('Waiting for rosservice %s...' % srv_name)
  rospy.wait_for_service (srv_name)

  rospy.loginfo ('Calling rosservice %s...' % (srv_name))

  try:
    srv = rospy.ServiceProxy (srv_name, GetModelState)

    req = GetModelStateRequest ()
    # Don't need to set relative_entity_name if getting wrt world
    req.model_name = model_name

    resp = srv (req)

  except rospy.ServiceException, e:
    rospy.logerr ('sample_gazebo get_model_state(): Service call to %s failed: %s' %(srv_name, e))
    return False, None

  if resp.success:
    # Too much printouts. Comment out if you want to see other debug info.
    #rospy.loginfo ('rosservice call succeeded')
    return True, resp.pose
  else:
    rospy.logwarn ('rosservice call failed')
    return False, None


def set_fixed_hand (fixed):

  srv_name = '/reflex_gazebo/set_fixed_hand'

  #rospy.loginfo ('Waiting for rosservice %s...' % srv_name)
  rospy.wait_for_service (srv_name)

  rospy.loginfo ('Calling rosservice %s, passing in %s...' % (srv_name,
    str(fixed)))

  try:
    srv = rospy.ServiceProxy (srv_name, SetFixedHand)

    req = SetFixedHandRequest ()
    req.fixed = fixed

    resp = srv (req)
    time.sleep (0.3)
    #rospy.sleep (0.3)

  except rospy.ServiceException, e:
    rospy.logerr ('sample_gazebo set_fixed_hand(): Service call to %s failed: %s' %(srv_name, e))
    return False

  if resp.success:
    # Too much printouts. Comment out if you want to see other debug info.
    #rospy.loginfo ('rosservice call succeeded')
    return True
  else:
    rospy.logwarn ('rosservice call failed')


############################################################### Utility fns ##

# Parameters:
#   pose1, pose2: geometry_msgs/Pose
def compare_poses (pose1, pose2):

  # Allow for some noise
  if abs (pose1.position.x - pose2.position.x) < 1e-1 and \
    abs (pose1.position.y - pose2.position.y) < 1e-1 and \
    abs (pose1.position.z - pose2.position.z) < 1e-1 and \
    abs (pose1.orientation.x - pose2.orientation.x) < 1e-1 and \
    abs (pose1.orientation.y - pose2.orientation.y) < 1e-1 and \
    abs (pose1.orientation.z - pose2.orientation.z) < 1e-1 and \
    abs (pose1.orientation.w - pose2.orientation.w) < 1e-1:

    return True

  else:
    return False


# Ref: http://stackoverflow.com/questions/492519/timeout-on-a-python-function-call
def timeout_handler (signum, frame):
  raise Exception ("Timed out")


################################################################ Core class ##

# Teleports the hand in Gazebo to a specified sequence of absolute poses.
#   Collects sets of triangles and save them to csv and pcd files using
#     triangles_collect.py. Uses triangles_collect_semiauto.py to interface
#     with triangles_collect.py to close gripper and collect triangles.
#   Saves probabilities data for active touch using io_probs.py.
# Used by triangle_sampling sample_gazebo.py and active_touch
#   execute_actions_gazebo.py
class SampleGazeboCore:

  # Parameters:
  #   wrist_poses: List of 7-tuples, absolute poses to teleport wrist poses to.
  #     Core functionality of this object.
  def __init__ (self, cfg, recorderNode, cyl_grid_node, io_probs_node,
    hand_name='reflex', goal_poses=None):

    self.cfg = cfg
    self.recorderNode = recorderNode
    self.cyl_grid_node = cyl_grid_node
    self.io_probs_node = io_probs_node

    self.hand_name = hand_name

    # List of all wrist poses to teleport to
    self.goal_poses = goal_poses

    self.lastSeenIdx = -1


    # Broadcast custom tf frames so can see things in RViz
    # Each tf_broadcaster.py node only broadcasts one transform constantly.
    #   If need multiple, need to start multiple tf_broadcaster.py nodes.
    #   In launch file, just remap the topic from /tf_broadcaster/transform to
    #   whatever name I choose here.
    # /base to /base_link
    self.tf_bc_hand_pub = rospy.Publisher ('/tf_broadcaster/transform_hand',
      Transform, queue_size=5)
    # /base_link to /left_gripper
    self.tf_bc_wrist_pub = rospy.Publisher ('/tf_broadcaster/transform_wrist',
      Transform, queue_size=5)

    self.vis_pub = rospy.Publisher ('/visualization_marker',
      Marker, queue_size=2)


    # For calling log_this_run()
    self.obj_cat = ''
    self.obj_basename = ''


    # No longer used. For get_next_wrist_pose ()

    #self.txh = [0.2, 0.2, -0.2, -0.2]
    #self.tyh = [0.2, -0.2, 0.2, -0.2]
    #self.tzh = [0.1, 0.05, 0.1, 0.05]

    #self.qxh = [0, 0, 0, 0]
    #self.qyh = [0, 0, 0, 0]
    #self.qzh = [0, 0, 0, 0]
    #self.qwh = [1, 1, 1, 1]

    #self.wrist_pose_seen_idx = -1


  def set_obj_info_for_log (self, obj_cat, obj_basename):

    self.obj_cat = obj_cat
    self.obj_basename = obj_basename


  # Parameters:
  #   poses: n x 7 numpy array. I'm not sure what frame this is in anymore.
  #     I thought execute_poses() wants /left_gripper frame, as suggested by
  #     comments, but when I give it that, it's too close to object.
  def set_goal_poses (self, poses):

    self.goal_poses = deepcopy (poses)


  # Execute poses in self.goal_poses
  # Parameters:
  #   start_print_idx: only used for printing. Poses, if need to be picked up
  #     from a prev run, should be determined by caller. So we always start at
  #     pose index [0], but we can print at an offset.
  def execute_poses (self, start_print_idx=0):

    if self.goal_poses is None:
      print ('%sERROR: SampleGazeboCore.execute_poses(): No goal_poses were set. Set them before calling this function. Doing nothing.%s' % (
        ansi_colors.FAIL, ansi_colors.ENDC))
      return

    # Shorthand
    cfg = self.cfg


    # Prep hand in first preshape, opened
    print ('%sPutting fingers in preshape %g...%s' % ( \
      ansi_colors.OKCYAN, cfg.preshapes [0], ansi_colors.ENDC))
    call_move_preshape (cfg.preshapes [0], wait_time=0.2)
 
    print ('%sOpening ReFlex fingers...%s' % (ansi_colors.OKCYAN,
      ansi_colors.ENDC))
    call_smart_commands ('open', 0.2)
 
    n_pos_accumulated = 0

    ell_doTerminate = False

    start_time = time.time ()
    # For ret val, in case there are 0 poses, then elapsed_time won't be set
    elapsed_time = 0.0
 
    # Collect one object
    # Loop through each pose
    for p_i in range (0, self.goal_poses.shape [0]):

      print ('%sExecuting pose index: [%d] out of %d (0 based)%s' % (
        ansi_colors.OKCYAN, p_i+start_print_idx,
        self.goal_poses.shape [0]+start_print_idx-1, ansi_colors.ENDC))
      self.lastSeenIdx = p_i

      # rospy.Time.now() doesn't work. Gives me 0 for elapsed_time
      #tart_time = time.time ()
      if cfg.PROFILER:
        print ("Loop starts")

      isLastPos = (p_i == self.goal_poses.shape [0] - 1)

      # Establish output file names, in case ends unexpected
      #   Checking recorderNode.timestring exists checks that we got any
      #   contact at all. Otherwise no need to record.
      # Always use timestamp name first, let user choose whether to overwrite
      #   the permanent-name files at the end of program execution.
      if cfg.RECORD_PROBS and not self.io_probs_node.get_configured () and \
          self.recorderNode.timestring:

        # Config once per object file
        self.io_probs_node.set_costs_probs_filenames (
          self.recorderNode.timestring)

 
      #prev_ceil_done = cyl_grid_node.ceil_done

      #####
      # Set goal pose. No movement is done.
      #####

      # 7-item list. TWw
      goal_pose = []

      # If using cylinder
      if cfg.USE_CYL_GRID:

        # No movement is done by collect(). Just sets goal pose.
        self.cyl_grid_node.collect (dry=True)
        if cfg.PROFILER: print ('1: '), print (time.time() - start_time)
  
        # Broadcast goal pose as a tf frame
        self.cyl_grid_node.broadcast_goal_frame ()
        # Visualize the object wall and grid pts
        self.cyl_grid_node.visualize ()
        if cfg.PROFILER: print ('2: '), print (time.time() - start_time)

        # goal_pose is set by collect(). It calculates where /left_gripper
        #   wrist frame should go, wrt /base robot frame. It is in /base
        #   frame.
        if not self.cyl_grid_node.goal_pose:
          print ('cyl_grid_node.goal_pose == None. This is not supposed to happen.')

        else:
          goal_pose = [self.cyl_grid_node.goal_pose.pose.position.x,
            self.cyl_grid_node.goal_pose.pose.position.y,
            self.cyl_grid_node.goal_pose.pose.position.z,
            self.cyl_grid_node.goal_pose.pose.orientation.x,
            self.cyl_grid_node.goal_pose.pose.orientation.y,
            self.cyl_grid_node.goal_pose.pose.orientation.z,
            self.cyl_grid_node.goal_pose.pose.orientation.w]

      # If using ellipsoid
      else:
        if isLastPos:
          print ('Saw last point in ellipsoid. Done with one object.')
          ell_doTerminate = True

        # TODO For active_touch, using this else stmt causes off-by-one error,
        #   last pose is not executed! I think with ellipsoid grid training, it
        #   should also execute without else-stmt, since now the poses are
        #   passed in to set_goal_poses(). It should all break by for-loop.
        #else:

        # 1 x 7 vector
        goal_pose = self.goal_poses [p_i, :].tolist ()

        # Visualize, for screen recording for IROS video. yellow ball
        marker_goal = Marker ()
        create_marker (Marker.SPHERE, 'curr_goal', '/base', 0,
          goal_pose[0], goal_pose[1], goal_pose[2],
          1, 1, 0, 0.9, 0.03, 0.03, 0.03,
          marker_goal, 0)  # Use 0 duration for forever
        self.vis_pub.publish (marker_goal)
        # Pause for RViz to propagate
        time.sleep (0.2)


      # end if USE_CYL_GRID


      #####
      # Move hand to goal pose and collect triangles
      #####
      if len (goal_pose) > 0:

        #####
        # Convert goal pose, which is computed as a pose of wrist /left_gripper
        #   wrt base in triangles_collect_semiauto.py, to be of ReFlex hand
        #   /base_link wrt base.
        #####

        # triangles_collect_semiauto.py define goals for /left_gripper, `.`
        #   Baxter IK requires it. Here, we don't have a Baxter, we want to
        #   use ReFlex base_link, which is rotated 90 deg wrt /left_gripper.
        # So rotate it back.
 
        '''
        # Orientation
        # tf quaternion in Python is just a tuple
        #   http://answers.ros.org/question/69754/quaternion-transformations-in-python/
        quat_left_gripper_wrt_base = goal_pose [3:7]
        # 4-tuple
        quat_base_link_wrt_base = transform_quat_wrist_to_hand ( \
          quat_left_gripper_wrt_base)
 
        pos_base_link_wrt_base = goal_pose [0:3]
        '''


        # TODO Need to be tested with sample_gazebo.py. Only tested with
        #   active_predict.py.
        # New attempt. I think this is more correct. 22 Aug 2016
        #   It's same as the transform_quat_wrist_to_hand() above, except this
        #   also transforms the position. Above just sets it the same which
        #   isn't right
        # TWh = TWw * Twh
        #   We want hand pose wrt world, `.` will teleport hand in empty world!
        # The other direction used in active_touch execute_actions.py:
        # TWw = TWh * Thw
        #     = TWh * (Twh)^-1
        #   This wants wrist pose wrt world, `.` for motion planning or IK, need
        #   goal endpoint (wrist) pose wrt robot /base (world frame).
        #   Actually oppo dir not used in execute_actions.py anymore, `.` using
        #   it is wrong, not using it actually produced right results!
        # Uhhhhhhh I think I see a big typo!!!! Why is the last step calling
        #   _from_matrix() on TWw???? It should be calling TWh!!!! That's why
        #   putting the reverse in execute_actions.py made it wrong!!! `.` in
        #   here we applied it again!!
        TWw = tf.transformations.quaternion_matrix (goal_pose [3:7])
        TWw [0:3, 3] = goal_pose [0:3]
       
        Twh = tf.transformations.quaternion_matrix (
          wrist_to_hand_transforms.quat_hand_wrt_wrist)
        Twh [0:3, 3] = wrist_to_hand_transforms.pos_hand_wrt_wrist
       
        TWh = np.dot (TWw, Twh)
        # 19 Jan 2016: TYPO?????? Should be calling on TWh, not TWw!!!
        #   Change to TWh, then uncomment the block in execute_actions.py that
        #   does the reverse (from Twh to TWw), and see if active_predict.py
        #   still works.
        pos_base_link_wrt_base = tf.transformations.translation_from_matrix (
          TWw)
        quat_base_link_wrt_base = tf.transformations.quaternion_from_matrix (
          TWw)


        # TODO: This is wrong. Cannot add directly onto xyz, the fn doesn't
        #   even know what frame the position is in! Definitely need tf
        #   away from /base frame before calling the function, so that the
        #   function's shift, which is wrt hand or wrist frame, is actually
        #   applied onto a position that is wrt hand or wrist frame!
        #   Otherwise, the goal_pose is currently in robot /base frame.
        #   Applying a hard x+= y+= z+= just makes the hand move in some
        #   arbitrary direction, regardless of which direction it's facing!
        #   It causes problems like, in one or two quadrants, the hand 
        #   becomes too far from the object, because you just moved it in
        #   some arbitrary direction!!!
        #   Should just delete this line.
        #   Shouldn't even do it for USE_CYL_GRID! But because I haven't
        #   tested it commented out with cylinder grid, I'll leave it in.
        # Position. Move closer to object
        # triangles_collect_semiauto.py doesn't use this, because position
        #   is controlled by constants that were tuned to be the right
        #   distance from object, which already includes the hand's and
        #   wrist's thickness.
        # Here, it turns out goal_pose position is too far from object in sim,
        #   so I don't want to include the wrist's thickness, only the hand.
        #   Easy way to undo this tuned-wrist-thickness is to transform from
        #   hand to wrist (yes it's reverse of what I'm trying to do), because
        #   hand is closer to object than wrist is. So putting hand in wrist
        #   frame (instead of hand frame; which means you increase z more)
        #   means you move closer to object. This transformation puts
        #   hand in sim closer to object.
        if cfg.USE_CYL_GRID:
          pos_base_link_wrt_base = transform_pos_hand_to_wrist_tuple ( \
            goal_pose [0:3])
 
        if cfg.PROFILER: print ('3: '), print (time.time() - start_time)


        # Loop through each hand rotation wrt z
        for z_i in range (0, len (cfg.wrist_rots)):

          #####
          # Apply current rotation to hand goal pose
          #####

          goal_pos_hand = deepcopy (pos_base_link_wrt_base)
          goal_rot_hand = deepcopy (quat_base_link_wrt_base)

          # Rotate the goal rotation wrt z axis of hand.
          #   Goal pose is wrt /base frame. How can I rotate it wrt hand's
          #   axis?
          # Just rotate the amount directly on quat_base_link_wrt_base.
          # This makes sense because quat_base_link_wrt_base is a description
          #   of a frame (a more confusing way of thinking about it would be,
          #   a transformation btw two frames, which makes sense too, it's
          #   transformation of base_link frame with respect to base frame),
          #   so quat_base_link_wrt_base can be seen as an "intermediate
          #   frame". Think of when you multiply a frame oriented aligned to
          #   base_frame axes by 2 quaternions to define the goal pose; the
          #   pose in btw the 2 multiplications is the intermediate frame.
          #   Multiplying the 2nd quaternion gives you a frame OF the goal
          #   pose, wrt base frame.
          #   Similarly here, the quat_base_link_wrt_base is already an
          #   intermediate frame, OF hand base_link, wrt base. So multiplying
          #   a quat by it will just further rotate this intermediate frame,
          #   along the axis of the *intermediate frame*, by however much
          #   you specify in the quaternion. It will not rotate wrt base.
          #   So this should transform the intermediate frame further, into
          #   the desired frame -
          #   which in English is, you apply an additional rotation to the
          #   wrist, and the rotation can be defined wrt the wrist's current
          #   axis, which acts like an intermediate frame btw base and your
          #   final goal wrist pose.
          # Yes! Worked!! Yay :D :D Much easier than I first thought.
          curr_rot_quat = tf.transformations.quaternion_from_euler ( \
            *(cfg.wrist_rots [z_i]))
          goal_rot_hand = tf.transformations.quaternion_multiply ( \
            goal_rot_hand, curr_rot_quat)


          #####
          # Compute absolute pose for training for active learning
          #####

          if cfg.RECORD_PROBS:

            # Absolute pose wrt fixed robot frame /base
            abs_pose = ( \
              goal_pos_hand [0],
              goal_pos_hand [1],
              goal_pos_hand [2],
              goal_rot_hand [0],
              goal_rot_hand [1],
              goal_rot_hand [2],
              goal_rot_hand [3])

            # Get triangles within a WRIST pose, where z1 is observed.
            #   Don't distinguish btw FINGER poses (i.e. preshapes),
            #   accumulate them under each WRIST pose. `.` action "a"
            #   in p(z2 | y, z1, a) doesn't consider finger preshape.
            # So this var is initialized inside the wrist_rots loop,
            #   outside N_PRESHAPES loop
            contacts_at_curr_wrist_pose = []

          # end if RECORD_PROBS

         
          #####
          # Teleport hand to goal pose, until correct (sometimes hand gets
          #   bumped by object during move, esp if fingers are stuck on object
          #   and cannot open all the way, then teleporting bumps hand
          #   crooked)
          #####

          self.teleport_until_correct (self.hand_name, goal_pos_hand,
            goal_rot_hand, cfg.STRICT_POSE)

          if cfg.PROFILER: print ('4: '), print (time.time() - start_time)

          # Loop through each fore-finger preshape
          for ps_i in range (0, cfg.N_PRESHAPES):

            # Move fore-fingers to current preshape
            print ('%sPutting fingers in preshape %g...%s' % ( \
              ansi_colors.OKCYAN, cfg.preshapes [ps_i], ansi_colors.ENDC))
            call_move_preshape (cfg.preshapes [ps_i], wait_time=0.2)
         
            # Trying wait_time 0.5, instead of 0, to see if this is why
            #   contacts aren't cleared, that they are recorded even when
            #   hand is moved to safe place.
            # > Does created less noise!
            call_empty_rosservice ('/zero_tactile', wait_time=0.5)
            if cfg.PROFILER: print ('5: '), print (time.time() - start_time)



            #####
            # Set var cache_this_iter and n_pos_accumulated. It's a nested
            #   if-condition for # preshapes and # positions of gripper.
            # The order of nested for-loops tree is:
            #   goal_pose (N_ACCUM_POS = 2)
            #     wrist_rots (2)
            #       N_PRESHAPES (2)
            #         (8 total whole-hand configuration poses here, 2 * 2 * 2)
            #   At each wrist goal_pose, we rotate the wrist len(wrist_rots)
            #     times. Inside each rotation, we vary the fore-finger
            #     preshape N_PRESHAPES times.
            #   Accumulation N_ACCUM_POS is counting the number of goal_poses.
            #     It's really the number of "positions", not the pose of
            #     wrist. But I call it goal_pose here because in the
            #     terminology of forward kinematics, it's a pose. In the
            #     higher level sense of the 3-layer definition of varying
            #     gripper parameters for sampling, it's a position.
            #     At this level of reasoning, "pose" means the layer beneath
            #     N_PRESHAPES, it's not a wrist pose, it's an entire hand
            #     configuration pose, including fingers.
            #####

            cache_this_iter = False

            # If this is the last preshape at this position, check if stop
            #   accumulating contacts. Only check at last preshape, `.` if
            #   there are more than 1 preshape at each position, still need
            #   to cache across all the preshapes at the position!
            if ps_i == cfg.N_PRESHAPES - 1:

              # Cache multiple wrist positions
              if cfg.ACCUM_CONTACTS_BTW_POS:
             
                # If have accumulated N_ACCUM_POS-1 positions, this is the
                #   one to evaluate at.
                if n_pos_accumulated == cfg.N_ACCUM_POS - 1:
                  cache_this_iter = False
                  n_pos_accumulated = 0
             
                # Last position ever on the ellipsoid grid, then whether we
                #   have accumulated N_ACCUM_POS, need to evaluate triangles.
                # TODO: USE_CYL_GRID is temporary, until I write a
                #   geo_cylinder.py for cylinder. Might just add to
                #   geo_ellipsoid.py and pass in different constants for
                #   different shapes for initialize() method!
                elif not cfg.USE_CYL_GRID and isLastPos:
                  cache_this_iter = False
                  n_pos_accumulated = 0
             
                else:
                  cache_this_iter = True
                  n_pos_accumulated += 1
             
              # Cache all preshapes at a wrist pose
              # ACCUM_CONTACTS_BTW_POS overwrites
              #   ACCUM_CONTACTS_BTW_PRESHAPES. If accumulate across wrist
              #   positions, then must accumulate btw all preshapes in a
              #   wrist position.
              elif cfg.ACCUM_CONTACTS_BTW_PRESHAPES:
                cache_this_iter = False
              # Not caching at all
              else:
                cache_this_iter = False
         
            # Else, not at the last pose yet, keep accumulating
            else:
              if cfg.ACCUM_CONTACTS_BTW_POS or cfg.ACCUM_CONTACTS_BTW_PRESHAPES:
                cache_this_iter = True
              # Not caching at all
              else:
                cache_this_iter = False


            # Copied from triangles_collect_semiauto.py
         
            # Record once in current preshape
            # Calls guarded_move, open, and records data to file.
            # Need to add a little wait time, otherwise hand is still opening
            #   when I teleport it, making it tilt, then the pose is not the
            #   desired pose anymore.
            # wait_time_before_record=2.5 seconds reduces most of the noise,
            #   but not all, maybe ~15%-25% noise remain. If you want no
            #   noise, it is necessary to wait, BEFORE recording, to allow
            #   contacts to pass from Gazebo contact sensors, to
            #   reflex_driver_node.py, to detect_reflex_contacts.py, and to
            #   here. It takes a while.
            #   > But in the end, it is not worth it to wait 2.5 seconds!!!
            #   Data collection takes forever!!! The bit of noise reduced is
            #   probably nothing in the whole histogram, only a little tail
            #   probably, as Ani conjectured, and I agree. Also, noise is
            #   only bad when obj is small, during which fingers act more
            #   abnormally and get more bad collisions. On large objects,
            #   there is almost no noise. So this wait is really unnecessary
            #   and takes toll on time.
            self.cyl_grid_node.close_gripper_and_record ( \
              guarded_move_wait_time=0.2, open_wait_time=0.2,
              wait_time_before_record=0.5,
              cache_only_dont_eval=cache_this_iter)


            # Record absolute wrist goal_pose and triangle observations to
            #   probabilities matrix for active touch.
            if cfg.RECORD_PROBS:
              # Compute triangles using latest contact points at this finger
              #   pose. Don't prune duplicates, so each wrist pose collectis
              #   the honest triangle observations at that pose.
              contacts_at_curr_wrist_pose.extend (
                self.recorderNode.contacts_latest_hand)

            # end RECORD_PROBS

         
            #####
            # If hand got stuck on object and not opened all the way, move
            #   hand away to an empty space, then open it all the way, and
            #   move back to where it's supposed to be
            #####
         
            # 3-element list
            prox_pos = self.recorderNode.get_prox_joint_pos ()
            # If any one finger isn't opened all the way, teleport somewhere
            #   else to open hand, then teleport back.
            if prox_pos [0] > self.recorderNode.TENDON_MIN or \
              prox_pos [1] > self.recorderNode.TENDON_MIN or \
              prox_pos [2] > self.recorderNode.TENDON_MIN:
         
              print ('%sFingers not all the way opened. Teleporting away to open fully%s' % ( \
                ansi_colors.OKCYAN, ansi_colors.ENDC))
         
              # Teleport hand somewhere safe
              print ('%sMoving hand away from object birth place%s' % ( \
                ansi_colors.OKCYAN, ansi_colors.ENDC))
              self.teleport_hand (self.hand_name, *cfg.SAFE_PLACE)
              # Wait a bit, to let hand get settled. Otherwise fingers are
              #   crooked
              time.sleep (0.2)
              #rospy.sleep (0.2)
         
              # Open fingers
              print ('%sOpening ReFlex fingers...%s' % (ansi_colors.OKCYAN,
                ansi_colors.ENDC))
              # Wait slightly longer, to let hand get settled, otherwise still
              #   not opened all the way
              call_smart_commands ('open', 0.3)
         
              # Teleport hand back to wrist position, if this is not the
              #   last preshape in this wrist position.
              #   (If this is the last preshape, next iter will move hand to
              #   next pose. This moves to old pose, but nothing else needs to
              #   be done at old pose. It just goes to new pose afterward
              #   anyway. So this is unnecessary, just wastes time.)
              if ps_i != cfg.N_PRESHAPES - 1:
                self.teleport_hand (self.hand_name, goal_pos_hand,
                  goal_rot_hand)

            # end checking hand is fully opened and unstuck
         
            print ('')
         
            if cfg.PROFILER: print ('7: '), print (time.time() - start_time)

          #end N_PRESHAPES


          if cfg.RECORD_PROBS:

            # Store observations, (l0, l1, a0) parameters of triangle.
            #   There may be more than 1 triangle observed, so a list of the
            #   3 params.
            # Triangles are pulled from triangles_collect.py.
            #   New triangles are only computed when cache is turned off.
            #   When cache is on, then the contact points are stored, no
            #   triangles are computed from the points.

            # Don't use get_latest_triangles_h(), `.` it's not cleared out
            #   after calculation at the end of an uncached iteration.
            #   So whether in a cached or uncached iteration, it'll return
            #   stuff, meaning in cached iterations, it'll return same data
            #   as the latest uncached iteration's calculations! This results
            #   in many many duplicates!! Waste of space and bad data.


            # Copied from triangles_collect.py record_tri()
            # 3 is nTriParams, number of pts needed to make a triangle
            print ('RECORD_PROBS: Sampling triangles from %d points independently of actual triangles data stored' % len (contacts_at_curr_wrist_pose))

            (_, l0, l1, l2, a0, a1, a2, _) = \
              sample_reflex.sample_tris (contacts_at_curr_wrist_pose, 3)
            nTris = len (l0)

            print ('RECORD_PROBS: Got %d triangles' % nTris)

            # If got more than 0 triangles, add to probs and costs matrix.
            #   I think this is okay... We don't need to add costs to a 
            #   relative action, if there's no triangles there, right? `.`
            #   active shouldn't even consider the pose, if the pose didn't
            #   see any triangles! Then no need to trade off cost with anyth.
            if nTris > 0:
              # nTris x 6 NumPy matrix. 6 is for l0, l1, l2, a0, a1, a2.
              tris_at_curr_wrist_pose = np.zeros ((0, 6))
             
              for i in range (0, nTris):
                # 1 x 6 row vector. This must be 2D, otherwise axis=1 passed
                #   to np.all() below will be out of bounds, when seen and
                #   tri_curr are both 1D.
                tri_curr = np.array ([[l0[i], l1[i], l2[i], a0[i], a1[i], a2[i]]])
                tris_at_curr_wrist_pose = np.append (tris_at_curr_wrist_pose, tri_curr, axis=0)
             
              # tris_at_curr_wrist_pose should have collected from N_PRESHAPES
              #   finger poses, inside only 1 wrist pose (which is what we
              #   want for probs, z1 must be observed from 1 wrist pose only,
              #   not accumulated from many)
              self.io_probs_node.add_abs_pose_and_obs (abs_pose,
                tris_at_curr_wrist_pose)

            # end nTris > 0

          # end RECORD_PROBS

        # end wrist_rots


        # Sleep a bit, to let hand rest before moving again. Else it's a
        #   visible sudden jump right after opening to moving. Not good.
        time.sleep (0.1)
        #rospy.sleep (0.1)

      # end if goal_pose

 
      # Copied from triangles_collect_semiauto.py
      # If using cylinder grid and terminating, or if using ellipsoid grid,
      #   write output files.
      # recorderNode.doTerminate indicates there's an error.
      if (cfg.USE_CYL_GRID and self.cyl_grid_node.doTerminate) or \
         (not cfg.USE_CYL_GRID and ell_doTerminate) or \
         (self.recorderNode and self.recorderNode.doTerminate):

        # Copied from triangles_reader.py
        # Print out running time
        # Seconds
        end_time = time.time ()
        # rospy.Time.now().to_sec() - start_time.to_sec () doesn't work.
        #   Gives me 0 for elapsed_time
        #end_time = rospy.Time.now ().to_sec ()
        elapsed_time = end_time - start_time
        print ('Total time this run: %f seconds.' % (elapsed_time))
 
        self.cyl_grid_node.write_obj_params ()

        if self.recorderNode:
          # Close output files
          self.recorderNode.doFinalWrite = True
          # Write final PCD file, with header
          self.recorderNode.collect_one_contact ()
          print ('')

          # Record log
          if self.recorderNode.timestring:
            # Print time string for easier copy-pasting to start next pickup
            #   run
            print ('%sTimestring for this run: %s%s' % ( \
              ansi_colors.OKCYAN, self.recorderNode.timestring,
              ansi_colors.ENDC))
 
            # If no timestring, that means no contacts, don't log a line,
            #   it'd have no timestring - empty string! Then the human
            #   readable file is wrong.
            if cfg.do_log_this_run:
              cfg.log_this_run (
                self.obj_cat, self.obj_basename, cfg.PALM_THICKNESS,
                self.recorderNode.timestring, cfg.pickup_from_file,
                cfg.pickup_ell_idx, cfg.pickup_z, cfg.pickup_theta,
                self.recorderNode.nPts_ttl, self.recorderNode.nTris_h_ttl,
                elapsed_time)

        break
      # end if terminating

    # end while 1  # collect one object


    # Return full path to triangles file recorded
    return self.recorderNode.trih_name, elapsed_time


  # Teleport hand to a specified pose, then check Gazebo topics to see actual
  #   hand pose matches desired pose. Re-teleport if necessary, until actual
  #   pose matches desired.
  # Parameters:
  #   STRICT_POSE: Whether to check hand's actual pose. If False, will just
  #     teleport once and return, without verifying whether hand's actual pose
  #     matches commanded pose.
  def teleport_until_correct (self, hand_name, pos_base_link_wrt_base,
    quat_base_link_wrt_base, STRICT_POSE=True):
 
    while 1:
  
      # Teleport hand to goal pose
      self.teleport_hand (hand_name, pos_base_link_wrt_base,
        quat_base_link_wrt_base)
        #*pos_base_link_wrt_base, *quat_base_link_wrt_base)
  
  
      # If pose correction not required, then just teleport once and
      #   move on
      if not STRICT_POSE:
        break
  
      success = False
      while not success:
        success, pose_actual = get_model_state (hand_name)
  
      correct = compare_poses (Pose (Point (pos_base_link_wrt_base [0],
        pos_base_link_wrt_base [1], pos_base_link_wrt_base [2]),
        Quaternion (quat_base_link_wrt_base [0],
          quat_base_link_wrt_base [1],
          quat_base_link_wrt_base [2],
          quat_base_link_wrt_base [3])),
        pose_actual)
  
      if not correct:
        continue
      else:
        break


  def teleport_hand (self, hand_name, t, q): # tx, ty, tz, qx, qy, qz, qw):
 
    # Allow hand to move
    set_fixed_hand (False)
    print ('')
 
 
    srv_name = '/reflex_gazebo/set_model_pose'
 
    #rospy.loginfo ('Waiting for rosservice %s...' % srv_name)
    rospy.wait_for_service (srv_name)
 
    rospy.loginfo ('Calling rosservice %s to move hand...' % srv_name)
 
    try:
      srv = rospy.ServiceProxy (srv_name, SetModelPose)
 
      req = SetModelPoseRequest ()
      req.model_name = hand_name
 
      req.pose = Pose (Point (t[0], t[1], t[2]),
        Quaternion (q[0], q[1], q[2], q[3]))
 
      # True for relative wrt parent, or False for world pose
      # For the hand, it doesn't matter, since it's parent IS the world. So
      #   either mode should do the same thing
      req.relative = False
 
      resp = srv (req)
 
      time.sleep (0.5)
      #rospy.sleep (0.5)
 
    except rospy.ServiceException, e:
      rospy.logerr ('sample_gazebo teleport_hand(): Service call to %s failed: %s' %(srv_name, e))
      return False
 
 
    # Don't allow hand to be moved if bumped by object when open fingers
    # Equivalent of:
    #   rostopic pub /reflex_gazebo/set_fixed_hand std_msgs/Bool -1 1
    set_fixed_hand (True)
    print ('')
 
 
    if resp.success:
      # Too much printouts. Comment out if you want to see other debug info.
      #rospy.loginfo ('rosservice call succeeded')
 
      # Broadcast hand frame /base_link wrt /base, to put a fixed "robot" frame,
      #   so data collection triangles_collect.py can record a PCD file (any
      #   model file records a fixed frame as reference, else points collected
      #   wrt a moving frame are just garbage).
      # Since my rosservice in world_plugin.cpp teleports by absolute coordinates
      #   (when req.relative=false), this effectively places /base robot frame
      #   at origin of Gazebo world.
      msg = Transform ()
      msg.pose = Pose (Point (t[0], t[1], t[2]),
        Quaternion (q[0], q[1], q[2], q[3]))
      msg.parent_frame = '/base'
      msg.child_frame = '/base_link'
      self.tf_bc_hand_pub.publish (msg)
 
      # Broadcast fake wrist frame /left_gripper wrt /base_link.
      #   Ideally, /left_gripper should be parent of /base_link, as on real
      #   robot. But that is hard to do, because I'd have to listen to the
      #   transform of /base_link wrt /base. Then put this relative pose into
      #   fixed robot frame /base, and then broadcast /left_gripper wrt /base.
      #   Then get /base_link wrt the new /left_gripper, and re-broadcast
      #   /base_link with parent of /left_gripper!! It's not straight forward,
      #   and it's a waste of time waiting for tf listener twice, to just
      #   broadcast two frames!
      # So will just compromise with broadcasting /left_gripper as child of
      #   /base_link. I only need the pose of the frames, don't care about
      #   which one is parent or child.
      msg2 = Transform ()
      msg2.pose.position.x = wrist_to_hand_transforms.pos_wrist_wrt_hand [0]
      msg2.pose.position.y = wrist_to_hand_transforms.pos_wrist_wrt_hand [1]
      msg2.pose.position.z = wrist_to_hand_transforms.pos_wrist_wrt_hand [2]
      msg2.pose.orientation.x = wrist_to_hand_transforms.quat_wrist_wrt_hand [0]
      msg2.pose.orientation.y = wrist_to_hand_transforms.quat_wrist_wrt_hand [1]
      msg2.pose.orientation.z = wrist_to_hand_transforms.quat_wrist_wrt_hand [2]
      msg2.pose.orientation.w = wrist_to_hand_transforms.quat_wrist_wrt_hand [3]
      msg2.parent_frame = '/base_link'
      msg2.child_frame = '/left_gripper'
      self.tf_bc_wrist_pub.publish (msg2)
 
      return True
    else:
      rospy.logwarn ('rosservice call failed')
      return False



  # Not in use anymore. If triangles_collect_semiauto.py is not available for
  #   some reason, this snippet will let you test whether teleport_hand works:
  #
  # Loop through each test pose, to test set_model_state works for moving
  #   hand to arbitrary pose.
  #for i in range (0, 4):
  #  pose_tuple = thisNode.get_next_wrist_pose ()
  '''
  #  # Unpack the tuple into separate params
  #  teleport_hand (hand_name, *pose_tuple)
  def get_next_wrist_pose (self):

    #self.wrist_pose_seen_idx += 1

    #i = self.wrist_pose_seen_idx

    #return (self.txh[i], self.tyh[i], self.tzh[i], self.qxh[i], self.qyh[i],
    #  self.qzh[i], self.qwh[i])



    # Copied from triangles_collect_semiauto.py

    # Get the angle for this theta-column
    theta = self.cyl_grid_node.wall_angles [theta_idx]
    # If this is a new column, calculate new column
    # Get x y and list of heights for this theta-column.
    #   x and y stay the same throughout all z's.
    if z_idx == 0:
      (self.cyl_grid_node.x, self.cyl_grid_node.y, self.cyl_grid_node.z_vec) = \
        self.cyl_grid_node.calc_column (self.cyl_grid_node.wrist_wall_r, theta)

    # Get current grid point's z position
    z = self.cyl_grid_node.z_vec [z_idx]


    print ('%sz_idx: %d out of %d (0 based), z: %f%s' % \
      (ansi_colors.OKCYAN, z_idx, len (self.cyl_grid_node.z_vec),
       self.cyl_grid_node.z_vec[z_idx], ansi_colors.ENDC))
    print (self.cyl_grid_node.z_vec)
    print ('%stheta_idx: %d out of %d (0 based), angle: %f%s' % \
      (ansi_colors.OKCYAN, theta_idx, len (self.cyl_grid_node.wall_angles)-1,
      self.cyl_grid_node.wall_angles [theta_idx] * 180.0 / np.pi,
      ansi_colors.ENDC))


    #####
    # Book-keeping: Update theta_idx and z_idx
    #####

    # If prev grid point was done, increment indices to go to next grid point
    #
    #   This control sequence is equivalent to this loop:
    #
    #   for theta in self.wall_angles:
    #     # Loop through each height on this theta-column
    #     for z in z_vec:
    #if self.prev_move_success:

    # Inner loop. Loop through height along a theta-column
    z_idx += 1

    # If inner loop completed, increment to next theta-column
    if z_idx == len (self.cyl_grid_node.wall_heights):
      theta_idx += 1
      # Reset height to start fresh in the new column
      z_idx = 0

      # If all theta-columns done, we can terminate program
      if theta_idx == len (self.cyl_grid_node.wall_angles):
        self.cyl_grid_node.doTerminate = True
        print ('Finishing object... (collect_wall() detects all grid points done)')
        return None

    self.cyl_grid_node.theta_idx = theta_idx
    self.cyl_grid_node.z_idx = z_idx

    return [self.cyl_grid_node.x, self.cyl_grid_node.y, ]
  '''


# Not using this anymore.
# This uses gazebo_ros_pkgs's rosservice, which sucks. It kills Gazebo after
#   a while. I do not trust this package at all. It set_physics_properties also
#   makes my hand's twist params all nans. Sucks!!!!! My WorldPlugin did the
#   same job without this stupid error.
# Equivalent of
#   $ rosservice call /gazebo/set_model_state '{model_state: {model_name: reflex, pose: {position: {x: 0.5, y: 0, z: 0}, orientation: {x: 0, y: 0, z: 0, w: 1}}}}'
'''
def teleport_hand_ros (hand_name, tx, ty, tz, qx, qy, qz, qw):

  srv_name = '/gazebo/set_model_state'

  rospy.loginfo ('Waiting for rosservice %s...' % srv_name)
  rospy.wait_for_service (srv_name)

  rospy.loginfo ('Calling rosservice %s to move hand...' % srv_name)

  try:
    srv = rospy.ServiceProxy (srv_name, SetModelState)

    req = SetModelStateRequest ()
    req.model_state.model_name = hand_name
    req.model_state.pose.position.x = tx
    req.model_state.pose.position.y = ty
    req.model_state.pose.position.z = tz
    req.model_state.pose.orientation.x = qx
    req.model_state.pose.orientation.y = qy
    req.model_state.pose.orientation.z = qz
    req.model_state.pose.orientation.w = qw

    resp = srv (req)

    time.sleep (0.5)
    #rospy.sleep (0.5)

  except rospy.ServiceException, e:
    rospy.logerr ('sample_gazebo teleport_hand_ros(): Service call to %s failed: %s' %(srv_name, e))
    return False

  if resp.success:
    # Too much printouts. Comment out if you want to see other debug info.
    #rospy.loginfo ('rosservice call succeeded')
    return True
  else:
    rospy.logwarn ('rosservice call failed: %s' % resp.status_message)
    return False
'''


