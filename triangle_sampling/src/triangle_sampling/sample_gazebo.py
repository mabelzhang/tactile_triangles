#!/usr/bin/env python

# Mabel Zhang
# 4 Nov 2015
#
# Sample triangles from object mesh models in Gazebo, using my simulated ReFlex
#   Hand.
#
# Read meta list of objects. For each object, generates a temporary Gazebo SDF
#   file that imports the object's .dae mesh file.
#   Spawn the SDF file (i.e. load the object) in Gazebo.
# Move hand around the object in a cylinder grid (by calling routines in
#   triangles_collect.py), enclose fingers onto object using simulated
#   guarded_move
#
#
# To use this file:
# 1. Start the hand in a Gazebo world
#    roslaunch reflex_gazebo reflex_world.launch
#
# 2. Controllers, reflex sim rosservices, tf broadcaster, RViz Markers:
#    roslaunch reflex_gazebo reflex.launch
#
# 3. Publish my custom contact msgs, so triangles_collect.py can record them
#    rosrun tactile_map detect_reflex_contacts.py
#
# 4. This file, move hand around and collect data
#    rosrun triangle_sampling sample_gazebo.py
#  
# Parameters are set in sample_gazebo_utils.py, in SampleGazeboConfigs.
# Parameters for sampling:
#   Grid density:
#     cyl_density: arc length on cylinder (if using cylinder sampling)
#     ell_deg_step: degree step on ellipsoid grid (if using ellipsoid sampling)
#   preshapes[] (search for N_PRESHAPES to get to it quickly), for fore-finger
#     preshape angle
#   wrist_rots[], for rotation around wrist frame
#   ACCUM_*, for accumulating contacts for the (n choose 3) in triangle
#     calculation. If False, no accumulation, just calculate from contacts
#     in each grasp (or grasps until have at least 3 contact points). If True,
#     accumulate for however many wrist positions you specify, or accumulate
#     for all preshapes at each wrist pose.
#
# Parameters for replayability to avoid time-consuming re-training:
#   do_log_this_run: True to record obj, pickup file name, stats, etc to
#     ../out/gz/per_run.csv.
#
# Parameters for recording probabilities:
#   RECORD_PROBS: True to record probabilities in .pkl files, False to skip.
#     Saving these files to disk can take a long I/O time, so set to False if
#     you aren't training probabilities data!


# New print() to allow writing to file http://stackoverflow.com/questions/6159900/correct-way-to-write-line-to-file-in-python
#   Usage: print ('hello', file=f)
from __future__ import print_function

# ROS
import rospy
from std_msgs.msg import Bool
import tf
import rosnode
from visualization_msgs.msg import Marker, MarkerArray

# Gazebo
from gazebo_msgs.srv import DeleteModel, DeleteModelRequest, \
  SetPhysicsProperties, SetPhysicsPropertiesRequest, \
  GetWorldProperties, GetWorldPropertiesRequest

# Python
import os, shutil
import subprocess
import time
import signal
import argparse

# NumPy
import numpy as np

# My packages
from util.ansi_colors import ansi_colors
from tactile_collect import tactile_config
from triangle_sampling.config_paths import get_robot_obj_params_path
from triangle_sampling.parse_models_list import parse_meta_one_line
from tactile_map.create_marker import create_marker
from reflex_gazebo_msgs.srv import SetBool, SetBoolRequest
from tactile_map.tf_get_pose import *
from triangle_sampling.sample_gazebo_core import SampleGazeboCore, \
  pause_physics, get_model_size, remove_model, timeout_handler, \
  load_rescale_config, load_object_into_scene, call_empty_rosservice
from triangle_sampling.sample_gazebo_utils import SampleGazeboConfigs
from triangle_sampling.triangles_collect import TrianglesCollect
from triangle_sampling.triangles_collect_semiauto import \
  TrianglesCollectSemiAuto
from triangle_sampling.geo_ellipsoid import EllipsoidSurfacePoints
from triangle_sampling.io_probs import IOProbs


def load_meta_file (pkg_path, meta_base):

  train_rootpath = tactile_config.config_paths ('custom', '')

  meta_name = os.path.join (pkg_path, 'config/', meta_base)

  print ('%sLoading meta file %s...%s' % (
    ansi_colors.OKCYAN, meta_name, ansi_colors.ENDC))

  # Size: number of categories
  catnames = []
  catcounts = []
  # Size: number of object files
  catids = []
  # Full path to .dae files
  in_dae_names = []

  # Read meta list file line by line
  with open (meta_name, 'rb') as metafile:

    for line in metafile:

      line = line.strip ()

      # Parse line in file, for base name and category info
      #   Ret val is [basename, cat_idx], cat_idx for indexing catnames.
      parse_result = parse_meta_one_line (line, catnames, catcounts, catids)
      if not parse_result:
        continue

      # Make sure file extension is dae, if not, replace with dae (`.`
      #   sometimes it could be pcd if I'm not careful in copying)
      if os.path.splitext (line) [1] != '.dae':
        print ('%sFile in meta list does not have .dae extension. Will try to replace, but you should fix the meta file!%s' % (ansi_colors.FAIL, ansi_colors.ENDC))
        line = os.path.splitext (line) [0] + '.dae'

      dae_name = os.path.join (train_rootpath, line)
      if not os.path.isfile (dae_name):
        print ('%sFile does not exist. Correct it in meta file and try again. Offending file: %s %s' % (ansi_colors.FAIL, dae_name, ansi_colors.ENDC))
        return [], [], []

      # Get full path. SDF export needs this
      in_dae_names.append (dae_name)

      #catnames [cat_idx]

  return in_dae_names, catids, catnames


######################################################## Gazebo environment ##

def reset_simulation ():

  # Equivalent of: rosservice call /gazebo/reset_simulation
  return call_empty_rosservice ('/gazebo/reset_simulation')


# Don't disable gravity this way. This way makes all twist params of robot
#   go to nan, and then you can't move it or do anything on it anymore!
# Use Gazebo WorldPlugin world_plugin.cpp to set via
#   gazebo::physics::PhysicsEngine().
def disable_gravity ():

  # Equivalent of: rosservice call /gazebo/set_physics_properties '{gravity: {x: 0.0, y: 0.0, z: 0.0}}'

  srv_name = '/gazebo/set_physics_properties'

  #rospy.loginfo ('Waiting for rosservice %s...' % srv_name)
  rospy.wait_for_service (srv_name)

  rospy.loginfo ('Calling rosservice %s to disable gravity...' % srv_name)
  try:
    srv = rospy.ServiceProxy (srv_name, SetPhysicsProperties)

    req = SetPhysicsPropertiesRequest ()
    req.gravity.x = 0.0
    req.gravity.y = 0.0
    req.gravity.z = 0.0

    resp = srv (req)

    time.sleep (0.2)
    #rospy.sleep (0.2)

  except rospy.ServiceException, e:
    rospy.logerr ('sample_gazebo disable_gravity(): Service call to %s failed: %s' %(srv_name, e))
    return False

  if resp.success:
    # Too much printouts. Comment out if you want to see other debug info.
    #rospy.loginfo ('rosservice call succeeded')
    return True
  else:
    rospy.logwarn ('rosservice call failed: %s' % resp.status_message)
    return False


# SetBool.srv: http://docs.ros.org/jade/api/std_srvs/html/srv/SetBool.html
def call_setbool_rosservice (srv_name, data, wait_time=0.3):

  #rospy.loginfo ('Waiting for rosservice %s...' % srv_name)
  rospy.wait_for_service (srv_name)

  rospy.loginfo ('Calling rosservice %s, passing in %s...' % (srv_name,
    str(data)))

  try:
    srv = rospy.ServiceProxy (srv_name, SetBool)

    req = SetBoolRequest ()
    req.data = data

    resp = srv (req)
    time.sleep (wait_time)
    #rospy.sleep (wait_time)

  except rospy.ServiceException, e:
    rospy.logerr ('sample_gazebo call_setbool_service(): Service call to %s failed: %s' %(srv_name, e))
    return False

  if resp.success:
    # Too much printouts. Comment out if you want to see other debug info.
    #rospy.loginfo ('rosservice call succeeded')
    return True
  else:
    rospy.logwarn ('rosservice call failed with msg: %s' % (resp.message))
    return False


def get_world_properties (wait_time=0.3):

  srv_name = '/gazebo/get_world_properties'
  #rospy.loginfo ('Waiting for rosservice %s...' % srv_name)

  rospy.wait_for_service (srv_name)

  rospy.loginfo ('Calling rosservice %s...' % (srv_name))

  try:
    srv = rospy.ServiceProxy (srv_name, GetWorldProperties)

    req = GetWorldPropertiesRequest ()

    resp = srv (req)
    time.sleep (wait_time)
    #rospy.sleep (wait_time)

  except rospy.ServiceException, e:
    rospy.logerr ('sample_gazebo get_world_properties(): Service call to %s failed: %s' %(srv_name, e))
    return (False, None)

  if resp.success:
    # Too much printouts. Comment out if you want to see other debug info.
    #rospy.loginfo ('rosservice call succeeded')
    return (True, resp)
  else:
    rospy.logwarn ('rosservice call failed with msg: %s' % (resp.status_message))
    return (False, None)


# This doesn't correctly spawn hand though. Hand doesn't appear in Gazebo,
#   though it says it's spawned. It's nowhere to be found!
# Equivalent of this in .launch file:
#   <node name="spawn_urdf" pkg="gazebo_ros" type="spawn_model" args="-param robot_description -urdf -model reflex" />
#   $ rosrun gazebo_ros spawn_model -param robot_description -urdf -model reflex
def spawn_hand (x=0, y=0, z=0, R=0, P=0, Y=0, hand_name='reflex'):

  rospy.loginfo ('Calling rosrun to spawn robot hand...')

  # This prints stuff to rospy.loginfo, AND returns. Best so far
  p = subprocess.call (['rosrun', 'gazebo_ros', 'spawn_model', '-param',
    'robot_description', '-urdf', '-model', hand_name])
    #'-x', str(x), '-y', str(y), '-z', str(z),
    #'-R', str(R), '-P', str(P), '-Y', str(Y)])

  time.sleep (2)
  #rospy.sleep (2)

  rospy.loginfo ('rosrun spawn_model returned')

  return hand_name


# Parameters:
#   hand_name: Name of existing hand in world to delete, and of new hand to
#     load. The new hand will have same name as existing hand. If no hand
#     exists, just pass in name for new hand
def reload_hand (x=0, y=0, z=0, R=0, P=0, Y=0, old_hand_name='reflex',
  new_hand_name='reflex'):

  print ('%sReloading hand, old model named %s, new model named %s...%s' % ( \
    ansi_colors.OKCYAN, old_hand_name, new_hand_name, ansi_colors.ENDC))

  # Sanity check
  if not old_hand_name:
    print ('%sERROR: sample_gazebo reload_hand() sees no old_hand_name specified. Must specify hand name.%s' % ( \
      ansi_colors.FAIL, ansi_colors.ENDC))
    return
  # If only specified old name, but no new name, then set new name to old
  elif not new_hand_name:
    new_hand_name = old_hand_name


  # Delete existing hand first

  # Delete the contact sensors manually, else new hand's contact sensors can't
  #   be loaded!
  #call_empty_rosservice ('/reflex_gazebo/remove_hand', wait_time=3)
  # Not an empty service anymore, now using RemoveModel msg type, so just
  #   call it like remove_model.
  remove_model (old_hand_name, srv_name='/reflex_gazebo/remove_hand')

  # Delete hand model
  # Don't delete using gazebo_ros rosservice anymore, buggy. Use WorldPlugin's
  #   RemoveModel(). The rosservice /gazebo/delete_model is bad, when it
  #   removes a model (e.g. hand), it doesn't remove all the contact sensors
  #   on it!! So when I reload the hand, Gazebo says the sensors already exist,
  #   but I can't delete them - it says they don't exist!!!!! Contradicting!
  #   http://answers.gazebosim.org/question/6369/how-to-delete-a-model-entirely-including-contact-filter/
  #   https://bitbucket.org/osrf/gazebo/pull-requests/1106/added-world-removemodel-to-fix-issue-1177/diff
  # > RemoveModel() doesn't work. Gives internal pointer NULL error. Gosh they
  #   can't even make a fix right.
  remove_model (old_hand_name)
  # Wait for hand to be completely removed
  time.sleep (1)
  #rospy.sleep (1)


  # Spawn new hand
  new_hand_name = spawn_hand (x, y, z, R, P, Y, new_hand_name)


  # I don't think this is necessary. removing the hand seems to do these
  #   automatically. Just need to set respawn=true for everything in
  #   reflex.launch.

  # Restart any existing Gazebo joint controllers, so they can connect with the
  #   new hand. This is needed if the hand was removed and then reloaded.

  # Kill the nodes, to let roslaunch respawn them automatically.
  # Assumption: reflex_gazebo pkg reflex.launch should have set <node
  #   respawn="true">, so that these will be respawned when they are killed.
  #   This code is only responsible for killing them, not restarting them!
  #   This is the ROS way to go. Not good to manually start nodes, because
  #   there's no such thing as rosnode run, so you must start nodes by
  #   calling roslaunch. But you can't kill every node in a roslaunch
  #   without hardcoding all of their names in program, and kill using rosnode
  #   kill! Here, I only have to hardcode two, this is the minimum that would
  #   work.
  # Ref rosnode API: http://docs.ros.org/jade/api/rosnode/html/
  #
  # Not killing anything, things print working, but the fingers don't actually
  #   close.
  # Killing only /rhr_flex_model/controller_spawner seems to freeze Gazebo.
  # Killing both /robot_state_publisher and /reflex_driver_node, fingers 
  #   still don't move.
  # Doesn't freeze anymore! Solution is to kill all 3!!! Then works!!!!
  #   Fingers close now!! Hand is reloaded!!!
  successes, fails = rosnode.kill_nodes (['/rhr_flex_model/controller_spawner',
    '/robot_state_publisher', '/reflex_driver_node'])
    #'/reflex_tf_broadcaster', '/hand_visualizer'])
  print ('Killed successfully:')
  print (successes)
  print ('Kill failed:')
  print (fails)

  # Need to kill detect_reflex_contacts too, because it's connected to the
  #   old reflex_driver_node! Need to reconnect to new one, else no contacts
  #   are published!
  # TODO NOW HERE: Still not getting anything published from
  #   detect_reflex_contacts.py!!! Even if I rosrun an instance. Is it that
  #   triangles_collect (recorderNode) needs to be re-instantiated too??
  #   But it's not just the problem with connection to it, I think. It simply
  #   isn't even detecting contacts from /reflex_hand!!! Nothing is being
  #   published on /tactile_map/detect_reflex_contacts/contacts topic!!
  #   Is the new reflex_driver_node not registered with Gazebo contact sensors??
  #   But they're just rostopic msgs! Should be registered, since I'm
  #   restarting reflex_driver_node, it should subscribe to whatever that's
  #   existing!!
  # Do I need to kill reflex_tf_broadcaster and hand_visualizer too??
  #   Tried. Didn't help.
  #
  # Oh that's not even the problem! Even rostopic echo /reflex_hand doesn't
  #   return any True's!! No wonder detect_reflex_contacts doesn't detect
  #   any contacts! The problem is on reflex_driver_node not seeing contacts
  #   published by Gazebo sensors. But the joint position values do change,
  #   so it IS publishing reflex_hand at current time.
  # Are gazebo contact sensors publishing contacts then??? Maybe they stop
  #   after I kill hand?? TODO NOW HERE do this:
  #   $ rostopic echo /reflex_gazebo/contact
  #
  #   This topic is published by the contact sensors, in my reflex_gazebo pkg
  #   contact_sensor_plugin.cpp. If these don't publish anything, then it's a
  #   problem btw the killed hand and contact sensors restarting. Gazebo does
  #   give red errors when I kill hand, about contact sensors existing:
  #
  #     Error [ContactManager.cc:271] Filter with the same name already exists! Aborting
  #     Handle name: f1s6_plugin
  #     Parsed: 1 6
  #     [ INFO] [1451905795.560939196, 73.064000000]: Gazebo contact sensor plugin initialized for finger 1 sensor 6
  #
  #   I think this means, when I load new hand, a new sensor is added with the
  #   same name as the old plugin on the old hand. But the old plugin was never
  #   automatically deleted when the hand was deleted. So the new plugin is NOT
  #   loaded! (The plugin intialized is printed by me in
  #   contact_sensor_plugin.cc, so it doesn't mean anything, I probably didn't
  #   detect for failure.) Then when the new hand moves around, the old plugins
  #   are ghosts, not attached to any physical body anymore. They don't move
  #   with the hand. And on the real hand, there are no new sensor plugins
  #   loaded, so that's why no contacts were published! So I should remove the
  #   old plugin manually when I remove the hand! BTW Gazebo really should do
  #   this automatically!!
  # 
  #   Would I have to kill the contact plugins too?? I don't know how to kill
  #   them separately. Killing the hand should have killed them... If not,
  #   might have to write a rosservice in world_plugin.cpp to bring down the
  #   contact sensors explicitly?? Can probably find the contact sensors by
  #   name, then dynamic cast to sensors::ContactSensor. Then find a destroy
  #   function. Maybe have to manually DisconnectUpdated()? Or SetActive(false)?
  #   https://osrf-distributions.s3.amazonaws.com/gazebo/api/dev/classgazebo_1_1sensors_1_1ContactSensor.html
  #
  #   Oh gee looks like a known problem!
  #   http://answers.gazebosim.org/question/8193/multiple-bumper-sensors-filter-with-the-same-name-already-exists/
  #   pull request merged:
  #     https://bitbucket.org/osrf/gazebo/pull-requests/1413/allow-multiple-contact-sensors-per-link/diff
  #   scpeters made a fix in gazebo4 and gazebo5, but maybe it's not the same
  #   as my problem. Maybe I should manually remove sensor from world. How to
  #   completely remove it?
  #
  #   WorldPlugin can do it via RemovePlugin():
  #   https://osrf-distributions.s3.amazonaws.com/gazebo/api/dev/classgazebo_1_1physics_1_1World.html
  #     void 	RemovePlugin (const std::string &_name)
 	#     Remove a running plugin. More...
  #
  #   Maybe a better option than deleting the hand is World->Reset()?
  #
  #
  #   RemovePlugin didn't work. Maybe need to remove the actual sensors.
  #   Sensors cannot be removed by World object, as they are loaded in .gazebo
  #   file, which probably puts them directly into the world, instead of being
  #   attached to part of a model! Maybe that's why Gazebo can't delete them
  #   automatically when you delete a model. World obj can remove models, but
  #   not sensors.
  #   Googled "gazebo remove sensor"
  #   Sensors have their own API:
  #   https://osrf-distributions.s3.amazonaws.com/gazebo/api/dev/group__gazebo__sensors.html
  #     GAZEBO_VISIBLE void 	gazebo::sensors::remove_sensor (const std::string &_sensorName)
 	#     Remove a sensor by name.
  #   Gave error 
  #     Error [SensorManager.cc:354] Unable to remove sensor[default::reflex::Distal_1/sensor_1::finger_1_sensor_6] because it does not exist.
  #
  #   Maybe I should just call DisconnectUpdated(), in destructor of my
  #     contact_sensor_plugin.cpp?
  #   https://osrf-distributions.s3.amazonaws.com/gazebo/api/dev/classgazebo_1_1sensors_1_1ContactSensor.html
  #     void 	DisconnectUpdated (event::ConnectionPtr &_c)
 	#     Disconnect from a the updated signal. More...
  #   Didn't work either
  #
  #   Googled the ContactManager.cc line of error again, got this! This is
  #   what I want to ask!
  #   http://answers.gazebosim.org/question/6369/how-to-delete-a-model-entirely-including-contact-filter/
  #   pull request merged:
  #     https://bitbucket.org/osrf/gazebo/pull-requests/1106/added-world-removemodel-to-fix-issue-1177/diff
  #   Looks like World::RemoveModel() might work better? I've been removing
  #   using rosservice /gazebo/delete_model, which is probably provided by
  #   gazebo_ros, which as I found out before with gravity disabling, is very
  #   buggy and unreliable!!!! Best way to go is with plugins.
  #   So I'll try World::RemoveModel() in my world_plugin.cpp now, instead of
  #   using the gazebo_ros rosservice!!! Gosh what does the gazebo_ros pkg do,
  #   if they aren't using plugins!!!
  #   
 
  #rosnode.kill_nodes (['/detect_reflex_contacts'])

  return new_hand_name


############################################################### Utility fns ##

# Parameters:
#   obj_prefix: basename without extension, e.g. 'sphere_3cm'
def move_files_to_perm_names (recorderNode, obj_prefix, obj_cat,
  obj_params_path, obj_params_name,
  costs_root, costs_name_old, probs_root, probs_name_old):

  print ('Renaming timestamp-named files to permanent object names...')

  # Create the folders if they don't exist yet
  tri_path = os.path.join (recorderNode.tri_path, obj_cat)
  if not os.path.exists (tri_path):
    os.makedirs (tri_path)
  pcd_path = os.path.join (recorderNode.pcd_path, obj_cat)
  if not os.path.exists (pcd_path):
    os.makedirs (pcd_path)

  perm_trih_name = os.path.join (tri_path, obj_prefix + '_hand.csv')
  perm_trir_name = os.path.join (tri_path, obj_prefix + '_robo.csv')
  perm_pcd_name = os.path.join (pcd_path, obj_prefix + '.pcd')

  # Ref: http://stackoverflow.com/questions/8858008/moving-a-file-in-python
  #   If in Python 2.7.3, need to catch error, shutil.move() fails if file
  #   already exists.
  # Ref: http://stackoverflow.com/questions/82831/check-whether-a-file-exists-using-python
  perm_params_name = ''
  if obj_params_path:
    # Create the folder if doesn't exist yet
    params_path = os.path.join (obj_params_path, obj_cat)
    if not os.path.exists (params_path):
      os.makedirs (params_path)

    perm_params_name = os.path.join (params_path, obj_prefix + '.csv')

  else:
    print ('%sWARN: Path for collection parameters file is never set (triangles_sampling_semiauto.py self.obj_params_path). Most likely no files saved. Will not rename the timestamps file to permanent object name!%s' % (ansi_colors.WARNING, ansi_colors.ENDC))


  costs_path = os.path.join (costs_root, obj_cat)
  costs_name = os.path.join (costs_path, obj_prefix + '.pkl')

  probs_path = os.path.join (probs_root, obj_cat)
  probs_name = os.path.join (probs_path, obj_prefix + '.pkl')


  overwrite = False

  # Check if any file already exists
  if os.path.isfile (perm_trih_name) or \
     os.path.isfile (perm_trir_name) or \
     os.path.isfile (perm_pcd_name) or \
     os.path.isfile (perm_params_name) or \
     os.path.isfile (costs_name) or \
     os.path.isfile (probs_name):

    uinput = raw_input ('%sWARN: At least one PERMANENT object file exists, out of these:\n  %s\n  %s\n  %s\n  %s\nIf you want to keep the old files, manually move them elsewhere before overwriting. Overwrite ALL triangles, pcd, params files for this object? (Y/N): %s' % ( \
      ansi_colors.WARNING, perm_trih_name, perm_trir_name, perm_pcd_name, perm_params_name, ansi_colors.ENDC))
   
    if uinput.lower () == 'y':
      overwrite = True
      print ('Will overwrite existing files.')
    else:
      overwrite = False
      print ('Will NOT overwrite. No files renamed.') 
    
  else:
    # No files exist, just write new file
    overwrite = True

  if overwrite:
    shutil.move (recorderNode.trih_name, perm_trih_name)
    shutil.move (recorderNode.trir_name, perm_trir_name)
    shutil.move (recorderNode.pcd_name, perm_pcd_name)
    if obj_params_name:
      shutil.move (obj_params_name, perm_params_name)
    shutil.move (costs_name_old, costs_name)
    shutil.move (probs_name_old, probs_name)

    print ('Timestamp-named files now moved to:\n  %s\n  %s\n  %s\n  %s\n  %s\n  %s\n' % ( \
      perm_trih_name, perm_trir_name, perm_pcd_name, perm_params_name,
      costs_name, probs_name))


# Do nothing, allow files to finish being written
# Ref: docs.python.org/2/library/signal.html
def sigterm_handler (signal_number, stack_frame):
  return


###################################################################### Main ##

def main ():

  # Set disable_signals=True to use my own Ctrl+C signal handler, so that
  #   I can finish writing files to disk!
  #   Note there will be no rospy.exceptions to catch Ctrl+C, need to use
  #     KeyboardInterrupt.
  #   Tested in test_shutdown.py.
  rospy.init_node ('sample_gazebo', anonymous=True, disable_signals=True)

  signal.signal (signal.SIGTERM, sigterm_handler)

  #####
  # Parse cmd line args
  #####

  arg_parser = argparse.ArgumentParser ()

  #arg_parser.add_argument ('--meta', type=str, default='models_active.txt',
  # models_simple.txt is for running on sphere and cube, known simple objects,
  #   to compare PCL and Gazebo data in 1d histogram intersection plots.
  #arg_parser.add_argument ('--meta', type=str, default='models_simple.txt',
  # models_hist_inter.txt is to test a few objects and plot 1d histogram
  #   intersection btw PCL data and a few Gazebo runs with different sampling
  #   parameters, for the same object. 1d hist inters lets you see whether
  #   Gazebo data converges to PCL data, and how dense your sampling params
  #   need to be.
  #arg_parser.add_argument ('--meta', type=str, default='models_hist_inter.txt',
  arg_parser.add_argument ('--meta', type=str, default='models_gazebo_dae.txt',
  # Cubes and spheres, for testing active_touch
  #arg_parser.add_argument ('--meta', type=str, default='models_active_simple.txt',
  #arg_parser.add_argument ('--meta', type=str, default='models_active.txt',
    help='Meta file to read object filenames from')

  # Pick which shape grid to sample in
  arg_parser.add_argument ('--cyl_grid', action='store_true', default=False)
  arg_parser.add_argument ('--ell_grid', action='store_true', default=False)

  # Default to empty string, otherwise triangles_collect_semiauto.py will load
  #   the file.
  arg_parser.add_argument ('--pickup_from_file', type=str, default='',
    help='Prefix of file (timestamp or permanent object name) to append this run to, e.g. cube_3cm, or cup_c635e584. Make sure config file has only ONE object uncommented, else behavior undefined, might concat multiple objects or overwrite the wrong files!')

  # If using cylinder grid
  arg_parser.add_argument ('--pickup_z', type=int, default=0,
    help='z_idx in cylinder grid to pick up at. This should be where program was killed that you are picking up from.')
  arg_parser.add_argument ('--pickup_theta', type=int, default=0,
    help='theta_idx in cylinder grid to pick up at. This should be where program was killed that you are picking up from.')

  # If using ellipsoid grid
  arg_parser.add_argument ('--pickup_ell_idx', type=int, default=0,
    help='index in ellipsoid grid to pick up at. This should be where program was killed that you are picking up from.')

  # Need to do this for argparse to work with roslaunch. parse_args() without
  #   args will error with roslaunch!
  args = arg_parser.parse_args (rospy.myargv () [1:])


  # If true, use elliptic cylinder (currently in triangles_collect_semiauto.py,
  #   but needs to be refactored into a EllipticCylinderSurfacePoints class
  #   later).
  #   Else use ellipsoid (in geo_ellipsoid.py).
  # Default to use ellipsoid grid.
  USE_CYL_GRID = False

  if args.cyl_grid and args.ell_grid:
    print ('%sBoth --cyl_grid and --ell_grid were specified. Must choose one.%s' % \
      (ansi_colors.FAIL, ansi_colors.ENDC))
    return
  if not args.cyl_grid and not args.ell_grid:
    print ('%sNeither --cyl_grid and --ell_grid were specified. Must choose one.%s' % \
      (ansi_colors.FAIL, ansi_colors.ENDC))
    return

  if args.ell_grid:
    USE_CYL_GRID = False
  elif args.cyl_grid:
    USE_CYL_GRID = True

  if USE_CYL_GRID:
    print ('%sSampling grid set to CYLINDER%s' % ( \
      ansi_colors.OKCYAN, ansi_colors.ENDC))
  else:
    print ('%sSampling grid set to ELLIPSOID%s' % ( \
      ansi_colors.OKCYAN, ansi_colors.ENDC))


  meta_base = args.meta


  # Init user adjustable parameters
  cfg = SampleGazeboConfigs (USE_CYL_GRID)


  cfg.pickup_leftoff = False

  cfg.pickup_z = 0
  cfg.pickup_theta = 0

  cfg.pickup_ell_idx = 0

  # Sanity check
  if ((args.pickup_z != 0 or args.pickup_theta != 0) and \
    # Allow user  to issue a ell_idx without a pickup file. `.` sometimes
    #   want to skip the beginning ones where there's no contact, esp for hammer
    #or args.pickup_ell_idx != 0) and \
      (not args.pickup_from_file)):
    print ('%spickup parameters were specified, but --pickup_from_file was not. If you are picking up from before, must specify it. Check your args and rerun. Terminating...%s' % (ansi_colors.FAIL, ansi_colors.ENDC))
    return

  if args.pickup_from_file:
    cfg.pickup_leftoff = True
    cfg.pickup_from_file = args.pickup_from_file
    print ('%sPicking up from before, previous file %s%s' % ( \
      ansi_colors.OKCYAN, args.pickup_from_file, ansi_colors.ENDC))

    if USE_CYL_GRID:
      cfg.pickup_z = args.pickup_z
      cfg.pickup_theta = args.pickup_theta
      print ('%sPicking up cylinder grid from z_idx %d, theta_idx %d%s' % ( \
        ansi_colors.OKCYAN, cfg.pickup_z, cfg.pickup_theta, ansi_colors.ENDC))

    else:
      cfg.pickup_ell_idx = args.pickup_ell_idx
      print ('%sPicking up ellipsoid grid from ell_idx %d%s' % ( \
        ansi_colors.OKCYAN, cfg.pickup_ell_idx, ansi_colors.ENDC))

  # These two are convenience vars used only for cyl_grid_node ctor. Really
  #   should use if (pickup_leftoff and USE_CYL_GRID), but ctor has too many
  #   args
  pickup_cyl_leftoff = False
  if args.pickup_from_file and USE_CYL_GRID:
    pickup_cyl_leftoff = True


  #####
  # Init ROS stuff
  #####

  #tf_listener = tf.TransformListener ()

  vis_arr_pub = rospy.Publisher ('/visualization_marker_array',
    MarkerArray, queue_size=2)


  #####
  # Init class objects
  #####

  # The TrianglesCollect class actually writes things to file!
  # The file by itself can record manually, responding to user input from
  #   tactile_map keyboard_interface.py node.
  # Initialize within while-loop, to enable one recorder per object. Easiest 
  #   way to re-init recorder for each object.
  recorderNode = None


  # wall_range is how far around object to move
  cyl_grid_node = TrianglesCollectSemiAuto (robot_on=False, sim_robot_on=True,
    reflex_on=True,
    recorderNode=recorderNode, do_ceil=False, do_wall=True,
    #wall_range=270.0, zero_tactile_wait_time=0.1)
    # Testing if zero_tactile more would make noise on cylinder grid disappear
    wall_range=360.0, zero_tactile_wait_time=0.5,
    pickup_leftoff=pickup_cyl_leftoff,
    pickup_from_file=cfg.pickup_from_file)

  # Set to 0, because in simulation, there's no table to avoid.
  cyl_grid_node.PALM_UPPER_WIDTH = 0
  cyl_grid_node.PALM_LOWER_WIDTH = 0
  # 0.08 is too close to cup, closed finger gets stuck and can't open again
  cyl_grid_node.PALM_THICKNESS = cfg.CYL_PALM_THICKNESS

  cyl_grid_node.density = cfg.cyl_density


  # If using ellipsoid grid
  ell_node = EllipsoidSurfacePoints ()


  # Probs and costs for active touch

  # pkl files take a long time to save at the end. Set to False if don't really
  #   need the probability files and just want to quickly produce triangle .csv
  #   and .pcd files
  io_probs_node = IOProbs ('_gz', (cfg.obj_x, cfg.obj_y, cfg.obj_z),
    save_debug_abs_poses=True, sequential_costs=False,
    pickup_leftoff=cfg.pickup_leftoff, pickup_from_file=cfg.pickup_from_file,
    discretize_m_q_tri=(0.06, 0.05, 0.08))
    # (2, 3, 2) is too fine. compute_costs_probs_n() takes long time, SIGTERM
    #   gets issued before it's finished. I have no way to bar SIGTERM from
    #   passing control in main() to sigterm_handler(), which can't pick up the
    #   unfinished function call anymore (maybe it can, if I mess with the
    #   stack frame passed to it, but I don't know how, don't have time now to
    #   look it up)
    #discretize_m_q_tri=(2, 3, 2))



  #####
  # Initialize Gazebo scene for pipeline
  #####

  # Unpause physics, so hand can be moved around correctly
  print ('%sUnpausing physics%s' % (ansi_colors.OKCYAN, ansi_colors.ENDC))
  pause_physics (pause=False)
  print ('')


  # Check existing models
  old_hand_name = ''
  old_obj_name = ''
  success, world_props = get_world_properties ()
  if success:
    for name in world_props.model_names:
      if name.startswith ('reflex'):
        old_hand_name = name
      if name.startswith ('object'):
        old_obj_name = name


  hand_name = 'reflex'

  # TODO: world_plugin.cpp is not ready to handle this... Need to pass in
  #   hand name to some of the services that deal with the hand. Right now
  #   world_plugin.cpp just looks for "reflex" model, which isn't there when
  #   we use a random number suffix!
  # Generate a random number suffix, as a workaround for the "not removing
  #   cleanly" problem of Gazebo. If keep same name before and after removal
  #   of a model, it cannot be reloaded - it will be invisible, even
  #   transport:: publishing msgs::Visual messages don't work!
  #hand_name = 'reflex' + str (np.random.randint (0, 10000))

  # Spawning in roslaunch file for now
  # Load ReFlex Hand simulator
  print ('%sSpawning hand%s' % (ansi_colors.OKCYAN, ansi_colors.ENDC))
  # If no hand is loaded yet, load one.
  if not old_hand_name:
    hand_name = spawn_hand (hand_name=hand_name)
  # TESTING!! TEMPORARY
  # Still bad. Gazebo stops responding when I delete hand now. I don't know why.
  #   With random number name, the new hand is displayed, but then Gazebo
  #   freezes.
  '''
  if not old_hand_name:
    print ('%sWARN: Did not find existing model name starting with reflex. Will assume old hand to remove has name reflex. If this is incorrect, you will end up with 2 hands, watch out and delete the old one manually in Gazebo GUI.%s' % ( \
      ansi_colors.WARNING, ansi_colors.ENDC))
    old_hand_name = 'reflex'
  hand_name = reload_hand (old_hand_name=old_hand_name, new_hand_name=hand_name)
  '''
  print ('')


  # Remove any existing object
  remove_model (old_obj_name)


  # Load object names from meta file
  in_dae_names, catids, catnames = load_meta_file (cfg.pkg_path, meta_base)

  if not in_dae_names:
    rospy.logerr ("No lines in meta list file. Did you uncomment any lines?")
    return

  print ('Categories:')
  print (catnames)
  print ('Corresponding category IDs:')
  print (catids)
  print ()


  # Read rescale config file for 3DNet data
  rescale_map = load_rescale_config (cfg.pkg_path)



  #####
  # Main loop
  #####

  obj_idx = 0

  HERTZ = 10.0
  wait_rate = rospy.Rate (HERTZ)

  # Seconds
  start_time = time.time ()

  ended_unexpected = False

  # Loop through each object, collect triangles data
  while not rospy.is_shutdown ():

    # Re-init recorderNode, to clear all data, without having to worry
    #   about clearing vars individually
    recorderNode = TrianglesCollect (csv_suffix='gz_',
      pickup_from_file=cfg.pickup_from_file,
      sample_robot_frame=cfg.sample_robot_frame)
    cyl_grid_node.recorderNode = recorderNode

    # Pass in the pointer of the nodes, so changes in instance are changed in
    #   main() too
    # TODO: Should really refactor cyl_grid_node's file, to separate the
    #   sampling from the cylinder grid. I'm not going to be using cylinder grid
    #   on real robot after putting in collision detection anyway. This is a
    #   great time to do it `.` I'm not using the old pipeline on real robot
    #   anymore!
    thisNode = SampleGazeboCore (cfg, recorderNode, cyl_grid_node,
      io_probs_node, hand_name)

    # Move hand sufficiently far from object, so they don't collide
    print ('%sMoving hand away from object birth place%s' % (ansi_colors.OKCYAN,
      ansi_colors.ENDC))
    thisNode.teleport_hand (hand_name, *cfg.SAFE_PLACE)
    print ('')


    try:

      if obj_idx >= len (in_dae_names):
        break


      obj_catid = catids [obj_idx]
      obj_cat = catnames [obj_catid]
      obj_path = in_dae_names [obj_idx]

      model_name, _ = load_object_into_scene (obj_path,
        obj_cat, rescale_map, cfg, thisNode.vis_pub)
      thisNode.set_obj_info_for_log (obj_cat, obj_path)

      # Get object bbox size
      #   Center is wrt Gazebo world frame, which we publish /base at.
      success = False
      while not success:
        (success, model_size, model_center) = get_model_size (model_name)
      if success:

        # TODO: Draw in rviz to see which dimension the sizes correspond with,
        #   `.` I rotate the model 90 wrt X, so I don't know which dim is
        #   which
        # cup_2b46c83c.dae reference correct dimensions in world:
        #   0.08 (diameter w/o handle), 0.1 (diameter with handle), 0.04 (height)
        print ('Got model size: %f %f %f, model center: %f %f %f' % ( \
          model_size.x, model_size.y, model_size.z,
          model_center.x, model_center.y, model_center.z))

        if cfg.USE_CYL_GRID:
         
          # Set parameters
          #cyl_grid_node.obj_radius = np.max ([model_size_aug[0],
          #  model_size_aug[1]]) #0.06
          #cyl_grid_node.obj_height = model_size_aug[2] #0.1
         
          # Circular cylinder
          # Good for some objs
          #cyl_grid_node.obj_radius = np.max ([model_size.x,
          #  model_size.y])
          # Good for highly elliptical objs
          #cyl_grid_node.obj_radius = np.min ([model_size.x,
          #  model_size.y])
         
          # Elliptical cylinder
          cyl_grid_node.obj_radius_x = model_size.x
          cyl_grid_node.obj_radius_y = model_size.y
         
          # This way doesn't work, because the hand's position ends up way high,
          #   almost not touching object anymore. So instead of doing this max,
          #   reduce PALM_LOWER_WIDTH and PALM_UPPER_WIDTH to 0 for sim.
          # Simulation palm height doesn't matter, because there's no table
          #   below to collide with hand. So set a min object height that's at
          #   least PALM_LOWER_WIDTH + PALM_UPPER_WIDTH, so we get at least one
          #   wall height (z)! Otherwise cyl_grid_node.wall_heights[] will be
          #   empty, no grid points to touch!
          #cyl_grid_node.obj_height = np.max ([model_size.z,
          #  cyl_grid_node.PALM_LOWER_WIDTH + cyl_grid_node.PALM_UPPER_WIDTH])
          cyl_grid_node.obj_height = model_size.z
         
          # Account for when actual geometry of model is not centered at origin
          #   of model file. Then add the actual geometry bbox center to the
          #   object center where I load the file (i.e. its origin) in Gazebo.
          # model_center is in world frame, which in Gazebo sim is same as
          #   fake robot frame '/base', which I create.
          # cyl_grid_node.obj_center needs to be wrt robot frame /base, which in
          #   this case is just same as Gazebo world origin.
          # Therefore I can just set them equal, don't need to += a specific
          #   offset or anything.
          cyl_grid_node.obj_center [0] = model_center.x
          cyl_grid_node.obj_center [1] = model_center.y
          cyl_grid_node.obj_center [2] = model_center.z
         
          # Decrement object center by half the height, because
          #   triangles_collect_semiauto.py assumes obj_center is at bottom 
          #   center of object, not true center. (But we needed obj_center to
          #   be true center above, because SDF puts origin of DAE at where we
          #   specify, and DAE *PROBABLY* has origin at object center.
          # TODO: Check that all DAE files I'm using have their center at obj
          #   origin, otherwise hand's position wrt object won't be right. You
          #   can try using ModelPlugin to get center of model, but I don't know
          #   whether that gives the mesh center, or the center of the DAE file,
          #   which might not be at center of mesh!
          # > Shouldn't need this anymore. Now I use model_center above,
          #   returned from my rosservice /reflex_gazebo/get_model_size! I use
          #   model_center to set cyl_grid_node.obj_center, so it doesn't
          #   matter where the actual geometry is in the model file, I'll
          #   always have the correct object geometry center, around wich to
          #   place the hand!
          cyl_grid_node.obj_center [2] -= (model_size.z * 0.5)
                 
          print ('obj_center: %f %f %f' % (cyl_grid_node.obj_center[0],
            cyl_grid_node.obj_center[1], cyl_grid_node.obj_center[2]))

        else:
          # Use deg_step=60 for easier debugging, fewer arrows to track.
          # rings_along_dir='h' gives me horizontal circular slices.
          #   alternate_order=False makes horizontal circles rotate in
          #   consistent direction.
          #   (If use rings_along_dir='v', then must use alternate_order=True)
          ell_node.initialize_ellipsoid ( \
            (model_center.x, model_center.y, model_center.z),
            (model_size.x + cfg.ELL_PALM_THICKNESS,
             model_size.y + cfg.ELL_PALM_THICKNESS,
             model_size.z + cfg.ELL_PALM_THICKNESS),
            deg_step=cfg.ell_deg_step,
            alternate_order=False, rings_along_dir='h')
            #alternate_order=True, rings_along_dir='v')

          print ('Initialized ellipsoid sampling grid with center (%g %g %g), radii (%g %g %g)' % \
            (ell_node.cx, ell_node.cy, ell_node.cz,
             ell_node.rx, ell_node.ry, ell_node.rz))

          # Visualize ellipsoid, for debugging in RViz
          ell_marker_arr = ell_node.visualize_ellipsoid ('/base',
            vis_quat=False, vis_idx=False, vis_z_frames=True,
            extra_rot_if_iden=cfg.ell_extra_rot_base)
          vis_arr_pub.publish (ell_marker_arr)
          # Wait for RViz Marker msg to propagate
          time.sleep (0.5)
          #rospy.sleep (0.5)

        # end if USE_CYL_GRID

 
      # Skip this object
      else:
        print ('Could not find bounding box size of model. Skipping this object!! (You might want to look into why and fix this.)' % \
          (ansi_colors.FAIL, ansi_colors.ENDC))
        obj_idx += 1
        time.sleep (1.0 / HERTZ)
        #rospy.sleep (1.0 / HERTZ)
        continue


      ################################# Init vars for training active touch ##

      #####
      # Initialize vars to store training data for active learning
      #####

      if cfg.RECORD_PROBS:
        # Reset vars for each object
        io_probs_node.reset_vars ()


      ############################################# Move hand around object ##
 
      #####
      # Sample object by moving hand in a elliptical cylinder grid
      #####
 
      print ('%sMoving hand around%s' % (ansi_colors.OKCYAN, ansi_colors.ENDC))
 
      # Reconfig cylinder grid space
      cyl_grid_node.reconfig ()
 
      # If picking up from where program left off before termination last time
      if cfg.pickup_from_file:
        if cfg.USE_CYL_GRID:
          cyl_grid_node.pickup_leftoff = True
          cyl_grid_node.z_idx = cfg.pickup_z
          cyl_grid_node.z_idx_dry = cyl_grid_node.z_idx
          cyl_grid_node.theta_idx = cfg.pickup_theta
          cyl_grid_node.theta_idx_dry = cyl_grid_node.theta_idx
        else:
          ell_node.set_next_idx (cfg.pickup_ell_idx)


      if not cfg.USE_CYL_GRID:

        # n x 7 NumPy array. Wrist poses
        ell_poses = np.zeros ([0, 7])

        while True:
          # ell_q rotation is an axis-angle rotation, btw the world z-axis and
          #   the goal hand z-axis. The axis is cross prod of the two. Angle
          #   is the angle btw them. This effectively rotates hand in all 360
          #   degs (at least in rings_in_dir='h' horizontal slices mode, it
          #   goes all 360 degs) as it goes around the object. This covers a
          #   lot more variety than keeping hand oriented such that fingers are
          #   parallel to the ground!!! So I'm keeping this.
          # NOTE if ell_q is ever changed in get_next_point(), then code it
          #   here. Use get_relative_rotation_v, use change-of-basis method,
          #   which will fix n, the cross prod btw the 2 input vecs, and rotate
          #   the angle btw the 2 vecs.
          #   http://math.stackexchange.com/questions/1246679/expression-of-rotation-matrix-from-two-vectors?rq=1
          ell_pt, _, ell_q, _, _ = ell_node.get_next_point ( \
            quat_wrt=(0,0,1), extra_rot_if_iden=cfg.ell_extra_rot_base)
       
          # geo_ellipsoid.py only returns None when there's no more points to
          #   return, meaning we're done with whole ellipsoid.
          if ell_pt is None:
            break
       
          # Append 3-vector and 4-vector into 1 x 7 vector. Append to n x 7 mat
          ell_poses = np.append (ell_poses, [np.append (ell_pt, ell_q)],
            axis=0)


      # Execute actions, move hand in Gazebo to sample triangles
      # NOTE: Not implemented for USE_CYL_GRID, since I only use ellipsoid now
      thisNode.set_goal_poses (ell_poses)
      thisNode.execute_poses (cfg.pickup_ell_idx)


      #####
      # Do n*n calculation for pairwise movement cost and observation
      #   probability. Write probs and costs data to file.
      #####

      if cfg.RECORD_PROBS:

        print ('Recording n x n probabilities data for entire object, this may take minutes...')

        # Use recorderNode.timestring as a time-saving condition, `.` if we
        #   aren't saving files, then we don't need to waste time computing
        #   the probabilities in RECORD_PROBS (it can take many minutes in
        #   those two sets of n*n for-loops!)
        if recorderNode.timestring:
 
          #catid = catids [obj_idx]
          io_probs_node.compute_costs_probs (obj_cat, #obj_catid,
            obj_path,
            np.array ([model_center.x, model_center.y, model_center.z]))

          io_probs_node.write_costs_probs ()
 
        else:
          print ('%sError: No recorderNode.timestring. No costs and probs file saved. Did you run rosrun instead of roslaunch for sample_gazebo.py?%s' % (ansi_colors.FAIL, ansi_colors.ENDC))
       
        # end if recorderNode.timestring
      # end if RECORD_PROBS


      # Wrap up this object

      if not cfg.USE_TIMESTAMP_FILENAMES:
        # Not tested with RECORD_PROBS=True!!
        move_files_to_perm_names (recorderNode, obj_prefix, obj_cat,
          cyl_grid_node.obj_params_path, cyl_grid_node.obj_params_name,
          io_probs_node.get_costs_root(), io_probs_node.get_costs_name(),
          io_probs_node.get_probs_root(), io_probs_node.get_probs_name())

      # Reset, so if ros shuts down before next object gets contacts, we don't
      #   write again to these files, which can overwrite previous completed
      #   files with empty files!
      recorderNode.timestring = ''

      # One object ends
      print ('')

     
      obj_idx += 1
 
      # If this doesn't get awaken, it's because your Gazebo physics is paused.
      #   Just unpause, and this will return from sleep.
      #   Ref: http://answers.ros.org/question/11761/rospysleep-doesnt-get-awaken/
      #wait_rate.sleep ()
      time.sleep (1.0 / HERTZ)
      #rospy.sleep (1.0 / HERTZ)

    # end try

    #except rospy.exceptions.ROSInterruptException, err:
    # Using my custom signal handler, won't have rospy.exceptions error
    except KeyboardInterrupt:
      ended_unexpected = True
      #pass
      break

  # End while rospy.is_shutdown() loop


  # Custom signal handler, to save all files. Must put it in a separate fn
  #   than main(), `.` SIGTERM gets sent to main() and still kills files in
  #   the middle of being saved!
  sigint_handler (start_time, ended_unexpected,
    thisNode, cyl_grid_node, recorderNode, io_probs_node,
    cfg, obj_cat, obj_path, model_center, model_name)

  return


#   in_dae_name: Full path to .dae file. But we only use the basename without
#     extension.
def sigint_handler (start_time, ended_unexpected,
  thisNode, cyl_grid_node, recorderNode, io_probs_node,
  cfg, obj_cat, in_dae_name, model_center, model_name):

  # Copied from triangles_reader.py
  # Print out running time
  # Seconds
  end_time = time.time ()
  elapsed_time = end_time - start_time
  print ('Total time this run: %f seconds.' % (elapsed_time))

  obj_basename = os.path.basename (in_dae_name)
  obj_prefix = os.path.splitext (obj_basename)


  #####
  # Save all files, before calling ros signal_shutdown manually
  #####

  # If collection sequence ended unexpectedly, still save what we have
  #   so far
  if ended_unexpected:

    uinput = raw_input ('%sPress any key to continue saving files before shutdown, press Q to shutdown immediately and discard unsaved data (you may lose files not finished being written!): %s' % (
      ansi_colors.WARNING, ansi_colors.ENDC))

    # If user wants to discard all unsaved data
    if uinput.lower () == 'q':
      # Send manual shutdown signal to ROS. This immediately shuts down,
      #   anything after this line is not guaranteed to be executed.
      rospy.signal_shutdown ('normal shutdown by user command, without finishing saving files')

    else:
      cyl_grid_node.write_obj_params ()
 
      if recorderNode and recorderNode.timestring:
        # Close output files
        recorderNode.doFinalWrite = True
        # Write final PCD file, with header
        recorderNode.collect_one_contact ()
 
        # Record log
        if cfg.do_log_this_run:
          cfg.log_this_run (
            obj_cat, obj_basename, cfg.PALM_THICKNESS,
            recorderNode.timestring, cfg.pickup_from_file,
            cfg.pickup_ell_idx, cfg.pickup_z, cfg.pickup_theta,
            recorderNode.nPts_ttl, recorderNode.nTris_h_ttl, elapsed_time)
 
        if cfg.RECORD_PROBS:
          io_probs_node.compute_costs_probs (obj_cat,
            in_dae_name,
            np.array ([model_center.x, model_center.y, model_center.z]))
          io_probs_node.write_costs_probs ()
 
  #end if ended_unexpected
  print ('')
 
 
  # Move pcd and triangle .csv files written by recorderNode, which are named
  #   by timestamp, to permanent file locations.
  if recorderNode.timestring:
 
    if not cfg.USE_TIMESTAMP_FILENAMES:
      # Not tested with RECORD_PROBS=True!!
      move_files_to_perm_names (recorderNode, obj_prefix, obj_cat,
        cyl_grid_node.obj_params_path, cyl_grid_node.obj_params_name,
        io_probs_node.get_costs_root(), io_probs_node.get_costs_name(),
        io_probs_node.get_probs_root(), io_probs_node.get_probs_name())
 
    # Print time string for easier copy-pasting to start next pickup run
    print ('%sTimestring for this run: %s%s' % ( \
      ansi_colors.OKCYAN, recorderNode.timestring, ansi_colors.ENDC))
 
  # For convenience of restarting, if want to pick up from this run later
  lastSeenIdx = thisNode.lastSeenIdx
  # -1 indicates all goal points have finished
  if lastSeenIdx != -1:
    lastSeenIdx += cfg.pickup_ell_idx
  if not cfg.USE_CYL_GRID:
    print ('%sLast seen ell_idx: [%d]%s' % ( \
      #ansi_colors.OKCYAN, ell_node.get_next_idx () - 1, ansi_colors.ENDC))
      ansi_colors.OKCYAN, lastSeenIdx,
      ansi_colors.ENDC))
 

  # Remove object from world
  print ('%sRemoving object from world%s' % (ansi_colors.OKCYAN, ansi_colors.ENDC))
  remove_model (model_name)
  print ('')
 
  # Don't remove hand from world, because that disables the controllers. If
  #   you remove hand, then you also have to reload reflex.launch!!!
  # Remove hand from world
  #print ('%sRemoving hand from world%s' % (ansi_colors.OKCYAN, ansi_colors.ENDC))
  #remove_model (hand_name)
  #print ('')
 
 
  # Manually tell ROS to cleanup properly
  # This line is required when using custom Ctrl+C signal handler (via
  #   disable_signals=True to init_node())
  rospy.signal_shutdown ('normal shutdown by user command')


if __name__ == '__main__':
  main ()

