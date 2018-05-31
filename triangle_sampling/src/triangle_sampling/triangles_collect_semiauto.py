#!/usr/bin/env python

# Mabel Zhang
# 4 Sep 2015
#
# Semi-automatic collection of triangles tactile data on ReFlex + Baxter.
#
# This file, triangles_collect_semiauto.py, is the semi-automated version
#   of triangles_collect.py.
# This file is responsible for the IK movement of robot ONLY.
#   It calls triangles_collect.py to do the actual calculations and
#   recording of data to file on disk.
#
# Pass in --gauge 1 to enable interactive gauging for object center, before
#   starting collection.
#
# When visualize in RViz:
# Turn on tf frame visualization for /base and /left_gripper frames (do NOT
#   turn on /base_link, it will confuse you! IK goals are for /left_gripper
#   frame ONLY, by Baxter IK API).
#
# When on robot:
# START THE HAND ABOVE OBJECT, so that it doesn't knock object over when moving
#   to be above object using IK.
#
#
# Wrist moves around a cylinder larger than object for gripper clearance.
#   Hand closes and opens to sample the object, in grid points on the wrist
#   movement space cylinder, and in various orientations.
#
# Sampling is done from above object, and from around object walls.
#   From above, wrist rotates 360 degs.
#   From side walls, wrist has only one orientation per grid point, for
#     now.
#
#
# Adjustable parameters:
# In addition to the ones you can see by running this script with flag -h:
#   wall_end_angle: this defines how much of the cylinder you move to.
#     pi/2 to move 90 degs around obj, 2*pi to move 360 degs all around.
#   wall_start_angle: this defines where wrt object you move to. It's defined
#     wrt whatever frame you use self.wall_angles in.
#     If used in robot frame, which has x front, y left, z up, setting this
#     angle to 0 means you start at the side of object facing away from robot,
#     i.e. backside of object. Setting this to 90 puts it on y-axis, which
#     means you start at left of object.
#

# ROS
import rospy
import tf
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point, Quaternion
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty
from std_msgs.msg import String, Bool

# Python
import sys
import argparse
import csv, os
import time
from copy import deepcopy

# NumPy
import numpy as np

# Baxter packages
from baxter_core_msgs.srv import SolvePositionIKRequest
import baxter_interface
from baxter_core_msgs.msg import EndpointState

# My package
from baxter_reflex.reflex_control import call_smart_commands
from tactile_map.spin_seam import spin_cloud
from tactile_map.create_marker import create_marker
from util.ansi_colors import ansi_colors
from baxter_reflex.baxter_control import enable_robot, disable_robot, \
  move_arm, solve_ik
from tactile_map_msgs.msg import Contacts
from tactile_collect import tactile_config
from baxter_reflex.wrist_hand_transforms import *
from triangle_sampling.config_paths import get_robot_obj_params_path
from triangle_sampling.triangles_collect import TrianglesCollect

# Local
from gauge_object_center import gauge_obj_center


class TrianglesCollectSemiAuto:

  # This function only defines CONSTANTS. Put all non-constants in one of
  #   config*() or reconfig().
  # Parameters:
  #   obj_r, obj_h: Object radius and height. Unit: meters
  #   density: sampling density, i.e. how far apart are grid points on the
  #     sampling space cylinder. Grid points are goal poses for wrist to move
  #     to, not points on object.
  #   robot_on: Whether a robot connection is up, whether in real world or
  #     simulation. This decides whether we try to call enable_robot, which
  #     exits whole program (!! Annoying!) if it doesn't find a connection
  #     to robot!
  #   reflex_on: Whether ReFlex connection is up.
  #   recorderNode: Must be specified in order to record any contact
  #     data to disk! If not specified, will just move robot, but nothing
  #     will be recorded!!
  def __init__ (self, obj_c=[0.0, 0.0, 0.0], obj_rx=0.05, obj_ry=0.05,
    obj_h=0.15, density=0.03,
    robot_on=False, sim_robot_on=False, reflex_on=False, recorderNode=None,
    do_ceil=False, do_wall=False, wall_range=30.0, zero_tactile_wait_time=2,
    pickup_leftoff=False, pickup_from_file=''):

    #####
    # Wrist space configuration
    #####

    self.configured = False

    # For quicker debugging of the walls - just skip the ceiling, except the
    #   very first move, which is identical to the last move on the ceiling
    #   after 2*pi rotation on ceiling.
    self.skip_ceil_all = not do_ceil
    self.skip_ceil_all_but_first = not do_ceil
    self.skip_wall = not do_wall
    # Enforce if robot_on, meaning on real robot.
    # In Gazebo sim, since we aren't running with Baxter, it's okay there's no
    #   collision-aware planning. Hand isn't attached to Baxter anyway.
    if do_wall and do_ceil and robot_on:
      print ('Both ceil=1 and wall=1 were specified. You must pick one, because of the current lack of collision detection. Terminating...')
      self.doTerminate = True
      return

    # At bottom cener of object
    self.obj_center = np.array (obj_c)

    # Default 5 cm
    self.obj_radius_x = obj_rx
    self.obj_radius_y = obj_ry
    # Default 15 cm
    self.obj_height = obj_h

    # Caller can specify True, and set z_idx and theta_idx from caller, after
    #   calling reconfig() from caller. Then collect() will pick up this flag,
    #   z_idx, and theta_idx, and start from there.
    # Caller can set this flag to True after calling reconfig() too.
    self.pickup_leftoff = pickup_leftoff
    self.pickup_from_file = pickup_from_file


    # Thickness of palm. This should be measured from the IR camera on wrist
    self.PALM_THICKNESS = 0.11
    # Upper width of palm, this is on the side of the wrist that doesn't have
    #   IR sensor - this side has the hand mounted. Because hand must be
    #   mounted off-centered to avoid the camera, center of wrist is not
    #   center of palm. Therefore upper and lower widths from /left_gripper
    #   are not the same.
    # This needs to be subtracted from Z, when on wall (but not when on
    #   ceiling!), so that the hand is not too high above object, that its 2
    #   fingers grasp nothing.
    # Best way to do this is to take this offset into account when defining
    #   self.wall_heights.
    # Upper width: 6 cm from /left_gripper to top of finger base, 7 cm to top
    #   of soft palm rubber.
    self.PALM_UPPER_WIDTH = 0.06
    # Lower width of palm is from /left_gripper to bottom of soft palm rubber,
    #   not to the bottom of IR camera mount.
    # If you mount object on the corner of the stool, then hand should be able
    #   to move very close to object, without IR camera hitting the stool.
    #   If you can't do that, then you should use the width all the way to
    #   bottom of IR camera mount.
    # Lower width: 4 cm from /left_gripper to bottom of soft palm rubber,
    #   8 cm to bottom of IR sensor mount.
    self.PALM_LOWER_WIDTH = 0.04

    # How many degrees to sample around the object
    self.wall_range = wall_range

    # Sampling density, 2 cm default
    self.density = density


    #####
    # ROS
    #####

    self.bc = tf.TransformBroadcaster ()

    self.robot_frame_id = '/base'
    self.wrist_frame_id = '/base_link'
    self.obj_frame_id = '/obj'
    self.goal_frame_id = '/grid_goal'

    self.wrist_ceil_pub = rospy.Publisher ('/wrist_ceil', PointCloud,
      queue_size=2)
    self.wrist_ceil_grid_pts = None

    self.wrist_wall_pub = rospy.Publisher ('/wrist_wall', PointCloud,
      queue_size=2)
    # This will be populated by dry run. Don't put in reconfig(), that'll erase
    #   the grid points we got in dry run!
    self.wrist_wall_grid_pts = PointCloud ()
    self.wrist_wall_grid_pts.header.frame_id = self.robot_frame_id

    self.vis_pub = rospy.Publisher ('/visualization_marker', Marker,
      queue_size=5)

    # Recorder needs this
    # Threshold for this msg is in detect_reflex_contacts.py . Contact is
    #   only published if a pressure is greater than the threshold set
    #   in that node.
    rospy.Subscriber ('/tactile_map/detect_reflex_contacts/contacts',
      Contacts, self.contactsCB)

    self.reflex_on_pub = rospy.Publisher ('/detect_reflex_contacts/hand_on',
      Bool, queue_size=2)
    self.reflex_on_msg = Bool ()
    self.reflex_on_msg.data = reflex_on

    self.robot_on_pub = rospy.Publisher ('/detect_reflex_contacts/robot_on',
      Bool, queue_size=2)
    self.robot_on_msg = Bool ()
    # If sim_robot_on, publish robot_on to detect_reflex_contacts.py, so that
    #   it still looks for reference frame wrt /base.
    self.robot_on_msg.data = (robot_on or sim_robot_on)

    # Prompt to display at keyboard_interface.py prompt
    self.prompt_pub = rospy.Publisher ('keyboard_interface/prompt', String,
      queue_size=5)
    self.prompt_msg_default = 'Press p to pause, r to resume, '
    self.prompt_msg = String ()
    self.prompt_msg.data = self.prompt_msg_default

    rospy.Subscriber ('/keyboard_interface/key', String, self.keyCB)


    #####
    # Baxter
    #####

    self.robot_state = None
    self.robot_on = robot_on

    # Just to access SEED_CURRENT
    self.ikreq = SolvePositionIKRequest ()

    # Don't need this! SEED_CURRENT does this!
    # Ref: http://sdk.rethinkrobotics.com/wiki/API_Reference#Arm_Joints
    #rospy.Subscriber ('/robot/joint_states', JointState, self.jointStateCB)

    # Don't need this. Seed is JointState type, not Pose.
    # Endpoints are <side>_gripper tf frames.
    #   Ref http://sdk.rethinkrobotics.com/wiki/API_Reference#Cartesian_Endpoint
    # Always left side, `.` ReFlex hand is only on left
    self.arm_side = 'left'
    rospy.Subscriber ('/robot/limb/' + self.arm_side + '/endpoint_state',
      EndpointState, self.endpointCB)


    #####
    # ReFlex Hand and tactile recording
    #####

    self.reflex_on = reflex_on
    self.zero_tactile = rospy.ServiceProxy('/zero_tactile', Empty)
    self.recorderNode = recorderNode
    self.ZERO_TACTILE_WAIT_TIME = zero_tactile_wait_time

    self.UNKNOWN_PRESHAPE = -1
    self.CYL = 0
    self.SPH = 1

    self.obj_params_path = ''
    self.obj_params_name = ''



  # Configure everything in here, `.` after dry run, caller should call this
  #   fn to reconfigure everything.
  # Don't put anything in __init__(), other than the constants. `.` don't want
  #   dry run to change some member vars, and forget to change it back for real
  #   run.
  # So put all non-constant vars here, so that it's easier. Caller can just
  #   do dry run, then call this fn to reset all vars for fresh.
  def reconfig (self):

    # Enable robot if not enabled
    if self.robot_on and self.robot_state is None:
      (self.robot_state, self.robot_init_enabled) = enable_robot ()


    # Vertical dimension of sampling space
    # Always from small to big, i.e. bottom up.
    # When want to change order to top down, make a copy and reverse list order
    self.wall_heights = []

    # Horizontal dimension of sampling space, all around a cylinder
    # Angle in radians. 0 is left of object, count counter clock wise,
    #   i.e. left - front - right - behind.
    self.wall_angles = []

    self.ceil_angles = []


    #####
    # Data collection
    # Start from left of object, go counter-clockwise. I suspect we may not
    #   be able to move to the right of object for some. Keep a minimum range
    #   of pi / 2, i.e. 90 degrees, from left to front.
    #####

    self.pauseCollect = False

    # Init to True so the first move can ever get started!
    self.prev_move_success = True

    # Don't have time to figure this out
    # TODO use these, instead of checking (not self.goal_pose), for 1st
    #   ceiling pt.
    #self.first_ceil_pt = True
    #self.first_wall_pt = False

    self.ceil_done = False
    self.wall_done = False
    self.doTerminate = False

    self.obj_params_written = False

    # Every gripper closure counts as a move, even if wrist position didn't
    #   change.
    self.nMoves = 0


    # Order of traversal along the height of cylinder space. Pose a priori
    #   upright, so height = z.
    self.TOP_DOWN = 0
    self.BOTTOM_UP = 1
    # Init at BOTTOM_UP, so that the first round switches it to TOP_DOWN
    self.z_order = self.TOP_DOWN

    self.theta_idx = 0
    # Init to -1, so first iter will inc to 0
    self.z_idx = 0

    # Same vars for dry run
    self.theta_idx_dry = 0
    # Init to -1, so first iter will inc to 0
    self.z_idx_dry = 0

    self.x = 0
    self.y = 0
    self.z_vec = []


    # Pose wrt robot frame, `.` IK requires poses to be in robot frame
    # geometry_msgs/PoseStamped
    self.goal_pose = None
    self.preshape = self.UNKNOWN_PRESHAPE

    #self.endpoint = None
    #self.joint_state = None

    self.config_wrist_space ()
    #self.config_wrist_quaternions ()

    # Make visualization object that won't change during program duration
    self.config_constant_visualization ()


  # Defines grid point positions for wrist
  def config_wrist_space (self):

    #####
    # Define wrist space dimensions
    #   These are scalar floats, wrt no frame. i.e. wrt the cylinder itself.
    #####

    # Movement above object. Radii used only for plotting RViz Marker
    self.wrist_ceil_rx = self.obj_radius_x
    self.wrist_ceil_ry = self.obj_radius_y
    self.wrist_ceil_h = self.PALM_THICKNESS

    # Movement around cylinder wall
    self.wrist_wall_rx = self.obj_radius_x + self.PALM_THICKNESS
    self.wrist_wall_ry = self.obj_radius_y + self.PALM_THICKNESS
    self.wrist_wall_h = self.obj_height


    #####
    # Define wrist space cylinder ceiling
    #   wrt robot frame
    #####

    # Assume upright pose
    # Bottom center of object + height of wrist space
    self.ceil_pt = Point (self.obj_center[0], self.obj_center[1],
      self.obj_center[2] + self.obj_height + self.wrist_ceil_h)
    
    print ('obj_center z: %f' % (self.obj_center[2]))
    print ('obj_height: %f' % (self.obj_height))
    print ('ceil_pt z: %f' % (self.ceil_pt.z))

    self.wrist_ceil_grid_pts = PointCloud ()
    self.wrist_ceil_grid_pts.header.frame_id = self.robot_frame_id
    self.wrist_ceil_grid_pts.header.stamp = rospy.Time.now ()
    self.wrist_ceil_grid_pts.points.append (self.ceil_pt)


    #####
    # Define wrist space cylinder walls
    #####

    # arange() lets you specify the period in a step. Max is excluded.
    # linspace() lets you specify how many items you want. Max is included.
    # Do NOT use linspace, `.` it doesn't let you specify the sampling density!
    #   arange does.


    # Heights, in object frame

    # TODO: After deadline, make a sanity check that end_height > start_height,
    #   in case object is so small that PALM_LOWER_WIDTH and PALM_UPPER_WIDTH
    #   makes the sample room negative!

    # Take palm's widths into consideration, so it does not move below object
    #   (and hit table, in the case object is upright), and does not move
    #   way above object that the finger closes all the way and get contact
    #   points on the palm.
    wall_start_height = 0 + self.PALM_LOWER_WIDTH
    wall_end_height = self.obj_height - self.PALM_UPPER_WIDTH
    print ('wall_start_height %f, wall_end_height %f' % (wall_start_height,
      wall_end_height))

    self.wall_heights = np.arange (wall_start_height, wall_end_height,
      self.density)
    print ('wall_heights[]:')
    print (self.wall_heights)

    # Makes sure last elt is object height
    # Add a floating point guard because Python sucks at that
    if np.abs (self.wall_heights [len (self.wall_heights) - 1] -
      wall_end_height) > 1e-6:

      self.wall_heights = np.append (self.wall_heights, wall_end_height)


    # Angles of the columns on cylinder

    # Don't just use 2*pi divided by something. `.` we want to make sure we
    #   sample at density-length on cylinder arch. Simply dividing would just
    #   give you some arc length, but not the one you want. It'll give you
    #   sth that's evenly divided among 2*pi. But I want an angle that gives me
    #   the exact arc length for most of the cylinder, except for the last
    #   column, which can be narrower. This is how I ensure the cylinder
    #   surface is sampled the same density in both radius and height
    #   directions.

    # Calculate circumference
    # If circle
    if self.wrist_wall_rx == self.wrist_wall_ry:
      circ = 2 * np.pi * self.wrist_wall_rx
    # If ellipse, can only approximate
    # Ref Approximation 3 here:
    #   https://www.mathsisfun.com/geometry/ellipse-perimeter.html
    else:
      h = (self.wrist_wall_rx - self.wrist_wall_ry) ** 2 / ( \
           self.wrist_wall_rx + self.wrist_wall_ry) ** 2
      circ = np.pi * (self.wrist_wall_rx + self.wrist_wall_ry) * \
        (1 + (3 * h) / (10 + np.sqrt (4 - 3 * h)))

    # These two ratios are equal:
    #   circumference / arc_length = 2 pi / angle_step
    # arc length == density
    # angle_step is the only unknown.
    # For circular cylinder, can simplify eqn above:
    #   2 pi r / arc_length = 2 pi / angle_step
    #   r / arc length = 1 / angle_step
    #   arc_length / r = angle_step
    # For elliptical cylinder, cannot simplify, `.` circumference needs to be
    #   approximated:
    #   2 pi / (circumference / arc_length) = angle_step
    #   2 pi / circumference * arc_length = angle_step
    if self.wrist_wall_rx == self.wrist_wall_ry:
      self.angle_step = self.density / self.wrist_wall_rx
    else:
      self.angle_step = 2 * np.pi / circ * self.density
    print ('%sAngle step: %f%s' % (ansi_colors.OKCYAN, self.angle_step,
      ansi_colors.ENDC))

    # Angle for wrist to rotate in-place on top of object
    # Get max and min of joint limits using:
    #   $ rosrun baxter_examples joint_position_keyboard.py
    #   Press n and . all the way to the end, then run this, look at left_w2:
    #   $ rostopic echo /robot/joint_states
    # Bonus: If want to see Cartesian pose, use this:
    #   $ rostopic echo /robot/limb/left/endpoint_state
    # Mult by 2 to make this go faster. I don't think it's necessary to rotate
    #   30+ times. More than enough, and most importantly, it'd take forever.
    self.ceil_angles = np.arange (-3.0589910537836067, 3.0589967568411645,
      self.angle_step * 2)


    # Angle for wrist to move to on cylinder wall

    # Allow wrist z (axis out of palm) to point slightly away from robot.
    # Don't start with z pointing exactly to the right, very hard to be
    #   reachable for IK. Start a little more than 90, with palm slightly
    #   facing away from robot.
    self.wrist_ang_buffer = 30.0 * np.pi / 180.0

    # This is wrt whatever frame you end up using this variable in! Robot
    #   frame or object frame. These angles decide which side of the object
    #   you start in! So if you start the range at 0, that means the start
    #   location is on the x-axis, which in robot frame pts front - this
    #   means you start at the back side of object. If you start the range at
    #   pi/2, that is where y axis lies. In robot frame, y-axis points to
    #   the left, which means you will start on left of object!
    # wrt robot /base.
    #   Use this for defining POSITIONS only, not orientation of wrist on wall.
    #   Latter probably only works if you do all rotations after the 1st
    #   one (in collect_wall) wrt wrist frame, just like you do in collect_ceil.
    wall_start_angle = np.pi * 0.5 + self.wrist_ang_buffer

    # This defines how big a sector of the cylinder you will move to, around
    #   the wall. e.g. If you add pi/2, that means you'll move 90 degs around
    #   object. If you add 2*pi, you'll move all the way around, 360 degs.
    # Some objects may make it impossible to go all 2*pi around, so you'll
    #   want to adjust this. 90 degs from left of obj to front is safe
    #   minimum. Preferred max is 360 degs all around, but that may not be
    #   possible to move to.
    # Just do 30 degs. The elbow up/down is more than human assistance can
    #   handle. Also couldn't find a spot in workspace that's reachable without
    #   flipping elbow, when collision avoidance is not present. Just record
    #   object in multiple runs of this program, each time doing 30 degs only.
    wall_end_angle = wall_start_angle + (self.wall_range * np.pi / 180.0)

    self.wall_angles = np.arange (wall_start_angle, wall_end_angle,
      self.angle_step)

    # Make sure last elt is 2*pi
    #   This closes the loop so that we can see how far the error has drifted
    if self.wall_angles [len (self.wall_angles) - 1] != wall_end_angle:

      self.wall_angles = np.append (self.wall_angles, wall_end_angle)


    #####
    # Print out some info
    #####

    print ('%sWrist space configuration:%s' % (ansi_colors.OKCYAN,
      ansi_colors.ENDC))

    print ('')
    print ('Palm clearance: %f m' % self.PALM_THICKNESS)
    print ('Object height: %f m' % self.obj_height)
    print ('  Wrist ceiling space height: %f. Wrist wall space height: %f' % \
      (self.wrist_ceil_h, self.wrist_wall_h))
    print ('Object radius rx: %f m, ry: %f m' % ( \
      self.obj_radius_x, self.obj_radius_y))
    print ('  Wrist ceiling space radius: rx %f ry %f. Wrist wall space radius: rx %f ry %f' % \
      (self.wrist_ceil_rx, self.wrist_ceil_ry,
       self.wrist_wall_rx, self.wrist_wall_ry))
    print ('')

    print ('Sampling density: %f m' % self.density)
    print ('Number of heights: %d' % (len (self.wall_heights)))
    print (self.wall_heights)
    print ('')

    print ('Arc length steps (should == density): %f m' % \
      (circ / len (self.wall_angles)))
    print ('Circumference: %f m' % (circ))
    print ('Number of ceiling angles: %d' % (len (self.ceil_angles)))
    print (self.ceil_angles * 180.0 / np.pi)
    print ('Number of wall angles: %d' % (len (self.wall_angles)))
    print (self.wall_angles * 180.0 / np.pi)

    print ('Product of above two grid sizes: %d' % (len (self.wall_heights) * \
      len (self.wall_angles)))
    print ('')

    print ('Number of ceiling grid points: %d' % (len \
      (self.wrist_ceil_grid_pts.points)))


    self.configured = True


  def config_constant_visualization (self):

    #####
    # Wrist space cylinders, as RViz Markers
    #####

    # dark blue
    self.marker_w_wall = Marker ()
    create_marker (Marker.CYLINDER, 'wrist_space', self.robot_frame_id, 0,
      self.obj_center[0], self.obj_center[1],
      # They put center at center height of cylinder. Shift it up
      self.obj_center[2] + self.obj_height * 0.5,
      0, 0, 1, 0.2, self.wrist_wall_rx * 2, self.wrist_wall_ry * 2,
      self.obj_height,
      self.marker_w_wall, 0)  # Use 0 duration for forever

    self.marker_w_ceil = Marker ()
    create_marker (Marker.CYLINDER, 'wrist_space', self.robot_frame_id, 1,
      self.obj_center[0], self.obj_center[1],
      # They put center at center height of cylinder. Shift it up
      self.obj_center[2] + self.obj_height + self.wrist_ceil_h * 0.5,
      0, 0, 1, 0.2, self.wrist_ceil_rx * 2, self.wrist_ceil_ry * 2,
      self.wrist_ceil_h,
      self.marker_w_ceil, 0)  # Use 0 duration for forever


  def call_zero_tactile (self):

    if self.reflex_on:
      print ('Zeroing tactile values...')
      self.zero_tactile ()
      rospy.sleep(self.ZERO_TACTILE_WAIT_TIME)


  def get_n_moves (self):
    return self.nMoves


  def broadcast_obj_frame (self):

    # TODO: May add orientation for asymmetric objects, but... it really
    #   doesn't matter, since I don't have a model for the real objects to
    #   set a reference frame against.

    # http://mirror.umd.edu/roswiki/doc/diamondback/api/tf/html/python/tf_python.html
    self.bc.sendTransform (self.obj_center, [0, 0, 0, 1],
      rospy.Time.now (), self.obj_frame_id, self.robot_frame_id)


  # Broadcast goal grid point, and orientation
  #   Display this tf frame so you can see the RGB axes and see if it's the
  #     correct goal pose for WRIST.
  def broadcast_goal_frame (self):

    if not self.goal_pose:
      return

    pos = self.goal_pose.pose.position
    rot = self.goal_pose.pose.orientation

    #print ('Broadcasting goal pose in frame %s' % self.goal_pose.header.frame_id)
    #print ('  Broadcasting goal z: %f' % pos.z)

    # Broadcast the tf frame so can see it in RViz, to help debug
    self.bc.sendTransform ([pos.x, pos.y, pos.z], [rot.x, rot.y, rot.z, rot.w],
      rospy.Time.now (), self.goal_frame_id, self.goal_pose.header.frame_id)


  # Don't need this! SEED_CURRENT does this!
  #def jointStateCB (self, msg):
  #  self.joint_state = JointState (msg.header, msg.name, msg.position,
  #    msg.velocity, msg.effort)
  #  self.joint_state.header.frame_id = '/base'


  # Remove if don't need
  # Copied from tactile_map explore_fixed_pos.py
  # Parameters:
  #   msg: type baxter_core_msgs/EndpointState. frame_id is blank, but it's
  #     wrt /base.
  def endpointCB (self, msg):

    self.endpoint = PoseStamped (msg.header, msg.pose)
    self.endpoint.header.frame_id = '/base'

    #print ('Received endpoint pose:')
    #print (self.endpoint.pose.position)


  # Recorder needs this
  def contactsCB (self, msg):

    if self.recorderNode:
      self.recorderNode.store_contacts_as_lists (msg)


  # Copied from weights_collect.py
  # ROS node: make sure to call this every iteration
  def pub_keyboard_prompt (self):
    self.prompt_pub.publish (self.prompt_msg)


  def keyCB (self, msg):

    # Let user pause the auto movements, in case something bad happens
    if msg.data.lower () == 'p':
      self.pauseCollect = True

    # Let user resume from pause
    elif msg.data.lower () == 'r':
      self.pauseCollect = False


  def write_obj_params (self):

    print ('Writing object params...')

    if not self.recorderNode:
      return

    # If recorder node has not started recording, print warning
    if not self.recorderNode.timestring:
      rospy.logwarn ('triangles_collect_semiauto.py write_obj_params(): Timestring not yet set in recorder node. Have you started rosrun tactile_map detect_reflex_contacts.py?')
      return
 
    self.obj_params_path = get_robot_obj_params_path ( \
      self.recorderNode.csv_suffix)
    self.obj_params_name = os.path.join (self.obj_params_path,
      self.recorderNode.timestring + '.csv')
    obj_params_file = open (self.obj_params_name, 'wb')

    column_titles = ['obj_center_x', 'obj_center_y', 'obj_center_z',
      'obj_radius_x', 'obj_radius_y', 'obj_height', 'sampling_density',
      'n_grid_pts', 'n_moves', 'n_contacts', 'n_triangles_hand']
    obj_params_writer = csv.DictWriter (obj_params_file,
      fieldnames=column_titles, restval='-1')
    obj_params_writer.writeheader ()


    n_grid_pts = 0
    # Assumption:
    #   This if-elif stmt assumes ceiling and wall are exclusive, not both done
    #   in one run. If you add collision detection and therefore can do both
    #   in one run, you should change how you count n_grid_pts.
    if self.skip_ceil_all:
      # Wall points
      # Assumption: The number of grid points is total possible values of
      #   z_idx, multiplied by total possible values of theta_idx. And the
      #   lengths of these arrays are equal to those total number of poss vals.
      n_grid_pts += (len (self.wall_angles) * len (self.z_vec))

    elif self.skip_ceil_all_but_first:
      # Wall points + ceiling points
      n_grid_pts += (len (self.wall_angles) * len (self.z_vec) + \
        len (self.wrist_ceil_grid_pts.points))

    if self.skip_wall:
      # Ceiling points
      # Assumption: This has 1 point, even if you don't run dry run.
      n_grid_pts += len (self.wrist_ceil_grid_pts.points)


    n_moves = self.nMoves
    n_contacts = self.recorderNode.get_n_contacts ()
    n_triangles_hand = self.recorderNode.get_n_tris_h ()

    # Debug info
    print ('This run: n_moves %d, n_contacts %d, n_triangles_hand %d' % ( \
      n_moves, n_contacts, n_triangles_hand))


    # Read the file we're picking up from.
    #   Add n_moves, n_contacts, n_triangles_hand from prev file to current
    #   values.
    if self.pickup_leftoff and self.pickup_from_file:

      leftoff_name = os.path.join (self.obj_params_path,
        self.pickup_from_file + '.csv')
      print ('Summing contact stats from previous pcd_params file we are picking up from: %s' % leftoff_name)

      with open (leftoff_name, 'rb') as leftoff_file:

        leftoff_reader = csv.DictReader (leftoff_file)

        # There's only 1 data row in file
        for row in leftoff_reader:
          n_moves += (int (row ['n_moves']))
          n_contacts += (int (row ['n_contacts']))
          # Already cumulative above, `.` now load all old ones into memory
          #   for duplicate checks. Shouldn't have to add here... not sure why
          #   the result is not wrong.
          n_triangles_hand += (int (row ['n_triangles_hand']))

          print ('From previous file: n_moves %d, n_contacts %d, n_triangles_hand %d' % ( \
            int (row ['n_moves']), int (row ['n_contacts']), int (row ['n_triangles_hand'])))

        print ('Final parameters: n_moves %d, n_contacts %d, n_triangles_hand %d' % ( \
          n_moves, n_contacts, n_triangles_hand))


    # Write file
    row = dict ()
    row.update (zip (column_titles, [self.obj_center[0], self.obj_center[1],
      self.obj_center[2], self.obj_radius_x, self.obj_radius_y,
      self.obj_height, self.density,
      n_grid_pts, n_moves, n_contacts, n_triangles_hand]))
    obj_params_writer.writerow (row)

    obj_params_file.close ()

    print ('%sOutputted object parameters to %s%s' % ( \
      ansi_colors.OKCYAN, self.obj_params_name, ansi_colors.ENDC))

    self.obj_params_written = True


  # Entry point from outside, in ROS loop, call this function after
  #   instantiating an instance of this class.
  def collect (self, dry=False):

    print ('collect() is called')

    if self.pauseCollect:
      return

    if not self.configured:
      self.reconfig ()


    # For detect_reflex_contacts.py to know which frames to do tf to
    self.reflex_on_pub.publish (self.reflex_on_msg)
    self.robot_on_pub.publish (self.robot_on_msg)


    # Collect ceiling first
    if not self.ceil_done:

      if dry:
        self.theta_idx_dry = \
          self.collect_ceiling (self.theta_idx_dry, dry=dry)

      else:
        self.theta_idx = \
          self.collect_ceiling (self.theta_idx, dry=dry)

    # Then collect wall
    else:
      if not self.wall_done:

        # If dry run, this call will populate self.wrist_wall_grid_pts
        if dry:

          [self.theta_idx_dry, self.z_idx_dry] = self.collect_wall ( \
            self.theta_idx_dry, self.z_idx_dry, dry=dry)

          if self.doTerminate:
 
            print ('')
            print ('Number of wall grid points: %d' % (len \
              (self.wrist_wall_grid_pts.points)))
    
            print ('Assuming only 1/4 of wrist wall is reachable (left and front),')
            print ('  1/4 of grid wall points: %f' % \
              (0.25 * len (self.wrist_wall_grid_pts.points)))

            # Reset flag, so next time will reconfigure, to be safe to reset
            #   all params to start state
            self.configured = False
 
            return

        else:
          [self.theta_idx, self.z_idx] = self.collect_wall ( \
            self.theta_idx, self.z_idx, dry=dry)

          if self.doTerminate:

            # Make sure object parameters are also being recorded
            # (No param file is written if collection never starts, or
            #   recorderNode is not instantiated, i.e. no real data recorded).
            if self.recorderNode:
              self.write_obj_params ()

            return

      # If both done, we're done
      else:
        self.doTerminate = True
        print ('Terminating... (collect() detects both ceil and wall are done.)')
        return


  # First move, using IK to place wrist at top of ceiling, such that palm
  #   faces downwards, wrist joint position is at its min.
  # All subsequent moves, use FK to rotate just left_w2 joint, a little each
  #   iteration, from min to max. This is about 360 degrees.
  # Then done.
  def collect_ceiling (self, theta_idx, dry=False):

    if self.pauseCollect:
      return theta_idx
  
    # Reset for this move
    self.prev_move_success = False

    # If skipping whole thing
    if self.skip_ceil_all:
      self.prev_move_success = True
      self.ceil_done = True
      # Reset index for wall
      if not self.pickup_leftoff:
        theta_idx = 0
      return theta_idx


    # If this is the first ever grid point to traverse
    if not self.goal_pose:
      print ('%sCollecting ceiling...%s' % (ansi_colors.OKCYAN,
        ansi_colors.ENDC))

      # Put fingers in spherical preshape, for better coverage from top
      if self.reflex_on:
        print ('Zeroing tactile values...')
        self.zero_tactile ()
        rospy.sleep(self.ZERO_TACTILE_WAIT_TIME)
 
        if self.preshape != self.SPH:
          rospy.loginfo ('Putting fingers in spherical preshape...')
          call_smart_commands ('spherical')
          self.preshape = self.SPH


    print ('theta_idx: %d' % self.theta_idx)
    print ('Angle: %f' % self.ceil_angles [theta_idx])
  
    self.goal_pose = PoseStamped ()

    # Initial pose. Fore-fingers pointing back, thumb forward.
    # Zero reference for ceiling point. Start with this pose, then rotate wrt
    #   z (pointing down) incrementally to move to each rotation in
    #   ceil_angles[].
    # Goal orientation: z down, y left, x pts to back
    #   Quaternion from robot frame /base: 180 degs wrt y
    #   See my paper notes p.96.
    # Ref: http://answers.ros.org/question/69754/quaternion-transformations-in-python/
    ceil_zero_quat = tf.transformations.quaternion_from_euler (0, np.pi, 0)

    # Must specify wrt robot frame, `.` it's the only fixed frame, to
    #   specify a pre-defined pose for wrist.
    self.goal_pose.header.frame_id = self.robot_frame_id

    # self.ceil_pt is wrt robot frame
    # The grid points are 13 cm from object, and this distance is measured
    #   from base of ReFlex mounting board, which I think is where
    #   /left_gripper is. So the grid point positions are ready for IK, no
    #   need to call augment_pos() on these.
    self.goal_pose.pose.position = Point (self.ceil_pt.x, self.ceil_pt.y,
      self.ceil_pt.z)
    print ('goal pose z: %f' % (self.goal_pose.pose.position.z))

    # Init to zero reference frame
    quat = ceil_zero_quat
    # Rotate to the goal orientation, wrt z axis
    quat2 = tf.transformations.quaternion_from_euler (\
      0, 0, self.ceil_angles [theta_idx])
    quat = tf.transformations.quaternion_multiply (quat, quat2)


    # Transform the goal (calculated for /base_link on paper) to /left_gripper
    quat = self.augment_IK_sol_mult_hand_to_wrist (quat)


    # Ref API: http://answers.ros.org/question/12903/quaternions-with-python/
    #   *q is equivalent to Quaternion (q[0], q[1], q[2], q[3]). * operator
    #     is for unpacking a list into separate elements.
    self.goal_pose.pose.orientation = Quaternion (*quat)

    self.goal_pose.header.stamp = rospy.Time.now ()

    #print ('goal pose z after hand-to-wrist augment: %f' % \
    #  (self.goal_pose.pose.position.z))

    # Broadcast the tf frame so can see it in RViz, to help debug
    #wait_rate = rospy.Rate (10)
    for i in range (0, 5):
      self.broadcast_goal_frame ()
      #wait_rate.sleep ()
      time.sleep (0.1)


    if dry or not self.robot_on:
      self.prev_move_success = True

    # Call IK
    else:

      # If first one, moving from arbitrary wrist position to goal, just use
      #   all possible seeds.
      if theta_idx == 0:
        # Obtained using rosrun baxter_examples joint_positions_keyboard.py
        #   The corresponding endpoint state:
        '''
        position: 
          x: 0.735770728878
          y: 0.44130031154
          z: 0.300934618437
        orientation: 
          x: 0.66430638486
          y: 0.74557650427
          z: 0.0129003951511
          w: 0.0514420365007
        '''

        seed_angles = JointState ()
        seed_angles.name = ['left_e0', 'left_e1', 'left_s0', 'left_s1',
          'left_w0', 'left_w1', 'left_w2']
        seed_angles.position = [0.01089305480494307, 1.3987008086536905,
          -0.5166293591863651, -0.8091106563415531, 3.0393680270579644,
          -0.9066062733455071, -1.3496074623528482]

        seed_mode = self.ikreq.SEED_AUTO

        if self.move_to_goal (seed_angles=seed_angles, seed_mode=seed_mode):

          # Using FK,
          # turn wrist to min of joint limit, so that it doesn't get stuck
          #   midway of sampling ceiling - then I'd have to detect and turn the
          #   other way, too much unworthy work.

          joint_command = {'left_w2': self.ceil_angles [theta_idx]}
  
          # This function keep moving joint till reach goal, and it smoothes out
          #   movement using low-pass filter. Function blocks till reaches goal.
          #   Better for IK.
          limb = baxter_interface.Limb ('left')
          limb.move_to_joint_positions (joint_command)

          # Record current contact
          self.close_gripper_and_record ()

          self.prev_move_success = True

        # If can't even move to ceiling, user should adjust object position
        else:
          self.prev_move_success = False


      # After wrist is already at ceiling, use current joint positions to seed,
      #   to eliminate elbow up / elbow down switches
      else:

        # FK
        joint_command = {'left_w2': self.ceil_angles [theta_idx]}
 
        # This function keep moving joint till reach goal, and it smoothes out
        #   movement using low-pass filter. Function blocks till reaches goal.
        #   Better for IK.
        limb = baxter_interface.Limb ('left')
        limb.move_to_joint_positions (joint_command)

        # Record current contact
        self.close_gripper_and_record ()

        self.prev_move_success = True


    # Advance for next iteration
    if self.prev_move_success:

      # Reset for next round
      self.prev_move_success = False

      # This control sequence is equivalent to this loop:
      #   # Skip first one, already populated above
      #   for theta_idx in range (1, len (self.ceil_angles)):
      theta_idx += 1
     
      # If finished all angles, signal done with ceiling
      if theta_idx == len (self.ceil_angles) or self.skip_ceil_all_but_first:
        self.ceil_done = True
        # Reset index for wall
        theta_idx = 0


    return (theta_idx)


  def collect_wall (self, theta_idx, z_idx, dry=False):

    if self.pauseCollect:
      return

    if self.skip_wall:
      self.wall_done = True
      self.doTerminate = True
      theta_idx = len (self.wall_angles)
      z_idx = 0
      return (theta_idx, z_idx)

    # If this is the first ever point on the wall
    if self.reflex_on and theta_idx == 0 and z_idx == 0:
      print ('%sCollecting wall...%s' % (ansi_colors.OKCYAN,
        ansi_colors.ENDC))

      # Now jsut doing 2 preshapes per grid point. So don't need to do this
      #   in advance here. Remove when decide to permanently use 2 preshapes
      #   per grid point.
      #rospy.loginfo ('Putting fingers in cylinder preshape...')
      #call_smart_commands ('cylinder')
      #self.preshape = self.CYL


    # Reset
    self.prev_move_success = False


    # Get the angle for this theta-column
    theta = self.wall_angles [theta_idx]
    # Calculate new column's x y z position.
    # If this is a new column, calculate new column.
    # Get x y and list of heights for this theta-column.
    #   x and y stay the same throughout all z's.
    if z_idx == 0 or self.pickup_leftoff:

      print ('%sCollecting one column...%s' % \
        (ansi_colors.OKCYAN, ansi_colors.ENDC))
 
      if self.reflex_on:
        print ('Zeroing tactile values...')
        self.zero_tactile ()
        rospy.sleep(self.ZERO_TACTILE_WAIT_TIME)

      (self.x, self.y, self.z_vec) = self.calc_column (self.wrist_wall_rx,
        self.wrist_wall_ry, theta)

    # Get current grid point's z position
    z = self.z_vec [z_idx]


    print ('%sz_idx: [%d] out of %d (0 based), z: %f%s' % \
      (ansi_colors.OKCYAN, z_idx, len (self.z_vec)-1, self.z_vec[z_idx],
       ansi_colors.ENDC))
    print (self.z_vec)
    print ('%stheta_idx: [%d] out of %d (0 based), angle: %f%s' % \
      (ansi_colors.OKCYAN,
      theta_idx, len (self.wall_angles)-1,
      self.wall_angles [theta_idx] * 180.0 / np.pi, ansi_colors.ENDC))


    #####
    # Visualize goal x y z position. Position ONLY.
    #   x and y are same for this entire theta-column. z is different for each
    #     iteration this function is called (provided previous move was
    #     successful. Otherwise we retry the same z).
    #####

    # Yellow
    marker_goal = Marker ()
    create_marker (Marker.POINTS, 'sample_pts', self.obj_frame_id, 0,
      0, 0, 0, 1, 1, 0, 0.8, 0.005, 0.005, 0.005,
      marker_goal, 0)  # Use 0 duration for forever
    marker_goal.points.append (Point (self.x, self.y, z))

    self.vis_pub.publish (marker_goal)


    #####
    # Define quaternion for goal pose
    #   wrt robot frame /base.
    #   The goal pose quaternion was calculated using ReFlex /base_link frame!
    #     So at the end, we have to multiply by the transform from /base_link
    #     to /left_gripper, before passing to Baxter IK Solver, which expects
    #     things to be for /left_gripper.
    #####

    # Calculate orientation of wrist in this column
    quat = self.calc_column_quat (theta_idx)


    #####
    # Collect position and quaternion from above. Broadcast frame.
    #####

    self.goal_pose = PoseStamped ()

    # If use robot frame, must add object offset
    self.goal_pose.header.frame_id = self.robot_frame_id

    # The grid points are 13 cm from object, and this distance is measured
    #   from base of ReFlex mounting board, which I think is where
    #   /left_gripper is. So the grid point positions are ready for IK, no
    #   need to call augment_pos() on these.
    self.goal_pose.pose.position = Point (self.obj_center [0] + self.x,
      self.obj_center [1] + self.y, self.obj_center [2] + z)

    # Ref API: http://answers.ros.org/question/12903/quaternions-with-python/
    #   *q is equivalent to Quaternion (q[0], q[1], q[2], q[3]). * operator
    #     is for unpacking a list into separate elements.
    self.goal_pose.pose.orientation = Quaternion (*quat)
    self.goal_pose.header.stamp = rospy.Time.now ()

    # Broadcast the tf frame so can see it in RViz, to help debug
    self.broadcast_goal_frame ()

    print ('%sGoal position: %f %f %f%s' % (ansi_colors.OKCYAN,
      self.goal_pose.pose.position.x, self.goal_pose.pose.position.y,
      self.goal_pose.pose.position.z, ansi_colors.ENDC))


    #####
    # Move to grid point pose
    #####

    # If dry run, just add to point cloud visualization
    if dry or not self.robot_on:
      # Goal point for hand frame /base_link is self.x, self.y, z.
      #   But I want to display the IK endpoint frame throughout program, which
      #   is wrist frame /left_gripper.
      #   So offset the position by the transformation from /base_link to
      #   /left_gripper, to put them wrt /left_gripper. This will align with
      #   goal_pose tf frame, which must be wrt /left_gripper `.` that's what
      #   Baxter IK interface wants as endpoint.
      self.wrist_wall_grid_pts.points.append ( \
        self.goal_pose.pose.position)
        #Point (self.x, self.y, z))

      self.prev_move_success = True

    # Move for real
    else:

      '''
      Corresponding cartesian endpoint pose:
      $ rostopic echo /robot/limb/left/endpoint_state
      position: 
        x: 0.589219835031
        y: 0.720979550839
        z: 0.309068758767
      orientation: 
        x: 0.610347106894
        y: -0.357511513889
        z: 0.62623653211
        w: 0.32785626789

      To move here from ceiling, use
        $ rosrun baxter_examples joint_position_keyboard.py
      Press a series of u, 8, k, to make arm straight out;
        Use 9 to back shoulder away.
        Bring wrist in toward robot, closer to object, by series of y;
        Use i to bend arm slightly, to get close to obj;
        Use 9 to back away shoulder joint;
        Use j to bend wrist1 joint all the way in;
        Use n and . to rotate wrist2 (last joint), until green (y) axis pts
          downwards.

        Now you should be pretty close to the goal pose at left of object,
          with base_link y axis down, z axis pointing toward object center.
        Use u and y to straighten out the green (y) axis, so it's perfectly
          vertical.
        Use 8 to lower shoulder, to put wrist lower.
        Use 9 to back shoulder away, i to bring elbow in, 

        Now the arm should be pretty much in the xy-plane.
        Use 8, i, j to adjust the bending of arm's 3 joints, from top view
          in RViz, to bring it exactly to the goal point on left of obj.
      '''

      seed_angles = JointState ()
      seed_angles.name = ['left_e0', 'left_e1', 'left_s0', 'left_s1',
        'left_w0', 'left_w1', 'left_w2']
      seed_angles.position = [-1.5597481032375864, 1.5627945738100584,
        0.5097856422182492, -0.029266980950972687, 3.059004556765064,
        -0.8767840674733343, 1.7313499799561871]

      success = self.move_to_goal ( \
        seed_angles=seed_angles, seed_mode=self.ikreq.SEED_AUTO)


      if success:

        # Record again in spherical preshape
        if self.preshape != self.CYL:
          rospy.loginfo ('Putting fingers in cylinder preshape...')
          call_smart_commands ('cylinder')
          self.preshape = self.CYL
        # Record current contact
        self.close_gripper_and_record ()

        # Record once in cylinder preshape
        if self.preshape != self.SPH:
          rospy.loginfo ('Putting fingers in spherical preshape...')
          call_smart_commands ('spherical')
          self.preshape = self.SPH
        # Record current contact
        self.close_gripper_and_record ()

        self.prev_move_success = True

      # If couldn't move to goal, don't record, just skip this point
      else:

        # For any wall point other than the very first, if IK cannot find
        #   solution, skip the point.
        #   (`.` if there is no IK solution, we want it to move on and skip this
        #   pt, not get stuck forever. For very first point, if there's no IK
        #   solution, that means there really is no way to move there, manually
        #   adjust arm pose, or object position, and rerun script. Similarly,
        #   this is true for ceiling point.)
        if not (theta_idx == 0 and z_idx == 0):

          print ('%sSKIPPING this wall point that had no IK solution%s' % ( \
            ansi_colors.WARNING, ansi_colors.ENDC))

          # Tell collect() that this iteration was "successful", so we will
          #   stop trying on this point.
          self.prev_move_success = True


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
    if self.prev_move_success:
      # Inner loop. Loop through height along a theta-column
      z_idx += 1

      # If inner loop completed, increment to next theta-column
      if z_idx == len (self.wall_heights):
        theta_idx += 1
        # Reset height to start fresh in the new column
        z_idx = 0

        # If all theta-columns done, we can terminate program
        if theta_idx == len (self.wall_angles):
          self.doTerminate = True
          print ('Terminating... (collect_wall() detects all grid points done)')
          return (theta_idx, z_idx)


    return (theta_idx, z_idx)


  # Transform the member variable IK goal (calculated for /base_link on paper)
  #   to /left_gripper
  def augment_IK_sol_mult_hand_to_wrist (self, quat):

    # Apply the 4x4 transformation matrix, to turn the frame, which
    #   is calculated for ReFlex /base_link frame, into the Baxter
    #   Endpoint frame /left_gripper, which is what Baxter IK wants.

    # Position
    # The grid points are 13 cm from object, and this distance is measured
    #   from base of ReFlex mounting board, which I think is where
    #   /left_gripper is. So the grid point positions are ready for IK, no
    #   need to call augment_pos() on these.

    # Orientation
    # Transform from ReFlex hand /base_link, to Baxter /left_gripper.
    # For some reason, in my code, I defined this as the transform from
    #   /left_gripper to /base_link instead! That is wrist to hand, not
    #   hand to wrist (which it should be, `.` it's this function's name)!
    #   Logically, the transform should be:
    #   quat_wrist_wrt_hand = tf.transformations.quaternion_from_euler ( \
    #     0, 0, -np.pi*0.5)
    #   but I used (0, 0, np.pi*0.5) instead, on the real robot.
    #   It basically swaps which side the thumb and fore-fingers are on.
    #   It should be -np.pi*0.5. Change to that after testing that on
    #   real robot and verifying it works.
    #   Maybe I did that because it was better on real robot, perhaps
    #   the joint limits weren't good the other way?
    quat = transform_quat_wrist_to_hand (quat)

    return quat


  # Transform a given position from /base_link to /left_gripper frame
  def augment_pos_mult_hand_to_wrist (self, pos):

    # Use this refactored function. Have not tested, because no one actually
    #   use this function! Remove above when works
    return transform_pos_hand_to_wrist (pos)


  # Use current arm joint positions as seed
  def move_to_goal (self, seed_angles=None, seed_mode=None):

    # See my tactile_map explore_fixed_pos.py for example, search for "\.w".
    #   But I want slightly
    #   different - I want them to use the IK solution closest to current
    #   joints, look for the functions I used on internet, see if there's
    #   option to specify closest, or can they give me the solutions, and I
    #   look for closest myself.
    # From my old code, looks like you only get one response... Try in
    #   simulation and see how many you get.
    # http://sdk.rethinkrobotics.com/wiki/IK_Service_-_Code_Walkthrough
    # https://github.com/RethinkRobotics/sdk-docs/wiki/IK-Service-Example
    # https://github.com/RethinkRobotics/sdk-docs/wiki/API-Reference#arm-joints
    # 
    # OHHHH.... I think you just SEED it using the current pose! That must be
    #   it! In my old code, I always seeded using a constant pose, which may
    #   be why sometimes it switched from elbow up to elbow down, because the
    #   seed wasn't the closest anymore, after it's moved for quite a few cm!


    # Don't need this! SEED_CURRENT does this!
    '''
    # If have not received current joint positions yet
    if self.joint_state is None:
      print ('demo_pick_pen demo(): Waiting for Baxter joint states on' + \
        '/robot/joint_states...')
      return False
    '''


    #####
    # Call Baxter IK rosservice
    #####

    print ('move_to_goal(): goal pose z: %f' % (self.goal_pose.pose.position.z))

    # Don't need this! SEED_CURRENT does this!
    # Seed using current joints. Hopefully this eliminates possibility of
    #   switching btw elbow up and elbow down, when a solution exists for both
    #seed = self.joint_state

    # Ref SEED_CURRENT means current joint angles:
    #   https://github.com/RethinkRobotics/baxter_examples/blob/master/scripts/ik_service_client.py


    pose_idx = 0

    has_sol, ik_sol = get_ik_sol ([self.goal_pose], seed_angles, seed_mode)
    if not has_sol [pose_idx]:
      return False


    # If SEED_CURRENT doesn't work, might just have to feed in 
    #   seed = self.joint_state, and set seed_mode=SEED_AUTO.
    #   Then if you want to know which seed the solution was generated from,
    #     code here:
    #   https://github.com/RethinkRobotics/baxter_examples/blob/master/scripts/ik_service_client.py


    #####
    # Move to IK joint positions
    #####

    move_arm ('left', ik_sol.joints [pose_idx].name,
      ik_sol.joints [pose_idx].position)

    print ('Latest endpoint z: %f' % self.endpoint.pose.position.z)


    '''
    #####
    # Close in ReFlex Hand on object
    #####

    # Only do this if hand has been zeroed, to avoid uncalibrated hand damaging
    #   motors.
    if (not self.no_hand) and (self.hand_zeroed):
      rospy.loginfo ('Closing in ReFlex fingers...')

      # Execute moves. If failed, return.
      if not call_smart_commands ('pinch'):
        print ('/reflex/command_smarts cylinder/pinch failed. Aborting demo.')
        return
      # Wait 5 secs
      if not call_smart_commands ('guarded_move', 5):
        print ('/reflex/command_smarts guarded_move failed. Aborting demo.')
        return

      rospy.loginfo ('Closing fingers in guarded_move succeeded')

    '''

    return True


  # Call this only once per column, before starting to collect the column
  # Internal helper.
  def calc_column (self, rx, ry, theta):

    # In object frame. Otherwise you have to add the offset to robot frame
    x = rx * np.cos (theta)
    y = ry * np.sin (theta)


    # Decide top-down or bottom-up order. Set to reverse of previous, so can
    #   zip zag up and down. Faster, fewer movements.
    if self.z_order == self.TOP_DOWN:
      self.z_order = self.BOTTOM_UP
      print ('Column heights will be traversed BOTTOM UP:')
      z_vec = self.wall_heights.copy ()
    elif self.z_order == self.BOTTOM_UP:
      self.z_order = self.TOP_DOWN
      print ('Column heights will be traversed TOP DOWN:')
      z_vec = np.flipud (self.wall_heights)
    print (z_vec)

    return (x, y, z_vec)


  def calc_column_quat (self, theta_idx):

    # Or can use object frame, which is now fixed wrt robot frame, so it's
    #   just like robot frame. Right now, it's not clear which one is more
    #   advantageous. I would think robot frame is better, because we aren't
    #   tracking object. So if object frame changes, we don't know. If
    #   you want to stress that the descriptor is independent of obj mvmt,
    #   then you shouldn't track the object, just let it move if it wants to.
    #   So you should definitely go with robot frame, once you start to
    #   consider moving objects! (unless you plan to track the object)
    #self.goal_pose.header.frame_id = self.obj_frame_id
    #self.goal_pose.pose.position = Point (self.x, self.y, z)

    # Goal orientation: x pts to back, y down, z right (so gripper is on
    #   left of object, palm facing right, therefore z right).
    #   Quaternion from robot frame /base: +90 degs wrt x, 180 wrt z
    #   See my paper notes p.96.
    # Ref: http://answers.ros.org/question/69754/quaternion-transformations-in-python/
    quat1 = tf.transformations.quaternion_from_euler (0.5 * np.pi, 0, 0)
    quat2 = tf.transformations.quaternion_from_euler (0, 0, np.pi)

    # Ref API quaternion_multiply():
    #   http://answers.ros.org/question/196149/how-to-rotate-vector-by-quaternion-in-python/
    quat = tf.transformations.quaternion_multiply (quat1, quat2)

    # Then rotate by the current amount around the wall
    # Frame: wrt robot frame, rotating wrt world vertical axis (`.` we assume
    #   object is upright). If you track object, you'd rotate wrt object's z
    #   up axis. If you don't track object, you'd rotate wrt robot frame.
    # But, since this is a sequential multiplication, following the two above,
    #   you have to rotate this wrt intermediate axis. The wrist axis that pts
    #   down is y-axis. Therefore you rotate wrt y! Rotate negative amount, `.`
    #   y points downwards, so rotating CCW is negative.
    quat3 = tf.transformations.quaternion_from_euler (0,
      - self.angle_step * theta_idx - self.wrist_ang_buffer, 0)
    quat = tf.transformations.quaternion_multiply (quat, quat3)


    #####
    # Transform the goal (calculated for /base_link on paper) to /left_gripper
    #####

    quat = self.augment_IK_sol_mult_hand_to_wrist (quat)


    return quat


  # Parameters:
  #   cache_only_dont_eval: Cache the new contact points, do not sample
  #     triangles from them yet. This is useful if you want to accumulate more
  #     than just the minimum (3) points across multiple grasps.
  #     If False, evaluate whenever have >= 3 points (set in record_tri) in
  #     cache, the min number of points to make a triangle.
  def close_gripper_and_record (self, guarded_move_wait_time=5,
    open_wait_time=2, wait_time_before_record=0.5, cache_only_dont_eval=False):

    # If no recorder was passed in, do nothing.
    if not self.recorderNode:
      return

    # Close gripper
    # Wait 5 seconds
    rospy.loginfo ('Closing in ReFlex fingers...')
    call_smart_commands ('guarded_move', guarded_move_wait_time)

    if wait_time_before_record != 0:
      rospy.sleep (wait_time_before_record)

    # Record
    self.recorderNode.doCollectOne = True
    self.recorderNode.collect_one_contact (cache_only_dont_eval)

    # Make sure object parameters are also being recorded
    # (No param file is written if collection never starts, or recorderNode
    #   is not instantiated, i.e. no real data recorded).
    # Write this one time at beginning, then write again at the end, to make
    #   sure SOMETHING is written, even if don't have all the values
    if not self.obj_params_written:
      self.write_obj_params ()

    # Count each gripper closure as a move
    self.nMoves += 1

    # Open gripper
    rospy.loginfo ('Opening ReFlex fingers...')
    call_smart_commands ('open', open_wait_time)
 

  # Use this at beginning of program execution, to let user move object around,
  #   until all 8 extreme points of the cylinder are reachable via IK. Or
  #   even if not all 8 are reachable, user can make the call of whether to
  #   keep this configuration anyway. (Unreachable points during collection
  #   will be automatically skipped.)
  def test_eight_IK_points (self, obj_center):

    #####
    # 1 point on ceiling
    #####

    #goal_list_ceil = # list of 1 PoseStamped

    # TODO: I probably won't have seed angles, because I don't know where the
    #   object is! But I COULD put in the seed angles for ceiling, and
    #   seed angles for wall, for the default object center.
    seed_angles_ceil = None
    seed_mode_ceil = None

    has_sol, _ = get_ik_sol (goal_list_ceil, seed_angles_ceil,
      seed_mode_ceil)

    # Find out which point wasn't feasible, output to user - or just render in
    #   RViz, much easier and bug-proof!
    for i in range (0, len (has_sol)):
      if not valid:
        print ('IK solution not found for goal pose %d' % i)


    #####
    # 8 extreme points on cylinder wall
    #####

    #goal_list_wall = # list of 8 PoseStamped

    seed_angles_wall = None
    seed_mode_wall = None

    has_sol, _ = get_ik_sol (goal_list_wall, seed_angles_wall,
      seed_mode_wall)

    for i in range (0, len (has_sol)):
      if not valid:
        print ('IK solution not found for goal pose %d' % i)




  def visualize (self):

    # Object bbox
    # white
    marker_obj = Marker ()
    create_marker (Marker.CYLINDER, 'object', self.robot_frame_id, 0,
      self.obj_center [0], self.obj_center [1],
      # They put center at center height of cylinder. Shift it up
      self.obj_center[2] + self.obj_height * 0.5,
      # Scale: X diameter, Y diameter, Z height
      1, 1, 1, 0.2,
      self.obj_radius_x * 2, self.obj_radius_y * 2, self.obj_height,
      marker_obj, 0)  # Use 0 duration for forever
    self.vis_pub.publish (marker_obj)

    # Wrist space cylinder
    # dark blue
    self.vis_pub.publish (self.marker_w_wall)
    self.vis_pub.publish (self.marker_w_ceil)


    # Wrist space grid points
    # display this as green 0 1 0 in RViz
    self.wrist_ceil_grid_pts.header.stamp = rospy.Time.now ()
    self.wrist_ceil_pub.publish (self.wrist_ceil_grid_pts)

    self.wrist_wall_grid_pts.header.stamp = rospy.Time.now ()
    self.wrist_wall_pub.publish (self.wrist_wall_grid_pts)


def get_ik_sol (goal_list, seed_angles=None, seed_mode=None):

  # Default return value
  has_sol = [False] * len (goal_list)

  #try:
  has_sol, ik_sol = solve_ik ('left', goal_list, seed=seed_angles,
    seed_mode=seed_mode)
  #except rospy.ServiceException, e:
  #  rospy.logerr (
  #    'baxter_control solve_ik(): Service call to Baxter IK failed: %s' %e)
  #  return (has_sol, None)


  if ik_sol is None:
    rospy.logerr ('baxter_control solve_ik(): Service call to Baxter IK failed')

    # Return true to signal we tried, and terminate program.
    # In simulation: Return false to keep trying repeatedly until
    #   succeed. Needed in simulation `.` tf frame publishing is too 
    #   slow, and IK service doesn't wait for frames to exist or checks
    #   canTransform() like I do in my tf_get_pose(). It just returns
    #   no IK solution immediately when it can't find a tf transform.
    return (has_sol, ik_sol)

  else:
    if all (has_sol):
      rospy.loginfo ('Found IK solution for all poses requested.')
    else:
      rospy.logerr ('No IK solution found for at least one pose. ' +
        'Try moving arm to another starting position.')

  return (has_sol, ik_sol)




def main (argv):

  rospy.init_node ('triangles_collect_semiauto', anonymous=True)

  vis_pub = rospy.Publisher ('/visualization_marker', Marker)


  #####
  # Parse command line args
  #   Ref: Tutorial https://docs.python.org/2/howto/argparse.html
  #        Full API (better) https://docs.python.org/dev/library/argparse.html
  #####

  arg_parser = argparse.ArgumentParser ()

  # Required args
  arg_parser.add_argument ('ceil', type=int, default=False,
    help='Boolean int, 0 or 1. Only one of ceil and wall can be 1. Specify this flag to collect ceiling part of object. Gripper will move open loop to above object center, by the specified height, leaving some clearance.')
  arg_parser.add_argument ('wall', type=int, default=False,
    help='Boolean int, 0 or 1. Only one of ceil and wall can be 1. Specify this flag to collect wall part of object. Gripper will move open loop to around object, by the specified radius, leaving some clearance. The start angle is defined by wall_start_angle and wrist_ang_buffer in code. The total angle around the object is defined by wall_end_angle in code.')

  arg_parser.add_argument ('--gauge', action='store_true', default=False,
    help='Boolean int: 1 or 0. Whether to first go through UI to gauge object center')
  # Optional args
  arg_parser.add_argument ('-rx', '--obj_radius_x', type=float, default=0.05,
    help='Float. Object radius in x direction in meters. For symmetric object, specify same value for rx and ry.')
  arg_parser.add_argument ('-ry', '--obj_radius_y', type=float, default=0.05,
    help='Float. Object radius in y direction in meters. For symmetric object, specify same value for rx and ry.')
  arg_parser.add_argument ('-z', '--obj_height', type=float, default=0.2,
    help='Float. Object height in meters')
  arg_parser.add_argument ('-d', '--density', type=float, default=0.03,
    help='Float. Sampling density length in meters')

  arg_parser.add_argument ('--robot_on', action='store_true', default=False,
    help='Boolean flag. Whether robot is on')
  arg_parser.add_argument ('--reflex_on', action='store_true', default=False,
    help='Boolean flag. Whether ReFlex Hand is on')

  args = arg_parser.parse_args ()


  do_gauge = args.gauge

  obj_radius_x = args.obj_radius_x
  obj_radius_y = args.obj_radius_y
  obj_height = args.obj_height
  sampling_density = args.density
  if obj_radius_x <= 0 or obj_radius_y <= 0 or obj_height <= 0 or sampling_density <= 0:
    print ('%sObject radius, height, and sampling density must be positive numbers. Re-specify. Terminating...%s' % (ansi_colors.WARNING, ansi_colors.ENDC))
    return

  robot_on = args.robot_on
  if robot_on:
    print ('%sRobot is set to ON. Will try to enable and move robot.%s' % \
      (ansi_colors.OKCYAN, ansi_colors.ENDC))
  else:
    print ('%sRobot is set to OFF. Will NOT try to enable and move robot.%s' % \
      (ansi_colors.OKCYAN, ansi_colors.ENDC))

  reflex_on = args.reflex_on

  do_ceil = bool (args.ceil)
  do_wall = bool (args.wall)
  # Enforce if robot_on, meaning on real robot.
  # In Gazebo sim, since we aren't running with Baxter, it's okay there's no
  #   collision-aware planning. Hand isn't attached to Baxter anyway.
  if do_ceil and do_wall and robot_on:
    print ('Both ceil=1 and wall=1 were specified. You must pick one, because of the current lack of collision detection. Terminating...')
    return
  print ('%sceil set to %s, wall set to %s%s' % (ansi_colors.OKCYAN,
    do_ceil, do_wall, ansi_colors.ENDC))


  #####
  # Set a default object center first
  #####

  # Best one for simulation
  '''
  These were close to where wrist can move (using baxter_examples
    joint_position_keyboard.py), z is obj_height above obj_center:
  x: 0.735770728878
  y: 0.44130031154
  z: 0.300934618437
  '''
  # Both ceiling and wall are reachable for this.
  #obj_center = [0.64, 0.441, 0.05]

  # This wall is reachable, but not ceiling.
  # If x is too large, infeasible for wrist to be perpendicular to left surface
  #   of object. Better to keep x close to robot.
  # If you really want to start on exact left of obj, only way to achieve it
  #   is have arm bent like U-shape, with perfect 90 deg at the 2 angles of U.
  #   If you draw keyboard 9 shoulder backwards, then [j] wrist1 has to bend
  #   inwards more, but it will reach limit before it can compensate 9.
  #   Solution is to move object toward center of robot!
  #obj_center = [0.84, 0.2, 0.20]

  # A center previously used for purple ball
  #obj_center = [0.84, 0.504, -0.028]


  # Real objects
  # Stool is 0.15 m
  obj_center = [0.722, 0.347, -0.095]

  '''
  /left_gripper good height:
  -0.015 + 0.15 + 0.11 = 0.26 - 0.015 = 0.245

  But -0.015 is 7 cm too high for wall.

  '''


  #####
  # Interactive UI to gauge object center
  # User: Look in RViz to see where object currently is, and use baxter and
  #   hand's relative position to plotted object, to gauge when it's correct.
  #####

  if do_gauge:
    print ('Gauging from initial center: %f %f %f' % ( \
      obj_center[0], obj_center[1], obj_center[2]))
 
    obj_center = gauge_obj_center (obj_center, vis_pub, obj_height,
      obj_radius_x, obj_radius_y)

  print ('Gauging finished. Now initializing nodes for collection...')

  uinput = raw_input ('Press q to quit, any other key to continue: ')
  if uinput.lower () == 'q':
    return


  #####
  # Init nodes
  #####

  recorderNode = None
  if reflex_on:
    # The TrianglesCollect class actually writes things to file!
    # The file by itself can record manually, responding to user input from
    #   tactile_map keyboard_interface.py node.
    recorderNode = TrianglesCollect (csv_suffix='')

  thisNode = TrianglesCollectSemiAuto (obj_center, obj_radius_x, obj_radius_y,
    obj_height, sampling_density, robot_on=robot_on, reflex_on=reflex_on,
    recorderNode=recorderNode, do_ceil=do_ceil, do_wall=do_wall,
    wall_range=30.0)


  #####
  # Do a dry run to visualize wrist movement grid
  #####

  '''
  print ('')
  print ('================== DRY RUN ===================')

  thisNode.reconfig ()

  while 1:
    thisNode.collect (dry=True)

    if thisNode.doTerminate:
      break

  print ('')
  '''

  print ('=============== SKIPPING DRY RUN =============')


  #####
  # Main ROS loop
  #####

  wait_rate = rospy.Rate (10)

  print ('')
  print ('================== REAL RUN ===================')

  # Reset all member vars for real run
  thisNode.reconfig ()

  ended_unexpected = False

  while not rospy.is_shutdown ():

    thisNode.broadcast_obj_frame ()

    # Publish to keyboard_interface the prompt to display
    thisNode.pub_keyboard_prompt ()

    try:
      # Collect a grid point
      thisNode.collect ()

      thisNode.broadcast_goal_frame ()
      thisNode.visualize ()

      # Visualize data that's been collected to file
      if recorderNode:
        recorderNode.vis_arr_pub.publish (recorderNode.markers)
        # Don't publish this, baxter desktop gets too laggy
        #recorderNode.vis_arr_pub.publish (recorderNode.markers_cumu)
        # TODO test this
        recorderNode.cloud_pub.publish (recorderNode.collected_cloud)

      wait_rate.sleep ()

    except rospy.exceptions.ROSInterruptException, err:
      ended_unexpected = True
      break


    # If done with cylinder collection sequence
    if thisNode.doTerminate:
      if recorderNode:
        # Close output files
        recorderNode.doFinalWrite = True
        # Write final PCD file, with header
        recorderNode.collect_one_contact ()

        # Make sure latest contact / triangle stats are being recorded
        thisNode.write_obj_params ()

      break


  # If collection sequence ended unexpectedly, still save what we have so far
  if ended_unexpected:

    if recorderNode:
      # Close output files
      recorderNode.doFinalWrite = True
      # Write final PCD file, with header
      recorderNode.collect_one_contact ()

      thisNode.write_obj_params ()


if __name__ == '__main__':

  main (sys.argv)

