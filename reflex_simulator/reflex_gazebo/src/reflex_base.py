#!/usr/bin/env python

# Mabel Zhang
# 29 Oct 2015
#
# Simulates reflex_base_services.py in reflex-ros-pkg/reflex/src/reflex/ .
#   Smart command actual movements simulate reflex_smarts.py and reflex_base.py
#
#   Most function and class names in this file are same as those in real
#   hand's packages above.
#
# To test this file in bare shell:
#   roscore
#   rosrun reflex_gazebo reflex_base.py
#   rosservice call /reflex/command_smarts 'close'
#
# To test this file's simple commands with simulated hand in Gazebo:
#   roslaunch reflex_gazebo reflex_world.launch
#   roslaunch reflex_gazebo reflex.launch
#   rosservice call /reflex/command_smarts 'close'
#   rosservice call /reflex/command_smarts 'open'
#   rosservice call /reflex/command_smarts 'pinch'
#   rosservice call /reflex/command_smarts 'guarded_move'
#
#   For reflex.launch, you can substitute with rosrun reflex_base.py if you
#     don't need contact sensor data from rostopic /reflex_hand.
#     /reflex_hand is published by reflex_driver_node.py.
#     reflex.launch starts both of these .py files.
#

# ROS
import rospy
from std_srvs.srv import Empty
from std_msgs.msg import Float64

# ReFlex
from reflex_msgs.msg import Hand
from reflex_msgs.srv import CommandHand, MoveFinger, MovePreshape
from reflex_msgs.srv import CommandHandResponse

# Python
from copy import deepcopy

# Numpy
import numpy as np


# Copied from reflex-ros-pkgs/reflex/src/reflex/reflex_smarts.py
ROT_CYL = 0.0       # radians rotation
ROT_SPH = 0.8       # radians rotation
ROT_PINCH = 1.57    # radians rotation
OPEN = 0            # radians tendon spool
CLOSED = 3.5        # radians tendon spool
PROBE_POS = 0.8     # radians tendon spool

HOW_HARDER = 0.1    # radians step size for tightening and loosening


# Class for real hand is defined in reflex_base.py.
class ReFlex (object):

  def __init__ (self):

    super(ReFlex, self).__init__()

    # motion parameters
    self.FINGER_STEP = 0.05         # radians / 0.01 second
    self.FINGER_STEP_LARGE = 0.15   # radians / 0.01 second
    # Mabel added, for more speedy guarded_move for fast data collection
    self.FINGER_STEP_HUGE = 0.35    # radians / 0.01 second. ~20 degrees
    self.TENDON_MIN = 0.0           # max opening (radians)
    #self.TENDON_MAX = 4.0           # max closure (radians)
    # Mabel changed based on empirical. More than 3.14 penetrates palm!
    self.TENDON_MAX = 2.8           # max closure (radians)
    self.PRESHAPE_MIN = 0.0         # max opening (radians)
    # Copied to preshape_# joint <limit> tag in
    #   urdf/full_reflex_model.urdf.xacro
    self.PRESHAPE_MAX = 1.6         # max closure (radians)

    self.hand = Hand ()
    topic = '/reflex_hand'
    rospy.loginfo('ReFlex class is subscribing to topic %s', topic)
    rospy.Subscriber(topic, Hand, self.__hand_callback)


  def __hand_callback(self, msg):

    self.hand = deepcopy(msg)

    # Debug
    #for i in range (0, len (self.hand.finger)):
    #  if any (self.hand.finger [i].contact):
    #    print ('Finger %d is contacted' % i)



# Class for real hand is defined in reflex_smarts.py
# Here I merge functionalities of reflex_smart.py and reflex_base_services.py ,
#   just doing one file for simulated hand.
class ReFlex_Smarts (ReFlex):

  def __init__ (self):

    super(ReFlex_Smarts, self).__init__()

    self.N_FIN = 3
    self.N_SEN = 9


    # Controller names are from this package, config/reflex_sim_control.yaml

    preshape_1_pub = rospy.Publisher (
      '/rhr_flex_model/preshape_1_position_controller/command',
      Float64, queue_size=5)
    preshape_2_pub = rospy.Publisher (
      '/rhr_flex_model/preshape_2_position_controller/command',
      Float64, queue_size=5)

    self.preshape_pubs = []
    self.preshape_pubs.append (preshape_1_pub)
    self.preshape_pubs.append (preshape_2_pub)


    # You can rostopic echo these commands to see they are being published.
    prox_1_pub = rospy.Publisher (
      '/rhr_flex_model/proximal_joint_1_position_controller/command',
      Float64, queue_size=5)
    prox_2_pub = rospy.Publisher (
      '/rhr_flex_model/proximal_joint_2_position_controller/command',
      Float64, queue_size=5)
    prox_3_pub = rospy.Publisher (
      '/rhr_flex_model/proximal_joint_3_position_controller/command',
      Float64, queue_size=5)

    self.prox_pubs = []
    self.prox_pubs.append (prox_1_pub)
    self.prox_pubs.append (prox_2_pub)
    self.prox_pubs.append (prox_3_pub)


    # Command line example:
    #   $ rostopic pub /rhr_flex_model/distal_joint_1_position_ctroller/command std_msgs/Float64 1.0
    dist_1_pub = rospy.Publisher (
      '/rhr_flex_model/distal_joint_1_position_controller/command',
      Float64, queue_size=5)
    dist_2_pub = rospy.Publisher (
      '/rhr_flex_model/distal_joint_2_position_controller/command',
      Float64, queue_size=5)
    dist_3_pub = rospy.Publisher (
      '/rhr_flex_model/distal_joint_3_position_controller/command',
      Float64, queue_size=5)

    self.dist_pubs = []
    self.dist_pubs.append (dist_1_pub)
    self.dist_pubs.append (dist_2_pub)
    self.dist_pubs.append (dist_3_pub)



  # Entry point
  # If invalid command, return flag = True
  def command_smarts (self, action):

    if action == 'close':
      self.close ()
      return False

    elif action == 'open':
      self.open ()
      return False

    elif action == 'spherical':
      self.set_spherical ()
      return False

    elif action == 'cylinder':
      self.set_cylindrical ()
      return False

    elif action == 'pinch':
      self.set_pinch ()
      return False

    elif action == 'guarded_move':
      self.guarded_move ()
      return False

    else:
      return True


  # Commands the specified finger's proximal joint to move to the specified
  #   position. Distal joint position is calculated and command accordingly.
  # Function signature from reflex_base.py. Function content custom for Gazebo
  # finger_index is 0 based
  def move_finger (self, finger_index, goal_proximal_pos):

    # Publish msgs to Gazebo ros controller, to move finger

    #####
    # Proximal
    #####
    # Command line equivalent:
    #   $ rostopic pub -1 /rhr_flex_model/preshape_1_position_controller/command std_msgs/Float64 "data: 1"
    cmd_p = Float64 ()
    # Cap at max
    cmd_p.data = min (goal_proximal_pos, self.TENDON_MAX)
    self.prox_pubs [finger_index].publish (cmd_p)


    #####
    # Simulate distal joint, based on how much we're moving proximal joint
    #   (no calibration, just a crude estimation)
    #####

    dist_rads = 0.0

    # Want to move distal joint 3 degs, for every 10 proximal joint degs that
    #   are over 60 degs. That is 3 distal / 10 prox = 0.3 distal / 1 prox
    DIST_DEGS_PER_PROX_DEG = 0.3

    prox_degs = cmd_p.data * (180.0 / np.pi)

    if prox_degs >= 60.0:
      dist_degs = DIST_DEGS_PER_PROX_DEG * (prox_degs - 60.0)
      dist_rads = dist_degs * (np.pi / 180.0)


    #####
    # Distal
    #####
    cmd_d = Float64 ()
    cmd_d.data = dist_rads
    self.dist_pubs [finger_index].publish (cmd_d)


    print ('Moved finger %d to proximal %f, distal %f' % (finger_index + 1,
      cmd_p.data, cmd_d.data))


  # Commands the preshape joint to move to a certain position
  # Function signature from reflex_base.py. Function content custom for Gazebo
  def move_preshape (self, goal_pos):

    cmd = Float64 ()

    # Finger 2 moves negative, finger 1 moves positive. Both move at same time.
    #   Spool is a single joint.
    cmd.data = min (goal_pos, self.PRESHAPE_MAX)
    self.preshape_pubs [0].publish (cmd)

    cmd.data = -cmd.data
    self.preshape_pubs [1].publish (cmd)

    rospy.loginfo ('Moved preshape to %f' % (goal_pos))


  # reflex_smarts.py defines open(), close(), tighten(), loosen() for single
  #   fingers. Here I'll just define them to do all 3 fingers togehter, for
  #   simplicity.
  # Copied from reflex-ros-pkgs/reflex/src/reflex/reflex_smarts.py
  def open (self):
    # Close each finger
    for finger_index in range (0, 3):

      rospy.loginfo("reflex_smarts: Opening finger %d", finger_index + 1)
      self.move_finger (finger_index, OPEN)
 

  # Copied from reflex-ros-pkgs/reflex/src/reflex/reflex_smarts.py
  def close (self):
    # Close each finger
    for finger_index in range (0, 3):

      rospy.loginfo("reflex_smarts: Closing finger %d", finger_index + 1)
      self.move_finger (finger_index, self.TENDON_MAX)


  '''
  # Copied from reflex-ros-pkgs/reflex/src/reflex/reflex_smarts.py
  # spool means tendon length, for opening and closing fingers. Essentially
  #   it is proximal joint + distal joint, which are connected by a tendon.
  #   Only proximal is actively moved by a motor, so you can think of spool as
  #   mostly affecting proximal joint. (Distal gets pulled by the spool when
  #   it is tight.)
  def tighten (self, spool_delta=HOW_HARDER):
    # Tighten each finger
    for finger_index in range (0, 3):

      rospy.loginfo("reflex_smarts: Tighten finger %d", finger_index + 1)

      # TODO Ehhhh can't do this. Gazebo only has C++ plugin interface, not Python!
      #   so no way I can retrieve even just the current position of proximal
      #   joint from here. Would need a C++ node to do that, and publish it in
      #   a new topic like /reflex_gazebo/joints, and then have my
      #   reflex_driver_node.py Python file subscribe to it, and package it
      #   into the simualted /reflex_hand rostopic that I publish.
      # Get a simulated "spool" from Gazebo.
      # Since we don't have any tendons in Gazebo, this will have to be
      #   computed from joint position of proximal joint. Look for it in
      #   reflex_driver_node.cpp, which populates the /reflex_hand Hand msg.
      #curr_spool = 

      self.move_finger(finger_index,
                       curr_spool + spool_delta)
                       # Real hand gets it from /reflex_hand msg, which has
                       #   spool populated probably using info from motors.
                       #self.hand.finger[finger_index].spool + spool_delta)


  # Copied from reflex-ros-pkgs/reflex/src/reflex/reflex_smarts.py
  def loosen (self, spool_delta=HOW_HARDER):
    # Loosen each finger
    for finger_index in range (0, 3):

      rospy.loginfo("reflex_smarts: Loosen finger %d", finger_index + 1)
      self.move_finger(finger_index,
                       #self.hand.finger[finger_index].spool - spool_delta)
                       curr_spool - spool_delta)
  '''


  # Copied from reflex-ros-pkgs/reflex/src/reflex/reflex_smarts.py
  def set_cylindrical (self):
    rospy.loginfo("reflex_smarts: Going to cylindrical pose")
    self.move_preshape (ROT_CYL)

  # Copied from reflex-ros-pkgs/reflex/src/reflex/reflex_smarts.py
  def set_spherical (self):
    rospy.loginfo("reflex_smarts: Going to spherical pose")
    self.move_preshape(ROT_SPH)

  # Copied from reflex-ros-pkgs/reflex/src/reflex/reflex_smarts.py
  def set_pinch (self):
    rospy.loginfo("reflex_smarts: Going to pinch pose")
    self.move_preshape(ROT_PINCH)


  # Returns after all fingers feel at least one contact, or after MAX_TIME
  def guarded_move (self):

    # Record proximal joints' positions before guarded_move starts
    prox_pos_start = [self.hand.finger [0].proximal,
      self.hand.finger [1].proximal, self.hand.finger [2].proximal]
    prox_pos_curr = deepcopy (prox_pos_start)

    # Record which finger has stopped closing
    stopped = []
    for i in range (0, self.N_FIN):
      stopped.append (False)

    # Enforce a timeout in simulation, `.` less noise, a finger might not have
    #   any sensor fire at all, we don't want this service to hang forever.
    # Seconds
    MAX_TIME = 5.0
    start_time = rospy.Time.now()


    # Keep moving and checking contact sensor values, until all 3 fingers have
    #   stopped moving, or fingers are all the way closed already (proximal >=
    #   TENDON_MAX)
    while not all (stopped):

      # If exceeded alotted time, return anyway
      curr_time = rospy.Time.now()
      if (curr_time - start_time).to_sec () >= MAX_TIME:
        rospy.loginfo ('More than %f has elapsed, giving up guarded_move' % \
          (MAX_TIME))
        break

      # Check fingers one by one
      for i in range (0, self.N_FIN):

        #print stopped

        # If this finger has already been stopped, skip right away
        if stopped [i]:
          continue

        # If this finger has been closed to limit, stop closing
        if prox_pos_curr [i] >= self.TENDON_MAX:
          print ('Finger %d closed to max' % (i+1))
          stopped [i] = True
          continue

        # If any sensors in this finger are activated, stop closing
        if any (self.hand.finger [i].contact):
          print ('Stopping finger %d on contact' % (i+1))
          stopped [i] = True
          continue
 
        # Else keep enclosing finger
        else:
          # Increment proximal joint position
          prox_pos_curr [i] += self.FINGER_STEP_HUGE

          # Command proximal goal position. Distal will follow automatically
          self.move_finger (i, prox_pos_curr [i])

      # Seconds. Sleep a bit, to allow contact msgs to come in. Else close too
      #   quickly, hand already all the way closed before reacting to contacts!
      # < 0.1 if want smooth motion
      # Increase if want slower, but motion will be jagged. You'd have to
      #   decrease FINGER_STEP to make it not jagged.
      # If want slower closing, just make FINGER_STEP smaller!
      rospy.sleep (0.03)

    print ('guarded_move returning')

 

# Copied and slightly modified from reflex_base_services.py
class CommandSmartService:

  def __init__ (self, obj):
    self.obj = obj

    # If hand is being moved, don't allow other commands to move it
    # Copied feature from reflex_base_services.py
    self.locked = False

  def __call__ (self, req):

    rospy.loginfo("reflex_base:CommandSmartService:")

    if self.locked:
      rospy.loginfo("\tService locked at the moment (in use), try later")
      return CommandHandResponse ('', -1, rospy.Time())

    else:
      self.locked = True
      rospy.loginfo("\tRequested action %s is about to run", req.action)
  
      start_time = rospy.Time.now()
      # Make the movement
      flag = self.obj.command_smarts (req.action)
      end_time = rospy.Time.now()
      elapsed_time = end_time - start_time
      self.locked = False

      if flag:
        parse_response = 'ERROR: Given command was unknown'
      else:
        parse_response = 'Command parsed'

      # Ret vals are defined in CommandHand.srv:
      #   string parse_response
      #   int32 result
      #   duration elapsed_time
      return CommandHandResponse (parse_response, 1, elapsed_time)


# Copied and slightly modified from reflex_base_services.py
class MoveFingerService:

  def __init__ (self, obj):
    self.obj = obj
    self.locked = False

  def __call__ (self, req):

    rospy.loginfo ('reflex_base:MoveFingerService:')

    if self.locked:
      rospy.loginfo("\tService locked at the moment (in use), try later")
      return MoveFingerResponse (0, -1)

    else:
      self.locked = True
      start_time = rospy.Time.now()

      if req.finger_index < 0 or req.finger_index > 2:
        rospy.logwarn("Commanded finger index %d, only [0:2] allowed",
          req.finger_index)

      elif req.goal_pos < self.obj.TENDON_MIN \
        or req.goal_pos > self.obj.TENDON_MAX:
        rospy.logwarn("Commanded goal pos %f radians, [%f:%f] allowed",
          req.goal_pos, self.obj.TENDON_MIN, self.obj.TENDON_MAX)

      else:
        rospy.loginfo("\treflex_f%d is moving to %f radians",
          req.finger_index + 1, req.goal_pos)
        self.obj.move_finger(req.finger_index, req.goal_pos)

      end_time = rospy.Time.now()
      self.locked = False

      return (1, end_time - start_time)


# Copied and slightly modified from reflex_base_services.py
class MovePreshapeService:

  def __init__(self, obj):

    self.obj = obj
    self.locked = False

  def __call__(self, req):

    rospy.loginfo("reflex_base:MovePreshapeService:")

    if self.locked:
      rospy.loginfo("\tService locked at the moment (in use), try later")
      return (0, -1)

    else:
      self.locked = True
      start_time = rospy.Time.now()

      if req.goal_pos < self.obj.PRESHAPE_MIN\
         or req.goal_pos > self.obj.PRESHAPE_MAX:
         rospy.logwarn("Commanded goal pos %f radians, [%f:%f] allowed",
           req.goal_pos, self.obj.PRESHAPE_MIN, self.obj.PRESHAPE_MAX)

      else:
        rospy.loginfo("\tpreshape moving to %f radians", req.goal_pos)
        self.obj.move_preshape(req.goal_pos)

      end_time = rospy.Time.now()
      self.locked = False

      return (1, end_time - start_time)



# Copied from reflex-ros-pkgs/reflex/src/reflex/reflex_smarts.py
if __name__ == '__main__':

  # Start the whole command smarts server. Individual commands are handled by
  #   different methods, but all enter from CommandHand service type.

  rospy.init_node('ReflexServiceNode')
  reflex_hand = ReFlex_Smarts()

  sh1 = CommandSmartService(reflex_hand)
  s1 = "/reflex/command_smarts"
  rospy.loginfo("Advertising %s service", s1)
  s1 = rospy.Service(s1, CommandHand, sh1)

  sh2 = MoveFingerService(reflex_hand)
  s2 = "/reflex/move_finger"
  rospy.loginfo("Advertising %s service", s2)
  s2 = rospy.Service(s2, MoveFinger, sh2)

  sh3 = MovePreshapeService(reflex_hand)
  s3 = "/reflex/move_preshape"
  rospy.loginfo("Advertising %s service", s3)
  s3 = rospy.Service(s3, MovePreshape, sh3)

  rospy.spin ()

