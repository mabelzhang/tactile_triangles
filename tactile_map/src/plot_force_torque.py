#!/usr/bin/env python

# Mabel Zhang
# 5 Apr 2015
#
# Plot RViz arrows in directions of force and torque sensed in Baxter wrist.
#   Arrow lengths are in log scale.
#
# Force and torque are extracted from this rostopic. You can inspect the values
#   manually:
#   $ rostopic echo /robot/limb/left/endpoint_state
#

# ROS
import rospy
import tf
from geometry_msgs.msg import Wrench
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

# Baxter
from baxter_core_msgs.msg import EndpointState

# Python
import sys
import math
import numpy as np

# Local
from tactile_map.create_marker import create_marker, create_arrow
from tactile_map.tf_get_pose import tf_get_pose


class PlotForceTorque:

  def __init__ (self, arm_side):

    self.arm_side = arm_side

    rospy.Subscriber ('/robot/limb/' + arm_side + '/endpoint_state',
      EndpointState, self.endptCB)

    self.endpt_pos = Point ()
    self.wrench = Wrench ()

    self.vis_pub = rospy.Publisher ('/visualization_marker', Marker)
    # Use 0 duration for forever
    self.marker_duration = 60

    # Are we in simulation?
    self.use_sim_time = False
    if rospy.has_param ('/use_sim_time'):
      # Ref: http://wiki.ros.org/Clock
      self.use_sim_time = rospy.get_param ('/use_sim_time')
    rospy.loginfo ('use_sim_time found to be ' + str (self.use_sim_time))

    self.tfTrans = tf.TransformListener ()


  # Callback function for subscriber of endpoint state
  def endptCB (self, msg):

    self.endpt_pos = Point (msg.pose.position.x, msg.pose.position.y,
      msg.pose.position.z)
    self.wrench = Wrench (msg.wrench.force, msg.wrench.torque)

    # Debug. Seems to all be 0 in simulation. Maybe `.` I didn't enable robot..
    #if (msg.wrench.force.x != 0) and (msg.wrench.force.y != 0) and \
    #  (msg.wrench.force.z != 0):
    #  print (msg.wrench.force)


  # Plot in log scale length. Else too big
  # Parameters:
  #   force: scalar raw magnitude (of force or torque vector)
  # Returns scalar magnitude in log scale.
  def calc_arrow_magnitude (self, raw):

    # log(x+1) to shift 1 to the left, so 0 force is 0 length.
    # Use absolute value of force, so pos and neg forces have same scale.
    # Use a large log base, so that the log is shallower. Then force in z
    #   direction to counter gravity doesn't dominate too much.
    return math.log (abs (raw) + 1, 100) / 10.0


  # Parameters:
  #   vec: geometry_msgs/Vector3 type. Force or torque vector wrt /base, as
  #     received from Baxter rostopic /robot/limb/<side>/endpoint_state
  #   namespace, r, g, b: Marker properties
  #   text: text label for the arrow
  def plot_wrench_vec (self, vec, namespace, r, g, b, text):

    # Start wrench vector at robot endpoint
    # Instead of using tf transform, just get position of endpoint in
    #   endpoint_state.pose.position! Duh! That's why they provide it!
    #   That's already pose of endpoint wrt /base.
    # (tf_get_pose() throws error can't find transform. That's weird, I
    #   double guarded that function to make it work every time before.)
    arrow_start_np = np.asarray ([self.endpt_pos.x, self.endpt_pos.y,
      self.endpt_pos.z])

    # Calc unit vector
    vec_unit = np.asarray ([vec.x, vec.y, vec.z])
    vec_norm = np.linalg.norm (vec_unit)
    # If vector length is 0, plot an arrow with 0 length
    if vec_norm < 1e-6:
      arrow_len = 0

    else:
      vec_unit = vec_unit / vec_norm

      # Calc arrow length
      arrow_len = self.calc_arrow_magnitude (vec_norm)

    # Endpoint of wrench vector. Scale unit vector by length.
    # Add force vector (x y z) to start point (0 0 0) of wrist.
    # Force's x y z components define the force vector. It's simply a vector
    #   pointing toward (x y z) point.
    # Torque axis is simply the (x y z) vector, since they express it as a
    #   vector rather than a Quaternion. Makes life easier.
    arrow_end_np = arrow_start_np + vec_unit * arrow_len


    # Create RViz markers for wrench vector and text label
    marker_f = Marker ()
    marker_text = Marker ()
    create_arrow (arrow_start_np, arrow_end_np,
      namespace, '/base', [0, 1], r, g, b, 0.002, 0.004, 0, 0.02,
      text, self.marker_duration, [marker_f, marker_text])

    self.vis_pub.publish (marker_f)
    self.vis_pub.publish (marker_text)


  # Colors RGB by English color name: http://www.colorcombos.com/colors/FF0080
  def publish_markers (self):

    # fuschia for force
    self.plot_wrench_vec (self.wrench.force, 'wrist_force', 1, 0, 0.5, 'force')

    # teal for torque
    self.plot_wrench_vec (self.wrench.torque, 'wrist_torque', 0, 0.518, 0.510,
      'torque')


if __name__ == '__main__':

  rospy.init_node ('plot_force_torque', anonymous=True)

  arm_side = 'left'
  # Parse cmd line args
  for i in range (0, len (sys.argv)):
    if sys.argv[i] == '--left':
      arm_side = 'left'
    elif sys.argv[i] == '--right':
      arm_side = 'right'
  print ('Arm set to ' + arm_side + ' side')


  thisNode = PlotForceTorque (arm_side)

  wait_rate = rospy.Rate (10)
  while not rospy.is_shutdown ():

    thisNode.publish_markers ()

    try:
      wait_rate.sleep ()
    except rospy.exceptions.ROSInterruptException, err:
      break


