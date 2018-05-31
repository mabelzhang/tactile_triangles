#!/usr/bin/env python

# Mabel Zhang
# 3 Mar 2015
#
# During manual exploration of an object (i.e. using human hand to guide the
#   ReFlex Hand on the object surface), plots:
#   - a marker at anywhere a pressure sensor exceeds a threshold. Per iteration
#   - a point cloud of all such sensor positions. Cumulative.
#   - a normal marker in -z direction of sensor, with log-scale magnitude in
#     terms of the pressure value. Cumulative through entire run.
# Pressure sensor values are retrieved from rostopic /reflex_hand.
#
# Markers are plotted in the frame of the tactile sensor (!), not the robot.
#   Currently, robot isn't calibrated to hand. This excludes calibration errors
#   and puts focus on the hand itself. This allows us to see how much noise
#   there is in the hand itself, excluding hand-to-robot calibration noise.
# PointCloud is published. All points are same coordinates and frame as
#   Marker.
# Difference btw the two:
#   Markers only keep 1 frame of information, at any instance in time, same
#     info as /reflex_hand published by ReFlex Hand driver.
#     Useful for checking whether my code detects the correct place on hand
#     that is touched, when it is touched (set marker duration to small).
#   PointCloud keeps cumulative information throughout entire run.
#     Useful for reconstructing entire object, and visualizing afterwards in
#     PCD file (rosrun pcl_ros pointcloud_to_pcd input:=/topic).
#
# To pause plotting (publishing cloud and normals)
#   $ rostopic pub /tactile_map/pause std_msgs/Bool 1
#   $ rostopic pub /tactile_map/pause std_msgs/Bool 0
# This is useful when you need to move the hand around the object before next
#   exploration on the same object. You wouldn't want to record when you touch
#   the hand to move it, and it's not actually touching the object!
#


# ROS
import rospy
# http://docs.ros.org/api/visualization_msgs/html/msg/Marker.html
from visualization_msgs.msg import Marker
# http://mirror.umd.edu/roswiki/doc/diamondback/api/tf/html/python/tf_python.html
# http://wiki.ros.org/tf/TfUsingPython
import tf
#from geometry_msgs.msg import PointStamped
#from geometry_msgs.msg import Point
from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud
from std_msgs.msg import Bool

# ReFlex pkg
from reflex_msgs.msg import Hand

# Python
import math

# Local
from tactile_map.create_marker import create_marker
from tactile_map.get_contacts import get_contacts_live


class ManualExplore:

  def __init__ (self):

    ########
    # Constants

    self.contact_thresh = 100


    ########
    # ROS stuff

    self.tfTrans = tf.TransformListener ()

    self.vis_pub = rospy.Publisher ('/visualization_marker', Marker)
    self.cloud_pub = rospy.Publisher ('/tactile_map/contact/cloud',
      PointCloud)

    self.msg_id = 0
    # Note this is Base_link of ReFlex hand, not of whatever else.
    #   reflex_tf_broadcaster broadcasts a capitalized Base_link. Its parent is
    #   the lowercase base_link in URDF, published by robot_state_publisher.
    #self.marker_frame = '/base_link'
    # Use Baxter torso frame instead of hand frame, so that cloud and normals
    #   don't move around with the hand each time they're republished!
    self.marker_frame = '/base'

    # Cloud is CUMULATIVE for entire run.
    #   Marker is contacts at any ONE instance, from /reflex_hand.
    #   That is the difference btw the two.
    self.cloud = PointCloud ()
    self.cloud.header.frame_id = self.marker_frame

    # Surface normal arrow marker. CUMULATIVE for entire run.
    self.normals = Marker ()
    # Scale: LINE_LIST uses scale.x, width of line segments
    # LINE_LIST draws a line btw 0-1, 2-3, 4-5, etc. pairs of points[]
    # Pose is still used, just like POINTS still uses them. Make sure to set
    #   identity correctly, qw=1.
    create_marker (Marker.LINE_LIST, 'contacts_normals', self.marker_frame, 
      self.msg_id,
      0, 0, 0, 1, 1, 0, 0.5, 0.002, 0, 0,
      self.normals, 0)  # Use 0 duration for forever


    ########
    # Bookkeeping
    self.pause_recording = False


  def pauseCB (self, msg):
    self.pause_recording = msg.data

    if self.pause_recording:
      rospy.loginfo ('Contact detection PAUSED')
    else:
      rospy.loginfo ('Contact detection ENABLED')


  # Extract pressure sensor values
  # msg: reflex_msgs/Hand.msg
  #   fields:
  #     msg.palm.preshape: Joint angle. Scalar float
  #
  #     msg.palm.pressure: Pressure sensor values. 11-tuple float32
  #     msg.palm.contact: Booleans
  #
  #     msg.finger[finger_i].spool: Joint angle. Scalar float
  #     msg.finger[finger_i].proximal. Joint angle. Scalar float
  #     msg.finger[finger_i].distal. Joint angle. Scalar float
  #
  #     msg.finger[finger_i].pressure. Pressure sensor values. 9-tuple float32
  #     msg.finger[finger_i].contact. 9-tuple Booleans
  def handCB (self, msg):

    if self.pause_recording:
      return

    self.msg_id += 1

    # Use one marker for this entire msg. Add multiple points to it later
    #   Parameters: tx, ty, tz, r, g, b, alpha, sx, sy, sz
    marker_p = Marker ()
    create_marker (Marker.POINTS, 'contacts', self.marker_frame, self.msg_id,
      0, 0, 0, 1, 1, 0, 0.5, 0.01, 0.01, 0.01,
      marker_p, 60)#0)  # Use 0 duration for forever

    # Init points array to empty list
    # rospy msg "are deserialized as tuples for performance reasons, but you
    #   can set fields to tuples and lists interchangeably".
    # Ref: http://wiki.ros.org/msg
    marker_p.points = []

    # Comment out to keep cumulative cloud for whole run.
    #   Uncomment to look at cloud at one short instance.
    #self.cloud.points = []
    # Update next msg's seq, even if no new pts are added (in which case the
    #   timestamp of cloud won't change, by design), to let user know we're
    #   running normally.
    self.cloud.header.seq = self.msg_id
    self.normals.header.seq = self.msg_id


    # Get coordinates of sensors whose pressure values exceed threshold
    [contacts, normal_endpts, _, _, _] = get_contacts_live (msg,
      self.contact_thresh, self.marker_frame, self.tfTrans, True)
    self.add_points (marker_p, contacts, self.normals, normal_endpts)


    if len (marker_p.points) > 0:
      self.vis_pub.publish (marker_p)

      # Init point cloud msg
      #   API: http://docs.ros.org/api/sensor_msgs/html/msg/PointCloud.html
      # Including this in the if-stmt has advantage: then when you save to
      #   .pcd file, instead of saving a ton of the same one, it will overwrite
      #   all files with same timestamp. So you get unique files, don't need
      #   to dig through and delete repetitive ones.
      self.cloud.header.stamp = rospy.Time.now ()
      # Cloud will be published in next loop in __main__. Don't publish here,
      #   to save callback function lag time before receiving next msg in queue

      # Init normals msg
      self.normals.header.stamp = rospy.Time.now ()


    # To see whether marker is plotted in correct x y quadrant wrt base_link
    '''
    test_marker = Marker ()
    create_marker (Marker.POINTS, 'test', '/base_link', 1,
      0, 0, 0, 0, 1, 0, 0.5, 0.01, 0.01, 0.01,
      test_marker)
    test_marker.points = []
    test_point = Point ()
    test_point.x = 0.1
    test_point.y = 0.1
    test_point.z = 0
    test_marker.points.extend ([test_point])
    self.vis_pub.publish (test_marker)
    '''


  # Add contact point to Marker (current frame) and cloud (cumulative)
  # Parameters:
  #   pts: geometry_msgs/Point[]. Point on object surface (really, point on
  #     sensor. But we make naive assumption here that it's also the point on
  #     object surface).
  #   norm_endpts: geometry_msgs/Point[]. Normal of surface points from pts[i]
  #     on obj surface, to norm_endpts[i] in space.
  #   pts and norm_endpts should have same number of elts and corresponding idx.
  def add_points (self, marker_p, pts, marker_n, norm_endpts):

    for i in range (0, len (pts)):

      pt = pts [i]

      # Add point in /base_link frame to Marker. Marker stores new pts each iter
      # Marker.points is type geometry_msgs/Point[]
      marker_p.points.extend ([pt])
     
      # 1 mm
      point_exist_thresh = 0.001
      point_exists = False
     
      # Check if this point is already in cloud, by some threshold.
      #   This saves memory, prevents lag at run time. Cloud store cumulative pts
      for j in range (0, len (self.cloud.points)):
        # Euclidean distance
        if (math.sqrt ((self.cloud.points [j].x - pt.x) ** 2 +
                  (self.cloud.points [j].y - pt.y) ** 2 +
                  (self.cloud.points [j].z - pt.z) ** 2)) < point_exist_thresh:
          point_exists = True
          break
     
      if not point_exists:
        # Add point to cloud
        pt_32 = Point32 ()
        pt_32.x = pt.x
        pt_32.y = pt.y
        pt_32.z = pt.z
        self.cloud.points.extend ([pt_32])
     
        # Add normal
        # Start pt is just the contact point
        marker_n.points.extend ([pt])
        # End pt is z=-1
        marker_n.points.extend ([norm_endpts [i]])


  # Publish point cloud of contacts
  def publish_cloud (self):
    if len (self.cloud.points) > 0:
      #rospy.loginfo ('# points in contact cloud: ' + str(len(self.cloud.points)))
      self.cloud_pub.publish (self.cloud)

      # Publish the normals
      self.vis_pub.publish (self.normals)


if __name__ == '__main__':

  rospy.init_node ('manual_explore', anonymous=True)

  thisNode = ManualExplore ()

  rospy.Subscriber ('/reflex_hand', Hand, thisNode.handCB)

  rospy.Subscriber ('/tactile_map/pause', Bool, thisNode.pauseCB)


  wait_rate = rospy.Rate (10)
  while not rospy.is_shutdown ():

    thisNode.publish_cloud ()

    try:
      wait_rate.sleep ()
    except rospy.exceptions.ROSInterruptException, err:
      break


