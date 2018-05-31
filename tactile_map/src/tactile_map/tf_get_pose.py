#!/usr/bin/env python

# Mabel Zhang
# 19 Mar 2015
#
# Extracted from tactile_map manual_explore.py
#
# Note: On Baxter desktop, if it says extrapolation into the future error, most
#   cases, this is because Baxter's clock is faster. Short-term fix is
#   $ ssh ruser@baxter.local
#   $ date
#   Then manually adjust the desktop computer's clock to match output of date:
#   $ sudo date --set "16 Jul 2015 19:57:00"
#   Note since it takes a bit to enter sudo password, you want to set a few
#     seconds in advance. Or just get a feel of how long 1 second is, by
#     repeatedly printing the "date" command, and hit enter after inputting
#     password, at the exact second you're setting.
#   Worked for me!
#
# Ref date --set:
#   http://askubuntu.com/questions/349763/how-can-i-change-the-date-and-time-on-ubuntu-12-04
#

# ROS
import rospy
import tf
# http://docs.ros.org/api/geometry_msgs/html/msg/PoseStamped.html
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped

# Python
import time


# Parameters:
#   use_common_time: Use the latest common time found btw two frames in tf. If
#     False, will just use rospy.Time.now ().
#     Recommend set to True. Only set to False if True doesn't work.
#     If in simulation, either could work, in different cases.
#       Note: in sim, even if rosparam /use_sim_time is true, common time
#       passed back from tf still appears to be in wall clock time!
#     If on real robot, definitely use common time!
# stamp: Supply a timestamp to use for tf transformPose(). If specified,
#   use_sommon_time is ignored. Currently no functions use this!!
# pose_or_point: True for tf.transformPose, False for tf.transformPoint().
#   Latter will ignore qx qy qz qw inputs.
# Returns a geometry_msgs/PoseStamped or geometry_msgs/PointStamped, depending
#   on pose_or_point is True or False, respectively.
def tf_get_pose (from_frame, to_frame, tx, ty, tz, qx, qy, qz, qw, listener,
  use_common_time=True, stamp=None, pose_or_point=True):

  # Construct input to tf transformPose()
  if pose_or_point:
    from_pose = PoseStamped ()
    from_pose.header.frame_id = from_frame
    from_pose.pose.position.x = tx
    from_pose.pose.position.y = ty
    from_pose.pose.position.z = tz
    from_pose.pose.orientation.x = qx
    from_pose.pose.orientation.y = qy
    from_pose.pose.orientation.z = qz
    from_pose.pose.orientation.w = qw
  else:
    from_pose = PointStamped ()
    from_pose.header.frame_id = from_frame
    from_pose.point.x = tx
    from_pose.point.y = ty
    from_pose.point.z = tz

  # This is by design of this function. If you dont' do this, will error
  #   at transformPose or transformPoint, because the wrong timestamp (now())
  #   is set, instead of the right one (common_time).
  if stamp is None:
    use_common_time = True

  # I think transformPose() is dangerous, because it could alternate
  #   the frame's orientation in pose.orientation, which we can't use
  #   in Marker.POINTS, because all POINTS must share the same 
  #   orientation.
  # So it's safer to use transformPoint() instead.
  '''
  from_pt = PointStamped ()
  from_pt.header.frame_id = from_frame
  from_pt.point.x = tx
  from_pt.point.y = ty
  from_pt.point.z = tz
  '''

  #print ('Trying to get tf from %s to %s' % (from_frame, to_frame)

  if stamp is not None:
    from_pose.header.stamp = stamp

    rospy.loginfo ('tf_get_pose(): Waiting for transform from %s to %s at timestamp %d.%d' \
      %(from_frame, to_frame, stamp.secs, stamp.nsecs))

    # Wait till it's there.
    # If it's never there, and you're running between ReFlex and Baxter, check
    #   your clocks. See this file's header comment. Did you sync your desktop
    #   clock to Baxter robot's?
    # This intermitently doesn't work, for some reason, even when desktop is
    #   synced to Baxter, and rosrun tf tf_echo spits out time 3 seconds later
    #   than the one I'm requesting, at same time this script is hanging on
    #   waitForTransform(). It's not clear why the tf arrived tf_echo but 
    #   doesn't arrive my waitForTransform()!! This is the case even if I wait
    #   10 seconds.
    #   Minutes before this kind of behavior, it would work on the fly. Cause
    #   of problem unclear.
    try:
      # This doesn't get killed by Ctrl+C, for some reason
      listener.waitForTransform (to_frame, from_frame, stamp,
        rospy.Duration (6.0))
    except tf.Exception, err:
      # This doesn't work if cache is empty. Just pass in None, faster
      #common_time = listener.getLatestCommonTime (to_frame, from_frame)
      print ('waitForTransform() did not receive transform for timestamp %d.%d, using latest common time instead' % 
        (stamp.secs, stamp.nsecs))
      stamp = None
      use_common_time = True
 
    # lookupTransform() returns a 3-tuple and 4-tuple
    # Python API: http://docs.ros.org/jade/api/tf/html/python/tf_python.html
    #(transl, rot) = listener.lookupTransform (to_frame, from_frame, common_time)

  # Changed to if None, so in case when provided stamp didn't work, we'll use
  #   a retrieved stamp
  #else:
  if stamp is None:

    # Get pose wrt to_frame
    # To test this in bare python shell:
    '''
    import rospy
    import tf
    rospy.init_node ('manual_explore', anonymous=True)
    listener = tf.TransformListener ()
    listener.frameExists ('/base_link')
    listener.frameExists ('/Proximal_1')
    '''
    # Sample code that does correct timing:
    #   2.1 TransformListener in http://wiki.ros.org/tf/TfUsingPython
    while (not (listener.frameExists (to_frame) and \
      listener.frameExists (from_frame))) and \
      (not rospy.is_shutdown ()):
 
      rospy.loginfo ('tf_get_pose(): Waiting for existence of frames %s and %s' \
        %(from_frame, to_frame))
 
      try:
        time.sleep (0.1)
      except rospy.exceptions.ROSInterruptException, err:
        break
    #print ('got tf')
 
    # Set pose's timestamp to common time of the two frames
    try:
      common_time = listener.getLatestCommonTime (to_frame, from_frame)
    except tf.Exception, err:
      rospy.logerr (err)
      return None


    #'''
    # Sometimes even when the above passed, still get error cache is empty.
    #   Need to use canTransform().
    #   est_center_axis_ransac.py needs this on physical robot. Pass in
    #   use_common_time = True.
    # Ref: http://answers.ros.org/question/27699/wait-for-tf-listern-queue-to-not-be-empty/
    while (not listener.canTransform (to_frame, from_frame, common_time)) and \
      (not rospy.is_shutdown ()):
 
      rospy.loginfo ('tf_get_pose(): Waiting for transformation from %s to %s' \
        %(from_frame, to_frame))
 
      try:
        time.sleep (0.1)
        # Update time to a newer time
        common_time = listener.getLatestCommonTime (to_frame, from_frame)
      except rospy.exceptions.ROSInterruptException, err:
        break
    #'''

    if use_common_time:
      from_pose.header.stamp = common_time
      #from_pt.header.stamp = common_time
    else:
      from_pose.header.stamp = rospy.Time.now ()

    # Debug if getLatestCommonTime() gives wall time even in simulation
    #print ('common time: %d.%d' %(common_time.secs, common_time.nsecs))
    #print ('ros time now:')
    #print (rospy.Time.now ())


  # Note Python doesn't have an API that takes 5 params with the timestamp.
  if pose_or_point:
    to_pose = listener.transformPose (to_frame, from_pose)
  else:
    to_pose = listener.transformPoint (to_frame, from_pose)

  # Debug
  #print ('to_pose.header.frame_id: %s' % (to_pose.header.frame_id))
  #rospy.loginfo (to_pose.pose.position)
  #rospy.loginfo (to_pt.point)

  return to_pose


# Overloaded function. Accepts a Pose object instead of 7 separate floats
# Parameters:
#   pose: geometry_msgs/PoseStamped
def tf_get_pose_Pose (from_frame, to_frame, pose, listener,
  use_common_time=True, stamp=None):

  return tf_get_pose (from_frame, to_frame,
    pose.position.x, pose.position.y, pose.position.z,
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w,
    listener, use_common_time, stamp)


# Overloaded function. Accepts a Point object instead of 7 separate floats.
#   Quaternion values set to 0 0 0 w=1 identity.
# Parameters:
#   point: geometry_msgs/PointStamped
def tf_get_pose_Point (from_frame, to_frame, point, listener,
  use_common_time=True, stamp=None):

  return tf_get_pose (from_frame, to_frame,
    point.x, point.y, point.z, 0, 0, 0, 1,
    listener, use_common_time, stamp, pose_or_point=False)


# A call to lookupTransform(), with waitForTransform wrapped around
#   Ref: http://wiki.ros.org/tf/Tutorials/tf%20and%20Time%20%28Python%29
# Note the "reverse" in arguments from intuition. To get transformation of
#   obj frame (which is (0 0 0) in obj frame) wrt robot frame (which would
#   be (xo yo zo) in robot frame), the "source" frame should be obj,
#   "target" frame is robot. This is the correct way of reasoning.
#   It's not the intuitive thinking where you want to get a transform
#   "from robot" "to object" (robot -> obj).
#   It's a transform "of obj" "wrt robot" (obj -> robot)
#
#   tf_echo prints this as well:
#   $ rosrun tf tf_echo
#   Usage: tf_echo source_frame target_frame
#   This will echo the transform from the coordinate frame of the source_frame
#     to the coordinate frame of the target_frame. 
#   Note: This is the transform to get data from target_frame into the
#     source_frame.
#
#   So the target and source you pass to lookupTransform() and transformPose()
#   are reversed.
# Returns 7-tuple (3-vec translation tx ty tz, 4-vec rotation qx qy qz qw)
#   Ref: http://docs.ros.org/jade/api/tf/html/python/tf_python.html
# To test on bare Python shell:
'''
import rospy
import tf
from tactile_map.tf_get_pose import tf_get_transform
rospy.init_node ('manual_explore', anonymous=True)
listener = tf.TransformListener ()

tf_get_transform ('/base', '/left_gripper', listener)
'''
def tf_get_transform (from_frame, to_frame, listener, stamp=None):

  if stamp is not None:
    common_time = stamp

  # If stamp is not provided, look for the latest common time between two
  #   frames. This is better than simply using rospy.Time.now() and a timeout
  #   using lookupTransform(), because this deals with time difference btw
  #   two machines better. With our Baxter's timestamp always faster than the
  #   desktop computer, lookupTransform() never works, because rospy.Time.now()
  #   sent from the desktop is always extrapolation into the past for more
  #   than 10 seconds (default length of buffer that tf keeps), since desktop
  #   time is so much slower than robot time. Rather than changing tf buffer
  #   length, just write code that works for a more general case.
  else:

    # Get pose wrt to_frame
    # To test this in bare python shell:
    '''
    import rospy
    import tf
    rospy.init_node ('manual_explore', anonymous=True)
    listener = tf.TransformListener ()
    listener.frameExists ('/base_link')
    listener.frameExists ('/Proximal_1')
    '''
    # Sample code that does correct timing:
    #   2.1 TransformListener in http://wiki.ros.org/tf/TfUsingPython
    while (not (listener.frameExists (to_frame) and \
      listener.frameExists (from_frame))) and \
      (not rospy.is_shutdown ()):
 
      #rospy.loginfo ('tf_get_pose(): Waiting for existence of frames %s and %s' \
      #  %(from_frame, to_frame))
 
      try:
        time.sleep (0.1)
      except rospy.exceptions.ROSInterruptException, err:
        break
    #print ('got tf')

    # Set pose's timestamp to common time of the two frames
    #common_time = listener.getLatestCommonTime (to_frame, from_frame)


    # lookupTransform should be equivalent of this, since this function we just
    #   need to lookup, don't need to transformPose. Tested script without this
    #   and worked. If doesn't work later, uncomment this.

    # Sometimes even when the above passed, still get error cache is empty.
    #   Need to use canTransform().
    #   weights_collect.py needs this on physical robot. getLatestCommonTime()
    #   above gives separate-tree errors! waitForTransform() below doesn't
    #   work either, says extrapolation into past, even when Baxter desktop
    #   time is synced to robot.
    # Ref: http://answers.ros.org/question/27699/wait-for-tf-listern-queue-to-not-be-empty/
    common_time = rospy.Time.now ()
    while (not listener.canTransform (to_frame, from_frame, common_time)) and \
      (not rospy.is_shutdown ()):
 
      #rospy.loginfo ('tf_get_pose(): Waiting for transformation from %s to %s' \
      #  %(from_frame, to_frame))
 
      try:
        time.sleep (0.1)
        # Update time to a newer time
        common_time = listener.getLatestCommonTime (to_frame, from_frame)
      except rospy.exceptions.ROSInterruptException, err:
        break


  # To test lookupTransform() in bare Python shell:
  '''
  import rospy
  import tf
  rospy.init_node ('manual_explore', anonymous=True)
  listener = tf.TransformListener ()

  stamp = rospy.Time.now ()
  # The constant is the delay btw Baxter robot and desktop station. If get error saying extrapolation into the past, compare the two stamps in error message and adjust this offset.
  stamp.secs += 22
  listener.waitForTransform ('/left_gripper', '/base', stamp, rospy.Duration(4.0))
  (trans, rot) = listener.lookupTransform ('/left_gripper', '/base', stamp)
  '''
  #common_time = rospy.Time.now ()
  #listener.waitForTransform (to_frame, from_frame, common_time,
  #  rospy.Duration (4.0))

  # lookupTransform() returns a 3-tuple and 4-tuple
  # Python API: http://docs.ros.org/jade/api/tf/html/python/tf_python.html
  (transl, rot) = listener.lookupTransform (to_frame, from_frame, common_time)

  # Debug to see if correct
  #print ('From %s to %s:' % (from_frame, to_frame))
  #print (transl)

  # Return 7-tuple. Concatenate 3-vec translation and 4-vec rotation.
  return (transl + rot)


