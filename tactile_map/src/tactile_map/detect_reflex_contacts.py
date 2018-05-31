#!/usr/bin/env python

# Mabel Zhang
# 11 Jul 2015
#
# Refactored code that creates a standalone node. Subscribes to pressure sensor
#   values on ReFlex rostopic /reflex_hand, and publishes Contacts msg when it
#   detects pressure values significantly far from 0 (set by a threshold)
#   that indicates a contact.
#
# On Baxter with ReFlex Hand plugged in and connected:
# In each terminal:
# $ baxter
# $ . ~/.bashrc
# Terminal 1:
# $ roslaunch baxter_reflex baxter_reflex.launch
# Terminal 2:
# $ rosrun tactile_map detect_reflex_contacts.py
#

import roslib; roslib.load_manifest ('tactile_map')

# ROS
import rospy
import tf
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose
from std_msgs.msg import ColorRGBA

# ReFlex pkg
from reflex_msgs.msg import Hand

# Local
from tactile_map.get_contacts import get_contacts_live, eliminate_duplicates
from tactile_map_msgs.msg import Contacts
from tactile_map.create_marker import create_marker
from tactile_map.tf_get_pose import tf_get_pose_Point


class DetectReFlexContacts:

  def __init__ (self):

    self.tfTrans = tf.TransformListener ()
    self.bc = tf.TransformBroadcaster ()

    # Tells us what hardware is on. If not on, we don't need to do tf to that
    #   frame.
    self.handOn_topic = '/detect_reflex_contacts/hand_on'
    rospy.Subscriber (self.handOn_topic, Bool, self.handOnCB)
    rospy.Subscriber ('/detect_reflex_contacts/cam_on', Bool, self.camOnCB)
    rospy.Subscriber ('/detect_reflex_contacts/robot_on', Bool, self.robotOnCB)
    self.handOn = True
    self.camOn = False
    self.robotOn = True

    rospy.Subscriber ('/reflex_hand', Hand, self.handCB)
    self.handMsg = None
    # Use 0 if want to record EVERYTHING, but that's usually a bad idea. Lots
    #   of sparse useless data.
    self.contact_thresh = 10  #15

    # Tells us that hand is in movement. Do not look at current tactile values
    #   or finger positions as contact points.
    #   This is important because hand moves really fast. Delay in code is 
    #   longer than delay in hand opening. So you might be looking at contact
    #   points a moment ago, and now looking up tf, but the hand has started
    #   moving, so you end up getting a point in mid-air from tf, because
    #   that's where the finger was in that split second, but that's not where
    #   the contact was.
    rospy.Subscriber ('/tactile_map/pause', Bool, self.pauseCB)
    self.pause_contacts = False

    self.contacts_pub = rospy.Publisher ( \
      '/tactile_map/detect_reflex_contacts/contacts', Contacts, queue_size=5)
    self.contacts_seq = 0

    self.vis_pub = rospy.Publisher ('/visualization_marker', Marker,
      queue_size=2)
    self.vis_arr_pub = rospy.Publisher ('/visualization_marker_array',
      MarkerArray, queue_size=2)
    self.text_marker_ids_seen = 0

    # [0] is palm, [1:3] is fingers 0 to 2
    #self.pressures_latest = [[0] * 11, [0] * 9, [0] * 9, [0] * 9]
    self.sensor_marker_ids = [range(0,11), range(11,20), range(20,29),
      range(29,38)]

    # int[]
    self.existing_individual_marker_ids = []


  def handOnCB (self, msg):
    self.handOn = msg.data

  def camOnCB (self, msg):
    self.camOn = msg.data

  def robotOnCB (self, msg):
    self.robotOn = msg.data


  def handCB (self, msg):

    self.handMsg = msg


  def pauseCB (self, msg):

    if self.pause_contacts != msg.data:
      if msg.data:
        print ('Pause contact detection')
      else:
        print ('Resume contact detection')

    self.pause_contacts = msg.data


  # Publishes Contacts message
  # If no contacts are above threshold, the Pose[] lists in msg will be empty.
  # Ref: http://wiki.ros.org/msg
  def detect_contacts (self):

    if not self.handOn:
      print ('Hand is off, according to rostopic %s. Nothing to do.' % (self.handOn_topic))
      return

    if self.handMsg is None:
      print ('Waiting for first /reflex_hand msg...')
      return


    msg = Contacts ()

    msg.hand_frame_id = '/base_link'
    msg.robot_frame_id = '/base'
    # TODO define these when I know what they are
    msg.cam_frame_id = 'TODO'
    msg.obj_frame_id = '/obj'

    contact_frame = msg.hand_frame_id


    # Get ReFlex hand contacts and rough fingertip normals. Normal endpts are
    #   at magnitudes corresponding to pressure reading
    # [geometry_msgs/Point[], geometry_msgs/Point[]]. Arrays with same length
    [contacts, normal_endpts, pressures, finger_idx, sensor_idx] = \
      get_contacts_live (
      self.handMsg, self.contact_thresh, contact_frame, self.tfTrans, 0.1)

    ind_marker_ids = [0] * len (contacts)

    # If no contacts, nothing to publish. This makes it easier at subscribers'
    #   end, don't have to take care of cases where all arrays are empty.
    if not contacts:
      #print ('No contacts, returning...')

      # Erase all outdated RViz Markers, since there are no contacts to plot
      (self.existing_individual_marker_ids, erase_arr) = erase_dead_markers ( \
        self.existing_individual_marker_ids, ind_marker_ids, contact_frame,
        ['contacts_ind', 'sen_normals_ind', 'pressures_ind'])
      if len (erase_arr.markers) > 0:
        self.vis_arr_pub.publish (erase_arr)

      return


    # Have contacts to publish

    self.contacts_seq += 1

    msg.header.seq = self.contacts_seq
    # Hand msg doesn't have a timestamp header! So we have to use the time now.
    #   tf using this timestamp, to get the transform as close to contact time
    #   as possible.
    # TODO: tf_get_pose() I haven't tried using with a timestamp. I think the
    #   last time I tried, it didn't work, that's why no functions currently
    #   do that.
    #   When on robot, try to use it, passing in self.handMsg.header.stamp
    #   > Tried. I now realized why. Because hand's connected to desktop, which
    #     has timestamp many seconds slower than the Baxter robot. Maybe ask
    #     Leif whether they encountered this, and how they solved it.
    #     http://askubuntu.com/questions/349763/how-can-i-change-the-date-and-time-on-ubuntu-12-04
    #     $ sudo date --set "16 Jul 2015 19:57:00"
    #     This works! At least on command line, date prints out correct thing.
    # Header API: http://docs.ros.org/jade/api/std_msgs/html/msg/Header.html
    msg.header.stamp = rospy.Time.now ()

    msg.pressures = pressures [:]
    msg.finger_idx = finger_idx [:]
    msg.sensor_idx = sensor_idx [:]


    # Call tf to transform from hand frame to other frames. Fill in the msg

    # When timestamp is specified to tf_get_pose(), use_common_time is ignored.
    #stamp = msg.header.stamp
    #rospy.loginfo ('%d.%d' % (stamp.secs, stamp.nsecs))
    # Above intermittently doesn't work, not sure why. See tf_get_pose.py for
    #   description, in the block when stamp is provided. When it doesn't work
    #   and waitForTransform(6), it also doesn't get killed by Ctrl+C. Really
    #   stupid.
    # Passing in None is actually not that much later than contact time,
    #   from rospy.loginfo, looks like only some nanoseconds behind, not bad
    #   at all, totally doable. So will pass in None from now on.
    stamp = None
    use_common_time = False

    # geometry_msgs/Pose type
    #   API http://docs.ros.org/jade/api/geometry_msgs/html/msg/Pose.html
    for i in range (0, len (contacts)):

      # Update member var
      #self.pressures_latest [finger_idx [i]] [sensor_idx [i]] = pressures [i]
      ind_marker_ids [i] = self.sensor_marker_ids [finger_idx [i]] \
        [sensor_idx [i]]

      # Set quaternion to identity, since currently get_contacts_live doesn't
      #   provide that. I shouldn't need it. If need it later, add that
      #   feature to get_contacts_live

      msg.hand_valid = True
      msg.pose_wrt_hand.append (Pose ())
      msg.pose_wrt_hand [i].position = contacts [i]
      msg.pose_wrt_hand [i].orientation.w = 1
      msg.norm_endpt_wrt_hand.append (Pose ())
      msg.norm_endpt_wrt_hand [i] = normal_endpts [i]

      # TODO: Add a timeout in tf_get_transform to check if a tf frame doesn't
      #   exist after 10 secs, then it should return None. And we'll publish
      #   None here too.

      # TODO: eventually modify tf_get_pose so that it can take a list of
      #   poses. That makes it easier here, just call it once, pass in both
      #   contacts[i] and normal_endpts[i].

      msg.robot_valid = False
      if self.robotOn:
        # contact point
        msg.pose_wrt_robot.append (Pose ())
        msg.pose_wrt_robot [i].position = \
          tf_get_pose_Point (contact_frame, msg.robot_frame_id,
            contacts [i], self.tfTrans, use_common_time, stamp).point
        msg.pose_wrt_robot [i].orientation.w = 1
        # normal endpt
        msg.norm_endpt_wrt_robot.append (Pose ())
        msg.norm_endpt_wrt_robot [i] = \
          tf_get_pose_Point (contact_frame, msg.robot_frame_id,
            normal_endpts [i], self.tfTrans, use_common_time, stamp).point

        # If tf_get_pose() returned None, then the frame doesn't exist
        # Assumption: One point not None, then all others are valid too. So just
        #   check first call from tf_get_pose_Point
        if msg.pose_wrt_robot [i].position:
          msg.robot_valid = True


      # If cam is not on, the published Pose[] is just an empty list. Check 
      #   empty list x with "if not x".
      msg.cam_valid = False
      msg.obj_valid = False
      if self.camOn:

        # cam frame

        # contact point
        msg.pose_wrt_cam.append (Pose ())
        msg.pose_wrt_cam [i].position = \
          tf_get_pose_Point (contact_frame, msg.cam_frame_id,
            contacts [i], self.tfTrans, use_common_time, stamp).point
        msg.pose_wrt_cam [i].orientation.w = 1
        # normal endpt
        msg.norm_endpt_wrt_cam.append (Pose ())
        msg.norm_endpt_wrt_cam [i] = \
          tf_get_pose_Point (contact_frame, msg.cam_frame_id,
            normal_endpts [i], self.tfTrans, use_common_time, stamp).point

        if msg.pose_wrt_cam [i].position:
          msg.cam_valid = True


        # obj frame
       
        # contact point
        msg.pose_wrt_obj.append (Pose ())
        msg.pose_wrt_obj [i].position = \
          tf_get_pose_Point (contact_frame, msg.obj_frame_id,
            contacts [i], self.tfTrans, use_common_time, stamp).point
        msg.pose_wrt_obj [i].orientation.w = 1
        # normal endpt
        msg.norm_endpt_wrt_obj.append (Pose ())
        msg.norm_endpt_wrt_obj [i] = \
          tf_get_pose_Point (contact_frame, msg.obj_frame_id,
            normal_endpts [i], self.tfTrans, use_common_time, stamp).point

        if msg.pose_wrt_obj [i].position:
          msg.obj_valid = True


    self.contacts_pub.publish (msg)


    # Commented out `.` it lags MoveIt too much!!! Lags like crazy. Maybe `.`
    #   cmumulative markers??? Not sure why. need to look into it. This is on
    #   real robot too, fast computer.

    # Visualize contacts and endpoints
    # For each sensor, only visualize its latest. Otherwise markers' opacities
    #   overlap, and all will look solid; normals' lengths overlap, and you
    #   can't see latest length.
    #   Do this by specifying unique marker IDs for each of the 38 sensors.
    # Comment out this block if you don't want to see them in RViz.
    '''
    text_marker_ids = range (self.text_marker_ids_seen + 1,
      self.text_marker_ids_seen + 1 + len(contacts))

    visualize_contacts (contacts, normal_endpts, pressures,
      self.contact_thresh,
      self.vis_pub, contact_frame, self.contacts_seq, text_marker_ids)

    self.text_marker_ids_seen = self.text_marker_ids_seen + len(contacts)


    # Erase all markers in namespace first, before visualizing the new set
    # NOTE: This erases whatever you publish in visualize_individual_contacts().
    #   Make sure namespace and marker ID match with those in that fn.
    (self.existing_individual_marker_ids, erase_arr) = erase_dead_markers ( \
      self.existing_individual_marker_ids, ind_marker_ids, contact_frame,
      ['contacts_ind', 'sen_normals_ind', 'pressures_ind'])
    if len (erase_arr.markers) > 0:
      self.vis_arr_pub.publish (erase_arr)

    # Add new marker IDs to list
    for i in range (0, len (ind_marker_ids)):
      if ind_marker_ids [i] not in self.existing_individual_marker_ids:
        self.existing_individual_marker_ids.append (ind_marker_ids [i])


    # Visualize each sensor's latest contact individually
    visualize_individual_contacts (contacts, normal_endpts, pressures,
      self.contact_thresh,
      #self.pressures_latest, self.sensor_marker_ids, 
      self.vis_pub, contact_frame, ind_marker_ids)
    '''


# Latest, non-cumulative markers.
# Visualize at most 1 point, 1 normal, and 1 pressure text, per sensor.
#   So 38 * 3 max number of markers.
# Each sensor gets a unique ID. Each type of data is in its own namespace.
# Parameters:
#   marker_ids: List of IDs corresponding to the contacted sensors. Values are
#     0 to 37.
def visualize_individual_contacts (contacts, normal_endpts, pressures,
  contact_thresh,
  vis_pub, frame_id, marker_ids, marker_duration=0):

  # So that RViz doesn't print hundreds of errors
  if not frame_id:
    return


  # Each iteration uses the same unique ID, for each of the 3 namespaces.
  #   The ID corresponds to the sensor index, 0:37. So at any given time,
  #   only the latest contact is displayed, for any given sensor.
  for i in range (0, len (contacts)):

    # Contacted sensor
 
    # yellow
    marker_con = Marker ()
    # Parameters: tx, ty, tz, r, g, b, alpha, sx, sy, sz
    create_marker (Marker.POINTS, 'contacts_ind', frame_id, marker_ids[i],
      0, 0, 0, 1, 1, 0, 0.5, 0.01, 0.01, 0.01,
      marker_con, marker_duration)  # Use 0 duration for forever

    marker_con.points.append (contacts [i])
    # Per-point color with alpha
    marker_con.colors.append (ColorRGBA (1, 1, 0,
      calc_pressure_alpha (pressures [i], contact_thresh)))


    # Sensor normals

    # green
    marker_normals = Marker ()
    create_marker (Marker.LINE_LIST, 'sen_normals_ind', frame_id, marker_ids[i],
      0, 0, 0, 0, 1, 0, 0.5, 0.001, 0, 0,
      marker_normals, marker_duration)  # Use 0 duration for forever

    marker_normals.points.append (contacts [i])
    marker_normals.points.append (normal_endpts [i])


    # Pressure value texts

    text_height = 0.01

    # yellow?
    marker_text = Marker ()
    create_marker (Marker.TEXT_VIEW_FACING, 'pressures_ind', frame_id,
      marker_ids[i],
      normal_endpts [i].x, normal_endpts [i].y, normal_endpts [i].z,
      1, 1, 0, 0.5, 0, 0, text_height, marker_text, marker_duration)
    marker_text.text = str (pressures [i])


    vis_pub.publish (marker_con)
    vis_pub.publish (marker_normals)
    vis_pub.publish (marker_text)



# Not in the class, outside functions can call
# Visualize each contact iteration.
#   Contacts points are visualized as POINTS type, normals are as LINE_LIST,
#     to enable visualization of multiple points, and *multiple arrows*,
#     respectively, using the same marker ID for each set of points / arrows.
# Parameters:
#   contacts: geometry_msgs/Point[]
#   normal_endpts: geometry_msgs/Point[]
#   pressures: int[]
#   cumu_marker_id: int
#   text_marker_ids: int[]. One per sensor.
def visualize_contacts (contacts, normal_endpts, pressures, contact_thresh,
  vis_pub, frame_id, cumu_marker_id, text_marker_ids, marker_duration=0):

  if not frame_id:
    rospy.logwarn ('detect_reflex_contacts.py visualize_contacts() got empty frame_id passed in. This will result in RViz warnings of source_frame in tf2 frame_ids cannot be empty. You might not get correct contact XYZ positions!')
    return

  # Nothing to publish. Do not publish empty markers!
  if not contacts:
    return


  # Contacted sensors

  # Per-iteration, non-cumulative marker
  # yellow
  marker_con = Marker ()
  # Parameters: tx, ty, tz, r, g, b, alpha, sx, sy, sz
  create_marker (Marker.POINTS, 'contacts_curr', frame_id, 0,
    0, 0, 0, 1, 1, 0, 0.5, 0.01, 0.01, 0.01,
    marker_con, marker_duration)  # Use 0 duration for forever

  # Cumulative marker. Difference from non-cumulative in just namespace and
  #   marker ID (which should be unique) here is supplied by caller each time.
  # yellow
  marker_con_cumu = Marker ()
  # Parameters: tx, ty, tz, r, g, b, alpha, sx, sy, sz
  create_marker (Marker.POINTS, 'contacts', frame_id, cumu_marker_id,
    0, 0, 0, 1, 1, 0, 0.5, 0.01, 0.01, 0.01,
    marker_con_cumu, marker_duration)  # Use 0 duration for forever


  # Sensor normals

  marker_normals = Marker ()
  # green
  create_marker (Marker.LINE_LIST, 'sen_normals_curr', frame_id, 0,
    0, 0, 0, 0, 1, 0, 0.5, 0.001, 0, 0,
    marker_normals, marker_duration)  # Use 0 duration for forever

  marker_normals_cumu = Marker ()
  # green
  create_marker (Marker.LINE_LIST, 'sen_normals', frame_id, cumu_marker_id,
    0, 0, 0, 0, 1, 0, 0.5, 0.001, 0, 0,
    marker_normals_cumu, marker_duration)  # Use 0 duration for forever
 
 
  # Add points to markers

  for i in range (0, len (contacts)):

    # Contacted sensors

    marker_con.points.append (contacts [i])
    # Per-point color with alpha
    marker_con.colors.append (ColorRGBA (1, 1, 0,
      calc_pressure_alpha (pressures [i], contact_thresh)))

    marker_con_cumu.points.append (contacts [i])
    # Per-point color with alpha
    marker_con_cumu.colors.append (ColorRGBA (1, 1, 0,
      calc_pressure_alpha (pressures [i], contact_thresh)))


    # Sensor normals

    marker_normals.points.append (contacts [i])
    marker_normals.points.append (normal_endpts [i])

    marker_normals_cumu.points.append (contacts [i])
    marker_normals_cumu.points.append (normal_endpts [i])


    # Pressure value texts
    # Can't do multiple texts per marker, so will need to do many markers

    # yellow?
    marker_text = Marker ()
    create_marker (Marker.TEXT_VIEW_FACING, 'pressures_curr', frame_id, i,
      normal_endpts [i].x, normal_endpts [i].y, normal_endpts [i].z,
      1, 1, 0, 0.5, 0, 0, 0.02, marker_text, marker_duration)
    marker_text.text = str (pressures [i])
    vis_pub.publish (marker_text)

    marker_text_cumu = Marker ()
    create_marker (Marker.TEXT_VIEW_FACING, 'pressures', frame_id,
      text_marker_ids[i],
      normal_endpts [i].x, normal_endpts [i].y, normal_endpts [i].z,
      1, 1, 0, 0.5, 0, 0, 0.02, marker_text_cumu, marker_duration)
    marker_text_cumu.text = str (pressures [i])
    vis_pub.publish (marker_text_cumu)


  vis_pub.publish (marker_con)
  vis_pub.publish (marker_con_cumu)
  vis_pub.publish (marker_normals)
  vis_pub.publish (marker_normals_cumu)


# Only visualize a contact if it's on the fingertip, i.e. finger_idx 1 to 3
#   as defined in tactile_map Contacts.msg, and sensor_idx 8.
# This is useful for cumulative contacts at tip only.
# Calls visualize_contacts() (not visualize_individual_contacts(), which only
#   plots the latest contact, 1 per sensor).
def visualize_contacts_tips_only (finger_idx, sensor_idx,
  contacts, normal_endpts, pressures, contact_thresh,
  vis_pub, frame_id, cumu_marker_id, text_marker_ids, marker_duration=0,
  ball_center=None):

  # So that RViz doesn't print hundreds of errors
  if not frame_id:
    return

  if not contacts:
    return


  contacts_tip = []
  normals_tip = []
  pressures_tip = []
  text_marker_ids_tip = []

  for i in range (0, len (finger_idx)):

    # Record for visualization
    # Only visualize if contact is on the finger tip, [1:3][8]
    if finger_idx [i] != 0 and sensor_idx [i] == 8:
 
      contacts_tip.append (contacts [i])
      normals_tip.append (normal_endpts [i])
      pressures_tip.append (pressures [i])
      text_marker_ids_tip.append (text_marker_ids [i])

  # Visualize
  visualize_contacts (contacts_tip, normals_tip,
    pressures_tip, contact_thresh, vis_pub, frame_id, 
    cumu_marker_id, text_marker_ids_tip, marker_duration)

  if ball_center:
    visualize_ball_contact_normal (ball_center, contacts_tip,
      vis_pub, frame_id, cumu_marker_id)


# This is for ball with known center only.
#   For other things, this may not be the contact normal.
# Parameters:
#   ball_center: geometry_msgs/Point, for known ball center
#   All remaining params same as those of visualize_contacts_tips_only().
def visualize_ball_contact_normal (ball_center,
  contacts, vis_pub, frame_id, cumu_marker_id, marker_duration=0):

  # So that RViz doesn't print hundreds of errors
  if not frame_id:
    return

  # If empty, don't publish empty marker
  if not contacts:
    return


  # dark green (see noise_measure_legend.py)
  marker_normals = Marker ()
  create_marker (Marker.LINE_LIST, 'con_normals', frame_id, cumu_marker_id,
    0, 0, 0, 20/255.0, 90/255.0, 0, 0.5, 0.001, 0, 0,
    marker_normals, marker_duration)  # Use 0 duration for forever

  # Append points to marker
  for i in range (0, len (contacts)):

    # Contact normal = measured point on ball - known ball center
    marker_normals.points.append (ball_center)
    marker_normals.points.append (contacts [i])

  vis_pub.publish (marker_normals)



# Calculate color alpha of a point, as a function of pressure value
# Parameters:
#   pressure: magnitude 0 to ~300, if max, ~680
def calc_pressure_alpha (pressure, contact_thresh):

  if abs (pressure) < contact_thresh:
    alpha = 0

  # alpha is 1 to 0, 1 being opaque, 0 transparent invisible
  # In correspondence,
  # abs(pressure) is 300 to 0, 300 being a lot, 0 being nothing
  # So alpha is directly proportional to pressure
  #   But scale shouldn't be linear, because if contact is 15, I want to see
  #   it, that is the threshold. Above 100 can be solid alpha 1.
  #
  # Desired values (tried to make it linear so easier to infer visually):
  # Pressure	Opacity
  # < 15	0
  # 15		~0.2
  # 50		0.35
  # 100		0.5
  # 150		0.65
  # 200		0.8
  # >= 250	1
  #
  # y = mx + b
  # m = deltaY / deltaX = 0.15 / 50 = 0.004
  # b = 0.2 with exceptions
  # y = 0.003 * x + 0.15
  # 
  # I like m = 0.003 numerically and visually. 0.004 numerically looks good,
  #   visually might be okay too.
  else:
    alpha = 0.003 * abs (pressure)

    #print ('pressure: %f, alpha: %f, 0.003*abs(pressure) = %f' % ( \
    #  pressure, alpha, 0.003*abs(pressure)))

  # Cap at 1 if alpha exceeds 1
  return min ([1, alpha])


# Returns:
#   existing_ids: caller should reassign what they passed in with this ret val
#   erase_arr: MarkerArray. User should publish this to
#     visualization_marker_array ROS topic.
def erase_dead_markers (existing_ids, new_ids, frame_id, namespaces):

  erase_arr = MarkerArray ()

  # Indices to erase
  erase_existing_vals = []

  for i in range (0, len (existing_ids)):

    # If the existing marker will not be replaced by a new one, it should
    #   no longer be there, delete it.
    if existing_ids [i] not in new_ids:

      erase_existing_vals.append (existing_ids [i])

      # Erase this marker ID in every name space
      # Can't just change the namespace and append, must make brand new
      #   Markers, `.` Python append() doesn't make copies of objects, only
      #   references. This is unlike C++ push_back(), which makes a copy.
      for ns in range (0, len (namespaces)):

        erase_m = Marker ()
       
        erase_m.id = existing_ids [i]
        erase_m.action = Marker.DELETE
       
        erase_m.header.frame_id = frame_id
 
        erase_m.ns = namespaces [ns]
        erase_arr.markers.append (erase_m)

   
  # Delete AFTER the for-loop, otherwise for-loop counter doesn't know
  #   list size changed! Because range(0, len) is evaluated only once, at
  #   first iteration. Python is dumb like that; C++ would know.
  for i in range (0, len (erase_existing_vals)):
    # Erase by remove(), not del arr[idx], because del is error-prone. e.g.
    #   an array has 2 elts. You want to erase [0] and [1]. But once you've
    #   erased [0], now list has 1 elt. Then [1] is out of bounds! In order
    #   for del to be correct, every time you erase, you have to decrement
    #   each index in the list of indices to erase, and only decrement if
    #   that index is greater than the one you just erased! This is too
    #   hacky. remove() is much better.
    # It's okay to use remove() `.` marker IDs are unique. So won't have
    #   problems where two of the same values in list aren't both removed.
    existing_ids.remove (erase_existing_vals [i])

  return (existing_ids, erase_arr)



def main ():

  rospy.init_node ('detect_reflex_contacts', anonymous=True)

  thisNode = DetectReFlexContacts ()


  wait_rate = rospy.Rate (10)
  while not rospy.is_shutdown ():

    thisNode.detect_contacts ()

    try:
      wait_rate.sleep ()

    except rospy.exceptions.ROSInterruptException, err:
      break


if __name__ == '__main__':
  main ()


