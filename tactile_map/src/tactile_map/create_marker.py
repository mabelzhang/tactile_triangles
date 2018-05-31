#!/usr/bin/env python

# Mabel Zhang
# 10 Mar 2015
#
# Moved this function out from manual_explore.py to be standalone so can share
#   with other .py files easily.
#

# ROS
import rospy
# http://docs.ros.org/api/visualization_msgs/html/msg/Marker.html
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from interactive_markers.menu_handler import *

import numpy as np


# Code copied from C++ version grasp_point_calc/include/skeleton.h
# Parameters:
#   marker_type: int. Constant from visualization_msgs/Marker
#   frame: string
#   marker_id: int. Unique in this marker's namespace
#   tx, ty, tz: double. Position of marker in frame
#   r, g, b, alpha: float. Color of marker
#   sx, sy, sz: double. Scale
#   marker: visualization_msgs.Marker type. Ret val. Fields populated by this fn
#   ns: string. Namespace of marker
#   duration: double. Life time of marker in seconds. 0 for forever
#   qw, qx, qy, qz: double. Rotation in Quaternions
def create_marker (marker_type, ns, frame, marker_id,
  tx, ty, tz, r, g, b, alpha, sx, sy, sz,
  # Return value
  marker, duration=60.,
  qw=1, qx=0, qy=0, qz=0, frame_locked=False):

  marker.header.frame_id = frame
  # Treated differently by RViz. Do NOT use ros::Time::now(), that causes 
  #   marker to only be displayed when that time is close enough to current
  #   time, where close enough depends on tf. With ros::Time() (==0), marker
  #   will be displayed regardless of current time.
  # Ref: Python ros time http://www.ros.org/wiki/rospy/Overview/Time
  #   API (zero time example too!) http://mirror.umd.edu/roswiki/doc/diamondback/api/rospy/html/rospy.rostime.Time-class.html
  # If want marker to display forever
  #marker.header.stamp = rospy.Time ()
  # If want marker to display only for a duration lifetime
  marker.header.stamp = rospy.Time.now ()

  marker.ns = ns
  # Must be unique. Same ID replaces the previously published one
  marker.id = marker_id

  marker.type = marker_type
  marker.action = Marker.ADD

  marker.pose.position.x = tx
  marker.pose.position.y = ty
  marker.pose.position.z = tz

  marker.pose.orientation.x = qx
  marker.pose.orientation.y = qy
  marker.pose.orientation.z = qz
  marker.pose.orientation.w = qw

  marker.scale.x = sx
  marker.scale.y = sy
  marker.scale.z = sz

  marker.color.a = alpha
  marker.color.r = r
  marker.color.g = g
  marker.color.b = b

  marker.lifetime = rospy.Duration.from_sec (duration)

  # If this marker should be frame-locked, i.e. retransformed into its frame every timestep
  marker.frame_locked = frame_locked

  # Just let caller set it after function call, like they'd set points field
  #   after fn call.
  # Only if using a MESH_RESOURCE marker type:
  #  Note: Path MUST be on the computer running RViz, not the robot computer!
  #  Pass this in param
  #model_path_dae = 'package://MenglongModels/spraybottle_c_rotated_filledHole_1000timesSmaller.dae'
  # Equivalent to strcpy()
  #marker.mesh_resource = model_path_dae;


# Parameters:
#   startpt, endpt: Python list, tuple, or Numpy array of 3 elts. Whatever type
#     that can be accessed by [0] [1] [2]
#   marker_ids: Python list of 2 marker IDs, for arrow, arrow text,
#     respectively.
#   label: If don't want a label, then pass in markers[1]=None
#   markers: ret vals, 2-elt Python list
#   text_at_tail: Draw text at arrow tail (flat end), instead of at arrowhead.
def create_arrow (startpt, endpt, ns, frame_id, marker_ids,
  r, g, b, sx, sy, sz, text_height, label, marker_duration, markers,
  text_at_tail=False):

  # Make arrow

  # Parameters: marker_id, tx, ty, tz, r, g, b, alpha, sx, sy, sz
  # ARROW Marker: scale.x is shaft diameter, scale.y is head diameter, scale.z
  #   is head length if non-zero.
  create_marker (Marker.ARROW, ns, frame_id, marker_ids[0],
    0, 0, 0, r, g, b, 0.5, sx, sy, sz,
    markers[0], marker_duration)

  markers[0].points.append (Point (startpt[0], startpt[1], startpt[2]))
  markers[0].points.append (Point (endpt[0], endpt[1], endpt[2]))


  # Text label on arrow

  # Text at arrow head
  if not text_at_tail:
    text_x = endpt[0] + 0.01
    text_y = endpt[1] + 0.01
    text_z = endpt[2] + 0.01
  # Text at arrow tail
  else:
    text_x = startpt[0] + 0.01
    text_y = startpt[1] + 0.01
    text_z = startpt[2] + 0.01

  # Offset text a bit longer than head of arrow.
  # TEXT Marker: only scale.z is used, height of uppercase "A"
  if markers[1] is not None:
    create_marker (Marker.TEXT_VIEW_FACING, ns, frame_id, 
      marker_ids[1], text_x, text_y, text_z,
      r, g, b, 0.5, 0, 0, text_height, markers[1], marker_duration)

    markers[1].text = label


# 10 Jul 2015
# Parameters:
#   server: created by InteractiveMarkerServer()
#   position: geometry_msgs.Point type
#   markerArr: MarkerArray type with the set of Markers to be wrapped in
#     interactive controls.
#
# Ref: http://wiki.ros.org/rviz/Tutorials/Interactive%20Markers%3A%20Getting%20Started
#   http://wiki.ros.org/rviz/Tutorials/Interactive%20Markers%3A%20Basic%20Controls
# Code copied and modified from interactive_marker_tutorials/python/basic_controls.py
def create_interactive_marker (server, position, markerArr, frame_id):

  int_marker = InteractiveMarker()
  int_marker.header.frame_id = frame_id
  int_marker.pose.position = position
  int_marker.scale = 1

  int_marker.name = "view_facing"
  int_marker.description = ""  # This displays in RViz

  # make a control that rotates around the view axis
  '''
  control = InteractiveMarkerControl()
  control.orientation_mode = InteractiveMarkerControl.VIEW_FACING
  control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
  control.orientation.w = 1
  control.name = "rotate"
  int_marker.controls.append(control)
  '''

  # create a box in the center which should not be view facing,
  # but move in the camera plane.
  control = InteractiveMarkerControl()
  control.orientation_mode = InteractiveMarkerControl.VIEW_FACING
  #control.orientation_mode = InteractiveMarkerControl.FIXED
  control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE

  # Quick convert: http://www.onlineconversion.com/quaternions.htm
  control.independent_marker_orientation = False
  # Decided not worth it to compute. This is just visualization, not robotics.
  # http://www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToQuaternion/
  # x = bank, y = heading, z = attitude
  #bank = 0
  #heading = 0
  #attitude = np.pi / 2
  control.orientation.x = 0.7071067811865476
  control.orientation.y = -0.7071067811865476
  control.orientation.z = 0
  control.orientation.w = 0

  control.name = "move"

  # Copy markers from param into the interactive marker control
  for i in range (0, len (markerArr.markers)):
    control.markers.append (markerArr.markers [i])

  control.always_visible = True
  int_marker.controls.append(control)

  server.insert(int_marker, processFeedback)


# For RViz interactive marker
# Prints user UI feedback to terminal
#   From interactive_marker_tutorials/python/basic_controls.py
def processFeedback( feedback ):

    # Don't need it to print feedback to screen
    return


    # Print feedback to screen

    s = "Feedback from marker '" + feedback.marker_name
    s += "' / control '" + feedback.control_name + "'"

    mp = ""
    if feedback.mouse_point_valid:
        mp = " at " + str(feedback.mouse_point.x)
        mp += ", " + str(feedback.mouse_point.y)
        mp += ", " + str(feedback.mouse_point.z)
        mp += " in frame " + feedback.header.frame_id

    if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
        rospy.loginfo( s + ": button click" + mp + "." )
    elif feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
        rospy.loginfo( s + ": menu item " + str(feedback.menu_entry_id) + " clicked" + mp + "." )
    elif feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
        rospy.loginfo( s + ": pose changed")
# TODO
#          << "\nposition = "
#          << feedback.pose.position.x
#          << ", " << feedback.pose.position.y
#          << ", " << feedback.pose.position.z
#          << "\norientation = "
#          << feedback.pose.orientation.w
#          << ", " << feedback.pose.orientation.x
#          << ", " << feedback.pose.orientation.y
#          << ", " << feedback.pose.orientation.z
#          << "\nframe: " << feedback.header.frame_id
#          << " time: " << feedback.header.stamp.sec << "sec, "
#          << feedback.header.stamp.nsec << " nsec" )
    elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_DOWN:
        rospy.loginfo( s + ": mouse down" + mp + "." )
    elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
        rospy.loginfo( s + ": mouse up" + mp + "." )

    # Mabel: Put this in caller of create_interactive_marker, where server is
    #   created.
    #server.applyChanges()


