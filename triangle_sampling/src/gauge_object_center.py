#!/usr/bin/env python

# Mabel Zhang
# 7 Sep 2015
#
# Uses keyboard interface to let user gauge the position of an object.
# Can be called from outside function. Returns the final center.
#
# Called by triangles_collect_semiauto.py
#


# ROS
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# Numpy
import numpy as np

# Baxter
import baxter_external_devices

# My package
from tactile_map.create_marker import create_marker


# Parameters:
#   obj_height, obj_radius_x, obj_radius_y: Used solely for plotting in RViz
def move_delta (center, vis_pub, increment, obj_height, obj_radius_x,
  obj_radius_y):

  #print (center)
  #print (increment)
  center = center + increment

  #print ('New center: %f %f %f' % (center[0], center[1], center[2]))

  # Update RViz marker
  marker_center = Marker ()
  create_marker (Marker.POINTS, 'gauge_center', '/base', 0,
    0, 0, 0, 1, 1, 0, 0.8, 0.005, 0.005, 0.005,
    marker_center, 0)  # Use 0 duration for forever
  marker_center.points.append (Point (center[0], center[1], center[2]))
  vis_pub.publish (marker_center)

  # Object bbox
  # white
  marker_obj = Marker ()
  create_marker (Marker.CYLINDER, 'object', '/base', 0,
    center[0], center[1], center[2] + obj_height * 0.5,
    # Scale: X diameter, Y diameter, Z height
    1, 1, 1, 0.2, obj_radius_x * 2, obj_radius_y * 2, obj_height,
    marker_obj, 0)  # Use 0 duration for forever
  vis_pub.publish (marker_obj)


  return center


# Parameters:
#   obj_height, obj_radius_x, obj_radius_y: Used solely for plotting in RViz
def gauge_obj_center (init_pos, vis_pub, obj_height, obj_radius_x, 
  obj_radius_y):

  center = np.array (init_pos)

  # 1 mm
  DELTA = 0.001

  X_NEG = np.array ([-DELTA, 0, 0])
  X_POS = np.array ([DELTA, 0, 0])
  Y_NEG = np.array ([0, -DELTA, 0])
  Y_POS = np.array ([0, DELTA, 0])
  Z_NEG = np.array ([0, 0, -DELTA])
  Z_POS = np.array ([0, 0, DELTA])


  done = False

  # A dictionary, mapping from a character to a tuple.
  #   First element in tuple is a function, second is arguments to be
  #   unpacked and passed to the function, third is the string to display
  #   to user.
  bindings = {
  # key: (function, args, description)
    'h': (X_NEG, 'Move X by -DELTA'),
    'l': (X_POS, 'Move X by +DELTA'),
    'j': (Y_NEG, 'Move Y by -DELTA'),
    'k': (Y_POS, 'Move Y by +DELTA'),
    'm': (Z_NEG, 'Move Z by -DELTA'),
    'i': (Z_POS, 'Move Z by +DELTA'),
  }


  #####
  # Main UI loop
  #####

  print ('Welcome to interactive keyboard UI for gauging object center.')
  print ('Press ? for help, Esc to quit.')
  print ('')

  while not done and not rospy.is_shutdown ():

    # From baxter_examples joint_position_keyboard.py
    c = baxter_external_devices.getch ()

    if c:
      # Catch Esc or ctrl-c
      if c in ['\x1b', '\x03']:
        done = True
        print ('Done. Returning object center %f %f %f' % \
          (center[0], center[1], center[2]))
        break

      elif c in bindings:
        cmd = bindings[c]
        print("command: %s" % (cmd[1],))

        # Expand binding to something like "set_j(right, 's0', 0.1)"
        center = move_delta (center, vis_pub, cmd[0], obj_height, obj_radius_x,
          obj_radius_y)

        print ('New center: %f %f %f' % (center[0], center[1], center[2]))

      else:
        print("key bindings: ")
        print("  Esc: Quit")
        print("  ?: Help")
        for key, val in sorted(bindings.items(), key=lambda x: x[0]):
          print("  %s: %s" % (key, val[1]))


  return center


def main ():

  rospy.init_node ('gauge_object_center', anonymous=True)

  vis_pub = rospy.Publisher ('/visualization_marker', Marker)

  obj_center = [0.64, 0.441, 0.05]
  print ('Initial center: %f %f %f' % (obj_center[0], obj_center[1],
    obj_center[2]))

  obj_height = 0.15
  obj_radius_x = 0.05
  obj_radius_y = 0.05

  gauge_obj_center (obj_center, vis_pub, obj_height, obj_radius_x, obj_radius_y)


if __name__ == '__main__':
  main ()

