#!/usr/bin/env python

# Mabel Zhang
# 25 Apr 2015
#
# tf frame broadcaster. Broadcasts a random /obj frame, wrt /base frame.
#
# Useful when you need a separate node spitting out a tf
#   frame at high frequency, separate from the main node you want to test.
#
# First used in est_center_axis_pottmann.launch, to test
#   est_center_axis_pottmann.py.
#


import rospy
import tf

import numpy as np


# Broadcast a Quaternion offset from /base
def broadcast (bc, rand_quat):

  ## Create a tilted frame

  obj_frame = '/obj'

  # If in simulation and didn't start robot, '/base' frame might not exist
  #   yet. Publish it.
  #if not (self.tfTrans.frameExists ('/base')):
  #  self.bc.sendTransform ((0, 0, 0), (0, 0, 0, 0), rospy.Time.now (),
  #    '/base', '/world')

  # Broadcast new tf frame
  #   API: sendTransform(translation, rotation, time, child, parent)
  #   http://mirror.umd.edu/roswiki/doc/diamondback/api/tf/html/python/tf_python.html
  bc.sendTransform ((0, 0, 0),
    (rand_quat[0], rand_quat[1], rand_quat[2], rand_quat[3]),
    rospy.Time.now (),
    obj_frame, '/base')
  #print ('broadcasted /obj')


if __name__ == '__main__':

  rospy.init_node ('broadcast_frame', anonymous=True)

  tfTrans = tf.TransformListener ()
  bc = tf.TransformBroadcaster ()

  # Randomly generate a rotation in [0, 2*pi) interval
  rand_euler = np.random.rand (3,1) * np.pi * 2

  # Convert randomly generated rotation to Quaternion
  rand_quat = tf.transformations.quaternion_from_euler (
    rand_euler[0], rand_euler[1], rand_euler[2])

  # 1 Hz, slow enough for user to see each iteration in RViz.
  #   Adjust up for real time testing.
  wait_rate = rospy.Rate (10)
  while not rospy.is_shutdown ():

    broadcast (bc, rand_quat)

    try:
      wait_rate.sleep ()
    except rospy.exceptions.ROSInterruptException, err:
      break

