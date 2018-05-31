#!/usr/bin/env python

# Mabel Zhang
# 14 Jul 2015
#
# Refactored from tactile_collect/src/tactile_collect.py
#
# This standalone node allows keyboard interface to be in its own thread! Then
#   you can still have a keyboard prompt blocking for user input, without
#   suspending the main computation thread - which would be run in another
#   node.
#
#   Your main computation thread should be subscribed to the string msg this
#   node publishes, so that you get the effect of multi-threading, via your
#   callback function. Set a member variable flag in your callback function,
#   based on what string the user entered, then in your main computation
#   function, check for that member var and do things accordingly.
#
# To run this file:
#   $ rosrun tactile_map keyboard_interface
#


# ROS
import rospy
from std_msgs.msg import String


class KeyboardInterface:

  def __init__ (self):

    self.key_pub = rospy.Publisher ('/keyboard_interface/key', String)

    # Get prompt from whatever computation node that publishes its specific
    #   prompt
    rospy.Subscriber ('/keyboard_interface/prompt', String, self.promptCB)
    self.prompt = None


  def promptCB (self, msg):

    self.prompt = msg.data


  # Copied and modified from tactile_collect.py
  # This file doesn't do any recording, it just sets the flag self.doRecord,
  #   and returns False if need to quit program. Else return true, and main ROS
  #   loop keeps going, and each iter calls the functions to record data.
  def keyboard_input (self):

    if not self.prompt:
      print ('Waiting for rostopic /keyboard_interface/prompt... Publish this String in your main computation node to tell user what to press.')
      return True

    # Ask for user input
    uinput = raw_input (self.prompt + 'q to stop and quit program: ')

    # Publish the key. Even if it's 'q', still publish, so subscribers know
    #   user asks to terminate program!
    msg = String ()
    msg.data = uinput
    self.key_pub.publish (msg)
    print ('Published user input %s' % (uinput))

    # q is reserved for quitting keyboard interface. Do not use q for other
    #   things in your main computation node!
    if uinput.lower () == 'q':
      return False

    return True


def main ():

  rospy.init_node ('keyboard_interface', anonymous=True)

  # Send in a prompt specific for this program
  thisNode = KeyboardInterface () #'Press a to start recording, s to pause, ')


  wait_rate = rospy.Rate (10)
  while not rospy.is_shutdown ():

    # Get user input first, to start or stop recording
    if not thisNode.keyboard_input ():
      break

    try:
      wait_rate.sleep ()

    except rospy.exceptions.ROSInterruptException, err:
      break


if __name__ == '__main__':
  main ()

