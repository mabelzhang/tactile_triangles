#!/usr/bin/env python

# Mabel Zhang
# 28 Aug 2016
#
# Tests a custom ROS shutdown routine.
#
# Bypasses ROS's default signal (Ctrl+C) handler, by passing
#   disable_signals=True to init_node.
#   Then upon Ctrl+C, calls my own shutdown routine, which doesn't forcibly get
#   interrupted by ROS's default signal handler. It is allowed to finish
#   whatever it's doing, before shutting down.
#
# Ref: wiki.ros.org/rospy/Overview/Initialization%20and%20Shutdown
#

import rospy


# Custom shutdown routine
def drag_time ():

  print ('Starting custom shutdown routine')

  secs = 10

  for i in range (0, secs):
    print ('Dragging time... %d out of %d s' % (i, secs))

    try:
      rospy.sleep (1)
    except KeyboardInterrupt:
      print ('Wait a bit longer. Still performing task (looping here, but in reality you can write to a file without being interrupted, etc.).')

      uinput = raw_input ('Press any key to continue, press Q to shutdown (you may lose files not finished being written, in a real situation): ')
      if uinput.lower () == 'q':
        break
      else:
        pass

  # Maually tell ROS to shutdown
  # This line is required, for ROS to clean up properly.
  rospy.signal_shutdown ('normal shutdown')


def main ():

  # Set disable_signals=True to use my own Ctrl+C signal handler
  rospy.init_node ('test_shutdown', anonymous=True, disable_signals=True)

  print ('Running. Press Ctrl+C any time to test custom shutdown routine. Press Ctrl+C again afterwards, program should NOT shutdown until it has completed its task.')

  while not rospy.is_shutdown ():

    try:
      rospy.sleep (0.1)
    except KeyboardInterrupt:
      break

  drag_time ()

  return


if __name__ == '__main__':
  main ()

