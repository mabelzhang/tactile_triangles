#!/usr/bin/env python

# Mabel Zhang
# 31 Oct 2015
#
# Simulates /reflex_hand rostopic
#

# ROS
import rospy
from std_srvs.srv import Empty, EmptyResponse

# Gazebo
from gazebo_msgs.srv import GetJointProperties, GetJointPropertiesRequest

# ReFlex Hand
from reflex_msgs.msg import Hand, Finger, Palm

# My packages
from reflex_gazebo_msgs.msg import Contact


class ReflexDriverNode:

  def __init__ (self):#, hertz):

    rospy.Subscriber ('/reflex_gazebo/contact', Contact, self.contact_cb)

    self.sim_hand_pub = rospy.Publisher ('/reflex_hand', Hand, queue_size=5)

    #self.HERTZ_DESIRED = hertz

    self.N_FIN = 3
    self.N_SEN = 9

    # A constant, just to make it easier to see which sensors are fired.
    #   This should be above the threshold that you set for contact to be true.
    #   This is only set on sensors that have contact = True.
    self.PRESSED = 100.0

    # Time stamp when the current 1/HERTZ seconds segment started
    self.time_started = -1

    self.restart_timer = True

    self.contacts = []


  def reset_contacts (self):
    # Reset contacts array
    self.contacts = []
    for i in range (0, self.N_FIN):
      self.contacts.append ([])
      for j in range (0, self.N_SEN):
        self.contacts [j].append (False)
    

  def contact_cb (self, msg):

    if not self.contacts:
      self.reset_contacts ()

    #print ('Got a /reflex_gazebo/contact msg with contact on finger %d' % \
    #  (msg.fin_num))

    if self.restart_timer:

      self.restart_timer = False

      self.time_started = msg.header.stamp

      self.reset_contacts ()

    # Record this sensor was fired
    self.contacts [msg.fin_num - 1] [msg.sen_num - 1] = True

    # TODO: Bug. This never updates... because it only updates fingers that get
    #   a contact. But for fingers that got a contact before, but no longer
    #   contacted now, the contact is not being resetted to False.


    #print ('Finger %d sensor %d fired' % (msg.fin_num, msg.sen_num))


  # reflex_driver_node.cpp calls it reflex_hand_state_cb(), but since this is
  #   not a callback function, the _cb suffix is removed.
  def publish_reflex_hand_state (self):

    # If not initialized yet, that means no contacts yet. Just init all to 0s
    if not self.contacts:
      self.reset_contacts ()

    sim_hand_msg = Hand ()

    # Currently not used
    sim_hand_msg.palm = Palm ()


    #####
    # Joint data
    #####

    # Populate sim_hand_msg, by packaging all contacts since the last time this
    #   fn was called (caller should enforce calling this fn only every
    #   1/HERTZ seconds, so can publish /reflex_hand at HERTZ rate).

    # I don't have joints data right now. Not sure how to get it from Gazebo
    #   yet, probably can through ros controller or something
    # This is not too important for sim, because Gazebo sim automatically
    #   publishes tf, which is all I use. Hardware needs this, because otherwise
    #   software has no way of knowing.
    sim_hand_msg.joints_publishing = True

    success, preshape_1_pos = self.get_joint_pos ('preshape_1')
    success, proximal_1_pos = self.get_joint_pos ('proximal_joint_1')
    success, distal_1_pos   = self.get_joint_pos ('distal_joint_1')

    # Don't need preshape2, it should be enforced as negative of preshape1,
    #   `.` on real hand, there is only one joint.
    success, proximal_2_pos = self.get_joint_pos ('proximal_joint_2')
    success, distal_2_pos   = self.get_joint_pos ('distal_joint_2')

    success, proximal_3_pos = self.get_joint_pos ('proximal_joint_3')
    success, distal_3_pos   = self.get_joint_pos ('distal_joint_3')

    # Populate msg
    if preshape_1_pos:
      sim_hand_msg.palm.preshape = preshape_1_pos [0]
    if proximal_1_pos:
      sim_hand_msg.finger[0].proximal = proximal_1_pos [0]
    if distal_1_pos:
      sim_hand_msg.finger[0].distal = distal_1_pos [0]
    if proximal_2_pos:
      sim_hand_msg.finger[1].proximal = proximal_2_pos [0]
    if distal_2_pos:
      sim_hand_msg.finger[1].distal = distal_2_pos [0]
    if proximal_3_pos:
      sim_hand_msg.finger[2].proximal = proximal_3_pos [0]
    if distal_3_pos:
      sim_hand_msg.finger[2].distal = distal_3_pos [0]


    #####
    # Tactile data
    #####

    # True to indicate tactile values are accurate. For simulation, it's always
    #   accurate, `.` no zeroing is required.
    sim_hand_msg.tactile_publishing = True

    for i in range (0, len (self.contacts)):

      for j in range (0, len (self.contacts [i])):

        # If got a contact from this sensor, set to true, else false.
        if self.contacts [i] [j]:
          #sim_hand_msg.finger [i].contact.append (True)
          sim_hand_msg.finger [i].contact [j] = True
          sim_hand_msg.finger [i].pressure [j] = self.PRESSED

          #print ('Finger %d sensor %d set to True' % (i+1, j+1))

        else:
          #sim_hand_msg.finger [i].contact.append (False)
          sim_hand_msg.finger [i].contact [j] = False
          sim_hand_msg.finger [i].pressure [j] = 0.0

    # Debug: print first /reflex_hand msg
    #print (sim_hand_msg)


    #####
    # Publish msg
    #####

    # Publish 10 times, `.` UDP drops packages
    for i in range (0, 10):
      self.sim_hand_pub.publish (sim_hand_msg)

    #print ('Published /reflex_hand msg')

    # Restart a time segment, after having published this one
    self.restart_timer = True


  # Calls Gazebo rosservice to get joint position
  # Returns (True, position[]) if call is successful, else (False, []).
  def get_joint_pos (self, joint_name):

    srv_name = '/gazebo/get_joint_properties'

    rospy.wait_for_service (srv_name)

    #rospy.loginfo ('Calling rosservice %s on %s...' % (srv_name, joint_name))
    try:
      srv = rospy.ServiceProxy (srv_name, GetJointProperties)
      req = GetJointPropertiesRequest ()
      req.joint_name = joint_name
      resp = srv (req)

    except rospy.ServiceException, e:
      rospy.logerr ('sample_gazebo remove_model(): Service call to %s failed: %s' %(srv_name, e))
      return (False, [])

    if resp.success:
      #rospy.loginfo ('rosservice call succeeded')
      return (True, resp.position)
    else:
      rospy.logerr ('rosservice call failed: %s' % resp.status_message)
      return (False, [])


  def zero_tactile_fn (self, req):

    rospy.loginfo ('Zeroing tactile data...')

    # Clear all contacts. This is needed, otherwise rostopics may not be
    #   quick enough to arrive contact_cb to clear array, then
    #   publish_reflex_hand_state() would never publish a all-0 msg to clear
    #   the contacts! Then guarded_move in reflex_base.py might have outdated
    #   info from a prev contact, so it never closes the fingers with outdated
    #   contact=True info.
    self.reset_contacts ()

    # Publish a all-0-contacts msg to tell subscribers contacts are cleared
    self.publish_reflex_hand_state ()

    return EmptyResponse ()



def main ():

  rospy.init_node ('reflex_sim_driver_node', anonymous=True)

  #hertz = 30

  thisNode = ReflexDriverNode ()

  zero_tactile_name = "/zero_tactile"
  rospy.loginfo ("Advertising the %s service", zero_tactile_name)
  zero_tactile = rospy.Service (zero_tactile_name, Empty,
    thisNode.zero_tactile_fn)


  wait_rate = rospy.Rate (30)
  while not rospy.is_shutdown ():

    thisNode.publish_reflex_hand_state ()

    # If this doesn't get awaken, it's because your Gazebo physics is paused.
    #   Just unpause, and this will return from sleep.
    #   Ref: http://answers.ros.org/question/11761/rospysleep-doesnt-get-awaken/
    wait_rate.sleep ()


if __name__ == '__main__':

  # This enables Ctrl+C kill at any time
  try:
    main ()
  except rospy.exceptions.ROSInterruptException, err:
    pass

