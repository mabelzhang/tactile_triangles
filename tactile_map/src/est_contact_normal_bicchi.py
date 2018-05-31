#!/usr/bin/env python

# Mabel Zhang
# 8 May 2015
#
# Estimates contact centroid c at fingertip, contact normal n on fingertip at c.
# Inputs:
#   f, m: force measured at force-torque sensor
# Outputs:
#   c: contact centroid for a soft finger contact (area contact), a
#      displacement vector from force-torque sensor frame. It is also the lever
#      arm, connecting torque axis felt at sensor frame, to the force applied
#      at estimated contact centroid.
#   n: contact normal, at contact centroid c
#   q: torque at contact centroid
#   (p: force at contact centroid, p = f)
#
# Bicchi calculation is implemented in util_geometry.py .
#
# Set up of this file is similar to est_center_axis_pottmann.py .
#

# ROS
import rospy
from geometry_msgs.msg import Wrench
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
import tf

# Baxter
from baxter_core_msgs.msg import EndpointState

# Python
import sys

# Numpy
import numpy as np

# Local
from util_geometry import calc_contact_normal, calc_contact_normal_point, \
  calc_normal
from tactile_map.create_marker import create_marker, create_arrow
from tactile_map.tf_get_pose import tf_get_pose


class EstContactNormal_Bicchi:

  def __init__ (self, arm_side, A, R, e):

    self.arm_side = arm_side

    rospy.Subscriber ('/robot/limb/' + arm_side + '/endpoint_state',
      EndpointState, self.endptCB)

    self.tfTrans = tf.TransformListener ()

    self.endpt_pos = None
    self.wrench = None

    # Use 0 duration for forever
    self.marker_duration = 0
    self.vis_pub = rospy.Publisher ('/visualization_marker', Marker)
    self.vis_pub_arr = rospy.Publisher ('/visualization_marker_array',
      MarkerArray)

    '''
    # Are we in simulation?
    self.use_sim_time = False
    if rospy.has_param ('/use_sim_time'):
      # Ref: http://wiki.ros.org/Clock
      self.use_sim_time = rospy.get_param ('/use_sim_time')
    rospy.loginfo ('use_sim_time found to be ' + str (self.use_sim_time))

    self.tfTrans = tf.TransformListener ()
    '''

    # Get zero reference for wrench, before user starts grasping stuff
    self.force0 = None
    self.torque0 = None

    self.A = A
    self.R = R
    self.e = e

    # Sensor frame. B is sensor frame origin, as defined in paper.
    #   Calculations should always use sensor frame B, not robot base frame,
    #   `.` Bicchi paper assumes sensor frame is at B. If you need to plot
    #   things in robot /base frame, then do tf. Don't change calculation to
    #   base frame.
    self.sensor_frame = '/left_gripper'
    # Sensor location. B = (0, 0, 0) in sensor's frame. (Change it via tf if
    #   you decide to use a different frame!)
    # Note: Currently code assumes this is 0 0 0. If you change this (even
    #   though you shouldn't! `.` Bicchi paper defines B to be sensor frame), 
    #   then probably need to add B to plotting, like when plotting e, c, maybe
    #   subtract B from e and c when calling calc_contact_normal() as well,
    #   since that calculation assumes e is offset from B (0 0 0).
    self.B = np.asmatrix ([[0, 0, 0]]).T

    # Just testing: Do you need to zero c as well?
    self.c0 = None


  # Callback function for subscriber of endpoint state
  def endptCB (self, msg):

    self.endpt_pos = Point (msg.pose.position.x, msg.pose.position.y,
      msg.pose.position.z)
    self.wrench = Wrench (msg.wrench.force, msg.wrench.torque)

    # Debug. Seems to all be 0 in simulation. Maybe `.` I didn't enable robot..
    #if (msg.wrench.force.x != 0) and (msg.wrench.force.y != 0) and \
    #  (msg.wrench.force.z != 0):
    #  print (msg.wrench.force)


  def zero_wrench (self):

    # TODO: test this fn on real robot

    print ('Zeroing wrench - grabbing a zero-reference for force and torque.')

    wait_rate = rospy.Rate (1)
    while self.wrench is None:
      rospy.loginfo ('est_contact_normal_bicchi.py zero_wrench(): Waiting for an EndpointState msg to arrive...')
      try:
        wait_rate.sleep ()
      except rospy.exceptions.ROSInterruptException, err:
        return False

    self.force0 = np.asmatrix ([[self.wrench.force.x, self.wrench.force.y,
      self.wrench.force.z]]).T
    self.torque0 = np.asmatrix ([[self.wrench.torque.x, self.wrench.torque.y,
      self.wrench.torque.z]]).T

    print ('f0: %g %g %g, m0: %g %g %g' % (
      self.force0[0], self.force0[1], self.force0[2],
      self.torque0[0], self.torque0[1], self.torque0[2]))

    return True


  # Generates the case defined in Bicchi IJRR 1993 Section 8, Table 3
  '''
To test this fn with calc_contact_normal():

A = np.asmatrix (np.eye (3))
R = 1
cx = 0.125
cy = 0
cz = np.sqrt (R*R + cx*cx - cy*cy)
c_true = np.asmatrix ([[cx, cy, cz]]).T
f = np.asmatrix ([[-0.3, -0.04, 0.01]]).T
m = np.asmatrix ([[0.01, 0.01, 0.48]]).T
(c, n, q) = calc_contact_normal (f, m, A, R)
  '''
  def generate_and_estimate (self):

    print ('Center of sphere: e = %g %g %g' % (self.e[0], self.e[1], self.e[2]))

    # Note: Keep everything as Numpy matrix type (not array type), for 
    #   consistent matrix multiplications in util_geometry.py! Array type gives
    #   wrong dimensions for matrix multiplication.

    # Simulated measured force and torque in Table 3. 3 x 1 Numpy matrix.
    f = np.asmatrix ([[-0.3, -0.04, 0.01]]).T
    m2 = [0.48, 0.49, 0.50, 0.51]
    # Table 3 last column "Ellipsoid" method
    c_answer_lst = [[-0.6, 0.8, 0], [-0.66, 0.75, 0.04], [-0.74, 0.67, 0.09],
      [-0.76, 0.62, 0.2]]

    # For printing later
    m_lst = []
    c_lst = []
    q_lst = []
    n_lst = []
    cp_lst = []
    qp_lst = []
    np_lst = []

    wait_rate = rospy.Rate (1)

    # Produce each row of Table 3
    for i in range (0, len (m2)):

      m = np.asmatrix ([[0.01, 0.01, m2[i]]]).T

      # Section 5 method, for c
      (c, n, q, K) = calc_contact_normal (f, m, self.A, self.R, self.e)

      # Special m just to test section 4 method
      # Make m a vector perpendicular to f, so that the method finds valid
      #   answers. This is the easiest case to find m to test. Other m values
      #   give negative value in sqrt, no way to test method fully.
      tmp_vec = np.asarray ([1, 0, 0])
      if (abs (f - tmp_vec) < 1e-6).all ():
        tmp_vec = np.asarray ([0, 1, 0])
      m_method4 = np.asmatrix (np.cross (tmp_vec, f, axis=0))

      # Section 4 method, for c'
      # This line is just to test section 4 method. Usually, shouldn't call
      #   this fn directly. This fn is automatically called by
      #   calc_contact_normal() when q is detected to be 0.
      (c_prime, n_prime, q_prime, _) = calc_contact_normal_point (f, #m,
        m_method4,
        self.A, self.R, self.e)

      # A good sanity check is that distance formula of c - ball_pos should 
      #   equal self.R, because contact c must lie on the sphere with radius
      #   self.R.
      print ('Sanity checks:')
      print ('||c-e|| should be R (%f): %g' %(self.R, np.linalg.norm (c-e)))
      # eqn (13) should be satisfied
      #print ('c.T * A * A * c == R*R should be True: ' + \
      #  str(c.T * self.A * self.A * c == self.R * self.R))
      print ('')

      # Store some values for printing table later
      m_lst.append ([m[0,0], m[1,0], m[2,0]])
      c_lst.append ([c[0,0], c[1,0], c[2,0]])
      q_lst.append ([q[0,0], q[1,0], q[2,0]])
      n_lst.append ([n[0,0], n[1,0], n[2,0]])
      cp_lst.append ([c_prime[0,0], c_prime[1,0], c_prime[2,0]])
      qp_lst.append ([q_prime[0,0], q_prime[1,0], q_prime[2,0]])
      np_lst.append ([n_prime[0,0], n_prime[1,0], n_prime[2,0]])



      ## Plot inputs f, m

      self.plot_inputs (f, m)


      ## Plot result from my implementation of Bicchi section 5 method.
      #  n and q: n is contact normal, q is contact torque, both are at contact
      #    centroid c. n is unit vector, parallel to q, but not nec in same
      #    direction of q.

      self.plot_c_n_q_p (self.B, c, n, q, f, '/bicchi_outputs',
        self.sensor_frame, range(0,8), text_height=0.1)

      print ('My answer: ')
      print ('c=[%.2g %.2g %.2g], q=[%.2g %.2g %.2g]' %
        (c[0][0], c[1][0], c[2][0],
         q[0][0], q[1][0], q[2][0]))
     

      ## Plot result from my implementation of Bicchi section 5 method.

      self.plot_c_n_q_p (self.B, c_prime, n_prime, q_prime, f,
        '/bicchi_method4_outputs', self.sensor_frame, range(0,8),
        text_height=0.1,
        c_lbl='n_ans_4', n_lbl='n_ans_4', q_lbl='q_ans_4 || n_ans_4')

     
      ## Plot result from Bicchi Table 3, for comparison to my implementation
     
      c_answer = self.e + np.asmatrix ([c_answer_lst[i]]).T
      AAc_answer = self.A * self.A * (c_answer - self.e)
      n_answer = AAc_answer / np.linalg.norm (AAc_answer)  # eqn (10)
      # The q obtained this way does not have n_answer as its unit vector! Odd.
      #   That means it doesn't satisfy eqn (3)!
      q_answer = m - np.cross (c_answer, f, axis=0)  # eqn (3)
      K_answer = q_answer / AAc_answer  # eqn (10)
      # This is actually exactly same as above, using eqn(3) form. But funny
      #   that it's incorrect.
      #K_answer = (m - np.cross (c_answer, f, axis=0)) / AAc_answer  (eqn 11)
     
      self.plot_c_n_q_p (self.B, c_answer, n_answer, q_answer, f,
        '/bicchi_answers', self.sensor_frame, range(0,8), 0, 0.4, 0,
        text_height=0.1, c_lbl='c_ans', n_lbl='n_ans', q_lbl='q_ans || n_ans')

      print ('Correct answer: ')
      print ('c=[%.2g %.2g %.2g], q=[%.2g %.2g %.2g], K=[%.2g %.2g %.2g]' %
        (c_answer[0,0], c_answer[1,0], c_answer[2,0],
         q_answer[0,0], q_answer[1,0], q_answer[2,0],
         K_answer[0,0], K_answer[1,0], K_answer[2,0]))
     
      print ('')


      try:
        wait_rate.sleep ()
      except rospy.exceptions.ROSInterruptException, err:
        break


    # Print Table 3
    print ('F/T Measurements      Wrench Axis Mthd       Ellipsoid Mthd')

    for i in range (0, len (m2)):
      print ('f=[%.2g %.2g %.2g]' %
        (f[0], f[1], f[2]))

      print ('m=[%.2g %.2g %.2g]   c\'=[%.2g %.2g %.2g]  c=[%.2g %.2g %.2g]' %
        (m_lst[i][0], m_lst[i][1], m_lst[i][2],
        cp_lst[i][0], cp_lst[i][1], cp_lst[i][2],
        c_lst[i][0], c_lst[i][1], c_lst[i][2]))


    print ('\n')


  # On real robot
  def get_live_wrench_and_estimate (self):

    # TODO: Test the f - f0, m - m0, subtraction of zero reference wrench. See
    #   if result is even reasonable.

    if self.force0 is None or self.torque0 is None:
      if not self.zero_wrench ():
        rospy.logerr ('est_contact_normal_bicchi.py get_live_wrench_and_estimate(): Error while trying to find zero position for force and torque. Returning...')
        return

    # This is in /base frame, raw from Baxter rostopic
    f_raw = np.asarray ([[self.wrench.force.x, self.wrench.force.y,
      self.wrench.force.z]]).T - self.force0
    m_raw = np.asarray ([[self.wrench.torque.x, self.wrench.torque.y,
      self.wrench.torque.z]]).T - self.torque0

    # This is Baxter dependent. Baxter EndPointState rostopic has empty
    #   string frame_id, but it's really '/base'.
    raw_frame_id = '/base'


    ## Convert f, m into sensor frame B, which is assumed by Bicchi method

    f_pose = tf_get_pose (raw_frame_id, self.sensor_frame,
      f_raw[0,0], f_raw[1,0], f_raw[2,0],
      0, 0, 0, 0, self.tfTrans,
      use_common_time=True, stamp=None, pose_or_point=False)
    f = np.asmatrix ([[f_pose.point.x, f_pose.point.y, f_pose.point.z]]).T

    m_pose = tf_get_pose (raw_frame_id, self.sensor_frame,
      m_raw[0,0], m_raw[1,0], m_raw[2,0],
      0, 0, 0, 0, self.tfTrans,
      use_common_time=True, stamp=None, pose_or_point=False)
    m = np.asmatrix ([[m_pose.point.x, m_pose.point.y, m_pose.point.z]]).T

    # This gives more correct results. Maybe `.` f given by Baxter is internal
    #   force felt (I assume its arrow goes from endpoint outwards), whereas
    #   Bicchi wants f to be external force felt (arrow points from outside
    #   towards sensor frame B). So we should flip the force.
    f = -f


    ## Calculate contact centroid and contact torque

    print ('Inputs f: %.3g %.3g %.3g, m: %.3g %.3g %.3g' %(
      f[0], f[1], f[2], m[0], m[1], m[2]))

    (c, n, q, K) = calc_contact_normal (f, m, self.A, self.R, self.e)

    print ('Outputs c: %.3g %.3g %.3g, q: %.3g %.3g %.3g' %(
      c[0], c[1], c[2], q[0], q[1], q[2]))

    q_unit = q / np.linalg.norm (q)
    m_unit = m / np.linalg.norm (m)
    print ('m unit vector: %.3g %.3g %.3g' % (m_unit[0], m_unit[1], m_unit[2]))
    print ('q unit vector: %.3g %.3g %.3g' % (q_unit[0], q_unit[1], q_unit[2]))


    ## Plot inputs f, m

    self.plot_inputs (f, m)


    ## Plot results from Bicchi method

    self.plot_c_n_q_p (self.B, c, n, q, f, '/bicchi_outputs',
      self.sensor_frame, range(0,8), text_height=0.1)


  # Draw a sphere representing fingertip.
  # Fingertip is assumed to be a sphere, for the easiest case in Bicchi's 
  #   method. Other supported geometry include cylinder, plane, and ellipsoid.
  #   If have compound objects that can be described as ellipsoid sections, 
  #   see Bicchi section 7 for criteria.
  # Parameters:
  #   pos: Numpy array. 3 elts. (x y z) pos of fingertip
  #   R: radius of sphere fingertip
  def plot_fingertip (self, pos, R, frame_id):

    marker = Marker ()
    create_marker (Marker.SPHERE, '/fingertip_bicchi', frame_id, 0,
      pos[0], pos[1], pos[2], 1, 1, 1, 0.5, 2*R, 2*R, 2*R,
      marker, self.marker_duration)

    self.vis_pub.publish (marker)


  def plot_point_with_label (self, pt, ns, frame_id,
    marker_ids, pt_text, r=0, g=1, b=0, text_height=0.02):

    # POINTS Marker: scale.x is point width, scale.y is point height
    marker_pt = Marker ()
    create_marker (Marker.POINTS, ns, frame_id, marker_ids[0],
      0, 0, 0, r, g, b, 0.5, 0.1, 0.1, 0,
      marker_pt, self.marker_duration)
    marker_pt.points.append (Point (pt[0], pt[1], pt[2]))

    # Offset text a bit longer than head of arrow.
    # TEXT Marker: only scale.z is used, height of uppercase "A"
    marker_p_text = Marker ()
    create_marker (Marker.TEXT_VIEW_FACING, ns, frame_id, 
      marker_ids[1],
      pt[0] + 0.01, pt[1] + 0.01, pt[2] + 0.01,
      r, g, b, 0.5, 0, 0, text_height, marker_p_text, self.marker_duration)
    marker_p_text.text = pt_text

    self.vis_pub.publish (marker_pt)
    self.vis_pub.publish (marker_p_text)


  def plot_vec_with_label (self, pt, vec, ns, frame_id,
    marker_ids, vec_text, r=0, g=1, b=0, text_height=0.02, text_at_tail=False):

    marker_vec = Marker ()
    marker_v_text = Marker ()

    create_arrow (pt, pt+vec, ns, frame_id,
      marker_ids, r, g, b, 0.02, 0.04, 0, text_height,
      vec_text, self.marker_duration, [marker_vec, marker_v_text], text_at_tail)

    self.vis_pub.publish (marker_vec)
    self.vis_pub.publish (marker_v_text)


  # Plot Bicchi method inputs f, m (measured force and torque at 
  #   force-torque sensor), and frame origins B (sensor frame), e (ball
  #   fingertip sphere center)
  def plot_inputs (self, f, m):

    B_1D = np.squeeze (np.asarray (self.B))

    # Plot force-torque sensor frame location B
    #ball_pos = np.asarray ([0, 0, 0])
    ball_pos = np.squeeze (np.asarray (self.e))
    self.plot_point_with_label (B_1D, '/bicchi_inputs', self.sensor_frame,
      range(0,2), 'B', text_height=0.1)

    # Plot spherical fingertip
    self.plot_point_with_label (ball_pos, '/bicchi_inputs', self.sensor_frame,
      range(2,4), 'e', text_height=0.1)
    self.plot_fingertip (ball_pos, self.R, self.sensor_frame)

    # Convert Numpy 2D 3 x 1 matrix to 1D array, to pass to fn
    f_1D = np.squeeze (np.asarray (f))

    # Plot f, m. These are sensed at sensor location B.
    self.plot_vec_with_label (B_1D-f_1D, f_1D,
      '/bicchi_inputs', self.sensor_frame, range(4,6), 'f', text_height=0.1,
      text_at_tail=True)
    self.plot_vec_with_label (B_1D, np.squeeze (np.asarray (m)),
      '/bicchi_inputs', self.sensor_frame, range(6,8), 'm', text_height=0.1)


  # Plot Bicchi method outputs
  # Publish RViz Marker for a given point, and a vector starting at the point
  #   and ending at a second given point.
  # Parameters:
  def plot_c_n_q_p (self, B, c, n, q, f, ns, frame_id,
    marker_ids, r=0, g=1, b=0, text_height=0.02,
    c_lbl='c', n_lbl='n', q_lbl='q || n', p_lbl='p=f'):

    # Convert Numpy 2D 3 x 1 matrix to 1D array
    B_1D = np.squeeze (np.asarray (B))
    c_1D = np.squeeze (np.asarray (c))
    f_1D = np.squeeze (np.asarray (f))
    
    # Plot contact centroid c
    self.plot_vec_with_label (B_1D, c_1D,
      ns, frame_id, marker_ids[0:2], c_lbl, r, g, b, text_height)

    contact = B_1D + c_1D

    # Plot torque q at contact point
    self.plot_vec_with_label (contact, np.squeeze (np.asarray (q)),
      ns, frame_id, marker_ids[2:4], q_lbl, r, g, b, text_height)

    # Note n is unit length, so will be 1 m in RViz!
    self.plot_vec_with_label (contact, np.squeeze (np.asarray (n)),
      ns, frame_id, marker_ids[4:6], n_lbl, r, g, b, text_height)

    # Plot force at contact point, p = f
    self.plot_vec_with_label (contact-f_1D, f_1D,
      ns, frame_id, marker_ids[6:8], p_lbl, r, g, b, text_height,
      text_at_tail=True)


  # Sanity check my modification of Bicchi eqn (10), for a sensor frame B that
  #   is outside the sphere. Offset from B to center of sphere is e.
  def plot_sphere_normals (self):

    self.plot_fingertip (self.e, self.R, self.sensor_frame)

    marker_arr = MarkerArray ()

    marker_ids = np.array ([0, 1])

    # Wikipedia's is more correct than Wolfram MathWorld. z should be 
    #   cos(theta), not cos(phi) like Wolfram says!
    # http://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates
    for theta in np.arange (0, 2*np.pi, np.pi/10):
      for phi in np.arange (0, 2*np.pi, np.pi/10):

        marker_ids = marker_ids + 2

        # Point on sphere S
        # Calculate cartesian coordinate using polar coordinates
        p = np.matrix ([[0.0, 0.0, 0.0]]).T
        p [0,0] = self.R * np.sin (theta) * np.cos (phi)
        p [1,0] = self.R * np.sin (theta) * np.sin (phi)
        p [2,0] = self.R * np.cos (theta)
       
        # Normal at point r on sphere S. r is wrt frame, not specific to 
        #   center of sphere, so add e + r to get the world coords.
        (_, n) = calc_normal (self.A, self.e + p, self.e)
       
        # Publish RViz ARROW marker representing the normal
        marker_n = Marker ()
        create_arrow (
          np.squeeze (np.asarray (self.e+p)), np.squeeze (np.asarray (self.e+p+n)),
          '/sanity_normals', self.sensor_frame, marker_ids,
          0, 1, 0, 0.02, 0.04, 0, 0.1,
          '', self.marker_duration, [marker_n, None])
        marker_arr.markers.append (marker_n)

      self.vis_pub_arr.publish (marker_arr)


if __name__ == '__main__':

  rospy.init_node ('est_contact_normal_bicchi', anonymous=True)

  arm_side = 'left'
  # Parse cmd line args
  for i in range (0, len (sys.argv)):
    if sys.argv[i] == '--left':
      arm_side = 'left'
    elif sys.argv[i] == '--right':
      arm_side = 'right'
  print ('Arm set to ' + arm_side + ' side')


  # Generated shape or on real robot
  GEN = 0
  REAL = 1
  MODE = REAL

  # Surface satisfying eqn S(r) = r^T * A^T * A * r - R^2 = 0.
  #   A is 3 x 3 diagonal constant coeff matrix.
  #   R is scalar factor.
  # See Bicchi IJRR 1993 eqn (9), and section 5.1 for common shapes that fit in
  #   the assumption of ellipsoidal shapes (sphere, cylinder, plane).
  if MODE == GEN:
    # Spheres have A = 3 x 3 identity mat. Bicchi section 5.1.1.
    A = np.asmatrix (np.eye (3))
    R = 1

    # Generate some random offset e of ball center from sensor frame B.
    #   Put e within 2 meters of B.
    e_max = 4
    # np.random.rand() returns [0, 1). This generates numbers in range [-2, 2)
    e = ((np.random.rand (3, 1)) - 0.5) * e_max

    #e = np.asmatrix ([[0, 0, 0]]).T

    # This has solution from method 4
    #e = np.asmatrix ([[0.485031, 0.89146, 1.85679]]).T

    print ('Center of fingertip sphere set to %f %f %f' %(e[0], e[1], e[2]))

  # On real robot
  #   e needs to be redefined every time. Better way is to get it from Kinect
  else:
    A = np.asmatrix (np.eye (3))
    # TODO: define R and e when have a ball to use
    # In meters.
    #   SAIC dark blue stress ball is 7 cm diameter, 3.5 cm radius
    #   Borrowed big red foam ball is 13 cm diameter, 6.5 cm radius
    R = 0.065

    #e = np.asmatrix ([[0, 0, 0]]).T
    e = np.asmatrix ([[-0.035, -0.01, +0.155]]).T

  thisNode = EstContactNormal_Bicchi (arm_side, A, R, e)


  # 1 Hz, slow enough for user to see each iteration in RViz.
  #   Adjust up for real time testing.
  wait_rate = rospy.Rate (1)
  while not rospy.is_shutdown ():

    if MODE == GEN:
      #thisNode.plot_sphere_normals ()
      thisNode.generate_and_estimate ()
    else:
      thisNode.get_live_wrench_and_estimate ()

    try:
      wait_rate.sleep ()
    except rospy.exceptions.ROSInterruptException, err:
      break


