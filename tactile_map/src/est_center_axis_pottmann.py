#!/usr/bin/env python

# Mabel Zhang
# 21 Apr 2015
#
# Estimate center axis, the line (axis) that intersects the most lines
#   (normals), using Pottmann CAD 1999 paper's way.
# Pottmann calculation is implemented in util_geometry.py .
#
# To run:
#   $ roslaunch tactile_map est_center_axis_pottmann.launch
# Or
#   $ rosrun tactile_map broadcast_frame.py
#   $ rosrun tactile_map est_center_axis_pottmann.py
#


# ROS
import rospy
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point
import tf
from std_msgs.msg import Bool

# Python
import numpy as np

# ReFlex pkg
from reflex_msgs.msg import Hand

# Local
from tactile_map.create_marker import create_marker
from util_geometry import est_linear_complex
from tactile_map.spin_seam import spin_cloud, calc_rough_normals
from tactile_map.tf_get_pose import tf_get_pose
from tactile_map.get_contacts import get_contacts_live, eliminate_duplicates


class EstCenterAxis_Pottmann:

  def __init__ (self): #, tfTrans, bc):

    # Debug flag. Set to true to plot axes for all 6 Pottmann eigvals,
    #   from est_linear_complex(). As opposed to just the axis with min eigval,
    #   which is what Pottmann selects.
    #   Practice shows that the one with min eigval is most correct. If it's
    #   very off, then the other eigvals' axes are even more off. So it's 
    #   sufficient to just plot the correct one.
    #   This flag is still useful though when in doubt.
    self.PLOT_ALL_AXES = False

    self.vis_pub = rospy.Publisher ('/visualization_marker', Marker)
    self.NORMALS_MARKER_ID = 1
    self.TRUE_AXIS_ID = 0
    self.EST_AXIS_ID = 1
    self.marker_duration = 0

    self.cloud_spun_pub = rospy.Publisher (
      '/tactile_map/spin_seam/cloud_spun', PointCloud)

    self.tfTrans = tf.TransformListener ()
    self.bc = tf.TransformBroadcaster ()
    #self.tfTrans = tfTrans
    #self.bc = bc

    rospy.Subscriber ('/reflex_hand', Hand, self.handCB)
    self.handMsg = None
    self.contact_thresh = 15

    self.contacts = []
    self.contact_cloud = PointCloud ()
    self.contact_cloud.header.frame_id = '/base'
    self.contact_cloud_pub = rospy.Publisher ('/tactile_map/contact/cloud',
      PointCloud)

    self.normal_endpts = []
    self.marker_normals = Marker ()
    # green
    create_marker (Marker.LINE_LIST, 'normals', '/base',
      0, 0, 0, 0, 0, 1, 0, 0.5, 0.001, 0, 0,
      self.marker_normals, self.marker_duration)  # Use 0 duration for forever

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


  # Generate a cylinder shape and rough surface normals.
  # Returns surface normals of the geometry in this format:
  #   [[(x1 y1 z1), (x2 y2 z2)]_1, .... [(x1 y1 z1), (x2 y2 z2)]_n]
  #   A list of k lists of size-2 Numpy arrays. Each size-2 Numpy array 
  #   represents a line, described by 2 points on the line.
  #   [] denotes Python list, () denotes Numpy.
  #   lines[i][0] is line i point 1, lines[i][1] is line i point 2.
  # Returns also the ground truth unit axis of cylinder.
  def gen_cylinder (self):

    ## Constants to define the shape

    # Center point in base
    base_pt = np.array ([0.1, 0.1, 0.1])

    # 30 cm
    height = 0.3
    # 1 cm
    height_step = 0.01
    height_range = np.arange (0, height, height_step)
    # Just 2 heights, for testing util_geometry est_linear_complex(). Simpler
    #   for me to see. Actually harder for algo too.
    #height_range = [0, height]
    n_slices = len (height_range)

    # 5 cm
    radius = 0.05
    # 10 degrees
    radians_step = 10.0 / 180.0 * np.pi
    # Just replicate the same radius, for a straight cylinder
    #radii = [radius] * n_slices

    # Generate some random radii
    radius_max = 0.05
    radii = np.random.rand (n_slices, 1) * radius_max

    # Sort the radii to get a smooth shape
    radii.sort (0)



    ## Generate the shape, upright, in the tilted frame

    obj_frame = '/obj'

    # For now, test on an upright cylinder. Axis is z-axis.
    axis_unit = np.array ([0, 0, 1])

    # Points at center of cylinder bottom and top
    axis_pt1 = base_pt
    axis_pt2 = base_pt + height * axis_unit

    # Generate cloud
    cloud = PointCloud ()
    n_pts_per_slice = spin_cloud (axis_pt1, axis_pt2, height_range, radii,
      obj_frame, cloud, 1)
    self.cloud_spun_pub.publish (cloud)



    ## Convert the upright object from tilted /obj frame to robot /base frame.
    #    Now the object should be a tilted object in /base frame.
    #  This is to test if Pottmann algo's implementation est_linear_complex()
    #    makes any upright assumptions of object. If it does, then result
    #    will plot incorrectly. It shouldn't.

    cloud_tilt = PointCloud ()
    cloud_tilt.header.frame_id = '/base'

    for i in range (0, len (cloud.points)):

      pt_tilt = tf_get_pose (
        cloud.header.frame_id, cloud_tilt.header.frame_id,
        cloud.points[i].x, cloud.points[i].y, cloud.points[i].z, 0, 0, 0, 0,
        self.tfTrans, True, None, False)

      cloud_tilt.points.append (pt_tilt.point)

    cloud_tilt.header.stamp = rospy.Time.now ()


    ## Print ground truth axis in /base frame

    axis_pt1_tilt = tf_get_pose (
      cloud.header.frame_id, cloud_tilt.header.frame_id,
      axis_pt1[0], axis_pt1[1], axis_pt1[2], 0, 0, 0, 0,
      self.tfTrans, True, None, False)

    axis_pt2_tilt = tf_get_pose (
      cloud.header.frame_id, cloud_tilt.header.frame_id,
      axis_pt2[0], axis_pt2[1], axis_pt2[2], 0, 0, 0, 0,
      self.tfTrans, True, None, False)

    axis_unit_tilt_np = np.asarray (
      [axis_pt2_tilt.point.x - axis_pt1_tilt.point.x,
       axis_pt2_tilt.point.y - axis_pt1_tilt.point.y,
       axis_pt2_tilt.point.z - axis_pt1_tilt.point.z])
    axis_unit_tilt_np = axis_unit_tilt_np / np.linalg.norm (axis_unit_tilt_np)

    print ('Truth unit axis: ' + str (axis_unit_tilt_np))


    # Draw ground truth axis in /base frame

    # green
    marker_axis = Marker ()
    create_marker (Marker.LINE_LIST, 'center_axis', '/base',
      self.TRUE_AXIS_ID, 0, 0, 0, 0, 1, 0, 0.5, 0.01, 0, 0,
      marker_axis, self.marker_duration)  # Use 0 duration for forever
  
    marker_axis.points.append (axis_pt1_tilt.point)
    marker_axis.points.append (axis_pt2_tilt.point)
    
    self.vis_pub.publish (marker_axis)



    ## Get rough normals, for the tilted object

    marker = Marker ()
    # green
    # Use base frame, to be wrt robot. This tilting shape is to test if any
    #   function makes any upright assumptions. They shouldn't. If they do,
    #   then the normals will plot incorrectly.
    create_marker (Marker.LINE_LIST, 'normals', '/base',
      self.NORMALS_MARKER_ID, 0, 0, 0, 0, 1, 0, 0.5, 0.001, 0, 0,
      marker, self.marker_duration)  # Use 0 duration for forever

    # Get normals of generated shape
    normals = calc_rough_normals (axis_pt1, axis_pt2, cloud_tilt, n_pts_per_slice,
      marker)

    self.vis_pub.publish (marker)


    return (normals, axis_unit_tilt_np)


  # Randomly select a small set of normals, in the given set.
  # Parameters:
  #   N_SPARSE: How many to select
  # Returns the selected set.
  def sparsen_normals (self, normals, N_SPARSE=5):

    ## Randomly pick a small set of normals to pass to Pottmann

    # Unique list of random indices. choice() is random sampling.
    sparse_idx = np.random.choice (range (0, len(normals)), size=N_SPARSE,
      replace=False)

    normals_sparse = [normals [i] for i in sparse_idx]

    print ('Chose %d normals out of %d:' % (len(normals_sparse), len(normals)))
    print (sparse_idx)


    ## Plot the sparse set of normals chosen, in a different namespace

    LEN = 0.03

    marker_sparse = Marker ()
    # green
    create_marker (Marker.LINE_LIST, 'sparse_normals', '/base',
      0, 0, 0, 0, 0, 1, 0, 0.5, 0.001, 0, 0,
      marker_sparse, self.marker_duration)  # Use 0 duration for forever

    for i in range (0, len (normals_sparse)):
      endpt = normals_sparse[i][0] + LEN * normals_sparse[i][1]

      marker_sparse.points.append (Point (normals_sparse [i] [0] [0],
        normals_sparse [i] [0] [1], normals_sparse [i] [0] [2]))
      marker_sparse.points.append (Point (endpt[0], endpt[1], endpt[2]))

    self.vis_pub.publish (marker_sparse) 


    return normals_sparse


  # Estimate and plots the center axis of a generated shape.
  def estimate (self, normals, axis_true=None):

    if not self.PLOT_ALL_AXES:

      ## Estimate center axis
     
      (axis_pt1, axis_pt2) = est_linear_complex (normals)
     
     
      ## Draw the axis returned by est_linear_complex()
     
      # red
      marker_axis = Marker ()
      create_marker (Marker.LINE_LIST, 'center_axis', '/base',
        self.EST_AXIS_ID, 0, 0, 0, 1, 0, 0, 0.5, 0.01, 0, 0,
        marker_axis, self.marker_duration)  # Use 0 duration for forever
   
      marker_axis.points.append (Point (axis_pt1[0], axis_pt1[1], axis_pt1[2]))
      marker_axis.points.append (Point (axis_pt2[0], axis_pt2[1], axis_pt2[2]))
      
      self.vis_pub.publish (marker_axis)


    else:
      # Test every eigvec
      for i in range (0, 6):
        # Parameter: [[(x1 y1 z1), (x2 y2 z2)]_1, .... [(x1 y1 z1), (x2 y2 z2)]_n]
        #   A list of k lists of size-2 Numpy arrays. Each size-2 Numpy array 
        #   represents a line, described by 2 points on the line.
        #   [] denotes Python list, () denotes Numpy.
        #   lines[i][0] is line i point 1, lines[i][1] is line i point 2.
        (axis_pt1, axis_pt2) = est_linear_complex (normals, i)
     
     
        # Check correctness of estimate, against ground truth
     
        # Unit vector of estimated axis
        axis_unit = axis_pt2 - axis_pt1
        axis_unit = axis_unit / np.linalg.norm (axis_unit)
     
        # Default color: red
        if i == 0:
          r = 1; g = 0; b = 0;
        elif i == 1:
          r = 1; g = 1; b = 1;
        elif i == 2:
          r = 0; g = 0; b = 1;
        elif i == 3:
          r = 1; g = 1; b = 0;
        elif i == 4:
          r = 0; g = 1; b = 1;
        elif i == 5:
          r = 1; g = 0; b = 1;
     
        # If dot prod of true axis and estimated axis is +/- 1, then cos == 1,
        #   meaning estimated axis is correct
        if (axis_true is not None) and \
           (abs (abs (axis_true.dot (axis_unit)) - 1) < 1e-6):
          print ('Correct axis has eigval index %d' % i)
     
          # Correct color: green
          r = 0
          g = 1
          b = 0
     
     
        # Draw the axis returned by est_linear_complex()
     
        # red
        marker_axis = Marker ()
        create_marker (Marker.LINE_LIST, 'center_axis', '/base',
          i, 0, 0, 0, r, g, b, 0.5, 0.01, 0, 0,
          marker_axis, self.marker_duration)  # Use 0 duration for forever
      
        marker_axis.points.append (Point (axis_pt1[0], axis_pt1[1], axis_pt1[2]))
        marker_axis.points.append (Point (axis_pt2[0], axis_pt2[1], axis_pt2[2]))
        
        self.vis_pub.publish (marker_axis)


  # An example of how to call this class to test a generated shape. Caller could
  #   also call this fn directly to go through whole pipeline.
  def generate_and_estimate (self):

    normals, axis_true = self.gen_cylinder ()

    # Pass in all the normals
    #self.estimate (normals, axis_true)

    # Pass in a small subset of the normals
    normals_sparse = self.sparsen_normals (normals)
    self.estimate (normals_sparse, axis_true)


  def handCB (self, msg):

    self.handMsg = msg


  def pauseCB (self, msg):

    if self.pause_contacts != msg.data:
      if msg.data:
        print ('Pause contact detection')
      else:
        print ('Resume contact detection')

    self.pause_contacts = msg.data


  # An example of how to call this class on real robot.
  # When not in generation mode, caller should provide normals to estimate().
  #   axis_true is optional. Caller can choose to call sparsen_normals() to
  #   pick a small subset of the provided normals, before calling estimate()
  #   with the sparsened set.
  def get_live_normals_and_estimate (self):

    # If hand mover says values are invalid, wait till valid
    if self.pause_contacts:
      return

    # If haven't gotten a msg yet, wait till we get one
    if self.handMsg is None:
      return


    ## Update contact points

    # Get ReFlex hand contacts and rough fingertip normals. Normal endpts are
    #   at magnitudes corresponding to pressure reading
    # [geometry_msgs/Point[], geometry_msgs/Point[]]. Arrays with same length
    [contacts, normal_endpts, _, _, _] = get_contacts_live (
      self.handMsg, self.contact_thresh, '/base', self.tfTrans, 0.1)

    # Identify near-duplicate contacts
    [contacts_no_dups, normals_no_dups, _, old_idx, new_idx] = \
      eliminate_duplicates (
        self.contacts, contacts, self.normal_endpts, normal_endpts)

    # Add non-dups to cumulative member field
    self.contacts.extend (contacts_no_dups)
    self.normal_endpts.extend (normals_no_dups)

    # For near-dups, keep the new copy.
    #   This is to accept the fluctuation on real robot and not get stuck with
    #   one set of values that might be bad for Pottmann axis estimation.
    # TODO: Later can randomly select the old one or new one. Later select
    #   the one that yields highest confidence value for the estimated axis.
    for i in range (0, len (old_idx)):
      self.contacts [old_idx [i]] = contacts [new_idx [i]]
      self.normal_endpts [old_idx [i]] = normal_endpts [new_idx [i]]


    ## Plot contact cloud and rough object normals 

    self.contact_cloud.points = []
    self.marker_normals.points = []

    # Populate msgs
    for i in range (0, len (self.contacts)):
      self.contact_cloud.points.append (self.contacts [i])

      self.marker_normals.points.append (self.contacts [i])
      self.marker_normals.points.append (self.normal_endpts [i])

    # This way always uses the first-ever copy of a dup. Does not accept
    #   subsequent fluctuations. Remove when decide to keep fluctuations only.
    # Populate msgs
    #for i in range (0, len (contacts_no_dups)):
      # Plot contacts as cumulative PointCloud
      #self.contact_cloud.points.append (contacts_no_dups [i])

      # Plot normals as cumulative Marker
      #self.marker_normals.points.append (contacts_no_dups [i])
      #self.marker_normals.points.append (normals_no_dups [i])

    if len (self.contact_cloud.points) == 0:
      return

    self.contact_cloud.header.stamp = rospy.Time.now ()
    self.contact_cloud_pub.publish (self.contact_cloud)

    self.marker_normals.header.stamp = rospy.Time.now ()
    self.vis_pub.publish (self.marker_normals) 


    ## Estimate center axis

    # Convert normals to the format accepted by estimate()
    # List of size-2 lists of size-3 Numpy arrays.
    #   [[(x1 y1 z1), (n1 n1 n1)], ...]
    normals = [[
      np.asarray ([self.contacts[i].x, self.contacts[i].y, self.contacts[i].z]),
      np.asarray ([self.normal_endpts[i].x, self.normal_endpts[i].y, self.normal_endpts[i].z])]
      for i in range(0, len(self.contacts))]

    # Print for temporary recording
    print ('normals:')
    for i in range (0, len (normals)):
      print ('%g %g %g, %g %g %g' % (
        normals[i][0][0], normals[i][0][1], normals[i][0][2],
        normals[i][1][0], normals[i][1][1], normals[i][1][2],
      ))

    # Estimate and plot axis
    self.estimate (normals)



if __name__ == '__main__':

  rospy.init_node ('est_center_axis_pottmann', anonymous=True)

  #tfTrans = tf.TransformListener ()
  #bc = tf.TransformBroadcaster ()

  thisNode = EstCenterAxis_Pottmann () #tfTrans, bc)


  # Generated shape or on real robot
  GEN = 0
  REAL = 1
  MODE = GEN #REAL


  # 1 Hz, slow enough for user to see each iteration in RViz.
  #   Adjust up for real time testing.
  wait_rate = rospy.Rate (1)
  while not rospy.is_shutdown ():

    if MODE == GEN:
      thisNode.generate_and_estimate ()
    else:
      thisNode.get_live_normals_and_estimate ()

    try:
      wait_rate.sleep ()
    except rospy.exceptions.ROSInterruptException, err:
      break



