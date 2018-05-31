#!/usr/bin/env python

# Mabel Zhang
# 5 Apr 2015
#
# Estimate center axis of a rotationally symmetric object, using RANSAC.
#
# To run in simulation
#   $ roslaunch tactile_map est_center_axis_ransac.launch pcd:=1
#
#   (Launch file now does this for you)
#   $ rosrun pcl_ros pcd_to_pointcloud file.pcd 1 _frame_id:=/base
#   Use 1425948743.474750995_rectVase.pcd .
#
# To run on real robot
#   $ baxter_reflex
#   $ roslaunch tactile_map est_center_axis_ransac.launch sim:=false
#   $ rosrun tactile_map explore_fixed_pt.py
#   $ rosrun rviz rviz
#   Then in the keyboard interface in explore_fixed_pt.py, cycle between U/D,
#     G, O commands, to move hand up and down, open and close, to get tactile
#     measurements on object.
#


# ROS
import rospy
import tf
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from std_msgs.msg import Bool, Int32

# ReFlex
from reflex_msgs.msg import Hand

# Python
import sys
import random
import numpy as np
import time, datetime
import os

# Local
from tactile_map.get_contacts import get_contacts_live, GetContactsRecorded, \
  eliminate_duplicates
from tactile_map.create_marker import create_marker
from est_ransac_line import calc_3d_centroid, est_least_sqr_line, \
  est_ransac_line, calc_pt_to_line_dist, project_pts_onto_line
from spin_seam import spin_cloud


class EstCenterAxis_RANSAC:

  def __init__ (self, arm_side, PCD0_LIVE1_TXT2, sim,
    out_record_flag=False, out_record_path=''):

    ## Constants
    # 100 for default. 20 for very slanted objects like GSK 500 ml.
    #   Also change in reflex_driver/launch/reflex_driver.launch, so that
    #   guarded_move knows to terminate at same threshold.
    self.contact_thresh = 15

    self.arm_side = arm_side
    self.sim = sim
    self.out_record_flag = out_record_flag

    self.PCD0_LIVE1_TXT2 = PCD0_LIVE1_TXT2

    # Choose RANSAC or least squares for fitting line to find center axis of obj
    self.RANSAC0_LS1 = 0
    self.fit_least_sqrs_after_ransac = False


    # Used for pcd file rostopic
    self.pcd_cloud = None
    # Remainder of cloud from picking out a few each iteration
    self.pcd_cloud_remain = None

    # Used for live rostopic
    self.sensor_values = None

    # Used for input recorded contacts text file
    self.in_record_reader = None

    # If subscribe to PointCloud from pcd file
    if self.PCD0_LIVE1_TXT2 == 0:
      rospy.Subscriber ('/cloud1_pcd', PointCloud, self.pcdCB)
      
    # If subscribe to live rostopic
    elif self.PCD0_LIVE1_TXT2 == 1:
      rospy.Subscriber ('/reflex_hand', Hand, self.handCB)

    elif self.PCD0_LIVE1_TXT2 == 2:
      in_record_rosparam = '/tactile_map/est_center_axis_ransac/in_record_file'
      in_record_name = ''

      if rospy.has_param (in_record_rosparam):
        in_record_name = rospy.get_param (in_record_rosparam)

      else:
        print ('ERROR in EstCenterAxis_RANSAC __init__(): rosparam not found: %s . Cannot create object.' % in_record_name)
        return

      self.in_record_reader = GetContactsRecorded (in_record_name)


    # Output record text file
    self.out_record_file = None
    if self.out_record_flag:

      # Ref: http://stackoverflow.com/questions/13890935/timestamp-python
      timestamp = time.time ()
      timestring = datetime.datetime.fromtimestamp (timestamp).strftime (
        '%Y-%m-%d-%H-%M-%S')

      # Create output file
      out_record_name = os.path.join (out_record_path, timestring + '.txt')

      self.out_record_file = open (out_record_name, 'w')


    self.tfTrans = tf.TransformListener ()
    self.vis_pub = rospy.Publisher ('/visualization_marker', Marker)
    self.RANSAC_MARKER_ID = 0
    self.LS_MARKER_ID = 1
    self.N_ITERS_MARKER_ID = 2

    # On real robot, keep markers around forever, because usually need to debug
    #   or take screenshots.
    self.marker_duration = 60
    if not self.sim:
      self.marker_duration = 0

    # geometry_msgs/Point32[] for PCD, geometry_msgs/Point[] for live
    self.cached_contacts = []
    # List of Numpy arrays. [array([x1, y1, z1]), ... array([xn, yn, zn])]
    self.processed_contacts = []
    self.center_msg_id = 0
    # geometry_msgs/Point[]
    self.centroids = []

    self.processed_contacts_cloud = PointCloud ()
    self.contacts_pub = rospy.Publisher (
      '/tactile_map/contact/cloud', PointCloud)

    self.obj_pub = rospy.Publisher (
      '/tactile_map/est_center_axis_ransac/predicted_shape', PointCloud)

    # Tells us that hand is in movement. Do not look at current tactile values
    #   or finger positions as contact points.
    #   This is important because hand moves really fast. Delay in code is 
    #   longer than delay in hand opening. So you might be looking at contact
    #   points a moment ago, and now looking up tf, but the hand has started
    #   moving, so you end up getting a point in mid-air from tf, because
    #   that's where the finger was in that split second, but that's not where
    #   the contact was.
    rospy.Subscriber ('/tactile_map/pause', Bool,
      self.pauseCB)
    self.pause_contacts = False

    # Number of times we've looked at (including caching them when there aren't
    #   enough to calculate centroid in an iteration) the current contact.
    rospy.Subscriber ('/tactile_map/explore_fixed_pos/n_grasps', Int32,
      self.n_grasps_CB)
    self.n_grasps = -1
    self.n_sim_grasps = 0


  def close_input_file (self):
    self.in_record_reader.close_file ()

  def close_output_file (self):
    if self.out_record_file is not None:
      self.out_record_file.close ()

  def pcdCB (self, msg):
    self.pcd_cloud = PointCloud (msg.header, msg.points, msg.channels)

  def handCB (self, msg):
    self.sensor_values = Hand (msg.finger, msg.palm, msg.joints_publishing, 
      msg.tactile_publishing)


  def pauseCB (self, msg):

    if self.pause_contacts != msg.data:
      if msg.data:
        print ('Pause contact detection')
      else:
        print ('Resume contact detection')

    self.pause_contacts = msg.data


  def n_grasps_CB (self, msg):
    self.n_grasps = msg.data


  # Pick a small number of random points from the pcd file cloud
  # Returns Point32[]
  def get_contacts_pcd (self):

    # Ret val. Point32[]
    chosen_contacts = []

    MAX_N_CONTACTS = 3;

    # Generate a random number. This is the number of points to take from the
    #   pcd point cloud, in this iteration
    num_contacts = random.randint (1, MAX_N_CONTACTS)

    # Generate indices. These are the points to take from cloud
    idx = random.sample (range (1, len (self.pcd_cloud_remain.points)), num_contacts)
    # Sort the index in reverse order, so can easily remove these elts from list
    idx.sort (None, None, True)
    print ('Chosen indices: ' + str (idx))

    # Extract the chosen points from point cloud. Remove from remainder cloud.
    # Assumption: idx is sorted in reverse (decreasing) order, e.g. [9, 3, 1].
    #   So when remove in order, elements with larger index get removed first.
    #   This way, you don't need to decrement all remaining indices by 1, after
    #   removing each element (you'd have to if sorted in increasing order).
    for i in range (0, len (idx)):
      chosen_contacts.append (self.pcd_cloud_remain.points.pop (idx [i]))

    return [chosen_contacts, self.pcd_cloud_remain.header.frame_id]


  def estimate (self):

    ## Sanity checks

    if self.PCD0_LIVE1_TXT2 == 0:
      # No cloud received yet
      if self.pcd_cloud is None:
        return

      # If this is the first iteration
      if self.pcd_cloud_remain is None:
        self.pcd_cloud_remain = PointCloud (self.pcd_cloud.header,
          self.pcd_cloud.points, self.pcd_cloud.channels)

    elif self.PCD0_LIVE1_TXT2 == 1:
      # No sensor values received yet
      if self.sensor_values is None:
        return

      # If current contacts are invalid (happens when hand is in open/close 
      #   motion), don't look at them.
      if self.pause_contacts:
        return


    ## Get the contact points this iteration

    # PCD
    if self.PCD0_LIVE1_TXT2 == 0:
      # Ret val: geometry_msgs/Point32[]
      [chosen_contacts, contacts_frame] = self.get_contacts_pcd ()

    # Live rostopic
    elif self.PCD0_LIVE1_TXT2 == 1:
      contacts_frame = '/base'
      # Ret val: geometry_msgs/Point[]
      [chosen_contacts, _, _, _, _] = get_contacts_live (self.sensor_values,
        self.contact_thresh, contacts_frame, self.tfTrans, True)

    # Recorded text file
    elif self.PCD0_LIVE1_TXT2 == 2:
      chosen_contacts = self.in_record_reader.get_contacts_recorded ()


    # If no contacts, do nothing
    if len (chosen_contacts) < 1:
      return

    # Eliminate duplicates of captured contacts
    [chosen_contacts, _, isDup, _, _] = eliminate_duplicates (
      self.processed_contacts_cloud.points + self.cached_contacts,
      chosen_contacts)

    # If all contact points are duplicates, hand is still in previous contact
    #   with object. Nothing new to do.
    if all (isDup):
      return

    # TODO Now in eliminate_duplicates(). Remove this block when new fn works.
    '''
    # If there are any contact points detected this iteration, and if these
    #   are not the first ones, check if they are duplicates of captured ones.
    if (len (self.processed_contacts_cloud.points) > 0 or \
        len (self.cached_contacts) > 0):
      # Exclude new contact points that are same as previous
      #   ones. This happens if the touch on object is longer than one iter,
      #   which is like, always!
      # Bool[]
      isDup = check_dup_contacts (
        self.processed_contacts_cloud.points + self.cached_contacts,
        chosen_contacts)
        # cached_contacts: Point[]
        # processed_contacts_cloud.points: Point32[]
        # chosen_contacts: Point[]

      # If all contact points are duplicates, hand is still in previous contact
      #   with object. Nothing new to do.
      if all (isDup):
        return

      # If only some contact points are duplicates, hand has moved, we can
      #   process the non-duplicate contacts as normal.
      chosen_contacts = [chosen_contacts[i] \
        for i in range (0, len(chosen_contacts)) if not isDup[i]]
    '''


    ## If there are any contacts in cache (cache stores points that were too
    #    few to process in previous iterations)

    # Add cached points to current iter's points, clear cache
    if len (self.cached_contacts) > 0:
      chosen_contacts.extend (self.cached_contacts)
      del self.cached_contacts [:]


    ## Plot contacts in this iteration. Per-iteration noncumulative, by same 
    #    marker ID.

    # yellow
    marker_con = Marker ()
    # Parameters: tx, ty, tz, r, g, b, alpha, sx, sy, sz
    create_marker (Marker.POINTS, 'contacts', '/base', 0,
      0, 0, 0, 1, 1, 0, 0.5, 0.01, 0.01, 0.01,
      marker_con, self.marker_duration)  # Use 0 duration for forever

    for i in range (0, len (chosen_contacts)):
      marker_con.points.append (chosen_contacts [i])

    self.vis_pub.publish (marker_con)


    ## If we haven't returned at this point yet, there are new contact points
    #   in this iteration.
    self.n_sim_grasps += 1


    ## Calc centroid from contacts in current iteration
    #    Require at least 3 points. Else no centroid possible, by def of
    #    centroid.

    # If got >= 3 contacts, can calc centroid, else save in cache for next iter
    #   when have more points.
    if len (chosen_contacts) < 3:
      self.cached_contacts.extend (chosen_contacts)
      print ('Less than 3 new contacts. Caching.')
      return

    else:

      # Check variance in x y z components. Make sure >= 2 components
      #    have a good amount of variance.
      if not self.check_variance (chosen_contacts):
        print ('Stdev too low. Caching.')
        self.cached_contacts.extend (chosen_contacts)
        return


      # 1 geometry_msgs/Point
      # Pass in > 3 geometry_msgs/Point
      centroid = calc_3d_centroid (chosen_contacts)

      # Add to list of contact points seen. Convert to Numpy array.
      self.processed_contacts.extend ( \
        [np.asarray ([p.x, p.y, p.z]) for p in chosen_contacts])

      # Add to cumulative contact cloud
      self.processed_contacts_cloud.points.extend (chosen_contacts)

      # Add to list of centroids
      self.centroids.append (centroid)

      tmp_nContacts_str = format ('%d' %(len (chosen_contacts)))
      print (tmp_nContacts_str)
      if self.out_record_flag:
        # Record for testing in simulation
        self.out_record_file.write (tmp_nContacts_str + '\n')

      for i in range (0, len (chosen_contacts)):
        tmp_contact_str = str2 = format ('%g %g %g' \
          %(chosen_contacts [i].x, chosen_contacts [i].y,
          chosen_contacts [i].z))
        print (tmp_contact_str)

        if self.out_record_flag:
          self.out_record_file.write (tmp_contact_str + '\n')

    print ('%d contact points' %(len (self.processed_contacts)))
    #print (self.processed_contacts)
    print ('%d centroids' %(len (self.centroids)))
     

    ## Plot contact cloud so far. Cumulative from all iterations

    self.processed_contacts_cloud.header.stamp = rospy.Time.now ()
    self.processed_contacts_cloud.header.frame_id = '/base'

    self.contacts_pub.publish (self.processed_contacts_cloud)


    ## Plot centroid in this iteration. Cumulative from all iterations by
    #    unique marker id.

    # orange
    marker_cen = Marker ()
    self.center_msg_id = self.center_msg_id + 1
    create_marker (Marker.POINTS, 'centroids', '/base', self.center_msg_id,
      0, 0, 0, 1, 0.5, 0, 0.5, 0.01, 0.01, 0.01,
      marker_cen, self.marker_duration)  # Use 0 duration for forever

    marker_cen.points.append (centroid)
    self.vis_pub.publish (marker_cen)


    ## If only have one centroid so far, no line fitting to do, just return
    if len (self.centroids) < 2:
      return


    ## Find center axis, using the centroids found

    # 20 cm. Length for plotting in RViz
    LINE_LEN = 0.2

    if self.RANSAC0_LS1 == 1:

      ## Fit least squares line through the cumulative set of centroids.
      #    Note this isn't as robust as RANSAC line, if there are lots outliers.
      #      Least squares just fits to as many pts as possible, which isn't
      #      always the right answer.

      # Fit
      [vx, vy, vz, x0, y0, z0] = est_least_sqr_line (self.centroids)
     
      # Make a start point and end point
      axis_ls_start = Point (x0, y0, z0)
      axis_ls_end = Point (x0 + vx * LINE_LEN, y0 + vy * LINE_LEN,
        z0 + vz * LINE_LEN)
     
     
      ## Plot least squares fitted line through centroids. Per iteration
      #    non-cumulative, by using same marker ID.
     
      # LINE_LIST marker: only scale.x is used, for width of line segments.
      # orange
      marker_axis_ls = Marker ()
      create_marker (Marker.LINE_LIST, 'center_axis', '/base',
        self.LS_MARKER_ID, 0, 0, 0, 1, 0.5, 0, 0.5, 0.01, 0, 0,
        marker_axis_ls, self.marker_duration)  # Use 0 duration for forever
     
      marker_axis_ls.points.append (axis_ls_start)
      marker_axis_ls.points.append (axis_ls_end)
     
      self.vis_pub.publish (marker_axis_ls)


    else:

      ## Estimate center axis, using the sample points from current iteration
      #  TODO: perhaps better idea: after estimate enough centroids, just use
      #    all the centroids for RANSAc?? Yes!! That sounds so much more 
      #    reasonable than taking a bunch of points on the boundary! Then it's
      #    just a standard "ransac line" problem!

      # Ret val 1 is list of 2 indices, they index the param passed in
      # Ret val 2 is list of indices of inliers, includes ret val 1.
      [axis_idx, inliers_idx] = est_ransac_line (self.centroids)
      # Try RANSAC with all contact points, instead of just the centroids
      #[axis_idx, inliers_idx] = est_ransac_line (self.processed_contacts_cloud.points)

      # Fit a least squares line through the highest voted line and inliers
      vx = vy = vz = x0 = y0 = z0 = 0
      if self.fit_least_sqrs_after_ransac:
        inputs = [self.centroids [i] for i in inliers_idx]
        # Try RANSAC with all contact points, instead of just the centroids
        #inputs = [self.processed_contacts_cloud.points [i] for i in inliers_idx]
        # (vx vy vz) is a unit vector
        # (x0 y0 z0) is a pt on line
        (vx, vy, vz, x0, y0, z0) = est_least_sqr_line (inputs)

      # Don't do least squares after RANSAC
      else:
        pt1 = np.asarray ([
          self.centroids [axis_idx [0]].x,
          self.centroids [axis_idx [0]].y,
          self.centroids [axis_idx [0]].z])
        pt2 = np.asarray ([
          self.centroids [axis_idx [1]].x,
          self.centroids [axis_idx [1]].y,
          self.centroids [axis_idx [1]].z])
 
        vec = pt2 - pt1
        vec = vec / np.linalg.norm (vec)
        vx = vec [0]
        vy = vec [1]
        vz = vec [2]
 
        x0 = pt1 [0]
        y0 = pt1 [1]
        z0 = pt1 [2]

      # Make a start point and end point
      axis_rs_start = Point (x0, y0, z0)
      # Try RANSAC with all contact points, instead of just the centroids
      #axis_rs_start = calc_3d_centroid (self.processed_contacts_cloud.points)
      axis_rs_end = Point (
        axis_rs_start.x + vx * LINE_LEN,
        axis_rs_start.y + vy * LINE_LEN,
        axis_rs_start.z + vz * LINE_LEN)


      ## Plot RANSAC fitted line. Per iteration non-cumulative, by using same
      #    marker ID.

      # red for RANSAC
      marker_axis_rs = Marker ()
      create_marker (Marker.LINE_LIST, 'center_axis', '/base',
        self.RANSAC_MARKER_ID, 0, 0, 0, 1, 0, 0, 0.5, 0.01, 0, 0,
        marker_axis_rs, self.marker_duration)  # Use 0 duration for forever
     
      marker_axis_rs.points.append (axis_rs_start)
      marker_axis_rs.points.append (axis_rs_end)
     
      self.vis_pub.publish (marker_axis_rs)


      # Plot text saying how many contacts there has been

      marker_n_iters = Marker ()
      # red
      # TEXT Marker: only scale.z is used, height of uppercase "A"
      # Offset text a bit longer than head of arrow.
      create_marker (Marker.TEXT_VIEW_FACING, 'center_axis', '/base',
        self.N_ITERS_MARKER_ID,
        axis_rs_end.x + 0.01, axis_rs_end.y + 0.01, axis_rs_end.z + 0.01,
        1, 0, 0, 0.5, 0, 0, 0.02,
        marker_n_iters, self.marker_duration)  # Use 0 duration for forever

      # If publisher for n_grasps isn't running (happens when in sim), print
      #   n_sim_grasps, our best estimate for a "grasp"
      if self.n_grasps < 0:
        marker_n_iters.text = str (self.n_sim_grasps) + ' grasps'
      else:
        marker_n_iters.text = str (self.n_grasps) + ' grasps'
      self.vis_pub.publish (marker_n_iters)


      ## Plot inliers in a different color than yellow
      # TODO best way is to replace them, using their existing marker ID, 
      #   instead of plotting new ones. saves memory and no overlapping points 
      #   to confuse me.

      # Use inlier_idx


      ## Predict and plot what the object might look like, using the estimated
      #    center axis and perpendicular distance of contact points to the axis.
      self.predict (
        np.asarray ([axis_rs_start.x, axis_rs_start.y, axis_rs_start.z]), 
        np.asarray ([axis_rs_end.x, axis_rs_end.y, axis_rs_end.z]),
        self.processed_contacts)


  # Predict what we think the object looks like, based on the center axis
  #   estimate
  # Parameters:
  #   pt1, pt2: 1 x 3 Numpy array. Vector parallel to a line
  #   pts: nPts-item list of Numpy arrays. Contct points.
  #     Type: [array([x1, y1, z1]), ... array([xn, yn, zn])]
  def predict (self, pt1, pt2, pts):

    ## Calculate perpendicular distance from each point to axis.
    #  Perpendicular distance from a contact point to center axis tells us the
    #    *radius* of the object at that "object height" (object height := is
    #    measured wrt center axis)

    # 1 x nPts Numpy array. Index corresponds to pts idx.
    dists_to_axis = calc_pt_to_line_dist (pt1, pt2, pts)

    #print (dists_to_axis)
    #print ('Max dist to axis: %f, Min: %f' %(max (dists_to_axis), min (dists_to_axis)))


    ## Calculate what axis heights the points in "pts" array are at. i.e.
    #    distance from pt1 to the projection of point p onto axis (pt2 - pt1).
    #    Height is measured from pt1 (arbitrarilly selected. You can sel pt2,
    #    same thing).

    # 1 x nPts Numpy array. Index corresponds to pts idx
    heights = project_pts_onto_line (pt1, pt2, pts)

    #print (heights)
    #print ('Max object height from axis start point: %f, Min: %f' %(max (heights), min (heights)))


    ## TODO: will have to do something about outliers. I sometimes have points
    #    outside the object. Somehow exclude these from pts, and the
    #    corresponding elts in dists_to_axis, and heights arrays.


    ## Plot what the object might look like, i.e. plot additional points around 
    #    axis (pt2 - pt1), +/- direction of vector doesn't matter.
    #  Let each point on the axis a "height" on the axis. Then a set of points
    #    will be plotted at 360 degrees around each known height. A known height
    #    is a height at which there exists a point p in "pts" array projects
    #    perpendicularly to. The set of 360-degree points will have the same
    #    radius as the perpendicular distance of p to the axis.
    #  In plain words, at each axis height that there exists a point p in "pts"
    #    array, a circle will be plotted. Radius of circle = perpendicular
    #    distance of point p to the axis.

    predicted_shape = PointCloud ()
    spin_cloud (pt1, pt2, heights, dists_to_axis, '/base', predicted_shape)

    self.obj_pub.publish (predicted_shape)


  # Parameters:
  #   pts: geometry_msgs/Point[] or geometry_msgs/Point32[]
  # Returns True if variance in pts is greater than threshold for >= 2
  #   components in x y z, else False.
  def check_variance (self, pts):

    thresh = 0.02

    std_x = float (np.std (np.asarray ([p.x for p in pts])))
    std_y = float (np.std (np.asarray ([p.y for p in pts])))
    std_z = float (np.std (np.asarray ([p.z for p in pts])))

    print ('Stdev: x %f, y %f, z %f' %(std_x, std_y, std_z))

    # True + True = 2, in Python boolean type. (Numpy boolean type doesn't work)
    # Return True if sum of three booleans >= 2, meaning >= 2 components have
    #   stdev greater than threshold.
    return ((std_x > thresh) + (std_y > thresh) + (std_z > thresh) >= 2)


if __name__ == '__main__':

  rospy.init_node ('est_center_axis_ransac', anonymous=True)

  arm_side = 'left'
  # Parse cmd line args
  for i in range (0, len (sys.argv)):
    if sys.argv [i] == '--left':
      arm_side = 'left'
    elif sys.argv [i] == '--right':
      arm_side = 'right'

    elif sys.argv [i] == '--out_record_flag':
      i += 1
      if sys.argv [i] == '0' or sys.argv [i] == 'false' or \
        sys.argv [i] == 'False':
        out_record_flag = False
      else:
        out_record_flag = True

    elif sys.argv [i] == '--out_record_path':
      i += 1
      out_record_path = sys.argv [i]
  print ('Arm set to ' + arm_side + ' side')
  print ('Record-this-session set to ' + str (out_record_flag))
  if out_record_flag:
    print ('Output path set to ' + out_record_path)

  sim = False
  if rospy.has_param ('/use_sim_time'):
    sim = rospy.get_param ('/use_sim_time')
  rospy.loginfo ('/use_sim_time found to be ' + str (sim))


  # Mode. PCD or live /reflex_hand? Usually, 0 for sim, 1 for real robot.
  # Listen to a recorded full PointCloud from a .pcd file, or live contact
  #   points on rostopic (can be on real robot, or recorded in bag file).
  # If from .pcd file, will pick a small number of points per iteration, to
  #   treat as they are contacts in real time.
  PCD0_LIVE1_TXT2 = 0
  if rospy.has_param ('/tactile_map/est_center_axis_ransac/PCD0_LIVE1_TXT2'):
    PCD0_LIVE1_TXT2 = rospy.get_param (
      '/tactile_map/est_center_axis_ransac/PCD0_LIVE1_TXT2')
  rospy.loginfo ('PCD0_LIVE1_TXT2 found to be ' + str (PCD0_LIVE1_TXT2))


  thisNode = EstCenterAxis_RANSAC (arm_side, PCD0_LIVE1_TXT2, sim,
    out_record_flag, out_record_path)

  # 1 Hz, slow enough for user to see each iteration in RViz.
  #   Adjust up for real time testing.
  wait_rate = rospy.Rate (1)
  while not rospy.is_shutdown ():

    thisNode.estimate ()

    try:
      wait_rate.sleep ()
    except rospy.exceptions.ROSInterruptException, err:

      # Close opened file
      if PCD0_LIVE1_TXT2 == 2:
        thisNode.close_input_file ()

      if out_record_flag:
        thisNode.close_output_file ()

      break


