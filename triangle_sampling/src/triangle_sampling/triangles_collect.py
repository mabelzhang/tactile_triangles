#!/usr/bin/env python

# Mabel Zhang
# 25 Aug 2015
#
# Collect triangles data from physical ReFlex Hand contacts on 3 points on
#   real object surface.
# Two types of data are collected:
#   .pcd file:
#     XYZ points wrt robot. "absolute XYZs".
#   .csv file:
#     Triangles, without absolute XYZ points, thus without historical points
#     (in case object pose changed, which we wouldn't know. So absolute XYZs
#     are never known).
#
# Much starter code copied from ./weights_collect.py
#
# Usage:
#   First make sure baxter desktop time is synced with baxter robot.
#     See baxter_reflex README for how to do this, using date --set.
#
#   Launch all of these:
#
#   (See latest and more detailed descriptions in README)
#
#   1. Baxter+ReFlex driver:
#   $ roslaunch baxter_reflex baxter_reflex.launch
#
#   2. Baxter arm keyboard control, my custom one:
#   $ rosrun baxter_reflex joint_position_keyboard.py
#
#   3. Detect contacts on ReFlex, publish tactile_map_msgs.Contacts and RViz:
#   $ rosrun tactile_map detect_reflex_contacts.py
#
#   4. Launch this script:
#   $ rosrun tactile_collect triangle_collect.py
#
#   5. Launch keyboard controller for this node's triangle data recording:
#   $ rosrun tactile_map keyboard_interface.py
#
#
# Usage for creating a Python file outside that wants to use this node:
#   See main() here for what you can call to publish visualizations.
#   Your node needs to call collect_one_contact(), set doFinalWrite=True,
#     etc key functions.
#
#   Don't call prep_for_termination directly. Set doFinalWrite=True,
#     then call collect_one_contact().
#
#   See triangles_collect_semiauto.py for example usage for outside node.
#
#
#   Other essential functions to mirror in your file:
#
#   Your node needs to subscribe to contacts message, like contactsCB()
#     here, and set this node's contactsMsg member field to the msg
#     received.
#
#   Your node needs to set the program flow flags in this node, like
#     doCollectOne. Refer to the keyCB() functions in this node, for how
#     and which flags to set.
#

# ROS
import rospy
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import Empty
from sensor_msgs.msg import PointCloud  # TODO test this
from sensor_msgs.msg import JointState

# Python
import os, time, datetime
import csv
from copy import deepcopy
import shutil

# Numpy
import numpy as np

# ReFlex
from baxter_reflex.reflex_control import call_smart_commands

# My packages
from tactile_map.get_contacts import get_contacts_live
from tactile_map.detect_reflex_contacts import visualize_contacts, \
  visualize_individual_contacts
from tactile_map_msgs.msg import Contacts
from tactile_map.create_marker import create_marker
from triangle_sampling import sample_reflex
from tactile_collect import tactile_config
from util.ansi_colors import ansi_colors
from util.pcd_write_header import pcd_write_header, combine_many_pcd_files
from util.csv_combine import combine_many_csv_files
from triangle_sampling.config_paths import get_pcd_path, \
  get_robot_tri_path, get_robot_obj_params_path
from triangle_sampling.config_hist_params import TriangleHistogramParams as \
  HistP
from triangle_sampling.io_per_move import PerMoveConsts, get_per_move_name, \
  record_per_move_n_pts_tris


class TrianglesCollect:

  # Parameters:
  #   csv_suffix: '' for real robot data. 'gz_' for gazebo hand sampling
  #     2016 09 07: Trying 'bx_' for real robot data, see if anything gets
  #     messed up. bx is better indication of Baxter!
  #   sample_robot_frame: Specify False if you only want to sample hand frame.
  #     Our focus is hand frame. You can sample robot frame as well, for
  #     comparison. For one, robot frame is probably more accurate, so it's
  #     good for debugging. But if you are running in simulation and tight on
  #     time, then don't do robot frame.
  def __init__ (self, csv_suffix='', pickup_from_file='',
    sample_robot_frame=True):

    self.csv_suffix = csv_suffix
    self.pickup_from_file = pickup_from_file

    # This will be used whenever robot_valid is used.
    self.sample_robot_frame = sample_robot_frame

    #####
    # Keyboard interface
    #   Copied from weights_collect.py
    #####

    # Prompt to display at keyboard_interface.py prompt
    self.prompt_pub = rospy.Publisher ('keyboard_interface/prompt', String,
      queue_size=2)
    self.prompt_msg_default = 'Press l to collect a contact sample, z to zero pressures, o to open, g to guarded_move, c / s / p for cylinder / sphere / pinch preshapes, '
    self.prompt_msg = String ()
    self.prompt_msg.data = self.prompt_msg_default

    rospy.Subscriber ('/keyboard_interface/key', String, self.keyCB)

    self.doCollectOne = False
    self.doFinalWrite = False
    self.doTerminate = False

    self.zero_tactile = rospy.ServiceProxy('/zero_tactile', Empty)

    # [0] is palm, [1:3] is fingers 0 to 2
    #self.pressures_latest = [[0] * 11, [0] * 9, [0] * 9, [0] * 9]
    self.sensor_marker_ids = [range(0,11), range(11,20), range(20,29),
      range(29,38)]


    #####
    # Contact
    #   Copied from weights_collect.py
    #####

    # Threshold for this msg is in detect_reflex_contacts.py . Contact is
    #   only published if a pressure is greater than the threshold set
    #   in that node.
    rospy.Subscriber ('/tactile_map/detect_reflex_contacts/contacts',
      Contacts, self.contactsCB)
    self.contactsMsg = None
    # So that we don't process same msg more than once
    self.seenContactsSeq = 0

    # Empirically observed to be the proximal joint position where finger is
    #   touching palm.
    # Updated from my reflex_gazebo reflex_base.py. Observed in simulation only.
    #   What you command in reflex_base.py might be slightly higher than this,
    #   because if finger collides palm before closing all the way, then we'd
    #   see this lower number in /joint_states, not the absolute joint limit
    #   max in reflex_base.py.
    # This only accounts for when fore-finger preshape is cylinder
    #self.TENDON_MAX = 2.78
    # This also accounts for spherical fore-finger preshape, in which
    #   fore-fingers don't close as far before one finger is on top of the other
    self.TENDON_MAX = 2.35

    # Empirically observed to be the proximal joint position where finger is
    #   opened all the way
    # Used by sample_gazebo.py from outside.
    # 0.08 radians is 5 degrees. Less than 5 degs should be considered open
    self.TENDON_MIN = 0.08

    # This is published by hand_visualizer in reflex_visualizer package,
    #   by looking at the /reflex_hand rostopic. So it might be slower than
    #   /reflex_hand in timestamp.
    # In sim, you can use /rhr_flex_model/joint_states, which is published
    #   by Gazebo. But this won't work for real robot. So you need some kind
    #   of conditional here for sim or not. Too lazy to do that now. And
    #   this is better anyway, it makes sim and real do the same thing, which
    #   is always better!
    rospy.Subscriber ('/joint_states', JointState, self.jointsCB) 
    self.joint_states = None


    #####
    # Triangle sampling from contacted points
    #####

    # Number of points in PCD file. This is for writing PCD header. It doesn't
    #   mean how many points we see in this script. It means how many are
    #   actually written to file, regardless of how many seen by script.
    self.nPts = 0

    # Erroneous. `.` can double count. Not using this anymore! Just use
    #   len(seen[0]) to get current number of triangles!
    # Number of triangles visualized in RViz. This is for RViz marker_id
    # Note this can be different from the number of triangles written to
    #   file, which can be fewer, if near-duplicates were found.
    #self.nTris_robot = 0

    # geometry_msgs/Point[]
    self.contacts_latest_hand = list ()
    self.contacts_cached_robot = list ()
    self.contacts_cached_hand = list ()

    # Triangles seen
    # Used to do a list, but duplicate check was way too slow, so doing NumPy
    #   arrays and subtraction now. If you do choose to do a list, do NOT do
    #   "[[]] * 6"!!! Made that mistake. That refers to the same
    #   list instance, does not make a copy! Then adding to one inner list
    #   adds to all 6 inner lists! It'd drive you nuts for an hour, ask me
    #   how I know!
    # NumPy n x 6 matrix
    self.params_seen_robot = np.zeros ((0, 6))
    self.params_seen_hand = np.zeros ((0, 6))
    self.params_latest_robot = np.zeros ((0, 6))
    self.params_latest_hand = np.zeros ((0, 6))

    # Delta within which to count two measurements as duplicates.
    # Length is in meters
    #   1e-2 was adjusted for real robot, but I don't know if that's right
    #     anymore, so many triangles are left out, which might not be a good
    #     thing.
    self.DUP_DELTA_L = 1e-4
    # Angle is in radians.
    #   6e-2 was adjustd for real robot. It allows within ~3 degs to be counted
    #     as dups. 3 degs = 0.0523599 radians.
    #     But I don't know if that's right anymore, so many triangles are left
    #     out, which might not be a good thing.
    self.DUP_DELTA_A = 1.0 / 180.0 * np.pi

    self.L0_IDX = HistP.L0_IDX  #0
    self.L1_IDX = HistP.L1_IDX  #1
    self.L2_IDX = HistP.L2_IDX  #2
    self.A0_IDX = HistP.A0_IDX  #3
    self.A1_IDX = HistP.A1_IDX  #4
    self.A2_IDX = HistP.A2_IDX  #5


    #####
    # Visualization in RViz
    #####

    self.VISUALIZE = False

    # This doesn't determine what gets published on Contacts msg. It only
    #   determines what gets visualized in this file, ABOVE the
    #   threshold set in publisher of Contacts msg (detect_reflex_contacts.py)
    self.contact_thresh = 10

    #self.vis_pub = rospy.Publisher ('visualization_marker', Marker)
    self.vis_arr_pub = rospy.Publisher ('visualization_marker_array',
      MarkerArray, queue_size=2)
    self.text_marker_ids_seen = 0
    self.cumu_marker_dur = 60
    self.text_height = 0.008

    # Store latest markers and cumulative markers, for caller to publish in
    #   main ROS loop.
    # These are erased every iteration we record new triangles
    self.markers = MarkerArray ()
    # These are never erased, cumulative
    self.markers_cumu = MarkerArray ()

    # Publish the cumulative points collected into PCD as PointCloud, instead
    #   of Marker, to make Baxter desktop less laggy.
    # TODO test this
    self.cloud_pub = rospy.Publisher ('/triangles_collect/collected',
      PointCloud, queue_size=2)
    self.collected_cloud = PointCloud ()


    #####
    # File I/O for output files
    #####

    self.pcd_name = None
    self.pcd_tmp_name = None
    self.trih_name = None
    self.trir_name = None
    self.per_move_name = None

    self.useNormals = True

    self.timestring = None

    # Use 0, not -1, because per_run.csv would record the -1 if there are no
    #   contacts, then the human readable per_run.txt would have -1 instead
    #   of 0! Not good if I decide to sum at some point.
    self.nPts_ttl = 0
    self.nTris_h_ttl = 0
    self.nTris_r_ttl = 0

    # This is not the hardware number of moves. It's the number of times we
    #   inspected contact points from the rostopic. Most likely, caller (who
    #   controls the hardware) would call collect_one_contact() after each
    #   hardware movement. So this is an approximate number of hardware
    #   hand movements.
    self.nCollects = 0


  # Copied from weights_collect.py
  # ROS node: make sure to call this every iteration
  def pub_keyboard_prompt (self):

    self.prompt_pub.publish (self.prompt_msg)


  # Only in charge of file I/O commands.
  # For ReFlex control commands, use reflex_keyboard.py
  # Copied and updated from weights_collect.py
  def keyCB (self, msg):

    if msg.data.lower () == 'l':
      print 'Got user signal to collect a batch of data, starting collection...'
      self.doCollectOne = True

    elif msg.data.lower () == 'z':
      # Zero tactile, so next collection has accurate data
      # It's bad to do this in a CB function, but it has to be here, because
      #   you must zero when the finger is touching NOTHING. Else it's wrong
      #   zeroing! Only user knows when the finger is touching nothing, so
      #   this must be user-issued!
      print ('Zeroing tactile values...'), 
      self.zero_tactile ()
      rospy.sleep(2)
      print ('Done')

    #####
    # Preshapes

    # Cylinder preshape
    elif msg.data.lower () == 'c':
      call_smart_commands ('cylinder')

    # Spherical preshape
    elif msg.data.lower () == 's':
      call_smart_commands ('spherical')

    # Pinch preshape
    elif msg.data.lower () == 'p':
      call_smart_commands ('pinch')

    #####
    # Close and open

    elif msg.data.lower () == 'g':
      call_smart_commands ('guarded_move', 5)

    elif msg.data.lower () == 'o':
      call_smart_commands ('open')

    #####
    # Program flow

    elif msg.data.lower () == 'q':
      print 'Got user signal to terminate program...'
      self.doFinalWrite = True


  def contactsCB (self, msg):

    self.store_contacts_as_lists (msg)


  def store_contacts_as_lists (self, msg):

    self.contactsMsg = deepcopy (msg)

    # Convert all tuples to lists, so they are mutable. I will need to remove
    #   some elts, e.g. if the contact is btw finger and palm, then the contact
    #   pt is not part of object, we don't want this kind of contacts.
    self.contactsMsg.pressures = list (msg.pressures)
    self.contactsMsg.finger_idx = list (msg.finger_idx)
    self.contactsMsg.sensor_idx = list (msg.sensor_idx)

    self.contactsMsg.pose_wrt_hand = list (msg.pose_wrt_hand)
    self.contactsMsg.norm_endpt_wrt_hand = list (msg.norm_endpt_wrt_hand)

    self.contactsMsg.pose_wrt_robot = list (msg.pose_wrt_robot)
    self.contactsMsg.norm_endpt_wrt_robot = list (msg.norm_endpt_wrt_robot)

    self.contactsMsg.pose_wrt_cam = list (msg.pose_wrt_cam)
    self.contactsMsg.norm_endpt_wrt_cam = list (msg.norm_endpt_wrt_cam)

    self.contactsMsg.pose_wrt_obj = list (msg.pose_wrt_obj)
    self.contactsMsg.norm_endpt_wrt_obj = list (msg.norm_endpt_wrt_obj)


  def jointsCB (self, msg):
    self.joint_states = msg


  def get_n_contacts (self):
    return self.nPts

  def get_n_tris_h (self):
    return np.shape (self.params_seen_hand) [0]

  # Get latest triangles seen, calculated from contact pts wrt robot frame
  def get_latest_triangles_r (self):
    return self.params_latest_robot

  # Get latest triangles seen, calculated from contact pts wrt hand frame
  def get_latest_triangles_h (self):
    return self.params_latest_hand

  # Get latest triangles file name
  # Used by active_touch execute_actions.py
  def get_tri_h_filename (self):
    return self.trih_name


  # Parameters:
  #   cache_only_dont_eval: Cache the new contact points, do not sample
  #     triangles from them yet. This is useful if you want to accumulate more
  #     than just the minimum (3) points across multiple grasps.
  #     If False, evaluate whenever have >= 3 points (set in record_tri) in
  #     cache, the min number of points to make a triangle.
  def collect_one_contact (self, cache_only_dont_eval=False):

    # Collection is over. Write the final pcd
    if self.doFinalWrite:
      # If have anything at all to write
      if self.pcd_tmp_name:
        print ('Re-writing PCD file, now with headers, before terminating...')
        (self.pcd_name, self.pcd_file) = pcd_write_triangles_header ( \
          self.pcd_path, self.timestring, self.pcd_tmp_name, self.pcd_tmp_file,
          self.useNormals, self.nPts)

      self.prep_for_termination ()
      return


    if not self.doCollectOne:
      return

    if not self.contactsMsg:
      rospy.logwarn ('triangles_collect.py collect_one_contact(): Waiting for first contact message... Run rosrun tactile_map detect_reflex_contacts.py if you have not.')
      return

    # If have seen the contacts msg, nothing new to process. This is so that
    #   we don't process the same msg many times.
    # (Do not check if haven't seen the first contacts msg and return. `.`
    #   even before seeing first contact msg, we need to plot RViz colored
    #   guides to tell user how to start.)
    elif self.contactsMsg.header.seq <= self.seenContactsSeq:
      return


    # If first time recording in this run of the program, create output file.
    #   (This if-stmt makes it cleaner, so that if never press start, no
    #   empty file is ever created and hanging around confusing you.)
    if not self.pcd_tmp_name:
      self.create_output_files ()


    #####
    # Remove contacts on fingers, if the finger is touching palm. We don't want
    #   to record this noise on the robot body, we only want contacts on object.
    #####

    self.prune_finger_contacts_on_palm ()


    #####
    # Record contact XYZs to PCD file
    #####

    # Reset flag for next iteration
    self.doCollectOne = False

    robot_valid = self.contactsMsg.robot_valid
    hand_valid = self.contactsMsg.hand_valid

    n_new_pts = 0

    # Because this is ONE .pcd file, it will only have exactly ONE and the
    #   SAME ONE reference frame, EVER.
    #   So pick a frame that never changes, like robot frame.
    # Don't check sample_robot_frame here, need to record pcd no matter what!
    if robot_valid:
      if self.useNormals:
        n_new_pts = self.record_pcd (self.pcd_tmp_file,
          self.contactsMsg.pose_wrt_robot,
          self.contactsMsg.norm_endpt_wrt_robot)
      else:
        n_new_pts = self.record_pcd (self.pcd_tmp_file,
          self.contactsMsg.pose_wrt_robot)

    # Eh, this doesn't work. Since PCD's origin doesn't move, if you save
    #   wrt hand frame, your PCD will always just look like the hand. `.`
    #   while your hand moves in the world, changing its origin, the origin
    #   in PCD doesn't move, so its origin cannot fly around like your Gazebo
    #   world! Since the points are always wrt /base_link for hand frame,
    #   your points will always look like a hand.
    # ELSE IF, not IF, because there's only ONE PCD file! IF will put both
    #   robot frame and hand frame data in same file, and that's just garbage
    #   data.
    # Prioritize robot frame because it's fixed. If no robot frame available,
    #   then record from hand frame - because you'll always have that, since
    #   you're moving the hand around in order to touch things.
    #elif hand_valid:
    #  if self.useNormals:
    #    self.record_pcd (self.pcd_tmp_file, self.contactsMsg.pose_wrt_hand,
    #      self.contactsMsg.norm_endpt_wrt_hand)
    #  else:
    #    self.record_pcd (self.pcd_tmp_file, self.contactsMsg.pose_wrt_hand)


    #####
    # Calculate and record triangles to file
    #####

    # Add new contact points to cache
    if hand_valid:
      # Reset latest list
      self.contacts_latest_hand = []
      for i in range (0, len (self.contactsMsg.pose_wrt_hand)):
        # Store the latest to a separate member var, for active touch to access
        #   non-accumulated points from each individual wrist pose.
        self.contacts_latest_hand.append ( \
          self.contactsMsg.pose_wrt_hand [i].position)
      self.contacts_cached_hand.extend (self.contacts_latest_hand)

    if robot_valid and self.sample_robot_frame:
      for i in range (0, len (self.contactsMsg.pose_wrt_robot)):
        self.contacts_cached_robot.append ( \
          self.contactsMsg.pose_wrt_robot [i].position)

    # For logging.
    # If all are cached and no triangles computed, just leave it at 0. Then in
    #   the csv file, will just record number of points, and n tris = 0.
    #   Reader of the csv file needs to follow this pattern. On a line with
    #   non-zero number of points, if number of triangles is 0, that means the
    #   points are cached. On a line with non-zero number of triangles, that
    #   means triangles evaluated from all previous points that had 0 n tris,
    #   up to and not including the previous line with a non-zero n tris.
    n_new_tris_hand = 0
    n_new_tris_robot = 0

    # Evaluate triangles from cached points
    if not cache_only_dont_eval:

      # Record pt wrt both hand frame and robot frame, in separate files.
      #   `.` if use robot frame, then you can't
      #   claim that your triangles didn't know absolute xyz of contacts.
      #
      #   If I only record in hand frame, I'm afraid I cannot compare the
      #   difference btw how results will be if recorded in hand and robot frames.
      #   `.` even though I have pcd, since you don't know which points the
      #   triangle was from - `.` in the case of a contact that's less than 3
      #   pts, triangles could be from 5 pts from prev and curr iter, in some
      #   combination of 5 choose 3. So I wouldn't know how to reproduce the
      #   exact same thing in robot frame, using pcd. Then I cannot compare
      #   triangles btw hand frame (no absolute XYZ) and robot frame (have
      #   absolute XYZ).
      #
      #   It doesn't hurt to collect more. It'd be a nice 3-comparison for
      #   paper - dense sampling from abs xyz, live sparse triangles from
      #   abs xyz, triangles from no abs
      #   xyz. Much more painful to re-record than to have extra!
      if hand_valid:
        (recorded, self.params_seen_hand, self.params_latest_hand) = \
          self.record_tri (self.contacts_cached_hand, self.trih_writer,
            self.contactsMsg.hand_frame_id, self.params_seen_hand)
        print ('%s# triangles wrt hand frame: %d%s' % \
          (ansi_colors.OKCYAN, np.shape (self.params_seen_hand) [0], 
          ansi_colors.ENDC))
        n_new_tris_hand = np.shape (self.params_latest_hand) [0]
       
        if recorded:
       
          # Empty cache for next round. This is so that triangles do not take
          #   historical records, because object may move. Then historic 
          #   contacts shouldn't be mixed with new contacts after move!
          # TODO later: I have not decided what to do if object moves in btw
          #   caches, i.e. after cache has 1 pt, but before cache has 3 pts. Btw
          #   these two rounds, object moves. Hopefully the number of points
          #   will be few enough that the sampled triangles are outliers anyway.
          #   Also, the hope is that we get at least 3 contact points per touch,
          #   when a touch is present!! Otherwise the hardware is pretty
          #   hopeless!
          del self.contacts_cached_hand [:]
     
      if robot_valid and self.sample_robot_frame:
        (recorded, self.params_seen_robot, self.params_latest_robot) = \
          self.record_tri (self.contacts_cached_robot, self.trir_writer,
            self.contactsMsg.robot_frame_id, self.params_seen_robot)
        print ('%s# triangles wrt robot frame: %d%s' % \
          (ansi_colors.OKCYAN, np.shape (self.params_seen_robot) [0],
          ansi_colors.ENDC))
        n_new_tris_robot = np.shape (self.params_latest_robot) [0]
        if recorded:
          del self.contacts_cached_robot [:]

    self.nCollects += 1
    # Collect index is 1-based. Last index in file should = number of collects
    record_per_move_n_pts_tris (#self.nCollects,
      self.per_move_writer, n_new_pts, n_new_tris_hand, n_new_tris_robot)


    #####
    # Visualize contacts in RViz
    #   Copied from tactile_map detect_reflex_contacts.py
    #####

    # Just run detect_reflex_contacts for this.
    '''
    nContacts = len (self.contactsMsg.pressures)


    # Visualize the latest contact individually

    ind_marker_ids = [0] * nContacts
    for i in range (0, nContacts):
      ind_marker_ids [i] = self.sensor_marker_ids \
        [self.contactsMsg.finger_idx [i]] \
        [self.contactsMsg.sensor_idx [i]]

    visualize_individual_contacts (self.contactsMsg.pose_wrt_robot,
      self.contactsMsg.norm_endpt_wrt_robot,
      self.contactsMsg.pressures, self.contact_thresh,
      self.vis_pub, self.contactsMsg.robot_frame_id, ind_marker_ids)


    # Visualize cumulative contacts

    text_marker_ids = range (self.text_marker_ids_seen + 1,
      self.text_marker_ids_seen + 1 + nContacts)

    visualize_contacts (self.contactsMsg.pose_wrt_robot,
      self.contactsMsg.norm_endpt_wrt_robot, 
      self.contactsMsg.pressures, self.contact_thresh,
      self.vis_pub, self.contactsMsg.robot_frame_id,
      self.contactsMsg.header.seq, text_marker_ids)

    self.text_marker_ids_seen = self.text_marker_ids_seen + nContacts
    '''


    #####
    # Book-keeping
    #####

    # Update member var
    self.seenContactsSeq = self.contactsMsg.header.seq



  def create_output_files (self):

    # Use same timestamp for both files, for easier correspondence
    # Ref: http://stackoverflow.com/questions/13890935/timestamp-python
    timestamp = time.time ()
    self.timestring = datetime.datetime.fromtimestamp (timestamp).strftime (
      '%Y-%m-%d-%H-%M-%S')


    #####
    # XYZs PCD file
    #####

    #self.pcd_path = tactile_config.config_paths ('custom',
    #  'triangle_sampling/pcd_' + self.csv_suffix + 'collected/')
    self.pcd_path = get_pcd_path (self.csv_suffix)

    # This is file without headers, written along the way as new contacts are
    #   detected
    self.pcd_tmp_name = os.path.join (self.pcd_path, self.timestring + '_temp.pcd')
    self.pcd_tmp_file = open (self.pcd_tmp_name, 'wb')

    print ('PCD data without headers will be outputted to ' + self.pcd_tmp_name)


    #####
    # Triangle CSV file
    #####

    #self.tri_path = tactile_config.config_paths ('custom',
    #  'triangle_sampling/csv_' + self.csv_suffix + 'tri/')
    self.tri_path = get_robot_tri_path (self.csv_suffix)

    # Configurate constant strings to use as column titles in output csv files
    self.config_column_titles ()

    # Create output file, for hand frame
    self.trih_name = os.path.join (self.tri_path, self.timestring + '_hand.csv')
    self.trih_file = open (self.trih_name, 'wb')
    # If a field is non-existent, output '-1'
    self.trih_writer = csv.DictWriter (self.trih_file,
      fieldnames = self.tri_column_titles, restval='-1')
    self.trih_writer.writeheader ()

    # Create output file, for robot frame
    self.trir_name = os.path.join (self.tri_path, self.timestring + '_robo.csv')
    self.trir_file = open (self.trir_name, 'wb')
    # If a field is non-existent, output '-1'
    self.trir_writer = csv.DictWriter (self.trir_file,
      fieldnames = self.tri_column_titles, restval='-1')
    self.trir_writer.writeheader ()

    print ('Triangle data will be outputted to %s and %s' % (self.trih_name,
      self.trir_name))


    #####
    # Log file for per-collect (an approximation for per-move) number of
    #   points and number of triangles
    #####
 
    self.per_move_path, self.per_move_name = get_per_move_name (
      self.timestring, self.csv_suffix)
    self.per_move_file = open (self.per_move_name, 'wb')

    self.per_move_writer = csv.DictWriter (self.per_move_file,
      fieldnames=PerMoveConsts.per_move_column_titles, restval='-1')
    self.per_move_writer.writeheader ()

    print ('Per-collect (~ per-move) log will be outputted to %s' % \
      self.per_move_name)


    #####
    # Existing set of files to combine current files to
    #####

    if self.pickup_from_file:
      self.trih_leftoff_name = os.path.join (self.tri_path,
        self.pickup_from_file + '_hand.csv')
      self.trir_leftoff_name = os.path.join (self.tri_path,
        self.pickup_from_file + '_robo.csv')

      tmp_pcd_leftoff_name = os.path.join (self.pcd_path,
        self.pickup_from_file + '.pcd')
      if os.path.exists (tmp_pcd_leftoff_name):
        self.pcd_leftoff_name = tmp_pcd_leftoff_name
      # If permanent pcd file doesn't exist, maybe previous run ended badly,
      #   then a permanent .pcd with headers might have not been written.
      #   Use the _temp.pcd file. But if the previous run had a previous run,
      #   then that previous previous run's data will be missing! You have to
      #   manually concatenate it using pcd_write_header.py.
      else:
        print ('%sCannot find leftoff file permanent path %s. Will use the temp pcd instead. WARNING: This means lines from the previous file (if there is one) of the leftoff file will be missing! The *_temp.pcd file only contains data from a run, not concatenated with its previous run (if there is one). You will need to manually concatenate using pcd_write_header.py.%s' % \
          (ansi_colors.WARNING, tmp_pcd_leftoff_name, ansi_colors.ENDC))

        self.pcd_leftoff_name = os.path.join (self.pcd_path,
          self.pickup_from_file + '_temp.pcd')

      self.per_move_leftoff_name = os.path.join (self.per_move_path,
        self.pickup_from_file + '_per_move.csv')

      # Sanity check
      if not os.path.exists (self.trih_leftoff_name) or \
         not os.path.exists (self.trir_leftoff_name) or \
         not os.path.exists (self.pcd_leftoff_name) or \
         not os.path.exists (self.per_move_leftoff_name):
        print ('%sCannot find files left off from last time. Did you specify the correct file name? For example, could not find triangle .csv or .pcd file that looks similar to %s%s' % (ansi_colors.FAIL, self.trih_leftoff_name, ansi_colors.ENDC))
        self.doTerminate = True


      print ('%sLoading triangles from previous run in file %s%s' % (\
        ansi_colors.OKCYAN, self.trih_leftoff_name, ansi_colors.ENDC))
      # Load existing triangles from the previous run
      # Note this doesn't write the old triangles to the new file! You still
      #   need to concatenate them at end of program by pickup_csv().
      with open (self.trih_leftoff_name, 'rb') as f:

        h_reader = csv.DictReader (f)

        # Read all rows. Each row is 1 x 6 triangle params
        for row in h_reader:
          # 1 x 6 NumPy 2D array. Must be 2D to append correctly to n x 6 mat
          row_np = np.array ([[float (row [self.L0]),
                               float (row [self.L1]),
                               float (row [self.L2]),
                               float (row [self.A0]),
                               float (row [self.A1]),
                               float (row [self.A2])]])

          # n x 6 NumPy 2D array
          self.params_seen_hand = np.append (self.params_seen_hand, row_np,
            axis=0)
      print ('%sLoaded %d triangles in hand frame from previous run%s' % (\
        ansi_colors.OKCYAN, np.shape (self.params_seen_hand) [0],
        ansi_colors.ENDC))


      # Load existing triangles from the previous run
      # Note this doesn't write the old triangles to the new file! You still
      #   need to concatenate them at end of program by pickup_csv().
      with open (self.trir_leftoff_name, 'rb') as f:

        r_reader = csv.DictReader (f)

        # Read all rows. Each row is 1 x 6 triangle params
        for row in r_reader:
          # 1 x 6 NumPy 2D array. Must be 2D to append correctly to n x 6 mat
          row_np = np.array ([[float (row [self.L0]),
                               float (row [self.L1]),
                               float (row [self.L2]),
                               float (row [self.A0]),
                               float (row [self.A1]),
                               float (row [self.A2])]])

          # n x 6 NumPy 2D array
          self.params_seen_robot = np.append (self.params_seen_robot, row_np,
            axis=0)
      print ('%sLoaded %d triangles in robot frame from previous run%s' % (\
        ansi_colors.OKCYAN, np.shape (self.params_seen_robot) [0],
        ansi_colors.ENDC))


  # Triangle CSV file header titles
  # Copied and updated from weights_collect.py
  def config_column_titles (self):

    self.L0 = HistP.L0  #'l0'
    self.L1 = HistP.L1  #'l1'
    self.L2 = HistP.L2  #'l2'

    self.A0 = HistP.A0  #'a0'
    self.A1 = HistP.A1  #'a1'
    self.A2 = HistP.A2  #'a2'

    self.tri_column_titles = [self.L0, self.L1, self.L2,
      self.A0, self.A1, self.A2]


  def get_prox_joint_pos (self):

    prox_pos = []
    if self.joint_states:

      # Find index of proximal joints in list
      prox_1_idx = self.joint_states.name.index ('proximal_joint_1')
      prox_2_idx = self.joint_states.name.index ('proximal_joint_2')
      prox_3_idx = self.joint_states.name.index ('proximal_joint_3')
     
      # Access the indices found above
      prox_pos.append (self.joint_states.position [prox_1_idx])
      prox_pos.append (self.joint_states.position [prox_2_idx])
      prox_pos.append (self.joint_states.position [prox_3_idx])
     
      return prox_pos


  def prune_finger_contacts_on_palm (self):

    # Check /joint_states rostopic to see if any proximal_joint_# is
    #   nearly touching palm. If so, don't record the contact positions,
    #   `.` they are just on robot palm, not really on an object!
    #   Proximal joint ~2.8 is about touching palm.

    prox_pos = self.get_prox_joint_pos ()
    if not prox_pos:
      return

    fingers_np = np.asarray (self.contactsMsg.finger_idx)

    n_orig_contacts = len (self.contactsMsg.finger_idx)

    # Loop through each finger
    for i in range (0, 3):

      # Find if this finger has any contacts (fingers are 1 2 3 in msg, so i+1)
      fingers_mask = (fingers_np == i + 1)

      # If this finger has any contacts, and is nearly touching palm, remove
      #   its contact points from contactsMsg, so we don't record when finger
      #   is just touching palm.
      if any (fingers_mask) and prox_pos [i] >= self.TENDON_MAX:

        print ('%sRemoving contacts on finger %d because looks like it is touching palm%s' % (ansi_colors.WARNING, i+1, ansi_colors.ENDC))

        # Loop through all contact points, remove those that are on this finger
        #   that's touching the palm.
        # You can't remove things in a list in a for-loop in Python. Indices of
        #   list will change, but j's range won't, then you get index out of
        #   bounds in the later j's.
        #   You can do it from back of list though, and use a while loop, so j
        #   doesn't have a fixed range, and you can update j.
        j = len (self.contactsMsg.finger_idx) - 1
        while j >= 0:

          # If jth contact is on this finger
          if fingers_mask [j]:

            # del Ref: http://stackoverflow.com/questions/627435/how-to-remove-an-element-from-a-list-by-index-in-python

            del self.contactsMsg.pressures [j]
            del self.contactsMsg.finger_idx [j]
            del self.contactsMsg.sensor_idx [j]
           
            if self.contactsMsg.hand_valid:
              del self.contactsMsg.pose_wrt_hand [j]
              del self.contactsMsg.norm_endpt_wrt_hand [j]
           
            if self.contactsMsg.robot_valid:
              del self.contactsMsg.pose_wrt_robot [j]
              del self.contactsMsg.norm_endpt_wrt_robot [j]
           
            if self.contactsMsg.cam_valid:
              del self.contactsMsg.pose_wrt_cam [j]
              del self.contactsMsg.norm_endpt_wrt_cam [j]
           
            if self.contactsMsg.obj_valid:
              del self.contactsMsg.pose_wrt_obj [j]
              del self.contactsMsg.norm_endpt_wrt_obj [j]

           
          # Look at the next element
          j -= 1


  # Parameters:
  #   poses: geometry_msgs/Pose[]
  #   normals: geometry_msgs/Point[]
  def record_pcd (self, outfile, poses, normals=None):

    # Does PCL Python binding at least have a PointCloud type and a 
    #   writePCD() function? If so, then I can just keep adding pts to the 
    #   PointCloud type, then write to file at the end!
    #   `.` you cannot write a PCD file before knowing how many points
    #   you have in total! That information is required at top of file!
    # Alternative is, I can just write to a text file, then prepend the
    #   header after the run, using another node. It'd write the plain
    #   text to get number of points, then write to a new file with the
    #   header, and copy the text over.
    #   This is safer `.` if program ends or robot shut down whatever, I
    #   still have partial data!!!
    #   I also prefer this because I don't want to depend on PCL, as
    #   much as possible. It's a monster and it's horrible.
    #
    #   Actually I could even just output from this file, if it gets to
    #   the end! Then I can just keep code all in this node, don't have
    #   to write another, and don't have to always run another!
    #
    # Look at this file for reference of how to write a PCD file
    #   /home/master/graspingRepo/train/3DNet/cat10/train/pcd/apple/bd41ae80776809c09c25d440f3e4e51d.pcd

    # Print in yellow temporarily for debugging
    print ('\n%s%d contacts%s' % (ansi_colors.WARNING, len(poses),
      ansi_colors.ENDC))
    print ('Pressures:')
    print (self.contactsMsg.pressures)
    #print ('Recording these x y z [nx ny nz] lines to .pcd file:')

    nPts_before = self.nPts

    for i in range (0, len (poses)):

      # Copied from tactile_map est_center_axis_ransac.py

      # Keep 12 places after decimal. Robot has a lot of noise anyway, not worth
      #   keeping more. Not to mention Python also has floating point errors
      if not normals:
        line = format ('%.12g %.12g %.12g\n' \
          % (poses [i].position.x, poses [i].position.y, poses [i].position.z))

      # If normals are specified, use them
      # Assumption: user pass in or does not pass in normals consistently. i.e.
      #   if normal is passed in for first point, it must be passed in for all
      #   subsequent points. Otherwise file is invalid! Currently we do not
      #   check this. It is easy to check though!!
      else:
        line = format ('%.12g %.12g %.12g %.12g %.12g %.12g\n' \
          % (poses [i].position.x, poses [i].position.y, poses [i].position.z,
             normals [i].x, normals [i].y, normals [i].z))

      outfile.write (line)

      # Increment total number of points in the PCD file.
      #   One point per line. So best place to increment is when write a line.
      self.nPts += 1

      self.collected_cloud.points.append (poses [i].position)

      #print (line),

    print ('%s%d points total written to PCD file%s' % (ansi_colors.OKCYAN,
      self.nPts, ansi_colors.ENDC))

    # TODO test this.
    # This might have type error, as sensor_msgs/PointCloud expects Point32[],
    #   whereas visualization_msgs/Marker expects Point[]. So might need to
    #   create temp vars that are type Point32. But both types have fields
    #   x y z, and since python is dynamic typing, it should be okay?
    self.collected_cloud.header.frame_id = self.contactsMsg.robot_frame_id

    # Return number of points written in this call
    return (self.nPts - nPts_before)


  # Parameters:
  #   pts: geometry_msgs/Point[]. self.contacts_cached_*
  #   writer: a csv DictWriter
  #   accum_nTris: Only pass in True for one call, if you are calling this fn
  #     multiple times during one contact move. E.g. if you call this for
  #     the seen_hand AND seen_robot, nTris will be accumulated twice, adding
  #     double the number of triangles!
  #   seen: Triangles seen so far. This function will append to list
  # Returns: True if recorded, False if didn't record anything.
  #   Updated "seen" list.
  #   List of triangles just seen in this call.
  def record_tri (self, pts, writer, frame_id, seen):

    # Total points we have now
    nCached = len (pts)

    # If less than 3 points, cache them
    if nCached < 3:
      # Indicate to caller that nothing is recorded
      return (False, seen, [])


    #####
    # Sample some triangles from the cached contact points
    #####

    # Number of points to use to make a triangle
    nTriParams = 3

    print ('\nSampling triangles wrt frame %s to put in .csv file...' % \
      frame_id)

    # If have >= 3 points, sample triangles
    (triangles, l0, l1, l2, a0, a1, a2, vis_text) = \
      sample_reflex.sample_tris (pts, nTriParams)
    len_triangles = len (triangles)
    nTris = len (triangles) / 3


    #####
    # Write to file, and visualize
    #####

    #print ('\nSeen these parameters:')
    #print ('l0, l1, l2, a0, a1, a2:')
    #for i in range (0, len (seen [self.L0_IDX])):
    #  print ('%.2f %.2f %.2f %.2f %.2f %.2f' % \
    #    (seen [self.L0_IDX] [i], seen [self.L1_IDX] [i], seen [self.L2_IDX] [i],
    #    seen [self.A0_IDX] [i], seen [self.A1_IDX] [i], seen [self.A2_IDX] [i]))

    del self.markers.markers [:]

    #print ('nTris cumulative: %d' % self.nTris_robot)

    # Number of new triangles collected by the loop
    nTris_unseen = 0
    # New triangles collected by the loop
    seen_this_call = np.zeros ((0, 6))

    # Loop through each triangle newly sampled
    for i in range (0, nTris):

      # Duplicate check. If have seen this triangle (happens when hand hasn't
      #   moved since the last sampling), don't write it
      # Use NumPy matrix subtraction to check duplicates, instead of n x n
      #   nested for-loop! Later was taking forever for 5000 triangles, many
      #   minutes, to loop over 25,000,000 iterations!!!

      '''
      Test NumPy code in python shell:
import numpy as np
a = np.round (np.random.rand (5, 4) * 10)
b = np.round (np.random.rand (1, 4) * 10)
a - b  # This is legal
dup_thresh = np.array ((0,0,0,0))
bools = a - b <= dup_thresh  # This is legal
# The row that's all True's, gets True value in this array. This is the
#   duplicate!
np.all (bools, 1)
# This tells you where the True is. This is the index of the seen triangle that
#   the new triangle is a duplicate of.
np.where (np.all (bools, 1) == True)
      '''

      # 1 x 6 row vector. This must be 2D, otherwise axis=1 passed to np.all()
      #   below will be out of bounds, when seen and tri_curr are both 1D.
      tri_curr = np.array ([[l0[i], l1[i], l2[i], a0[i], a1[i], a2[i]]])

      dup_thresh = np.array (( \
        self.DUP_DELTA_L, self.DUP_DELTA_L, self.DUP_DELTA_L,
        self.DUP_DELTA_A, self.DUP_DELTA_A, self.DUP_DELTA_L))

      triangle_seen = False

      # seen is numSeen x 6 matrix
      if seen.size > 0:
        # Make sure both seen and tri_curr are initialized to 2D arrays, not
        #   just 1D! Otherwise np.all(bools,axis=1) will get axis out of bounds!
        bools = seen - tri_curr <= dup_thresh

        # If any row is all True, then this is a duplicate
        if np.any (np.all (bools, axis=1)):
       
          # This is the first index of a seen triangle that the new triangle is
          #   a duplicate of. Index [0][0] because seen and tri_curr are 2D,
          #   so bools is 2D too.
          #print ('Seen %dth new triangle at seen index %d' % ( \
          #  i, np.where (np.all (bools, 1) == True) [0] [0]))
       
          triangle_seen = True

      if triangle_seen:
        continue

      nTris_unseen += 1


      #####
      # Write to CSV file
      #####
     
      row = dict ()
      row.update ({self.L0: tri_curr[0, 0]})
      row.update ({self.L1: tri_curr[0, 1]})
      row.update ({self.L2: tri_curr[0, 2]})
      row.update ({self.A0: tri_curr[0, 3]})
      row.update ({self.A1: tri_curr[0, 4]})
      row.update ({self.A2: tri_curr[0, 5]})
     
      writer.writerow (row)


      #####
      # Book-keeping
      #####

      # seen[] is NumPy n x 6 matrix.
      # Add a (1 x 6) row to the (n x 6) matrix
      #   axis=0 for appending rows.
      seen_this_call = np.append (seen_this_call, tri_curr, axis=0)
      seen = np.append (seen, tri_curr, axis=0)


      #####
      # Visualize latest triangle
      #   Translated from triangle_sampling sample_pcl.cpp
      #####

      if self.VISUALIZE and frame_id == self.contactsMsg.robot_frame_id:

        # Last triangle seen
        #cumu_marker_id = len (seen [0]) - 1
        # seen is a NumPy matrix now. Not tested this
        cumu_marker_id = np.shape (seen) [0] - 1

        # Last three points just added
        marker_sample = Marker ()
        create_marker (Marker.POINTS, 'sample_pts', frame_id, 0,
          0, 0, 0, 1, 0, 0, 0.8, 0.005, 0.005, 0.005,
          marker_sample, 0)  # Use 0 duration for forever
        marker_sample.points.extend (triangles [i * nTriParams : \
          i * nTriParams + 3])
       
        # Make a copy for cumulative namespace
        '''
        marker_sample_cumu = Marker ()
        create_marker (Marker.POINTS, 'sample_pts_cumu', frame_id,
          cumu_marker_id, 0, 0, 0, 1, 1, 0, 0.8, 0.002, 0.002, 0.002,
          marker_sample_cumu, self.cumu_marker_dur)
        # Add all 3 points from this triangle
        marker_sample_cumu.points.extend (triangles [i * nTriParams : \
          i * nTriParams + 3])
        '''
       
       
        # Create a LINE_LIST Marker for the triangle
        # Simply connect the 3 points to visualize the triangle
        marker_tri = Marker ()
        create_marker (Marker.LINE_LIST, 'sample_tri', frame_id, 0,
          0, 0, 0, 1, 0, 0, 0.8, 0.001, 0, 0,
          marker_tri, 0)  # Use 0 duration for forever
        marker_tri.points.append (marker_sample.points [0])
        marker_tri.points.append (marker_sample.points [1])
        marker_tri.points.append (marker_sample.points [1])
        marker_tri.points.append (marker_sample.points [2])
        marker_tri.points.append (marker_sample.points [2])
        marker_tri.points.append (marker_sample.points [0])
       
        # Make a copy for cumulative namespace
        marker_tri_cumu = Marker ()
        create_marker (Marker.LINE_LIST, 'sample_tri_cumu', frame_id, 
          cumu_marker_id, 0, 0, 0, 1, 0, 0, 0.8, 0.001, 0, 0,
          marker_tri_cumu, self.cumu_marker_dur)  # Use 0 duration for forever
        marker_tri_cumu.points = marker_tri.points
       
       
        # Create text labels for sides and angle, to visually see if I
        #   calculated correctly.
       
        # Draw text at midpoint of side
        # NOTE: vis_text[i][0] is currently side btw pts [0] and [1]. Change if
        #   that changes.
        marker_s10 = Marker ()
        create_marker (Marker.TEXT_VIEW_FACING, 'text', frame_id, 0,
          (marker_sample.points [0].x + marker_sample.points [1].x) * 0.5,
          (marker_sample.points [0].y + marker_sample.points [1].y) * 0.5,
          (marker_sample.points [0].z + marker_sample.points [1].z) * 0.5,
          1, 0, 0, 0.8, 0, 0, self.text_height,
          marker_s10, 0)
        marker_s10.text = format ('%.2f' % (vis_text [i][0]))
       
        # Draw text at midpoint of side
        # NOTE: vis_text[i][1] is currently side btw pts [1] and [2]. Change if
        #   that changes.
        marker_s12 = Marker ()
        create_marker (Marker.TEXT_VIEW_FACING, 'text', frame_id, 1,
          (marker_sample.points [1].x + marker_sample.points [2].x) * 0.5,
          (marker_sample.points [1].y + marker_sample.points [2].y) * 0.5,
          (marker_sample.points [1].z + marker_sample.points [2].z) * 0.5,
          1, 0, 0, 0.8, 0, 0, self.text_height,
          marker_s12, 0)
        marker_s12.text = format ('%.2f' % (vis_text [i][1]))
       
        # NOTE: Angle currently is btw the sides [0][1] and [1][2], so plot
        #  angle at point [1]. If change definition of angle, need to change
        #  this too!
        marker_angle1 = Marker ()
        create_marker (Marker.TEXT_VIEW_FACING, 'text', frame_id, 2,
          marker_sample.points [1].x, marker_sample.points [1].y,
          marker_sample.points [1].z,
          1, 0, 0, 0.8, 0, 0, self.text_height,
          marker_angle1, 0)
        marker_angle1.text = format ('%.2f' % (vis_text [i][2] * 180.0 / np.pi))
       
       
        self.markers.markers.append (marker_sample)
        self.markers.markers.append (marker_tri)
       
        self.markers.markers.append (marker_s10)
        self.markers.markers.append (marker_s12)
        self.markers.markers.append (marker_angle1)
       
        #self.markers_cumu.markers.append (marker_sample_cumu)
        self.markers_cumu.markers.append (marker_tri_cumu)


    #if accum_nTris:
    #  self.nTris_robot += nTris_unseen

    print ('%s%d new unseen triangles from this round%s' % ( \
      ansi_colors.OKCYAN, nTris_unseen, ansi_colors.ENDC))

    return (True, seen, seen_this_call)


  # This is called after all files have been saved.
  def prep_for_termination (self):

    #####
    # Close file I/O
    #   Copied from tactile_collect.py
    # If this run was picking up from a previous run, concatenate the files.
    #####
 
    # If a final PCD was saved, print a line
    # (temp PCD file would have been closed by pcd_write_triangles_header())
    if self.pcd_name:
      if not self.pcd_file.closed:
        self.pcd_file.close ()

      # Concatenate to existing file
      self.nPts_ttl = self.nPts
      if self.pickup_from_file:
        self.nPts_ttl = self.pickup_pcd (self.pcd_leftoff_name, self.pcd_name)

      print ('%sOutputted %d XYZ positions to %s%s' % ( \
        ansi_colors.OKCYAN, self.nPts_ttl, self.pcd_name, ansi_colors.ENDC))
         
    # If only temp file was outputted, `.` program did not end normally (by user
    #   pressing Q), close the temp file, print a different line
    elif self.pcd_tmp_name:
      if not self.pcd_tmp_file.closed:
        self.pcd_tmp_file.close ()

      self.nPts_ttl = self.nPts
      if self.pickup_from_file:
        self.nPts_ttl = self.pickup_pcd (self.pcd_leftoff_name, self.pcd_tmp_name)

      print ('%sProgram ended unexpectedly. ' + \
        '%d XYZ positions (without headers) were outputted to %s%s' % ( \
        ansi_colors.WARNING, self.nPts_ttl, self.pcd_tmp_name, 
        ansi_colors.ENDC))
 
    if self.trih_name:
      # Close first before opening again
      self.trih_file.close ()

      self.nTris_h_ttl = np.shape (self.params_seen_hand) [0]
      if self.pickup_from_file:
        nTris_h_ttl = self.pickup_csv (self.trih_leftoff_name,
          self.trih_name)
        if nTris_h_ttl != -1:
          self.nTris_h_ttl = nTris_h_ttl

      print ('%sOutputted %d triangles wrt hand frame to %s%s'
        % (ansi_colors.OKCYAN, self.nTris_h_ttl, self.trih_name,
        ansi_colors.ENDC))
 
    if self.trir_name:
      # Close first before opening again
      self.trir_file.close ()

      self.nTris_r_ttl = np.shape (self.params_seen_robot) [0]
      if self.pickup_from_file:
        nTris_r_ttl = self.pickup_csv (self.trir_leftoff_name,
          self.trir_name)
        if nTris_r_ttl != -1:
          self.nTris_r_ttl = nTris_r_ttl

      print ('%sOutputted %d triangles wrt robot frame to %s%s'
        % (ansi_colors.OKCYAN, self.nTris_r_ttl, self.trir_name,
        ansi_colors.ENDC))

    if self.per_move_name:
      self.per_move_file.close ()

      if self.pickup_from_file:
        self.pickup_csv (self.per_move_leftoff_name, self.per_move_name)

      print ('%sOutputted per-move log to %s%s'
        % (ansi_colors.OKCYAN, self.per_move_name, ansi_colors.ENDC))


    # Signal to caller we are done, can terminate program.
    self.doTerminate = True


  # Read header and points from existing file. Store points in memory.
  #   Write to new temp file that has no header.
  # Return existing points read from file
  def pickup_pcd (self, prev_name, now_name):

    # Temp file to write to, so we aren't reading and writing to the same file!
    temp_outname = os.path.splitext (now_name) [0] + '_comboTemp' + \
      os.path.splitext (now_name) [1]

    # This fn needs to open files. Make sure all files are closed before
    #   calling the fn.
    nPts_ttl = combine_many_pcd_files ([prev_name, now_name], temp_outname)

    # Move from temp file to output file
    shutil.move (temp_outname, now_name)

    return nPts_ttl


  # Combines two csv files. Used for triangles files, and per-move file.
  def pickup_csv (self, prev_name, now_name):

    # Temp file to write to, so we aren't reading and writing to the same file!
    temp_outname = os.path.splitext (now_name) [0] + '_comboTemp' + \
      os.path.splitext (now_name) [1]

    # Collected data is first written to current timestamped file.
    #   Then, if pickup flag is specified, combine the specified previous
    #   file with current file.
    # This fn needs to open files. Make sure all files are closed before
    #   calling the fn.
    nLines_ttl = combine_many_csv_files ([prev_name, now_name], temp_outname,
      has_header=True)

    # If no error (-1 indicates error)
    if nLines_ttl != -1:
      # Move from temp file to output file
      shutil.move (temp_outname, now_name)

    return nLines_ttl



# Write headers of PCD file, along with all the points already written
# Parameters:
#   pcd_path: Final PCD file path prefix
#   timestring: To build final PCD file name
#   pcd_tmp_name, pcd_tmp_file: Existing PCD file (and name) with all the
#     points already written
#   useNormals, nPts: PCD header information
def pcd_write_triangles_header (pcd_path, timestring, pcd_tmp_name,
  pcd_tmp_file, useNormals, nPts):

  # Open the final PCD file
  #   This is file with headers, written if program ends normally by user
  #   pressing Q.
  pcd_name = os.path.join (pcd_path, timestring + '.pcd')

  # Write header to new file
  pcd_write_header (pcd_name, useNormals, nPts)


  # Read the temp file, write it to final file line by line

  # Close the temp PCD file instance that's in write mode
  pcd_tmp_file.close ()

  # Don't open file before calling pcd_write_header(), that function will
  #   open file again, opening twice without closing is not safe.
  # Open in append mode, so don't lose the header we just wrote!
  pcd_file = open (pcd_name, 'a')

  # Open a new instance of the temp PCD file, now in read mode
  with open (pcd_tmp_name, 'rb') as pcd_tmp_file:

    # Read a line
    for line in pcd_tmp_file:

      # Write the line to final PCD file
      pcd_file.write (line)

  # Close both when done
  pcd_tmp_file.close ()
  pcd_file.close ()

  return (pcd_name, pcd_file)



def main ():

  rospy.init_node ('triangles_collect', anonymous=True)

  thisNode = TrianglesCollect ()

  print ('triangles_collect.py Initialized. Starting to listen to messages...')
  print ('ATTENTION user: Run these things:')
  print ('  1. rosrun tactile_map keyboard_interface.py , to interact with this program.')
  print ('  2. rosrun tactile_map detect_reflex_contacts.py , to publish ReFlex contacts.')
  print ('  3. rosrun baxter_reflex joint_position_keyboard.py , to move Baxter by tele-op.')

  wait_rate = rospy.Rate (10)
  while not rospy.is_shutdown ():

    # Publish to keyboard_interface the prompt to display
    thisNode.pub_keyboard_prompt ()

    thisNode.collect_one_contact ()

    thisNode.vis_arr_pub.publish (thisNode.markers)
    thisNode.vis_arr_pub.publish (thisNode.markers_cumu)

    # TODO test this
    thisNode.cloud_pub.publish (thisNode.collected_cloud)


    # User input 'q' terminates program
    if thisNode.doTerminate:
      break

    # ROS loop control
    try:
      wait_rate.sleep ()
    except rospy.exceptions.ROSInterruptException, err:
      break


if __name__ == '__main__':
  main ()

