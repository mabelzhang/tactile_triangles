#!/usr/bin/env python

# Mabel Zhang
# 20 Apr 2016
#
# Writer functions for probabilities and costs data, for active touch training
#   in Gazebo.
# These data are read in active_touch package prob_of_observs.py.
#
# Can pick up a left off training in sample_gazebo.py.
#
# Refactored from sample_gazebo.py.
#
# Usage:
#   Call functions in IOProbs class like so, in this order (see
#     sample_gazebo.py):
#    
#     IOProbs ()
#     set_costs_probs_filenames ()
#     for each absolute pose:
#       # Get the observations at this abs pose... then call:
#       add_abs_pose_and_obs()
#     compute_costs_probs()
#     write_costs_probs()
#
#   For usage in combination with triangles_collect*.py, how to extract the
#     poses and triangle observations to call add_abs_pose_and_obs(), see
#     sample_gazebo_core.py, search for RECORD_PROBS.
#

# ROS
import rospy
import tf
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# Python
import os
import pickle
import sys
from copy import deepcopy

# NumPy
import numpy as np

# My packages
from tactile_collect import tactile_config
from util.ansi_colors import ansi_colors
from triangle_sampling.config_paths import get_probs_root
from tactile_map.create_marker import create_marker
from active_touch.costs_util import get_relative_pose, calc_mvmt_cost
from triangle_sampling.config_active_pkl import ActivePKLConsts as ActivPKL
from triangle_sampling.config_active_pkl import get_active_abs_poses_nxn
from triangle_sampling.config_hist_params import TriangleHistogramParams as \
  HistP
from tactile_map.tf_get_pose import tf_get_pose


class IOProbs:

  # Parameters:
  #   mode_suffix: '_gz' for Gazebo teleport ReFlex, '_pcd' for PCL training,
  #     '_bx' for Baxter.
  #   obj_center: This is NOT the geometry center in the object mesh file.
  #     This is where object mesh file is loaded in the world. 
  #     It is subtracted from absolute pose of hand in world, so that the
  #     recorded absolute poses are of hand wrt object, not of a world where
  #     the object can be in any arbitrary place - then you won't be able to
  #     touch the object if you use those coords directly at test time!
  #     Test time reader of .pkl files should add the object's position in
  #     world to the abs poses loaded, so that hand can move wrt object, no
  #     matter where object is in world.
  #   nxn_storage: If True, old-style storage, nActions x nActions are stored
  #     to disk, in a 4D dictionary (hashmap). This causes cumbersome file I/O,
  #     each requiring a 4-layer nested for-loop (because data structure is a
  #     hashmap, for O(1) access, not an actual matrix. Plus need more than
  #     integer indexing in matrix, we need float-tuples action/observation
  #     indexing).
  #     If False, new-style n-size storage. Stores n-size arrays as is stored
  #     in this class, without further doing to the cross product on actions
  #     to get n x n. This saves disk space and file I/O time tremendously!!!
  #     The n x n information will be calculated at run-time when data is
  #     loaded.
  #     False is the preferred option.
  #   pickup_leftoff: Set to true if picking up from a previous run of Gazebo
  #     training. The new data will automatically concatenate to the prev
  #     data, instead of creating new files (in which case you'd have to
  #     manually concatenate pickle binary files for dictionaries with
  #     overlapping keys, not fun!)
  #   discretize_m_q_tri: Discretization of 3D space into cells. If integers
  #     are specified, they are treated as number of decimal places to round.
  #     If floats are specified, they are treated as the cell width to
  #     discretize too. The floats are a more general way to discretize - but
  #     np.trunc / np.floor / nparray.astype(int) have rounding problems!! So
  #     perhaps the ints are a more correct way, though less control over how
  #     much you round.
  #     e.g. specify 1 to round to 1 decimal place, which is 0.1-width cells.
  #       Specify 0.5 to round to 0.5-width cells.
  #     m: position rounding, in decimals in meters
  #     q: orientation rounding, in decimals in quaternion values
  #     tri: decimals in triangle parameters
  #     This saves disk space and shortens time needed to compute
  #     n x n x m x m matrix entries in compute_costs_probs()!
  #     Especially the triangle discretization really speeds things up!!!
  #     The position and orientation are already discretized `.` ellipsoid
  #     sampling grid points are far enough apart.
  #   mean_quaternion: If True, for each unique absolute wrist position, 
  #     collect the wrist orientations (there may be more than one), take the
  #     mean quaternion. Replace the wrist orientations with their mean.
  #     This is useful for PCL probs, so that you only get 1 quaternion per
  #     wrist pose, simulating Gazebo training. Then you get a more reasonable
  #     size to save to file, else file is humungous, with 10 orientations at
  #     each wrist position, which isn't even realistic to the way we sample.
  def __init__ (self, mode_suffix, obj_center, nxn_storage=False,
    save_debug_abs_poses=True, sequential_costs=False,
    pickup_leftoff=False, pickup_from_file='', discretize_m_q_tri=(0, 0, 0)):
    #mean_quaternion=False):

    # Constant parameters

    self.obj_center = obj_center

    self.pickup_leftoff = pickup_leftoff
    self.pickup_from_file = pickup_from_file

    # Linear or pairwise costs. For the real run, you want pairwise. So set this
    #   to False. Linear is to check that basic code is working
    self.sequential_costs = sequential_costs

    self.nxn_storage = nxn_storage

    # Only used for nxn_storage=True mode
    # Save absolute pose for debug plots. Note this may increase file size by a
    #   lot! So you don't want to always keep this flag on!
    self.SAVE_DEBUG_ABS_POSES = save_debug_abs_poses

    self.tf_listener = tf.TransformListener ()

    # Only used for nxn_storage=True. i.e. no longer used!! n-storage data use
    #   the per-quantity setting RND_PLACES_* constants below.
    # meters. 1 m, 0.1 m = 10 cm, 0.01 m = 1 cm, 0.001 m = 1 mm
    self.ROUND_PLACES = 3

    # Discretization amount in number of decimal places. 0 means no
    #   discretization.
    #   Discretization of > 0 means the 3D space of wrist pose gets
    #   divided into cells, instead of continuous space. This saves time saving
    #   probabilities to disk, and saves disk space, as there will be fewer
    #   relative poses (action a) to save as dictionary keys!
    self.DISCRETIZE_M = discretize_m_q_tri [0]
    self.DISCRETIZE_Q = discretize_m_q_tri [1]
    self.DISCRETIZE_TRI = discretize_m_q_tri [2]

    # Not used anymore, `.` unsuccessful attempt to discretize to sig digs
    #   using np.round().
    #self.RND_PLACES_M = self.DISCRETIZE_M
    #self.RND_PLACES_Q = self.DISCRETIZE_Q
    #self.RND_PLACES_TRI = self.DISCRETIZE_TRI

    # If floating point, treat it as the discretization cell width.
    #   Find how many decimal places it is
    #if not isinstance (self.DISCRETIZE_M, int):
    #  self.RND_PLACES_M = n_decimal_places (self.DISCRETIZE_M)
    #if not isinstance (self.DISCRETIZE_Q, int):
    #  self.RND_PLACES_Q = n_decimal_places (self.DISCRETIZE_Q)
    #if not isinstance (self.DISCRETIZE_TRI, int):
    #  self.RND_PLACES_TRI = n_decimal_places (self.DISCRETIZE_TRI)

    #self.mean_quaternion = mean_quaternion


    # Paths to save active training data
    #   active_triangle/costs_gz/, active_triangle/probs_gz/
    self.costs_root, self.probs_root = get_probs_root (mode_suffix, nxn_storage)
    self.costs_name = ''
    self.probs_name = ''

    # As of 8 Aug 2016, I no longer use costs data. It is more efficient to
    #   compute at run-time. Takes like no time.
    self.STORE_COSTS = False


    # Initialize arrays for storing data. Load from leftoff file if there
    #   is one.
    self.reset_vars ()


  # Init / Re-init all data fields to empty arrays.
  #   Loads a file left off from prev run, if one is specified to ctor.
  # This function must be called before each object, to clear out all data
  #   fields from the previous object in Gazebo training (in case there
  #   was one).
  # Alternatively, you could just instantiate another instance of the
  #   class - probably safest option to clear out everything, but not
  #   space-efficient.
  def reset_vars (self):
       
    # All absolute grid poses for this object
    self.abs_poses = np.zeros ((0, 7))
    
    #####
    # Class-scope vars, so can save when program ends unexpectedly.
    #####

    # Not using anymore. Computing live now in active_touch cost_of_observs.py
    # Key: 7-tuple relative movement
    # Value: 2-tuple, (pos_cost, rot_cost)
    # Movement costs. Maps from 7-tuple relative_pose to (pos_cost, rot_cost),
    #   a 2-tuple of floats.
    # Saved to costs .pkl file.
    self.mvmt_costs = {}

    # Key: 7-tuple absolute pose
    # Value: List of 6-tuple triangle parameters, i.e. list of observations
    # Observations. Maps from absolute pose to list of 6-tuples. Each 6-tuple
    #   is a triangle.
    self.observs = {}

    # n-by-n-size storage data specific (nxn_storage=True)
    # Observation probabilities. 4-layer nested dictionary.
    # Key: Nested, [class][obs_src][action][obs_tgt] = [tally, src_abs_pose,
    #   tgt_abs_pose]
    # Saved to probs .pkl file.
    self.obs_probs = {}

    # n-size storage data specific (nxn_storage=False)
    self.pkl_n_wrapper = {}
    # {6-tuple observation: integer tally}
    self.obs_probs_n = {}


    # Initialize for first move of hand on an object, so we don't save the
    #   relative pose from an arbitrary training starting point to the first
    #   position around object.
    self.firstMove = True

    # Config once per object file
    # Must call set_costs_probs_filenames() before can save any data to file.
    self.costs_probs_paths_configed = False


    # Load existing vars from leftoff file
    if self.pickup_leftoff:
    
      # obs_probs
      probs_leftoff_name = os.path.join (self.probs_root,
        self.pickup_from_file + '.pkl')
 
      if os.path.exists (probs_leftoff_name):
        with open (probs_leftoff_name, 'rb') as f:

          if self.nxn_storage:
            self.obs_probs = pickle.load (f)

            # abs_poses, observs
            # Check if prev file saved absolute poses. If so, can store 
            #   abs_poses, otherwise no way of knowing... so probs and costs
            #   files stored will be missing the pairwise data btw new and old
            #   data (!).
            self.abs_poses, self.observs = get_active_abs_poses_nxn (
              self.obs_probs)
          else:
            self.pkl_n_wrapper = pickle.load (f)

            # NOTE: Assume there is only ONE object in the file!! `.` currently
            #   we only do one object per IOProbs instance. So each pkl file
            #   should contain only 1 object.
            catname = self.pkl_n_wrapper.keys () [0]

            # These need to correspond with write_costs_probs_n()
            # n x 7 NumPy array
            self.abs_poses = self.pkl_n_wrapper [catname] [ActivPKL.ABS_POSES_K]
            # {7-tuple absolute pose: [(obs0, ..., obs5), ...]}
            self.observs = self.pkl_n_wrapper [catname] [ActivPKL.OBSERVS_K]
            # {6-tuple observation: integer tally}
            self.obs_probs_n = self.pkl_n_wrapper [catname] [ActivPKL.OBS_PROBS_K]

          # Check if things are loaded from picked up .pkl file
          if self.abs_poses.shape [0] == 0:
            print ('%sNo absolute poses were saved in the active probabilities file left off from before. New probabilities file will be missing data of pairs between current run\'s and previous run\'s sampling poses.%s' % ( \
              ansi_colors.WARNING, ansi_colors.ENDC))
 
      else:
        print ('%sCannot find probs files left off from last time. Will assume starting from scratch. Offending file: %s%s' % (ansi_colors.WARNING, probs_leftoff_name, ansi_colors.ENDC))
 
      # mvmt_costs
      if self.STORE_COSTS:
        costs_leftoff_name = os.path.join (self.costs_root,
          self.pickup_from_file + '.pkl')
        if os.path.exists (costs_leftoff_name):
          with open (costs_leftoff_name, 'rb') as f:
            self.mvmt_costs = pickle.load (f)
        else:
          print ('%sCannot find costs files left off from last time. Will assume starting from scratch. Offending file: %s%s' % (ansi_colors.WARNING, costs_leftoff_name, ansi_colors.ENDC))

    # end if pickup_leftoff


  def get_configured (self):
    return self.costs_probs_paths_configed

  def get_probs_root (self):
    return self.probs_root

  def get_probs_name (self):
    return self.probs_name

  def get_costs_root (self):
    return self.costs_root

  def get_costs_name (self):
    return self.costs_name


  # Config once per object file
  # Must call this function before calling write_costs_probs() to write probs
  #   and costs to file, otherwise there is no file path to write to!
  # Parameters:
  #   timestring: A string from triangles_collect.py, generated after the
  #     first contact on object has been made. Used for filename prefix.
  def set_costs_probs_filenames (self, timestring):

    # Setting path same as root now. Keeping 2 separate vars `.` if user
    #   wants to name object by its permanent name, then will need to append
    #   object catname after the root path.
    if self.STORE_COSTS:
      costs_path = self.costs_root
      if not os.path.exists (costs_path):
        os.makedirs (costs_path)
      self.costs_name = os.path.join (self.costs_root, timestring + '.pkl')
    
    probs_path = self.probs_root
    if not os.path.exists (probs_path):
      os.makedirs (probs_path)

    self.probs_name = os.path.join (self.probs_root, timestring)
    #if not self.nxn_storage:
    #  self.probs_name += '_n'
    self.probs_name += '.pkl'

    self.costs_probs_paths_configed = True


  # Internal use.
  # Parameters:
  #   value: NumPy array, or tuple or list, some dimensions, doesn't matter.
  #     Values to discretize. Discretization is element-wise.
  #   cell_w: scalar floating point. Cell width of discretization grid.
  #   n_places: Unused
  # To test this fn:
  '''
value = np.arange (-10, 10, 0.6)
print (value)
cell_w = 0.5
disc_value = discretize (value, cell_w)
print (disc_value)
  '''
  def discretize (self, value, cell_w): #n_places):
    tmp = np.asarray (value) / float (cell_w)
    #print (tmp)
    #print (np.trunc (tmp))

    # NOTE: If cell_w is larger than value, then everything will fall into 1
    #   discretized bin! Caller is responsible for finding a valid bin width.
    # This also prints if this specific entry is discretized to the 0 bin.
    #   So look how many times this msg is printed, if from time to time, then
    #   it is fine. But if every time, then you should make width bigger.
    if np.all (np.asarray (value) < cell_w):
      print ('%sWARN: discretize(): bin size (%f) is larger than the values to discretize. If this prints for all values, then all values might fall into the same bin! You should specify a smaller DISCRETIZE_* to io_probs.py. %s' %(
        ansi_colors.WARNING, cell_w, ansi_colors.ENDC))
      print ('Values:')
      print (value)

    # Truncate decimals, multiply by bin width. This gives you the low
    #   limits of the bin that the position should fall in.
    # NOTE: NumPy has a floating point error. To reproduce:
    #   value = np.arange (-10, 10, 0.6)
    #   cell_w = 0.5
    #   Pass into this function, then you'll see 16.0 is truncated to 15, 4.0 is
    #   truncated to 3, etc. I don't know why that happens. Then they're put
    #   into wrong discretize bin - one bin lower! Not sure if that'll lead to
    #   a big problem in our probabilities data. But it sucks!!!!! I don't
    #   know how to get around this problem. nparray.astype(int), np.trunc(),
    #   np.floor() all have this problem.
    #   If you pass in fewer values in the np.arange, then this doesn't happen.
    #   It's also not Python's floating division problem, `.` they get 16.0 and
    #   4.0 fine. It's at numpy's truncate where this problem starts.
    # NOTE: Negative numbers will be put into one bin higher, with this
    #   discretization method. e.g. with bin width 0.5, -0.4 will get put into
    #    bin 0.0, but it should be in bin -0.5, `.` we always put them into the
    #    lower-end bin. This may result in bins around 0 getting more tallies.
    #    But as long as data is consistent across all objs, this might not be
    #    a big problem?
    #    To solve this, you could check if value < 0, then use np.ceil()
    #    instead of np.trunc(). But that'll slow down the discretization.
    disc_value = np.trunc (tmp) * cell_w

    # This doesn't work to get rid of floating point errors, `.` I tried:
    #   >>> np.around (-0.10000000000000001, 1)
    #   -0.10000000000000001
    # NumPy has floating point errors. Round to the exact number of decimals
    #   needed, so that poses like 0.050000003 and 0.050000001 get counted
    #   as the same pose! Otherwise you end up with thousands of abs_pose that
    #   are basically the same.
    #disc_value = np.round (disc_value, n_places)
    #print ('Rounded %s to %d places' % (disc_value, n_places))

    return disc_value


  # Entry function after calling set_costs_probs_filenames().
  # Add a 7-tuple absolute pose and a list of latest triangle observations
  #   to member vars, to be used to write to file at end of training.
  # Parameters:
  #   abs_pose: 7-tuple (px py pz qx qy qz qw), absolute pose of wrist.
  #   latest_tris: nTris x 6 NumPy matrix, returned from triangles_collect.py
  #     get_latest_tris_*(). 6 is for l0, l1, l2, a0, a1, a2, the 6 triangle
  #     parameters.
  def add_abs_pose_and_obs (self, abs_pose_continuous, latest_tris_continuous):

    # Don't record if this is first move from initial hand pose far
    #   from object, to first point next to object. This relative
    #   pose is useless, and similar direction for all objects. It
    #   just becomes the single far outlier for position costs,
    #   making color scale off in visualization.
    if self.firstMove:

      # Update for subsequent iterations
      self.firstMove = False

      return
 
    else:

      # Convert to list so can reassign values to it
      abs_pose = np.asarray (deepcopy (abs_pose_continuous))

      # Subtract object center in world, so the pose of hand is wrt object
      #   center, not wrt a fixed world frame!
      abs_pose [0] -= self.obj_center [0]
      abs_pose [1] -= self.obj_center [1]
      abs_pose [2] -= self.obj_center [2]

      # Discretize the pose

      # TODO: TEMP debug: See how many unique quaternions there are, maybe this
      #   is why there are 2000 absolute poses
      #abs_pose [0] = 0
      #abs_pose [1] = 0
      #abs_pose [2] = 0

      print ('Pose before discretizing: %.4g %.4g %.4g %.4g %.4g %.4g %.4g' % (
        abs_pose[0], abs_pose[1], abs_pose[2], abs_pose[3], abs_pose[4],
        abs_pose[5], abs_pose[6]))

      # Discretize the position
      if self.DISCRETIZE_M != 0:
        # If integer, treat it as the number of decimal places
        if isinstance (self.DISCRETIZE_M, int):
          abs_pose [0:3] = np.round (abs_pose [0:3],
            self.DISCRETIZE_M).tolist ()

        # If float, treat it as the cell width to discretize
        # To discretize a value to a cell with width self.DISCRETIZE_M, divide
        #   the value by the width, truncate the quotient to integer, then
        #   multiply the width back. That gives you the lower limit of the bin!
        #   This is the quickest way to discretize I can think of!
        else:
          abs_pose [0:3] = self.discretize (abs_pose [0:3], self.DISCRETIZE_M)
            #self.RND_PLACES_M)

      # Discretize the orientation
      if self.DISCRETIZE_Q != 0:
        if isinstance (self.DISCRETIZE_Q, int):
          abs_pose [3:7] = np.round (abs_pose [3:7],
            self.DISCRETIZE_Q).tolist ()
        else:
          abs_pose [3:7] = self.discretize (abs_pose [3:7], self.DISCRETIZE_Q)
            #self.RND_PLACES_Q)

      # Get rid of -0s!!! Important step, otherwise have too many unique poses
      #   that are 0 and -0!
      # Ref: http://stackoverflow.com/questions/26782038/how-to-eliminate-the-extra-minus-sign-when-rounding-negative-numbers-towards-zer
      #   Two ways. I think 2nd way is faster.
      #   1. abs_pose [abs_pose == 0.] = 0.
      #   2. abs_pose += 0.
      abs_pose += 0.

      # Convert to tuple to be hashable so can use as dictionary keys
      abs_pose = tuple (abs_pose.tolist ())

      print ('Pose after discretizing: %.4g %.4g %.4g %.4g %.4g %.4g %.4g' % (
        abs_pose[0], abs_pose[1], abs_pose[2], abs_pose[3], abs_pose[4],
        abs_pose[5], abs_pose[6]))


      # Absolute pose wrt fixed robot frame /base
      is_new_pose = True
      dif = self.abs_poses - np.asarray ([abs_pose])
      # Check if an entire row (np.all (, axis=1)) was True. That means
      #   abs_pose already exists in self.abs_poses; it has a close enough row
      #   in the matrix. If any entire row (np.any (rows)) was close, then 
      #   abs_pose is a duplicate.
      if np.any (np.all (np.fabs (dif) < 1e-4, axis=1)):
        is_new_pose = False

      if is_new_pose:
        # Append row. Cast row as 2D array, so can specify axis
        self.abs_poses = np.append (self.abs_poses, [abs_pose], axis=0)
        self.observs [abs_pose] = []

      # Debug
      #print ('io_probs.py debug:')
      #print ('A triangle before and after discretization:')
      #print (latest_tris_continuous [0, :])

      # Discretize the triangle parameters
      if self.DISCRETIZE_TRI != 0:
        if isinstance (self.DISCRETIZE_TRI, int):
          latest_tris = np.round (latest_tris_continuous, self.DISCRETIZE_TRI)
        else:
          latest_tris = self.discretize (latest_tris_continuous,
            self.DISCRETIZE_TRI) #self.RND_PLACES_TRI)

        # A triangle, the one at [0], before discretization
        #print (latest_tris [0, :])

        # Don't use this. This changes element order WITHIN a row too!! That's
        #   ridiculous, that messes up the triangle parameters' order!!! Doesn't
        #   work later when I try to index A0_IDX, it gives me A2 instead!!!
        # Remove duplicates, since now things are rounded
        # Assumption: Order of triangles doesn't matter
        # Ref: Find duplicate rows in NumPy array, use tuple()
        #   http://stackoverflow.com/questions/31097247/remove-duplicate-rows-of-a-numpy-array
        #latest_tris = np.unique ([tuple(row) for row in latest_tris])

        # This way works. It doesn't swap order.
        # Ref: http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
        # If this way turns out too slow, try:
        #   http://stackoverflow.com/questions/6776821/how-can-i-use-the-uniquea-rows-from-matlab-in-python
        tmp_tris = np.ascontiguousarray(latest_tris).view(np.dtype((np.void,
          latest_tris.dtype.itemsize * latest_tris.shape[1])))
        _, idx = np.unique(tmp_tris, return_index=True)
        latest_tris = latest_tris [idx]

        # A triangle, the one at [0], after discretization
        # Can be different from triangle above `.` np.unique doesn't preserve
        #   order
        #print (latest_tris [0, :])

      else:
        latest_tris = deepcopy (latest_tris_continuous)
   
      print ('Adding %d triangles to absolute pose %.4g %.4g %.4g %.4g %.4g %.4g %.4g in probs matrix, after discretization' % (
        latest_tris.shape [0], abs_pose[0], abs_pose[1], abs_pose[2],
        abs_pose[3], abs_pose[4], abs_pose[5], abs_pose[6]))

      # If not empty (shape [0] is for nTris, assuming matrix is nTris x 6).
      # Assumption: All 6 lists (rows) in latest_tris should have same length
      if latest_tris.shape [0] > 0:
        n_latest_tris = latest_tris.shape [0]

        # Save all 6 triangle parameters. Otherwise to get the other 3, I'd
        #   have to rerun Gazebo training, which takes forever! That would be
        #   stupid.
        self.observs [abs_pose].extend ([( \
          latest_tris [i] [0], latest_tris [i] [1], latest_tris [i] [2],
          latest_tris [i] [3], latest_tris [i] [4], latest_tris [i] [5])
          for i in range (0, n_latest_tris)])

        # Debug why triangles saved are wrong!
        #print ('A triangle added is:')
        #print (latest_tris[0])
  
  
      # Calculate sequential movement costs, for testing basic
      #   functionality of dictionary data.
      # If want pairwise cost, turn this off to save time and memory.
      #   This data will get overwritten by pairwise ones later anyway.
      # Note this was written before take_mean_quaternion(), so this doesn't
      #   take that into account. Sequential costs are not adjusted after
      #   poses are adjusted by the mean quaternion. I'm only using pairwise
      #   costs now, so the sequential costs don't matter.
      if self.sequential_costs and self.STORE_COSTS:
  
        # Relative pose wrt hand frame /base_link
        relative_pose = tf_get_pose ('/base', '/base_link',
          abs_pose[0], abs_pose[1], abs_pose[2],
          abs_pose[3], abs_pose[4], abs_pose[5], abs_pose[6],
          self.tf_listener)
       
        # Convert geometry_msgs/PoseStamped into a 7-tuple
        relative_pose = ( \
          relative_pose.pose.position.x,
          relative_pose.pose.position.y,
          relative_pose.pose.position.z,
          relative_pose.pose.orientation.x,
          relative_pose.pose.orientation.y,
          relative_pose.pose.orientation.z,
          relative_pose.pose.orientation.w)
       
        pos_cost, rot_cost = calc_mvmt_cost (*relative_pose)
        # Store cost to dictionary, index by the pose tuple being the key
        self.mvmt_costs [relative_pose] = (pos_cost, rot_cost)


  # Now called in caller of the class.
  # For each unique wrist position, collect all quaternions at the position (
  #   there may be multiple), take their mean. Then replace all instances of
  #   this unique wrist position with one instance that has this position and
  #   the mean quaternion. Combine all observatiosn at the wrist position into
  #   the new one. Discard all old entries (the ones with this position, and
  #   many unique quaternions).
  def take_mean_quaternion (self):

    # Find all the rows with unique positions
    tmp_p = np.ascontiguousarray(
      self.abs_poses[:, 0:3]).view(np.dtype((np.void,
      self.abs_poses.dtype.itemsize * 3)))
    _, uniq_idx = np.unique(tmp_p, return_index=True)

    # Make a new numpy array. This eliminates the need to remove rows from
    #   original abs_poses inside the for-loop, which causes u_i to be out of
    #   bound in later iterations!
    new_abs_poses = np.zeros ((0, self.abs_poses.shape[1]))
 
    for u_i in uniq_idx:
 
      # Ref: http://stackoverflow.com/questions/18927475/numpy-array-get-row-index-searching-by-a-row
      curr_pos_rowidx = (self.abs_poses [:, 0:3] == self.abs_poses [u_i, 0:3])
      curr_pos_rowidx = np.all (curr_pos_rowidx, axis=1)
      curr_pos_rowidx = np.where (curr_pos_rowidx) [0]
 
      # For wrist poses that have the current unique position, get their
      #   quaternions
      curr_qs = self.abs_poses [curr_pos_rowidx, 3:7]
 
      # axis=0 gets the mean downwards across each column
      # This takes the mean of all quaternions
      mean_q = np.mean (curr_qs, axis=0)
 
      # Create a new pose with current unique position, and the mean quaternion
      #   from all original poses with this position.
      new_pose = np.append (self.abs_poses [u_i, 0:3], mean_q)
 
 
      # Combine all observations from orig poses that have current position,
      #   to be under the new pose
      new_observ = []
      for pos_row_i in curr_pos_rowidx:
        # Remove observations of original pose with current position. Add 
        #   the observations to a temp list.
        # Need to convert the numpy array to tuple, to be hashable by dict.
        # Ref http://stackoverflow.com/questions/11277432/how-to-remove-a-key-from-a-dictionary
        #   pop(key) returns value
        new_observ.extend (self.observs.pop (
          tuple (self.abs_poses [pos_row_i].tolist ()), None))
 
      self.observs [tuple (new_pose.tolist())] = new_observ
 
      # Add the new pose
      new_abs_poses = np.append (new_abs_poses, [new_pose], axis=0)
 
    self.abs_poses = new_abs_poses


  # For debugging
  def visualize_abs_poses (self, vis_arr_pub):

    marker_arr = MarkerArray ()

    frame_id = '/base'
    alpha = 0.8
    duration = 0

    # Plot a blue square at discretized wrist absolute position
    marker_p = Marker ()
    create_marker (Marker.POINTS, 'wrist_pos_rounded', frame_id, 0,
      0, 0, 0, 0, 0, 1, alpha, 0.002, 0.002, 0.002,
      marker_p, duration)  # Use 0 duration for forever

    for i in range (0, len (self.abs_poses)):
      marker_p.points.append (Point (self.abs_poses[i, 0], self.abs_poses[i, 1],
        self.abs_poses[i, 2]))

      # Plot a cyan arrow for orientation. Arrow starts at wrist orientation,
      #   ends at average center of current triangle.
      marker_q = Marker ()
      create_marker (Marker.ARROW, 'wrist_quat_rounded', frame_id, i,
        self.abs_poses[i, 0], self.abs_poses[i, 1], self.abs_poses[i, 2],
        0, 0, 1, alpha,
        # scale.x is length, scale.y is arrow width, scale.z is arrow height
        0.03, 0.002, 0,
        marker_q, duration,  # Use 0 duration for forever
        qw=self.abs_poses[i, 6], qx=self.abs_poses[i, 3],
        qy=self.abs_poses[i, 4], qz=self.abs_poses[i, 5])
      marker_arr.markers.append (marker_q)

    marker_arr.markers.append (marker_p)

    vis_arr_pub.publish (marker_arr)
    rospy.sleep (0.1)


  # Call this function at the end of Gazebo training, after all poses are
  #   moved to, and all triangles observed.
  # Calculates n*n pairwise movement costs and observation probability, and
  #   stores them to file.
  # Parameters:
  #   catname: string name of category that the current object belongs to
  #   dae_name: string, full path to the object .dae file. Only used to write
  #     to probabilities file as meta.
  #   model_center: 3-elt 1D numpy array. Center of where object DAE
  #     model is placed in Gazebo.
  def compute_costs_probs (self, catname, dae_name, model_center):

    print ('Computing probabilities data...')

    if self.nxn_storage:
      self.compute_costs_probs_nxn (catname, dae_name, model_center)
    else:
      self.compute_costs_probs_n (catname, dae_name, model_center)


  # Internal function, do not call from outside
  def compute_costs_probs_n (self, catname, dae_name, model_center):

    # If this key doesn't exist in dictionary yet, init it
    if catname not in self.pkl_n_wrapper.keys ():
      self.pkl_n_wrapper [catname] = {}

    # 'dae_name'
    if ActivPKL.DAE_NAME_K not in self.pkl_n_wrapper [catname].keys ():
      self.pkl_n_wrapper [catname] [ActivPKL.DAE_NAME_K] = dae_name

    # Don't actually need this field anymore, `.` we subtract abs_pose by
    #   obj_center to put hand wrt object frame now!
    # 'obj_center'
    if ActivPKL.OBJ_CENTER_K not in self.pkl_n_wrapper [catname].keys ():
      self.pkl_n_wrapper [catname] [ActivPKL.OBJ_CENTER_K] = (model_center[0],
        model_center[1], model_center[2])

    # 'tri_params'
    if ActivPKL.TRI_PARAMS_K not in self.pkl_n_wrapper [catname].keys ():
      # ['l0', 'l1', 'l2', 'a0', 'a1', 'a2']
      self.pkl_n_wrapper [catname] [ActivPKL.TRI_PARAMS_K] = HistP.TRI_PARAMS


    print ('Populating observation probabilities...')

    # Populate probabilities data
    # self.observs is a dictionary, {pose: observation_list}
    # self.obs_probs_n is a dictionary, {observation: tally}
    # For each observation seen by each pose, add a tally.
    for obs_l in self.observs.values ():
      for obs in obs_l:
        if obs not in self.obs_probs_n.keys ():
          self.obs_probs_n [obs] = 0
        # {3-tuple observation (obs0, obs1, obs2): integer tally}
        self.obs_probs_n [obs] += 1


  # Internal function, do not call from outside
  # Populates the 4-layer nested dictionary obs_probs{}.
  def compute_costs_probs_nxn (self, catname, dae_name, model_center):

    # If this key doesn't exist in dictionary yet, init it
    if catname not in self.obs_probs.keys ():
      self.obs_probs [catname] = {}
   
    if self.SAVE_DEBUG_ABS_POSES:
      # Store this specific object's information, for RViz debug visualization
      # Only store one, not every object in the category, to use one as sample,
      #   since it's just for debugging...
      # 'dae_name'
      if ActivPKL.DAE_NAME_K not in self.obs_probs [catname].keys ():
        self.obs_probs [catname] [ActivPKL.DAE_NAME_K] = dae_name
      # A different structure for the dictionary, storing a plural list of
      #   dae_names, instead of one by one in the key that also stores the
      #   numerical data
      #if 'dae_names' not in self.obs_probs [catname].keys ():
        #self.obs_probs [catname] [ActivPKL.DAE_NAME_K] = []
      #self.obs_probs [catname] [ActivPKL.DAE_NAME_K].append (dae_name)
   
      if ActivPKL.OBJ_CENTER_K not in self.obs_probs [catname].keys ():
        #self.obs_probs [catname] [ActivPKL.OBJ_CENTER_K] = tuple (cyl_grid_node.obj_center)
        self.obs_probs [catname] [ActivPKL.OBJ_CENTER_K] = (model_center[0],
          model_center[1], model_center[2])
      # A different structure for the dictionary, storing a plural list of
      #   obj_centers, instead of one by one in the key that also stores the
      #   numerical data
      #if 'obj_centers' not in obs_probs [catname].keys ():
        #obs_probs [catname] ['obj_centers'] = []
      #obs_probs [catname] ['obj_centers'].append (tuple (cyl_grid_node.obj_center))

    # For progress bar printout
    #n_nested_iters = len (self.abs_poses) * len (self.abs_poses)
    n_bars = 50

    printed_p1 = False
    printed_p2 = False

    # Now placed in caller
    # Take mean of quaternions, to reduce number of poses
    #if self.mean_quaternion:
    #  self.take_mean_quaternion ()

    print ('Number of abs_poses to do n x n loop on: %d' % (
      len (self.abs_poses)))
    print (self.abs_poses)

    # Only for debug printout
    tmp_pos = np.ascontiguousarray(
      self.abs_poses[:, 0:3]).view(np.dtype((np.void,
      self.abs_poses.dtype.itemsize * 3)))
    _, uniq_idx = np.unique(tmp_pos, return_index=True)
    print ('Number of unique wrist positions: %d' % len (uniq_idx))
    print (self.abs_poses [uniq_idx, 0:3])

    tmp_pos = np.ascontiguousarray(
      self.abs_poses[:, 3:7]).view(np.dtype((np.void,
      self.abs_poses.dtype.itemsize * 4)))
    _, uniq_idx = np.unique(tmp_pos, return_index=True)
    print ('Number of unique wrist quaternions: %d' % len (uniq_idx))
    print (self.abs_poses [uniq_idx, 3:7])

    # Only for debug printout
    n_observs = 0
    for pose_k in self.observs.keys ():
      n_observs += len (self.observs [pose_k])
    print ('Total number of observations (triangles): %d' % n_observs)


    # n x n nested for-loop
    # TODO: Speed this up using NumPy tile(), then don't need to do n x n
    #   loop, slow Python array accesses. Can just do n*n linear loop.
    #   Currently this loop is the biggest overhead in training probs,
    #   takes minutes, not seconds!
    for p1i in range (0, len (self.abs_poses)):

      if rospy.is_shutdown ():
        break

      # Print progress bar
      #   Ref \r http://stackoverflow.com/questions/3002085/python-to-print-out-status-bar-and-percentage
      # Progress percentage
      # To test on command line:
      '''
import sys
from time import sleep
n_iters = 30
n_bars = 20
for i in range (0, n_iters):
  progress_pc = float (i) / n_iters
  sys.stdout.write ('\r')
  sys.stdout.write ('[%-s] %d%%' % ('='*int(progress_pc*n_bars), int(progress_pc * 100)))
  sys.stdout.flush ()
  sleep (0.1)

# Print the last number, 100%, when i == n_iters and loop breaks
sys.stdout.write ('\r')
sys.stdout.write ('[%-s] %d%%' % ('='*int(1*n_bars), int(1 * 100)))
sys.stdout.flush ()
      '''
      progress_pc = float (p1i) / len (self.abs_poses)
      sys.stdout.write('\r')
      sys.stdout.write("[%-s] %d%%" % ('='*int(progress_pc*n_bars), int(progress_pc * 100)))
      sys.stdout.flush()

      # Print an empty line if want to print the details inside loop. Printing
      #   anything will make progress bar print many times instead of once.
      print ('')

      # Reset outside the loop where the msg is printed, so it prints only once
      printed_p1 = False

   
      p1 = tuple (self.abs_poses [p1i, :].tolist ())
      # Quaternion
      q1 = (p1[3], p1[4], p1[5], p1[6])

      for p2i in range (0, len (self.abs_poses)):

        # Skip self cost, always 0. Skip self probabilities, because we don't
        #   want the predicted "next movement" to be stationary no movement!
        #   Then active exploration will never work, `.` that will always be
        #   the lowest cost!
        if p1i == p2i:
          continue

        if rospy.is_shutdown ():
          break
   
        p2 = tuple (self.abs_poses [p2i, :].tolist ())
        q2 = (p2[3], p2[4], p2[5], p2[6])
  
        relative_pose = get_relative_pose (p1, q1, p2, q2)
   
        if self.STORE_COSTS:
          pos_cost, rot_cost = calc_mvmt_cost (*relative_pose)
          self.mvmt_costs [relative_pose] = (pos_cost, rot_cost)
   
   
        #####
        # Calc observation probability, as integer tally
        #####
   
        # TODO: Might want higher precision for Quaternion, but we'll see
        #   after looking at visualization from read_movement_probs.py.
        relative_pose_rd = tuple (np.round (relative_pose,
          self.ROUND_PLACES).tolist ())
   
        # There were no triangles observed at one of the poses.
        #   This happens when no contacts occur at a hand pose.
        # Then skip it. `.` no point storing probability 0 to data. We're
        #   already storing sparse data. There are infinite number of poses
        #   where the hand would get 0 contact, for example, anywhere not
        #   near the object! We don't store any of those, `.` their
        #   probabilities are 0s. Therefore we shouldn't store poses on the
        #   grid that have probability 0 of getting triangles either.
        #   `.` the probabilities are in terms of OBSERVATIONS, not POSES!
        #   So if there are no observation z1, then p(z2|z1) doesn't exist!
        #   If there are no observation z2, then same thing. There are poses,
        #   but observation z at pose p1 doesn't exist.
        if (p1 not in self.observs.keys ()) or (p2 not in self.observs.keys ()):
          continue

        if not printed_p1:
          print ('%d observations in absolute pose %d out of %d' % (
            len (self.observs [p1]), p1i, len (self.abs_poses)))
          printed_p1 = True
        # Reset outside the loop where msg is printed, so it prints only once
        printed_p2 = False

        # Need n * n loop, `.` there may be more than one triangle observed
        #   at each pose.
        #   observs[p1] is a list of observations
        for observ1 in self.observs [p1]:

          if rospy.is_shutdown ():
            break

          # Round, so observations are actually binned, not all unique. If
          #   all numbers are unique, can never generalize!
          observ1_rd = tuple (np.round (observ1, self.ROUND_PLACES).tolist ())
          # If this key doesn't exist yet, init it
          if observ1_rd not in self.obs_probs [catname].keys ():
            self.obs_probs [catname] [observ1_rd] = {}

          if not printed_p2:
            print ('  %d observations in nested absolute pose %d out of %d' % (
              len (self.observs [p2]), p2i, len (self.abs_poses)))
            printed_p2 = True

          for observ2 in self.observs [p2]:

            if rospy.is_shutdown ():
              break
   
            #print ('obs_probs:')
            #print (obs_probs)
   
            # Round
            observ2_rd = tuple (np.round (observ2, self.ROUND_PLACES).tolist ())
            # If this key doesn't exist yet, init it
            if relative_pose_rd not in self.obs_probs [catname] [observ1_rd].keys ():
              self.obs_probs [catname] [observ1_rd] [relative_pose_rd] = {}
   
            # If this key doesn't exist yet, init it
            if observ2_rd not in self.obs_probs [catname] [observ1_rd] \
              [relative_pose_rd].keys ():
   
              # Just init the integer tally in single-element tuple
              if not self.SAVE_DEBUG_ABS_POSES:
                self.obs_probs [catname] [observ1_rd] [relative_pose_rd] \
                  [observ2_rd] = [0, ]
              # Init three-element tuple, for source and dest absolute poses
              else:
                self.obs_probs [catname] [observ1_rd] [relative_pose_rd] \
                  [observ2_rd] = [0, [], []]
   
            # Increment tally
            self.obs_probs [catname] [observ1_rd] [relative_pose_rd] \
              [observ2_rd] [0] += 1
   
            if self.SAVE_DEBUG_ABS_POSES:
              # Source pose. ABS_SRC_POSE is 1
              self.obs_probs [catname] [observ1_rd] [relative_pose_rd] \
                [observ2_rd] [ActivPKL.ABS_SRC_POSE].append (p1)
              # Destination pose. ABS_TGT_POSE is 2
              self.obs_probs [catname] [observ1_rd] [relative_pose_rd] \
                [observ2_rd] [ActivPKL.ABS_TGT_POSE].append (p2)


    # Print the last number, 100%, when i == n_iters and loop breaks
    sys.stdout.write ('\r')
    sys.stdout.write ('[%-s] %d%%' % ('='*int(1*n_bars), int(1 * 100)))
    sys.stdout.flush ()

    print ('')


  # Entry function after calling add_abs_pose_and_obs() on all poses of an
  #   object, and after calling compute_costs_probs().
  def write_costs_probs (self):

    print ('Writing probabilities data...')

    if self.nxn_storage:
      self.write_costs_probs_nxn ()
    else:
      self.write_costs_probs_n ()


  # Internal function, do not call from outside
  # Write n-size storage
  def write_costs_probs_n (self):

    if not self.costs_probs_paths_configed:
      print ('%sNo file names were configured for costs and probs. Was there any contacts? No costs and probs files will be saved!%s' % (
        ansi_colors.FAIL, ansi_colors.ENDC))
      return


    # Save n-size data structures to file
    #   Don't store mvmt_costs anymore, calculate at run-time easily
    #   Only store self.abs_poses, self.observs, self.obs_probs_n.
    if self.abs_poses != None and self.observs and self.obs_probs_n:

      print ('%sO(n)-size data will be outputted to %s ...%s' % (
        ansi_colors.OKCYAN, self.probs_name, ansi_colors.ENDC))

      # Assumption: There is only one category, `.` only one object is trained
      #   in a run, in order to save to a pkl file with the object name.
      #   So just index [0] to get the catname.
      #   For the caller, this simply means a new IOProbs instance should be
      #   created for each object it trains!
      catname = self.pkl_n_wrapper.keys () [0]

      with open (self.probs_name, 'wb') as f:

        # 'abs_poses'
        self.pkl_n_wrapper [catname] [ActivPKL.ABS_POSES_K] = self.abs_poses
        # 'observs'
        self.pkl_n_wrapper [catname] [ActivPKL.OBSERVS_K] = self.observs
        # 'obs_probs_n'
        self.pkl_n_wrapper [catname] [ActivPKL.OBS_PROBS_K] = self.obs_probs_n

        # HIGHEST_PROTOCOL is binary, good performance. 0 is text format,
        #   can use for debugging.
        pickle.dump (self.pkl_n_wrapper, f, pickle.HIGHEST_PROTOCOL)


  # Internal function, do not call from outside
  # Write nxn-size storage. n being number of absolute poses.
  def write_costs_probs_nxn (self):

    if not self.costs_probs_paths_configed:
      print ('%sNo file names were configured for costs and probs. Was there any contacts? No costs and probs files will be saved!%s' % (
        ansi_colors.FAIL, ansi_colors.ENDC))
      return


    # Save costs to file

    if self.mvmt_costs and self.STORE_COSTS:
    
      print ('%sCosts training data will be outputted to %s ...%s' % (
        ansi_colors.OKCYAN, self.costs_name, ansi_colors.ENDC))
      
      # Write costs to file
      with open (self.costs_name, 'wb') as f:
        # HIGHEST_PROTOCOL is binary, good performance. 0 is text format,
        #   can use for debugging.
        pickle.dump (self.mvmt_costs, f, pickle.HIGHEST_PROTOCOL)
    
    
    # Save probabilities to file
    # Note:
    #   Don't normalize tallies into probabilities here, normalize them in
    #   the code that reads these files. `.` Then we store raw integers,
    #   not floats. Integers from tally are more accurate and easier to
    #   manipulate if we change our mind for how to normalize! If only
    #   save probabilities, what if we want to sum something else later,
    #   then have to redo training! That's not good!

    if self.obs_probs:
    
      print ('%sProbabilities training data will be outputted to %s ...%s' % (
        ansi_colors.OKCYAN, self.probs_name, ansi_colors.ENDC))
      
      # Write probs to file
      with open (self.probs_name, 'wb') as f:
        # HIGHEST_PROTOCOL is binary, good performance. 0 is text format,
        #   can use for debugging.
        pickle.dump (self.obs_probs, f, pickle.HIGHEST_PROTOCOL)



# ============================================================ Util fns ==

def n_decimal_places (val):

  # If val has more than this number of decimal places, just ignore the rest,
  #   truncate at this many
  MAX_PLACES = 6

  for i in range (0, MAX_PLACES):
    # If 10^i multiplied by this number gets > 1, then i is the number of
    #   decimal digits in val.
    #   e.g. 0.1 * 10^1 >= 1, 0.01 * 10^2 >= 1, 0.05 * 10^2 >= 1
    if val * 10 ** i >= 1:
      break

  return i


