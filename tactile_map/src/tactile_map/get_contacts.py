#!/usr/bin/env python

# Mabel Zhang
# 6 Apr 2015
#
# Refactored from manual_explore.py
#


# ROS
from geometry_msgs.msg import Point

# Python
import math
import numpy as np

# Local
from tactile_map.tf_get_pose import tf_get_pose, tf_get_transform


## Define frame_ids of fingers and palms

# Finger tactile sensor tf frame names. Broadcasted by reflex package
#   reflex.yaml and reflex_tf_broadcaster.cpp. These are capitalized names.
#   Lowercase ones do not have the /sensor_ suffix, cannot get individual
#   sensor frames via lower case frames.
# 3 fingers, 5 proximal sensors each
proximal_sensors = ['/Proximal_'+str(i)+'/sensor_'+str(j) \
  for i in range(1,4) for j in range(1,6)]
# 3 fingers, 4 distal sensors each
distal_sensors = ['/Distal_'+str(i)+'/sensor_'+str(j) \
  for i in range(1,4) for j in range(1,5)]

# 3 x 9 list
# Access using [finger_i][sensor_j], i is 0 to 2, j is 0 to 8 inclusive
fingers_sensor_frames = [proximal_sensors[0:5] + distal_sensors[0:4],
  proximal_sensors[5:10] + distal_sensors[4:8],
  proximal_sensors[10:15] + distal_sensors[8:12]]

# 11-item list
# Capitalized frames: I made these names. Broadcasted by baxter_reflex package 
#   reflex_palm_tf_broadcaster.cpp.
palm_sensor_frames = ['/Palm_'+str(i) for i in range(1,12)]



# Called by get_contacts_live(). Call once for palm, once for fingers.
# Parameters:
#   pressure: float32[]
#   contact_thresh: scalar integer
#   sensor_frames: string[]. tf frame_id of each sensor, in same order as
#     pressures[].
#   target_frame: string. Frame to put return coordinates wrt.
# Returns [geometry_msgs/Point[], geometry_msgs/Point[], int[], int[]]
#   First elt is contact points;
#   Second elt is endpoints of normals, sized by magnitude of pressure, by a
#     formula defined in nested fn calc_norm_mag().
#   Points are wrt target_frame.
#   If no contacts, returned lists are empty.
def get_contacts_live_by_part (pressures, contact_thresh, sensor_frames,
  target_frame, tfTrans, norm_fac = 0.01, use_common_time=True, stamp=None):

  # Ret vals
  contacts = []
  normal_endpts = []
  pressures_fired = []
  sensor_idx = []

  # Constant to multiply pressure value by, to set magnitude of normal vector
  #   Pressure value 100 ~ 0.01 m (1 cm)
  #   log (100 * norm_fac + 1) * 0.01 = log (100 * 0.01 + 1) * 0.01 = log (2) * 0.01 = 0.69 * 0.01 = 0.0069 (0.69 cm)
  # The +1 shift makes log(0) = 0, instead of undefined.
  #   The 0.01 in front makes conversion 0.69 cm -> 0.0069 m
  def calc_norm_mag (pressure, norm_fac):
    return 0.01 * math.log (abs (pressure) * norm_fac + 1)


  # Get contact coordinates from pressure values
  for i in range (0, len (pressures)):

    # Get pressure sensor value
    pressure = pressures [i]

    # If a sensor's value exceeds threshold, count as a contact. Plot an
    #   RViz marker at the sensor's current position (this must be quick.
    #   Otherwise hand would have moved away by the time you plot).
    # This is a very rough estimate of a potential point on object surface.
    #   Very rough, `.` obviously this is a point inside hand, not on obj.
    #
    # TODO: We'll have to calibrate hand wrt robot, then here can tf
    #   immediately from hand frame to robot frame, then take our time in
    #   plotting it wrt robot frame.
    if abs (pressure) >= contact_thresh:

      #print ('Got contact')

      # Get frame name of this tactile sensor
      from_frame = sensor_frames [i]

      # Magnitude of normal vector
      norm_mag = calc_norm_mag (pressure, norm_fac) #abs (pressure) * norm_fac

      # Do tf here `.` has to be done immediately, before hand moves elsewhere
      # Get position of this tactile sensor wrt /base_link of hand
      # Parameters: tx ty tz qx qy qz qw
      sensor_pose_wrt_base = tf_get_pose (from_frame, target_frame,
        0, 0, 0, 0, 0, 0, 1, tfTrans, use_common_time, stamp)
      # Normal is -z. Get pos of (0 0 -1) of tactile sensor wrt /base_link
      surf_normal_wrt_base = tf_get_pose (from_frame, target_frame,
        0, 0, -norm_mag, 0, 0, 0, 1, tfTrans, use_common_time, stamp)

      contacts.append (sensor_pose_wrt_base.pose.position)
      normal_endpts.append (surf_normal_wrt_base.pose.position)
      pressures_fired.append (pressure)
      sensor_idx.append (i)

  if len (contacts) != len (normal_endpts):
    print ('Unequal lengths')

  return [contacts, normal_endpts, pressures_fired, sensor_idx]


# sensor_values: Hand msg received on rostopic /reflex_hand
# Calls get_contacts_live_by_part ()
# Returns [geometry_msgs/Point[], geometry_msgs/Point[], int[], int[], int[]]
#   If no contacts, returned lists are empty.
def get_contacts_live (sensor_values, contact_thresh, target_frame,
  tfTrans, norm_fac=0.01, use_common_time=True, stamp=None):

  # Ret vals
  contacts = []
  normal_endpts = []
  pressures = []
  # finger_idx is 0 for palm, 1~3 for each finger, following order defined by
  #   ReFlex.
  finger_idx = []
  sensor_idx = []

  # Palm. 11 sensors. Pressure is 11-tuple float32
  [contacts_p, normal_endpts_p, pressures_p, sensor_idx_p] = \
  get_contacts_live_by_part (
    sensor_values.palm.pressure,
    contact_thresh, palm_sensor_frames, target_frame,
    tfTrans, use_common_time)
  # Append to ret val
  contacts.extend (contacts_p)
  normal_endpts.extend (normal_endpts_p)
  pressures.extend (pressures_p)
  finger_idx.extend ([0] * len(contacts_p))
  sensor_idx.extend (sensor_idx_p)

  # 3 fingers
  for i in range (0, len (sensor_values.finger)):
    # Each finger has 9 sensors. Pressure is 9-tuple float32
    [contacts_f, normal_endpts_f, pressures_f, sensor_idx_f] = \
    get_contacts_live_by_part (
      sensor_values.finger [i].pressure,
      contact_thresh, fingers_sensor_frames [i], target_frame,
      tfTrans, use_common_time)
    # Append to ret val
    contacts.extend (contacts_f)
    normal_endpts.extend (normal_endpts_f)
    pressures.extend (pressures_f)
    finger_idx.extend ([i+1] * len(contacts_f))
    sensor_idx.extend (sensor_idx_f)

  if len (finger_idx) != len (sensor_idx):
    rospy.info ('get_contacts_live(): Unequal lengths finger_idx vs sensor_idx')
  if len (contacts) != len (finger_idx):
    rospy.info ('get_contacts_live(): Unequal lengths contacts vs finger_idx')
  if len (contacts) != len (sensor_idx):
    rospy.info ('get_contacts_live(): Unequal lengths contacts vs sensor_idx')

  return [contacts, normal_endpts, pressures, finger_idx, sensor_idx]


# Not used by any file. Useless. Often it's useful for caller to access the
#   raw contacts, not the non-dup ones. Caller can decide what to do with the
#   duplicates. i.e. usually caller calls eliminate_duplicates() itself, instead
#   of using this fn.
# sensor_values: Hand msg received on rostopic /reflex_hand
# Does NOT alter prev_ret_val in parameter! Do not update code in a way that
#   it would alter that! `.` caller might not want it changed.
# Parameters:
#   prev_ret_val: List previously returned by this fn or get_contacts_live().
#     The list to compare new contacts against, to find duplicates.
# Returns [geometry_msgs/Point[], geometry_msgs/Point[], int[], int[]].
#   First two lists:
#     New contacts that are not duplicates. If all were dups, then empty lists
#     are returned.
def get_contacts_live_no_dups (sensor_values, contact_thresh, target_frame,
  tfTrans, prev_ret_val, norm_fac=0.01, use_common_time=True, stamp=None):

  [contacts, normal_endpts] = get_contacts_live (
    sensor_values, contact_thresh, target_frame,
    tfTrans, norm_fac, use_common_time, stamp)

  [contacts_no_dups, normals_no_dups, _, old_idx, new_idx] = \
    eliminate_duplicates (
      prev_ret_val [0], contacts,
      prev_ret_val [1], normal_endpts)

  return [contacts_no_dups, normals_no_dups, old_idx, new_idx]


# Returns [geometry_msgs/Point[], geometry_msgs/Point[], Bool[], int[], int[]].
# Candidates that are non-duplicates.
# If *_contacts or *_normals don't exist, pass in None.
# Parameters:
#   keep_new: If set to true, will keep the new candidate duplicate, instead of
#     the orig copy in the list of existing items.
def eliminate_duplicates (existing_contacts, candidate_contacts,
  existing_normals=None, candidate_normals=None, keep_new=False):

  new_contacts = []
  new_normals = None
  if existing_normals is not None:
    new_normals = []

  # Exclude new contact points that are same as previous
  #   ones. This happens if the touch on object is longer than one iter,
  #   which is like, always!
  # Bool[]
  [isDup, old_idx, new_idx] = check_dup_contacts (
    existing_contacts, candidate_contacts)
    # contacts: Point[]
    # existing_contacts: Point[]


  # If only some contact points are duplicates, hand has moved, we can
  #   process the non-duplicate contacts as normal.
  new_contacts = [candidate_contacts[i] \
    for i in range (0, len(candidate_contacts)) if not isDup[i]]

  # Add the corresponding normals
  if existing_normals is not None:
    new_normals = [candidate_normals[i] \
      for i in range (0, len(candidate_normals)) if not isDup [i]]

  return [new_contacts, new_normals, isDup, old_idx, new_idx]


# Parameters:
#   new_contacts: Point[]
#   existing_contacts: Point[] or Point32[]? Or a mix of both in a list?
# Returns 3 lists:
#   1st list: Boolean array indicating whether each new item is a duplicate
#   2nd list: i's. Index of elts in existing list A that have new dups
#   3rd list: j's. Index of elts in new list B that are dups
#   Duplicates satisfy A[i] == B[j] (approximately equal)
#   len(A) == len(A) == len(isDup == True)
def check_dup_contacts (existing_contacts, new_contacts):

  # Ret val. Same size as new_contacts list.
  isDup = []
  old_dup_idx = []
  new_dup_idx = []

  # Sanity check
  # If there are any contact points detected this iteration, and if these
  #   are not the first ones, check if they are duplicates of captured ones.
  if (len (existing_contacts) == 0):
    isDup = [False] * len (new_contacts)
    return [isDup, old_dup_idx, new_dup_idx]


  # Convert to numpy array
  new_contacts_np = [np.asarray ([p.x, p.y, p.z]) for p in new_contacts]
  existing_contacts_np = [np.asarray ([p.x, p.y, p.z]) for p in existing_contacts]

  # Distance within which we'd call two points duplicates
  dup_thresh = 0.005

  # Loop through each new point
  for i in range (0, len (new_contacts_np)):

    # Check if there's an existing point same as current point
    #   array ([True, True, True], [False, False, True], ...)
    mask = abs (existing_contacts_np - new_contacts_np [i]) < dup_thresh

    # Index of existing item that has a duplicate.
    #   Check for elt s.t. all 3 components x y z are True, i.e. dup exists
    curr_old_idx = [i for i in range (0, len (mask)) if mask[i].all()]

    isDup.append (len (curr_old_idx) > 0)

    if (len (curr_old_idx) > 0):
      # If there are multiple existing elts that are all dups of this single new
      #   item, just pick first one. 
      #   (There shouldn't be multiple, since duplicates were eliminated. But
      #   just in case the user didn't call check_dup() at every step, then
      #   there might be duplicates).
      old_dup_idx.append (curr_old_idx [0])


    # Old simple way when I didn't have to return old_dup_idx.
    # Is there any element such that all 3 components x y z are True, 
    #   i.e. duplicate point exists
    #isDup.append (any ([m.all() for m in mask]))

  # Duplicate indices in new list are simply the ones with isDup[i] == True
  new_dup_idx = [i for i in range (0, len(new_contacts)) if isDup [i]]

  return [isDup, old_dup_idx, new_dup_idx]



# Need an object for this because need a persisting variable recording 
#   cursor in half-read file.
class GetContactsRecorded:

  def __init__ (self, filename):

    self.filename = filename
    self.inputfile = open (self.filename, 'r')

    self.fileEnded = False


  def close_file (self):
    self.inputfile.close ()
    self.fileEnded = True


  # Returns contact points in geometry_msgs/Point[]
  def get_contacts_recorded (self):

    if self.fileEnded:
      return []

    # Read number of contacts
    line = self.inputfile.readline ().strip ()

    # If this is the last line in file, close it.
    # Last line happens when readline() returns empty string. For an empty line,
    #   it returns '\n'. Endline character is not omitted from lines.
    # Ref: https://docs.python.org/2/tutorial/inputoutput.html
    if line == '':
      print ('Reached end of recorded contact file. That is all the contacts.')
      self.close_file ()
      return []


    # Ret val
    contacts = []

    # Number of contacts
    nContacts = 0
    try:
      nContacts = int (line)
    except ValueError, err:
      print ('ERROR in get_contacts.py get_contacts_recorded(): Unexpected file format in %s. Each block should start with positive integer indicating number of contact points. Stopping reading file...' % (self.filename))
      self.close_file ()
      return []

    # Read the contact points
    for i in range (0, nContacts):

      # '%f %f %f' for (x, y, z) coordinates of contact point
      line = self.inputfile.readline ()

      # Strip '\n' char and extra spaces. Split space-delimited elements
      line = line.split ()

      try:
        contacts.append (
          Point (float (line[0]), float (line[1]), float (line[2])))

      except ValueError, err:
        print ('ERROR in get_contacts.py get_contacts_recorded(): Unexpected file format in %s. Expected %f %f %f format for contact coordinates. Encountered non-float. Stopping reading file...' % (self.filename))
        self.close_file ()
        return []

    return contacts



# 12 Jul 2015
#   Refactored from tactile_collect/src/tactile_collect.py .
# Parameters:
#   msg: Hand msg received on rostopic /reflex_hand
#   joints, pressures: Lists to put joint values and pressure values in. These
#     CAN be the same list, as tactile_collect.py formats them this way.
#     This function only uses extend() on these lists.
# Returns:
#   Tuple of (list of 10 floats, list of 38 integers)
def get_joints_and_pressures (msg, joints, pressures):

  # Init ret vals, if not provided
  if joints is None:
    joints = list ()
  if pressures is None:
    pressures = list ()

  # Palm. 23 columns
  # Joint angle. Scalar float
  joints.extend ([msg.palm.preshape])
  # Pressur sensor values. Contact boolean (sparse) is converted to float.
  #   11-float tuple x 2, one for raw pressures, one for Booleans..
  pressures.extend (msg.palm.pressure)
  pressures.extend ([float(c) for c in msg.palm.contact])


  # Loop through each of the 3 fingers. 21 columns each
  #   (There are 3 fingers, this is hardcoded into reflex_msgs Hand.msg. So
  #   we'll hardcode too)
  for i_fing in range (0, 3):

    # To echo subfields on the command line:
    #   $ rostopic echo /reflex_hand/finger[0].spool

    # Joint angles. Scalar floats x 3
    joints.extend ([msg.finger[i_fing].spool, 
                  msg.finger[i_fing].proximal,
                  msg.finger[i_fing].distal])

    # Pressur sensor values. Contact boolean (sparse) is converted to float.
    #   9-float tuple x 2, one for raw pressures, one for Booleans.
    pressures.extend (msg.finger[i_fing].pressure)
    pressures.extend ([float(c) for c in msg.finger[i_fing].contact])
 
  return (joints, pressures)


# 22 Jul 2015
# Given finger index (0 for palm, 1 2 3 for fingers 0 1 2) and sensor index 
#   ([0:8] for fingers, [0:10] for palm), return the position of the
#   specified sensor, wrt specified frame.
# Parameters:
#   finger_idx: Python list. Specify as many fingers as you need.
#     0 for palm, 1 2 3 for fingers 0 1 2 (these are the values publisehd
#     in Contacts.msg).
#   sensor_idx: Python list. Same size as finger_idx. Sensor index on the 
#     corresponding finger in finger_idx.
#   target_frame: Usually robot body frame
#   tfTrans: tf TransformListener
#
# Returns list of size-3 lists. Outer list's size is same as len(finger_idx).
#   [[x1, y1, z1], ..., [xn, yn, zn]]
def get_sensor_frames (finger_idx, sensor_idx, target_frame, tfTrans,
  stamp=None):

  assert (len (finger_idx) == len (sensor_idx))

  sensor_pos = []

  # Loop through each sensor frame
  for i in range (0, len (finger_idx)):

    # 0 is palm
    if finger_idx [i] == 0:
      from_frame = palm_sensor_frames [sensor_idx [i]]
    # 1 2 3 is fingers 0 1 2, so subtract 1
    else:
      from_frame = fingers_sensor_frames [finger_idx [i] - 1] [sensor_idx [i]]

    # From sensor frame (0 0 0), to robot frame (x y z)
    sensor_pose = tf_get_transform (from_frame, target_frame, tfTrans, stamp)
    # Extract translation only
    sensor_pos.append (sensor_pose [0:3])

  return (sensor_pos)


