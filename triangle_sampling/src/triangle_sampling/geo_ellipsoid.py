#!/usr/bin/env python

# Mabel Zhang
# 14 Feb 2016
#
# Returns points on the surface of an ellipsoid
#

# ROS
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import tf

# NumPy
import numpy as np

# My packages
from tactile_map.create_marker import create_marker
from util.quat_utils import get_relative_rotation_v
from util.ansi_colors import ansi_colors
from util.mat_utils import find_dups_self


class EllipsoidSurfacePoints:

  def __init__ (self):

    # List of all points on ellipsoid surface
    self.surf_pts = []
    # Init to -1, so the first one returned would be [0]!
    self.next_idx_to_return = -1

    self.cx = -1
    self.cy = -1
    self.cz = -1

    self.rx = -1
    self.ry = -1
    self.rz = -1

    print ('Constructed. Caller should call initialize_ellipsoid() next.')

    # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.set_printoptions.html
    np.set_printoptions (precision=4)


  # Calculate all points on surface of ellipsoid, and store in a member var.
  #   Do this outside of __init__(), so can allow caller to initialize
  #   different ellipsoids, not just one. Then caller doesn't need to set
  #   individual parameters from outside, if they want a different ellipsoid
  #   than what they initialized this class instance with.
  # Parameters:
  #   c_xyz: 3-tuple or NumPy arr specifying center of ellipsoid, (cx, cy, cz)
  #   r_xyz: 3-tuple specifying radii of ellipsoid, (rx, ry, rz)
  #   deg_step: Angle in degrees to step on ellipsoid surface, to get points
  #     on surface. i.e. desired resolution of points on ellipsoid surface.
  #   deg_step_long, deg_step_lat: Set these instead of deg_step, if you have
  #     different deg_steps for longitude and latitude.
  #   range_long, range_lat: (optional) 2-tuples. Ranges for longitude and 
  #     latitude angles. By default, longitude is vertical circles, orange
  #     slices, -90 to 90 degrees. Latitude is full horizontal circles, -180 to
  #     180.
  #   alternate_order: Alternate the spinning direction of every other
  #     circular slice of points returned, so each point is always physically
  #     consecutive to the next, no jumping. This avoids jumping from one side
  #     of the ellipsoid to the opposite side. Useful when accumulating contact
  #     points in sample_gazebo.py, you only want to accumulate points in a
  #     local region reachable by the robot hand, not accumulate across the
  #     whole object!
  #     Note rings_along_dir='h' will always give you a full circle at each
  #     horizontal slice, so you automatically have consecutive points. The
  #     alternate_order will just make your circles go CW, CCW, CW, etc
  #     alternatively at each slice. So you don't need alternate order for
  #     rings_along_dir, if you just want consectuive points. It doesn't hurt
  #     you either.
  #   rings_along_dir: Along which direction to return the points in. 
  #     e.g. if along vertical ('v'), then return points in semi-circle slices
  #     in top to bottom or bottom to top. If along horizontal ('h'), then
  #     points are in sideways order, from one side of ellipsoid to the other.
  #     We can let you pick x y or z axis, because cos and sin have different
  #     signs in each quadrant. Would have to sort() one of the arrays, say x,
  #     and then record the index, reorder y and z using the same index. I
  #     don't have time to do that right now.
  def initialize_ellipsoid (self, c_xyz, r_xyz, deg_step=20,
    deg_step_long=None, deg_step_lat=None, deg_range_long=None, deg_range_lat=None,
    alternate_order=True, rings_along_dir='v'):

    print ('initialize_ellipsoid() called.')

    # Center of ellipsoid
    self.cx = c_xyz [0]
    self.cy = c_xyz [1]
    self.cz = c_xyz [2]

    # Radii of ellipsoid
    self.rx = r_xyz [0]
    self.ry = r_xyz [1]
    self.rz = r_xyz [2]

    if deg_step_long == None:
      deg_step_long = deg_step
    if deg_step_lat == None:
      deg_step_lat = deg_step
    self.radian_step_long = deg_step_long * np.pi / 180.0
    self.radian_step_lat = deg_step_lat * np.pi / 180.0

    # Default full ranges for longitude and latitude angles
    if deg_range_long == None:
      # Orange vertical slices
      range_long = (- np.pi * 0.5, np.pi * 0.5)
    else:
      # Convert to radians
      range_long = np.asarray (deg_range_long) * np.pi / 180.0
    if deg_range_lat == None:
      # Full horizontal circles
      range_lat = (-np.pi, np.pi)
    else:
      range_lat = np.asarray (deg_range_lat) * np.pi / 180.0

    # Reset vars
    # List of all points on ellipsoid surface
    self.surf_pts = []
    self.next_idx_to_return = 0

    # 1-based. If no extra rots are ever done, it's always 0
    self.extra_rots_done = 1


    #####
    # Calculate all surface points on the specified ellipsoid
    #   Parametric eqns http://en.wikipedia.org/wiki/Ellipsoid
    #####

    # Longitudes, u, theta. [-pi/2, pi/2]. pi/2-theta is polar angle.
    #   Vertical orange slices.
    # 1 x len(u), row vec
    theta = np.arange (range_long[0], range_long[1], self.radian_step_long)
    # Tack on np.pi * 0.5 at the end, if it's not covered
    if abs (theta [len (theta) - 1] - range_long[1]) > 0.01:
      theta = np.append (theta, range_long[1])

    # Latitudes, v, phi. [-pi, pi]. phi is azimuth angle of the point (x,y,z).
    #   Horizontal full ellipses.
    phi = np.arange (range_lat[0], range_lat[1], self.radian_step_lat)
    # Don't need to tack on np.pi. `.` -np.pi and np.pi are in same place, and
    #   -np.pi is already included! Adding np.pi would result in start and end
    #   points overlapping each other, giving you two of the same points!
    #   I don't want that.
    # Tack on np.pi at the end, if it's not covered
    #if abs (v [len (v) - 1] - np.pi) > 0.01:
    #  v = np.append (v, np.pi)
    # Transpose into column vector, so can multiply with row vector u
    # len(u) x 1, col vec
    phi = np.reshape (phi, (np.size (phi), 1))

    # Store input v for later seam-finding
    self.phi = phi

    # Take number of steps in 2*pi (360 degs) range as the total number of
    #   extra rotations to do.
    # Alternatively, can do
    #   np.shape (np.arange (0, 2 * np.pi, self.radian_step_lat)
    # as long as arange() is the way that surface points are generated!
    self.n_extra_rots = np.shape (phi) [0]

    #print ('u size %d, v size %d' % (np.size (u), np.size (v)))
    #print (u)
    #print (v)

    # You can check if the difference btw all points (including last elt
    #   manually appended) are the same, using this, all values printed should
    #   be the same:
    #u[1:np.size(u)] - u[0:np.size(u)-1]
    #v[1:np.size(v)] - v[0:np.size(v)-1]

    # len(v) x len(u), 2D array
    x = self.cx + self.rx * np.cos (theta) * np.cos (phi)
    # len(v) x len(u), 2D array
    y = self.cy + self.ry * np.cos (theta) * np.sin (phi)
    # 1 x len(u), row vec
    z = self.cz + self.rz * np.sin (theta)

    # repmat z to make it len(v) x len(u) 2D array as well
    z = np.tile (z, (np.size (phi), 1))


    #####
    # Reorder the points for returning
    #####

    if rings_along_dir == 'h':
      # Transpose to order the other way
      x = x.T
      y = y.T
      z = z.T

    # Flip every other row in the matrices
    if alternate_order:
      # For every odd row (1, 3, 5, 7, ...), flip the row
      for i in range (1, np.shape (x) [0], 2):
        # Must use flipud(), not fliplr(), because fliplr() requires 2D
        #   input. Row of a matrix is always 1D. Tested it does the right
        #   thing.
        x [i, :] = np.flipud (x [i, :])
        y [i, :] = np.flipud (y [i, :])
        z [i, :] = np.flipud (z [i, :])


    #####
    # Detect for overlapping points, eliminate the duplicates, keep only one.
    # This happens especially at the poles.
    #####

    # Reshape into column vectors for tiling, (len(u) * len(v)) x 1
    x = x.reshape (x.size, 1)
    y = y.reshape (y.size, 1)
    z = z.reshape (z.size, 1)

    xy = np.append (x, y, axis=1)	
    xyz = np.append (xy, z, axis=1)

    # Self duplicates are marked as non-duplicates (-1) in this function (`.`
    #   a row is always a duplicate of itself), which makes the caller's life
    #   easier - to eliminate duplicates, just eliminate all rows that are not
    #   -1. In other words, keep all rows that are non-dups (-1).
    dup_row_idxes = find_dups_self (xyz)
    # -1 means non-duplicates
    nondup_row_idxes = np.where (dup_row_idxes == -1) [0]

    # Keep only non-duplicates
    xyz = xyz [nondup_row_idxes, :]


    # Linearize x y z into a list of 3-tuples
    # len(u) * len(v) * 3
    # Ref: http://stackoverflow.com/questions/13704860/zip-lists-in-python
    #self.surf_pts = zip (x.flatten(), y.flatten(), z.flatten())
    self.surf_pts = zip (xyz[:, 0], xyz[:, 1], xyz[:, 2])


  def get_radii (self):
    return [self.rx, self.ry, self.rz]

  def get_latitude_angles (self):
    return self.v


  # Lets user set which point on ellipsoid to go to, if they don't want to
  #   start from [0].
  def set_next_idx (self, idx):

    self.next_idx_to_return = idx

    print ('EllipsoidSurfacePoints set_next_idx(): next_idx_to_return set to %d' % self.next_idx_to_return)


  def get_next_idx (self):

    return self.next_idx_to_return


  def get_n_points (self):

    return len (self.surf_pts)


  # Shortcut to get_next_point(). This calls get_next_point and returns four
  #   large NumPy arrays with all poses.
  # Returns:
  #   pts: n x 3 array, each row is (tx ty tz)
  #   vecs: n x 3 array, each row is the vector from center to each surface
  #     point? Or the other direction
  #   quats: n x 4 array, each row is (qx qy qz qw)
  #   mats: n x 4 x 4 array, each 4 x 4 matrix is the matrix form of the
  #     quaternion rotations
  def get_all_points (self, quat_wrt=(1.0, 0.0, 0.0), vec_inward=True,
    extra_rot_if_iden=None):

    pts = np.zeros ((len (self.surf_pts), 3))
    vecs = np.zeros ((len (self.surf_pts), 3))
    quats = np.zeros ((len (self.surf_pts), 4))
    mats = np.zeros ((len (self.surf_pts), 4, 4))

    # Restart the counter at 0, to make sure to return ALL points. This takes
    #   care of if get_next_point() was called previously, and the counter has
    #   advanced.
    self.set_next_idx (0)

    for i in range (self.get_n_points ()):
      curr_pt, curr_vec, curr_q, curr_mat, _ = self.get_next_point (
        quat_wrt, vec_inward, extra_rot_if_iden)

      # Safety guard. No more points left. Shouldn't happen `.` I
      #   set_next_idx(0) before loop, but just in case sth changes later.
      if curr_pt is None:
        break

      pts [i, :] = curr_pt
      vecs [i, :] = curr_vec
      quats [i, :] = curr_q
      mats [i, :, :] = curr_mat

    return pts, vecs, quats, mats


  # Returns point on surface, vector pointing from center of ellipsoid to
  #   point (or the other way, specified by vec_inward), and the quaternion
  #   representing the vector.
  # Parameters:
  #   quat_wrt: Axis with respect to which, to calculate the returning
  #     quaternion. If you're running nothing and just want to see ellipsoid,
  #     pass in identity quaternion, which is x-axis (1 0 0). If you are
  #     posing an end-effector to the returned quaternion directly, then use
  #     z-axis (0 0 1), which is the approach axis of the end-effector.
  #   vec_inward: If True, return vector pointing from a point on surface of
  #     ellipsoid to center. Else, return vector pointing the opposite dir.
  #   extra_rot_if_iden: Base vector to rotate by, should be same as quat_wrt.
  #     If quat_wrt == point on surface, when they are normalized, then the
  #     quaternion btw the two will be identity. Specify non-None if this is
  #     undesired.
  #     Used by sample_gazebo.py to make poses at bottom center of ellipsoid
  #     have different orientations, with quat_wrt=(0.0, 0.0, 1.0),
  def get_next_point (self, quat_wrt=(1.0, 0.0, 0.0), vec_inward=True,
    extra_rot_if_iden=None):

    # Sanity checks
    if not self.surf_pts:
      print ('get_next_point: Ellipsoid not initialized yet. Call initialize_ellipsoid() manually, before calling get_next_point().')
      return None, None, None, None, False

    if self.next_idx_to_return == len (self.surf_pts):
      print ('%sget_next_point(): Already returned last point in ellipsoid! No more points to return.%s' % (
        ansi_colors.WARNING, ansi_colors.ENDC))
      # Reset counter, in case user calls again. This happens if user called
      #   visualize_ellipsoid() before manually fetching each point.
      self.next_idx_to_return = 0
      return None, None, None, None, False

    print ('%sell_idx: [%d] out of %d (0 based) %s' % ( \
      ansi_colors.OKCYAN, self.next_idx_to_return, len (self.surf_pts) - 1,
      ansi_colors.ENDC))
    
    pt = np.array (self.surf_pts [self.next_idx_to_return])

    # Vector pointing from ellipsoid center to point on surface
    if vec_inward:
      vec = np.array ([self.cx, self.cy, self.cz]) - pt
    else:
      vec = pt - np.array ([self.cx, self.cy, self.cz])

    # 4-tuple. Pass in x, `.` quaternion 0 0 0 w=1 is x-axis.
    # mat is 4 x 4 NumPy matrix. Last col and row are 0 0 0 1.
    #   mat = tf.transformations.quaternion_matrix (q)
    quat, mat = get_relative_rotation_v (quat_wrt, vec)

    # When quat_wrt == vec, get_relative_rotation_v(quat_wrt, vec) will return
    #   identity. (Or when quat_wrt == -vec, it returns a 180-deg rotation wrt
    #   a random vector).
    #   That may not desired, e.g. if quat_wrt=(0,0,1), vec_inward=True, and
    #   the current surface point is at bottom center of ellipsoid. Then
    #   vec = c - pt = (0,0,0) - (0,0,-rz) = (0,0,rz), which normalizes to
    #   (0, 0, 1) in get_relative_rotation_v(), which is the same as
    #   quat_wrt! Then vec == quat_wrt, and you get identity 0 0 1 back. If
    #   there are many points at the bottom center of ellipsoid, and you wanted
    #   them to each have different orientations, this will not be possible!
    #   So in case quat_wrt == vec, flip quat_wrt, to produce different
    #   orientations at the surface point where quat_wrt == vec.
    if extra_rot_if_iden is not None:
      vec, quat, mat = self.apply_extra_rot_if_equal (quat_wrt, vec, quat, mat,
        extra_rot_if_iden)


    self.next_idx_to_return += 1

    # Whether this is the last point. Warns caller if caller needs to do sth
    #   special at last point.
    isLast = (self.next_idx_to_return == len (self.surf_pts) - 1)

    return (pt, vec, quat, mat, isLast)


  # Helper function
  def apply_extra_rot_if_equal (self, quat_wrt, vec, quat, mat, extra_rot_base):

    vec_norm = vec / np.linalg.norm (vec)
    quat_wrt_norm = quat_wrt / np.linalg.norm (quat_wrt)

    delta = np.array ([1e-6, 1e-6, 1e-6])
 
    # If curr_vec_norm == quat_wrt_norm (or the opposite, for bottom and top
    #   of ellipsoid in sample_gazebo.py), allowing floating point error
    if np.all (np.abs (vec_norm - quat_wrt_norm) < delta) or \
       np.all (np.abs (-vec_norm - quat_wrt_norm) < delta):
 
      # extra_rots_done is 1-based, to make sure never rotate by 0 (not
      #   rotating)!
      extra_rot_rad = self.radian_step_lat * self.extra_rots_done

      extra_rot_euler = np.array (extra_rot_base) * extra_rot_rad
      extra_rot_q = tf.transformations.quaternion_from_euler ( \
        *extra_rot_euler.tolist())
 
      extra_rot_mat = tf.transformations.quaternion_matrix (extra_rot_q)
 
      vec = np.dot (extra_rot_mat [0:3, 0:3], vec)
      quat = tf.transformations.quaternion_multiply (quat, extra_rot_q)
      mat = np.dot (mat, extra_rot_mat)

      # 1-based. Update for next time.
      self.extra_rots_done += 1
      if self.extra_rots_done > self.n_extra_rots:
        # Reset
        self.extra_rots_done = 1
 
    return vec, quat, mat


  # Returns a visualization_msgs/MarkerArray object. Caller is responsible for
  #   publishing it to a rostopic!
  # Can call this whenver after calling initialize_ellipsoid. The whole
  #   ellipsoid is already in memory.
  # Visualization is always the x-axis of the frame!! So even if you pass in
  #   quat_wrt=(0,0,1) for e.g. approach direction of an end-effector, the
  #   quaternion returned from get_next_point() will be to line up z-axis with
  #   the vector pointing at center of ellipsoid, but by definition the
  #   identity quaternion is always x-axis, so plotting with RViz Marker will
  #   always rotate quaternion from x-axis. Therefore visualization will always
  #   be of the x-axis.
  #   visualize_one_point() will plot the x-axis of the resulting frame. This
  #   might not be intuitive, should add feature to visualize_one_point() to
  #   take quat_wrt and visualize that frame, but I haven't done that yet. TODO
  # Parameters:
  #   duration: Use 0 for forever
  #   vis_frames: Visualize the rgb frames at each point. Would take a lot of
  #     memory in RViz to plot 3 lines at each point.
  #   extra_rot_if_iden: Only used if vis_frames=True. See
  #     get_next_point() header comment.
  def visualize_ellipsoid (self, frame_id, duration=0,
    quat_wrt=(1.0, 0.0, 0.0), vec_inward=True,
    vis_quat=True, vis_idx=True, vis_frames=False,
    extra_rot_if_iden=None):

    marker_arr = MarkerArray ()

    n_pts = 0

    while True:
      curr_pt, curr_vec, curr_q, curr_mat, _ = self.get_next_point ( \
        quat_wrt, vec_inward, extra_rot_if_iden)

      # Sanity check, termination check
      if curr_pt is None:
        # If no point was returned
        if n_pts == 0 and n_pts != self.get_n_points ():
          print ('%sNo more points remaining to return. Did you call before to get all the points already? Will reset set_next_idx(0) and visualize all the points.%s' % (
            ansi_colors.WARNING, ansi_colors.ENDC))

          # Restart the counter at 0. Subsequent iterations will get all pts
          self.set_next_idx (0)
          continue

        # If there was at least one point to visualize, None means hit the end,
        #   done
        else:
          break

      marker_arr = self.visualize_one_point (curr_pt, curr_vec, curr_q,
        curr_mat, marker_arr, n_pts, frame_id, duration,
        quat_wrt, vec_inward, vis_quat, vis_idx, vis_frames,
        extra_rot_if_iden)

      n_pts += 1

    return marker_arr


  # Convenience function so other files can call get_all_points(), manipulate
  #   the points, then visualize them using this.
  # Used by triangle_sampling plan_poses.py, execute_poses.py.
  # pts: n x 3 numpy array
  # vs: Only used if vis_quat=True
  # qs: n x 4 numpy array
  def visualize_custom_points (self, pts, vs, qs, frame_id, duration=0,
    quat_wrt=(1.0, 0.0, 0.0),
    vec_inward=True, vis_quat=True, vis_idx=True, vis_frames=False,
    extra_rot_if_iden=None):

    assert (pts.shape [0] == qs.shape [0])
    nPts = pts.shape [0]

    marker_arr = MarkerArray ()

    for i in range (nPts):
      #print ('Visualizing custom point [%d] out of %d' % (i, nPts))

      mat = tf.transformations.quaternion_matrix (qs [i, :])

      if vs is not None:
        vec = vs [i, :]
      else:
        vec = None

      marker_arr = self.visualize_one_point (pts [i, :], vec, qs [i, :],
        #mats [i, :, :],
        mat,
        marker_arr, i, frame_id, duration,
        quat_wrt, vec_inward, vis_quat, vis_idx, vis_frames, extra_rot_if_iden)

    return marker_arr


  #   curr_vec: Only used if vis_quat=True
  def visualize_one_point (self, curr_pt, curr_vec, curr_q, curr_mat,
    marker_arr, p_i, frame_id, duration=0,
    quat_wrt=(1.0, 0.0, 0.0),
    vec_inward=True, vis_quat=True, vis_idx=True, vis_frames=False,
    extra_rot_if_iden=None):

    # Plot RGB little frames at each point on ellipsoid, 1 cm long
    frm_len = 0.04

    alpha = 0.7
    frms_alpha = 0.4


    #r = np.cos (p_i) / 180.0 * np.pi)
    #g = np.sin (p_i) / 180.0 * np.pi)
    #b = np.fabs (r - g)
    # Take p_i divided by total number of points, to get [0, 1], then
    #   multiply by 2 * pi, so we stay in 360 range, to get a unique color
    #   per point.
    # Blue-yellow color theme
    #r = p_i / float (len (self.surf_pts))
    #g = p_i / float (len (self.surf_pts))
    #b = .5

    # Green-magenta color theme
    r = p_i / float (len (self.surf_pts))
    g = .5
    b = p_i / float (len (self.surf_pts))
    #print ('r %.1f g %.1f b %.1f' % (r, g, b))

    marker_p = Marker ()
    create_marker (Marker.POINTS, 'ellipsoid_pts', frame_id, p_i,
      0, 0, 0, r, g, b, alpha, 0.01, 0.01, 0.01,
      marker_p, duration)  # Use 0 duration for forever
    marker_p.points.append (Point (curr_pt[0], curr_pt[1], curr_pt[2]))
    marker_arr.markers.append (marker_p)
   
    # Visualize the quaternion to make sure it's correct
    if vis_quat:
      marker_v = Marker ()
      # Set where to start the arrow
      if not vec_inward:
        v_x = self.cx
        v_y = self.cy
        v_z = self.cz
      else:
        v_x = curr_pt[0]
        v_y = curr_pt[1]
        v_z = curr_pt[2]

      # Rotate the quaternion to the axis requested to be visualized.
      #   `.` curr_q is a regular quaternion, by default is from identity,
      #   which is x-axis (1, 0, 0).
      # Get the rotation between x-axis (1 0 0) to the desired axis to be
      #   visualized. e.g. if quat_wrt=(0 0 1), aug_q should be a rotation of
      #   -90 wrt y.
      aug_q, aug_mat = get_relative_rotation_v ((1.0, 0.0, 0.0), quat_wrt)
      #print (aug_mat)
      # Apply the additional rotation to curr_q, which is orientation of x-axis.
      #   This will make a quaternion that rotates to the desired axis to be
      #   visualized.
      curr_q_aug = tf.transformations.quaternion_multiply (curr_q, aug_q)

      create_marker (Marker.ARROW, 'ellipsoid_vec', frame_id, p_i,
        v_x, v_y, v_z, r, g, b, alpha,
        # scale.x is length, scale.y is arrow width, scale.z is arrow height
        np.linalg.norm (curr_vec), 0.02, 0.02,
        marker_v, duration,  # Use 0 duration for forever
        qw=curr_q_aug[3], qx=curr_q_aug[0], qy=curr_q_aug[1], qz=curr_q_aug[2])
      marker_arr.markers.append (marker_v)

    # For RGB frame; and for offsetting text index label so they don't all
    #   overlap, if the same position has multiple orientations to be plotted.
    # Multiply basis axes by the rotation matrix, to get the rotated
    #   frame's basis axes.
    x_ax = np.dot (curr_mat [0:3, 0:3], np.array ([1.0, 0.0, 0.0]))
    x_ax = x_ax / np.linalg.norm (x_ax) * frm_len


    # RGB frame that the rotation would create
    if vis_frames:

      y_ax = np.dot (curr_mat [0:3, 0:3], np.array ([0.0, 1.0, 0.0]))
      y_ax = y_ax / np.linalg.norm (y_ax) * frm_len

      z_ax = np.dot (curr_mat [0:3, 0:3], np.array ([0.0, 0.0, 1.0]))
      z_ax = z_ax / np.linalg.norm (z_ax) * frm_len

      marker_x_frm = Marker ()
      marker_y_frm = Marker ()
      marker_z_frm = Marker ()

      create_marker (Marker.ARROW, 'ellipsoid_frms_x', frame_id, p_i,
        # scale.x shaft diameter, scale.y head diameter, scale.z head length
        0, 0, 0, 1, 0, 0, frms_alpha, 0.007, 0.009, 0,
        marker_x_frm, duration)
      create_marker (Marker.ARROW, 'ellipsoid_frms_y', frame_id, p_i,
        0, 0, 0, 0, 1, 0, frms_alpha, 0.007, 0.009, 0,
        marker_y_frm, duration)
      create_marker (Marker.ARROW, 'ellipsoid_frms_z', frame_id, p_i,
        0, 0, 0, 0, 0, 1, frms_alpha, 0.007, 0.009, 0,
        marker_z_frm, duration)

      marker_x_frm.points.append (Point (curr_pt[0], curr_pt[1], curr_pt[2]))
      marker_x_frm.points.append (Point (curr_pt[0]+x_ax[0],
        curr_pt[1]+x_ax[1], curr_pt[2]+x_ax[2]))

      marker_y_frm.points.append (Point (curr_pt[0], curr_pt[1], curr_pt[2]))
      marker_y_frm.points.append (Point (curr_pt[0]+y_ax[0],
        curr_pt[1]+y_ax[1], curr_pt[2]+y_ax[2]))

      marker_z_frm.points.append (Point (curr_pt[0], curr_pt[1], curr_pt[2]))
      marker_z_frm.points.append (Point (curr_pt[0]+z_ax[0],
        curr_pt[1]+z_ax[1], curr_pt[2]+z_ax[2]))

      marker_arr.markers.append (marker_x_frm)
      #marker_arr.markers.append (marker_y_frm)
      marker_arr.markers.append (marker_z_frm)


    # Text showing point index
    if vis_idx:
      marker_t = Marker ()
      # Offset to be above x-axis tip, so that when the same point on ellipsoid
      #   has multiple orientations (in the case of custom points edited in
      #   caller of this class), the texts don't all overlap.
      create_marker (Marker.TEXT_VIEW_FACING, 'ellipsoid_idx', frame_id, p_i,
        curr_pt[0]+x_ax[0], curr_pt[1]+x_ax[1], curr_pt[2]+x_ax[2] + 0.01,
        r, g, b, alpha, 0, 0, 0.03,
        marker_t, duration)
      marker_t.text = '%d' % p_i
      marker_arr.markers.append (marker_t)

    return marker_arr


# ps: n x 3
def find_ellipsoid_seam (c, r, ps):

  # p = (x, y, z)
  # Ellipsoid standard equation:
  # x = c[0] + r[0] * np.cos (u) * np.cos (v)
  # y = c[1] + r[1] * np.cos (u) * np.sin (v)
  # z = c[2] + r[2] * np.sin (u)

  print (ps)
  print (c)
  print (r)

  print ('(ps[:, 2] - c[2]) / r[2]')
  print ((ps[:, 2] - c[2]) / r[2])

  # n x 1. Valid domain: [-1, 1]
  us = np.arcsin ((ps[:, 2] - c[2]) / r[2])
  # n x 1. use arssin and y value to calculate vs
  vs = np.arcsin ((ps[:, 1] - c[1]) / r[1] / np.cos (us))

  nan_idx = np.where (np.isnan (vs))

  # If cos gave 0, use arccos and x value to calculate vs.
  vs [nan_idx] = np.arccos ((ps[nan_idx, 0] - c[0]) / r[0] / np.cos (us [nan_idx]))

  print ('us:')
  print (us)
  print ('vs:')
  print (vs)

  seam_idx = np.zeros ((ps.shape [0],))

  # Assumption (implicit): Seam is defined at latitude angle self.v[0].
  #   At this angle, u changes. u is the same for all points on a horizontal
  #   ring. So a change in u indicates we have advanced to another horizontal
  #   ring. That is why this works - we can simply detect when u changes, to
  #   find where the seam is.
  curr_ring_seam_idx = 0
  for i in range (us.size):
    # If this is a new ring, update ring u value
    if np.abs (us [i] - us [curr_ring_seam_idx]) > 1e-6:
      curr_ring_seam_idx = i
    seam_idx [i] = curr_ring_seam_idx

  print (seam_idx)


  # Not very good. vs is 0 when it's not supposed to be. So you'd get 2 seams
  #   180 degs from each other
  '''
  # Define arbitrary seam to be points that have v = 0
  # Set boolean for rows in vs where vs == -np.pi
  #seam_bool [np.where (np.abs (vs + np.pi) < 1e-6)] = True
  seam_bool [np.where (np.abs (us + np.pi * 0.5) < 1e-6)] = True

  print ('seam_bool:')
  print (seam_bool)

  seam_idx = np.zeros (seam_bool.shape)
  prev_seam_idx = -1

  # For each point, set the index of the seam on its horizontal ring.
  # Assumption: seam point always appears as the first point on the ring.
  #   This assumption is true `.` initialize_ellipsoid() returns points in
  #   order, and v starts at -np.pi (where seam is intentionally defined).
  for i in range (seam_bool.shape [0]):
    if seam_bool [i]:
      prev_seam_idx = i

    seam_idx [i] = prev_seam_idx
  '''

  return seam_idx


# This tests that the class outputs points that are actually on the ellipse.
#   Human should check RViz plot to see that all the points look like an
#   ellipse.
def main ():

  rospy.init_node ('geo_ellipsoid', anonymous=True)

  vis_arr_pub = rospy.Publisher ('/visualization_marker_array',
    MarkerArray, queue_size=2)


  thisNode = EllipsoidSurfacePoints ()

  center = (0.1, 0.2, 0.3)
  # Use deg_step=60 for easier debugging, fewer arrows to track
  # 30 deg step: 90 pts
  # 25 deg step: 144 pts
  # 20 deg step: 190 pts
  thisNode.initialize_ellipsoid (center, (0.2, 0.4, 0.8), deg_step=25,
    alternate_order=True, rings_along_dir='h')

  frame_id = '/base'
  marker_arr = thisNode.visualize_ellipsoid (frame_id, quat_wrt=(0.0,0.0,1.0),
    vec_inward=True, vis_frames=True,
    extra_rot_if_iden=(0,0,1))

  while not rospy.is_shutdown ():

    vis_arr_pub.publish (marker_arr)



if __name__ == '__main__':

  main ()

