#!/usr/bin/env python

# Mabel Zhang
# 10 Mar 2015
#
# Make a seam from a cloud of the profile edge of a rotationally symmetric
#   object. Cloud is published by PointCloud_publisher.cpp on /cloud1_pcd
#   rostopic.
# Rotate the seam 360 degrees at some interval, wrt a center axis (found
#   somehow) at an adjustable radius.
#
# To test using a seam saved in a .pcd file, publish the .pcd file on rostopic:
#   $ rosrun pcl_ros pcd_to_pointcloud file.pcd 1 _frame_id:=/base
#   where
#     the (optional) integer is publish frequency in seconds;
#     ~frame_id should be set to self.marker_frame in manual_explore.py, or
#       whatever frame you saved the point cloud from.
#   This will publish a PointCloud2 onto topic /cloud_pcd, which
#     PointCloud_publisher.cpp listens to and publishes PointCloud onto topic
#     /cloud1_pcd, which this file listens to.
#
# Visualize the orig seam cloud by adding to RViz: PointCloud, topic
#   /cloud1_pcd.
#   Launch baxter_reflex.launch (if on real robot) or baxter_world.launch
#   (if in sim) first, so that the cloud has an existing reference frame_id.
#   Currently (10 Mar 2015), the ref frame that the pcd file is saved from is
#   /base on Baxter.
#
# To run this file, use the alias defined in aliases.bash:
#   $ spin_seam
#   Or use full command:
#   $ roslaunch tactile_map spin_seam.launch
#

# ROS
import rospy
from sensor_msgs.msg import PointCloud
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import tf

# Python
import numpy as np
# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
#from sklearn.decomposition import PCA
import math

# Local
from tactile_map.create_marker import create_marker
# Don't need this, this is included in the python tf package!
#   http://wiki.ros.org/tf/TfUsingPython
#from quaternion_from_matrix import quaternion_from_matrix
from est_ransac_line import project_pts_onto_line, calc_pt_to_line_dist


class SpinSeam:

  def __init__ (self):

    # Mode.
    # Hardcoded way was the originally coded way. Chooses a small sample of 
    #   contact points to spin. Assumes center axis is [0 0 1], therefore uses
    #   original z values in point cloud as height in obj to do the spinning.
    #   This mode is good for fool-proof testing. No dependency outside.
    # Flex way is the newer way after refactoring out spin_cloud() for outside
    #   function to call, for more general purpose. A point's height in object
    #   is computed by projecting the point onto center axis. Specific radius
    #   at a height is computed by computing the point's perpendicular
    #   distance to the center axis. This mode has outside function dependency.
    self.HARDCODE0_FLEX1 = 1

    self.seam_cloud = PointCloud ()
    #self.marker_frame = ''

    self.vis_pub = rospy.Publisher ('/visualization_marker', Marker)

    # To broadcast a new tf frame at center of cloud
    self.cloud_PCAed_frame = 'cloud_PCAed_center'
    self.cloud_PCAed_pub = rospy.Publisher (
      '/tactile_map/spin_seam/cloud_PCAed', PointCloud)
    self.bc = tf.TransformBroadcaster ()

    # To publish a downsampled cloud
    # Format: [[x1, y1, z1], [x2, y2, z2], ...] List of 3-lists.
    self.cloud_down_np = []
    self.cloud_down_pub = rospy.Publisher (
      '/tactile_map/spin_seam/cloud_downed', PointCloud)
    self.cloud_spun_pub = rospy.Publisher (
      '/tactile_map/spin_seam/cloud_spun', PointCloud)


  # msg: sensor_msgs PointCloud. Published by PointCloud_publisher.cpp, upon
  #   receiving msg /cloud_pcd.
  #   msg API: http://docs.ros.org/api/sensor_msgs/html/msg/PointCloud.html
  def pcdCB (self, msg):
    self.seam_cloud = msg
    #self.marker_frame = msg.header.frame_id
    #rospy.loginfo ('Got cloud, %d points.' % len (msg.points))

    self.downsample_cloud ()


  # Remove points within 5 mm
  def downsample_cloud (self):

    # 5 mm
    grid_size = 0.008

    # First pass, assign each point in the cloud to a voxel in the grid, by
    #   dividing by grid size
    grid_nonempty = []
    for i in range (0, len (self.seam_cloud.points)):
      # Store in sparse form. i.e. just store coords (gx, gy, gz) in a list
      grid_x = int (round (self.seam_cloud.points [i].x / grid_size))
      grid_y = int (round (self.seam_cloud.points [i].y / grid_size))
      grid_z = int (round (self.seam_cloud.points [i].z / grid_size))

      # Add each grid cell coordinate triplet only once
      if [grid_x, grid_y, grid_z] not in grid_nonempty:
        grid_nonempty.extend ([[grid_x, grid_y, grid_z]])

    # For each grid cell that had a point, put one point at center of
    #   this grid (or can take average of points in the grid, but.. meh too
    #   much work, just put at center for now).
    # Result is a downsampled point cloud.
    # Note: Points in this cloud may not exist in orig cloud, since I'm just
    #   replacing points with a point in center of voxel.
    cloud_down = PointCloud ()
    self.cloud_down_np = []
    for i in range (0, len (grid_nonempty)):
      pt = Point ()
      pt.x = grid_size * grid_nonempty [i][0] + (grid_size / 2)
      pt.y = grid_size * grid_nonempty [i][1] + (grid_size / 2)
      pt.z = grid_size * grid_nonempty [i][2] + (grid_size / 2)
      cloud_down.points.extend ([pt])

      # Keep a member copy to use after this fn ends
      self.cloud_down_np.append (np.asarray ([pt.x, pt.y, pt.z]))

    #print ('Downsampled point cloud size: ' + str (len (cloud_down.points)))

    cloud_down.header.stamp = rospy.Time.now ()
    cloud_down.header.frame_id = self.seam_cloud.header.frame_id
    self.cloud_down_pub.publish (cloud_down)



  # Hardcoded to test how good the contact points are, whether it is possible
  #   to use them to spin a cloud that looks like the original object at all.
  def spin_cloud_hardcode (self):

    if len (self.cloud_down_np) < 1:
      return

    # Assume a radius for the general radius
    # TODO later: cannot assume. have to calculate for general case
    radius = 0.05

    # Assume some direction (from seam) for center axis of object. Unit vector
    # TODO later: cannot assume. have to calculate for general case
    center_dir = np.asarray ([0, -1, 0])

    # Assume orientation of center axis is vertical, along z axis of /base
    # TODO later: cannot assume. have to calculate for general case
    

    # Median of each of x, y, z components in cloud
    x_vals = [p[0] for p in self.cloud_down_np]
    y_vals = [p[1] for p in self.cloud_down_np]
    z_vals = [p[2] for p in self.cloud_down_np]
    med_x = np.median (x_vals)
    med_y = np.median (y_vals)
    med_z = np.median (z_vals)
    #rospy.loginfo ('%f %f %f' % (med_x, med_y, med_z))

    # Assume center axis of obj is radius distance away from median pt in cloud
    # Compute where center axis is
    center_pos = np.asarray ([med_x, med_y, med_z]) + (radius * center_dir)

    # Assumption: center axis of obj is along z.
    axis_start = center_pos
    axis_end = center_pos + np.asarray ([0, 0, 1])


    if self.HARDCODE0_FLEX1 == 1:
      # Heights of points wrt center axis
      # Not unique. Var name left from before.
      uniques_z = project_pts_onto_line (axis_start, axis_end, self.cloud_down_np)
     
      # Radius
      spun_radius = calc_pt_to_line_dist (axis_start, axis_end, self.cloud_down_np)

    # Old hardcoded way, before having generic spin_cloud().
    #   Good for fool-proof check. Just plots along center axis [0 0 1]. No
    #   outside function dependencies.
    else:
      # Unique obj heights
      # Assumption: center axis of obj is along z. So just take unique z values
      uniques_z = np.unique (z_vals)
     
      # Specific radius at each unique height
      # Let's just test 1 point at each object-height
      # Get the specific radius for each height.
      #   self.radius is the general radius. There is a specific radius at each
      #   height of the object. They could all be different.
      spun_radius = []
      for i in range (0, len (uniques_z)):
        # Just pick first point in cloud with this height.
        #   (list.index() returns 1st appearance's index)
        pt_idx = z_vals.index (uniques_z [i])
     
        # Assumption: center axis of obj is along z. So just take distance
        #   formula of x and y components of point in cloud and center point.
        spun_radius.extend ([ \
          math.sqrt ((x_vals [pt_idx] - center_pos[0]) ** 2 + \
                     (y_vals [pt_idx] - center_pos[1]) ** 2)])


    # Spin cloud around center axis, which we assume is radius distance away in 
    #   center_dir direction from the median point in cloud.

    # Assumption: center axis of obj is along z. So just give 0 0 1 as dir.
    #   Scale 0 0 1 by the tallest height to make another point on axis.
    cloud_spun = PointCloud ()
    spin_cloud (center_pos, 
      center_pos + np.asarray([0, 0, max(uniques_z)]),
      uniques_z, spun_radius, self.seam_cloud.header.frame_id, cloud_spun,
      self.HARDCODE0_FLEX1)

    self.cloud_spun_pub.publish (cloud_spun)



  # This tries to project all points in the seam cloud onto first 2 PCs from
  #   PCA, to get a clean seam.
  # It doesn't work, because it's not easy to get the new tf frame of the 2
  #   PCs correct, because they might not be perpendicular!
  '''
  def make_seam (self):
    if len (self.seam_cloud.points) < 1:
      return

    # Convert geometry_msgs/Point32[] cloud to Python list
    cloud_list = []
    for i in range (0, len (self.seam_cloud.points)):
      cloud_list.extend ([[ \
        self.seam_cloud.points [i].x,
        self.seam_cloud.points [i].y,
        self.seam_cloud.points [i].z]])

    # Convert to numpy array
    cloud_np = np.asarray (cloud_list)

    # TODO: PCA. See if top 2 PCs can give me the seam plane to project to...

    pca = PCA (n_components = 2)
    pca.fit (cloud_np)
    #rospy.loginfo ('PCA PCs: ')
    #rospy.loginfo (pca.components_)
    #rospy.loginfo ('PCA mean: ')
    #rospy.loginfo (pca.mean_)
    print ('PCA PCs: ')
    print (pca.components_)
    print ('PCA mean: ')
    print (pca.mean_)

    # Draw mean of cloud
    marker_mean = Marker ()
    create_marker (Marker.POINTS, 'cloud_center', 
      self.seam_cloud.header.frame_id, 0,
      0, 0, 0, 1, 0, 1, 0.5, 0.01, 0.01, 0.01,
      marker_mean, 60)  # Use 0 duration for forever
    pt_mean = Point ()
    pt_mean.x = pca.mean_ [0]
    pt_mean.y = pca.mean_ [1]
    pt_mean.z = pca.mean_ [2]
    marker_mean.points.extend ([pt_mean])
    self.vis_pub.publish (marker_mean)

    # Draw top PCs
    marker_pcs = Marker ()
    # Scale: LINE_LIST uses scale.x, width of line segments
    # LINE_LIST draws a line btw 0-1, 2-3, 4-5, etc. pairs of points[]
    # Pose is still used, just like POINTS still uses them. Make sure to set
    #   identity correctly, qw=1.
    create_marker (Marker.LINE_LIST, 'cloud_PCs',
      self.seam_cloud.header.frame_id, 0,
      0, 0, 0, 1, 0, 1, 0.5, 0.002, 0, 0,
      marker_pcs, 60) # Use 0 duration for forever
    # Add the 2 PCs. Draw a line from mean to each PC
    marker_pcs.points.extend ([pt_mean])
    pc1 = Point ()
    pc1.x = pca.components_ [0] [0]
    pc1.y = pca.components_ [0] [1]
    pc1.z = pca.components_ [0] [2]
    marker_pcs.points.extend ([pc1])
    marker_pcs.points.extend ([pt_mean])
    pc2 = Point ()
    pc2.x = pca.components_ [1] [0]
    pc2.y = pca.components_ [1] [1]
    pc2.z = pca.components_ [1] [2]
    marker_pcs.points.extend ([pc2])
    self.vis_pub.publish (marker_pcs)


    # Apply dimensionality reduction on data.
    #   Result is a list of 2D points
    cloud_PCAed_np = pca.transform (cloud_np)

    # Broadcast a new tf frame at the top 2 PCs
    # z-axis can just be the cross product of the 2 PCs, we don't care which
    #   direction, because this frame will just be a plane. z of all pts = 0.
    #   Normalize the 2 PCs to unit vecs.
    # To convert from (x1 y1 z1) and (x2 y2 z2) to a rotation, just write the
    #   rotation matrix using unit vectors of the top 2 PCs, and unit vector of
    #   their cross product:
    #   [ x1 x2 x3
    #     y1 y2 y3
    #     z1 z2 z3 ]
    #   i.e. coords of new x-axis on 1st col, y on 2nd col, z on 3rd col.
    #     coords are wrt old frame (the original cloud before PCA).
    new_frame_x = pca.components_[0] / np.linalg.norm (pca.components_[0])
    new_frame_y = pca.components_[1] / np.linalg.norm (pca.components_[1])
    new_frame_z = np.cross (new_frame_x, new_frame_y)
    new_frame_z = new_frame_z / np.linalg.norm (new_frame_z)
    # I hate Python. It doesn't have anything. Doesn't have tf::Matrix3x3.
    # http://docs.ros.org/indigo/api/tf/html/c++/classtf_1_1Matrix3x3.html
    #new_frame_rot_mat = Matrix3x3 (
    #  new_frame_x[0], new_frame_y[0], new_frame_z[0],
    #  new_frame_x[1], new_frame_y[1], new_frame_z[1],
    #  new_frame_x[2], new_frame_y[2], new_frame_z[2])
    # Convert rotation matrix to Quaternion
    #new_frame_quat = Quaternion ()
    #new_frame_rot_mat.getRotation (new_frame_quat)
    new_frame_rot_mat = [ \
      #[new_frame_x[0], new_frame_y[0], new_frame_z[0], 0],
      #[new_frame_x[1], new_frame_y[1], new_frame_z[1], 0],
      #[new_frame_x[2], new_frame_y[2], new_frame_z[2], 0],
      [new_frame_x[0], new_frame_x[1], new_frame_x[2], 0],
      [new_frame_y[0], new_frame_y[1], new_frame_y[2], 0],
      [new_frame_z[0], new_frame_z[1], new_frame_z[2], 0],
      [0, 0, 0, 1]]
    new_frame_quat = tf.transformations.quaternion_from_matrix (new_frame_rot_mat)
    # Broadcast new tf frame
    #   API: sendTransform(translation, rotation, time, child, parent)
    #   http://mirror.umd.edu/roswiki/doc/diamondback/api/tf/html/python/tf_python.html
    self.bc.sendTransform ((pt_mean.x, pt_mean.y, pt_mean.z),
      #(new_frame_quat.x (), new_frame_quat.y (), new_frame_quat.z (), new_frame_quat.w ()),
      (new_frame_quat[0], new_frame_quat[1], new_frame_quat[2], new_frame_quat[3]),
      rospy.Time.now (),
      self.cloud_PCAed_frame, self.seam_cloud.header.frame_id)

    # Create PointCloud obj for the projected data
    cloud_PCAed = PointCloud ()
    cloud_PCAed.header.frame_id = self.cloud_PCAed_frame
    cloud_PCAed.header.stamp = rospy.Time.now ()
    # Assumption: # PCs = 2. Else you might need z, if # PCs == 3.
    for i in range (0, len (cloud_PCAed_np)):
      pt = Point ()
      pt.x = cloud_PCAed_np [i] [0]
      pt.y = cloud_PCAed_np [i] [1]
      pt.z = 0
      cloud_PCAed.points.extend ([pt])

    self.cloud_PCAed_pub.publish (cloud_PCAed)

    # TODO plot the result projected points, publish as a PointCloud rostopic
    #   tactile_map/spin_seam/projected_seam. Eh wait, but... what....... you
    #   project onto a plane... but how do you translate that to /base frame?
    #   I don't think you need to. Since points were projected from 3d points
    #   in terms of /base frame, the projected 2D points should still be
    #   reachable in /base frame. Think in 2D Cartesian, you project 2D into 
    #   2D, then when you visualize, you still keep origin at same (0, 0),
    #   right? Or do you move origin to center of cluster?? Assume former.
    # Just plot with (x, y, 0), in self.marker_frame frame.
    # Wait that won't work. I think what you need to do is to ADD A TF FRAME
    #   at center of the cloud. Then you plot the 2D points (x y 0) in this 
    #   frame. `.` after you apply dimensionality reduction, the points are in
    #   terms of the 2 PCs frame, not in the original cloud's frame (/base)
    #   anymore. This makes sense.
  '''



# A generic function for functions outside this file to call.
# Spin cloud around center axis (pt2 - pt1), with heights and radii having
#   corresponding indices. i.e. at heights[i] of center axis, the radius at
#   that height is radii[i].
# Parameters:
#   axis_pt1, axis_pt2: Numpy 3-arrays. vector (pt2 - pt1) defines the axis
#   heights: Numpy float array. Number of elements is number of segments on
#     cylindrical shape. Distances of cross-section slices from axis_pt1.
#     If negative, then projection is on -(pt2-pt1).
#   radii: radii[i] is the radius of a point on obj at axis height height[i].
#   HARDCODE0_FLEX1: Only for use by spin_seam_hardcode().
# Returns cloud_spun in parameter, type sensor_msgs/PointCloud, in frame
#   frame_id.
# Returns the number of points per slice, as ret val. This is useful if
#   caller wishes to call calc_rough_normals().
def spin_cloud (axis_pt1, axis_pt2, heights, radii, frame_id, cloud_spun,
  HARDCODE0_FLEX1=1, deg_step=10.0):

  # Center axis as unit vector
  axis = axis_pt2 - axis_pt1
  axis_norm = np.linalg.norm (axis)
  axis_unit = axis / axis_norm


  ## Calcualte two axes lying in a plane perpendicular to axis (pt2-pt1).
  #    This is the plane that all circles on the rotationally symmetric
  #    object will lie in, since all circles are perpendicular to center axis.

  # Take any arbitrary vector vec.
  #   Cross product c1 = (axis x vec) gives a vector perpendicular to both
  #     vectors, hence perpendicular to center axis.
  #   Cross product c2 = (axis x c1) gives a second vector perpendicular to
  #     center axis.
  #   c1 and c2 are perpendicular as well, they span the plane perpendicular
  #     to center axis.
  #   Hence c1 and c2 define a plane that each circle on object will lie on.
  #     There are infinite planes perpendicular to a line. Which one depends
  #     on at what height the plane intersects the line.
  # Ref: This math is copied from my grasp_point_calc package skeleton.cpp.
  some_vec = np.asarray ([0, 0, 1])
  # Check if the provided axis is actually 0 0 1, then use a different vector
  if (abs (some_vec - axis_unit) < 1e-6).all ():
    some_vec = np.asarray ([0, 1, 0])
  c1 = np.cross (axis_unit, some_vec)
  c2 = np.cross (axis_unit, c1)

  # Normalize
  c1 = c1 / np.linalg.norm (c1)
  c2 = c2 / np.linalg.norm (c2)


  ## Spin 360 degrees

  # 10 deg step. Convert to radians
  deg_step = deg_step / 180.0 * math.pi

  deg_range = np.arange (0, 2 * math.pi, deg_step)

  # Let's just test 1 point at each object-height
  # For each object height
  # Assumption: object orientation is vertical. Might need to change for-
  #   loop counter if not vertical. OR, easier way is to have a tf frame
  #   that's ROTATED!!! Its orientation should be same as object 
  #   orientation. Then you can always assume object center axis is z
  #   axis of its tf frame!
  for i in range (0, len (heights)):

    # (starting point pt1 on center axis) + (h * unit vector along center
    #   axis) = a point on center axis. This point is distance h from pt1.
    circle_center = axis_pt1 + axis_unit * heights[i]

    # Loop through a circle by step size deg_step
    for theta in deg_range:
      # We're going to ignore how many points are at the same object height.
      #   We'll just spin one arbitrary pt, for simplicity.
      # At this point, since we already got the radius for this height from
      #   the loop above,  we don't care where the original point in the 
      #   cloud was anymore. All we need is the radius.

      # Use circle equation.


      if HARDCODE0_FLEX1 == 1:
        # Think of a unit circle in a plane. cos(t) gives you distance along v1,
        #   sin(t) gives you distance along v2. In 3D, to get to that point on
        #   the circle, simply add the two vectors.
        #   In 2D, the actual circle equation would be:
        #     (x, y) = r * cos(t) * [1 0] + r * sin(t) * [0 1]
        #   Here, v1 is like r * [0 0 1], v2 is like r * [0 1 0], but not x and
        #     y necessarily, can be any arbitrary axis.
        spun_pt = circle_center + ((radii [i] * math.cos(theta) * c1) +
          (radii [i] * math.sin (theta) * c2))

        cloud_spun.points.append (Point (spun_pt[0], spun_pt[1], spun_pt[2]))

      # Old hardcoded way, before having generic spin_cloud().
      #   Good for fool-proof check. Just plots along center axis [0 0 1]. No
      #   outside function dependencies.
      else:
        # x = r cos theta, y = r sin theta. Shifted by center of circle.
        spun_pt = Point ()
        spun_pt.x = axis_pt1[0] + radii[i] * math.cos (theta)
        spun_pt.y = axis_pt1[1] + radii[i] * math.sin (theta)
        # Assumption: center axis is straight up. TODO Will need to change later
        # How do you draw a cylinder on a slanted axis? You don't! You draw a
        #   straight cylinder, and then rotate it afterwards! This makes
        #   generalizing to center axis of other orientation easier.
        spun_pt.z = heights[i]

        cloud_spun.points.append (spun_pt)

  cloud_spun.header.stamp = rospy.Time.now ()
  cloud_spun.header.frame_id = frame_id

  return len (deg_range)


# Parameters:
#   cloud: sensor_msgs/PointCloud generated by spin_cloud().
#   ns: Number of points per slice. A slice is a circle at one height of a
#     helical object.
#   marker: visualization_msgs/Marker. Initialized in caller by
#     marker = Marker (), and create_marker().
#   frame_id: frame_id for marker
def calc_rough_normals (axis_pt1, axis_pt2, cloud, ns, marker=None):

  # Assumption: cloud orders from bottom of object to top of object
  #   This affects py1 and py2 equations.

  # Number of slices. This should yield an integer. If not, cloud size is wrong.
  nh = len (cloud.points) / ns

  # Convert to Numpy array
  cloud_np = [np.asarray ([p.x, p.y, p.z]) for p in cloud.points]


  # Ret val
  normals = []

  LEN = 0.01

  # Loop through 1D array cloud
  for i in range (0, len (cloud_np)):

    # Find 4 neighboring points, to compute tangent. See my derivation on paper
    
    # Slice 0 (bottom slice). In 1D array, pts in slice 0 have index < ns
    if i < ns:
      # First point in slice 0
      if i == 0:
        px1 = cloud_np [i + (ns - 1)]
        px2 = cloud_np [i + 1]
        py1 = cloud_np [i]
        py2 = cloud_np [ns + i]

      # Last point in slice 0
      elif i == ns - 1:
        px1 = cloud_np [i - 1]
        px2 = cloud_np [i - (ns - 1)]
        py1 = cloud_np [i]
        py2 = cloud_np [ns + i]

      # Other points in slice 0
      else:
        px1 = cloud_np [i - 1]
        px2 = cloud_np [i + 1]
        py1 = cloud_np [i]
        py2 = cloud_np [ns + i]

    # Top slice
    elif i > (nh - 1) * ns - 1:
      # First point in slice
      if i == (nh - 1) * ns:
        px1 = cloud_np [i + (ns - 1)]
        px2 = cloud_np [i + 1]
        py1 = cloud_np [i - ns]
        py2 = cloud_np [i]

      # Last point in slice
      elif i == (nh - 1) * ns + (ns - 1):
        px1 = cloud_np [i - 1]
        px2 = cloud_np [i - (ns - 1)]
        py1 = cloud_np [i - ns]
        py2 = cloud_np [i]

      # Other points in slice
      else:
        px1 = cloud_np [i - 1]
        px2 = cloud_np [i + 1]
        py1 = cloud_np [i - ns]
        py2 = cloud_np [i]

    # Other slices in btw top and bottom slices
    else:

      # Current slice number. Starts at 0
      s = np.floor (i / ns)

      # First point in slice
      if i == [s * ns]:
        px1 = cloud_np [i + (ns - 1)]
        px2 = cloud_np [i + 1]
        py1 = cloud_np [i - ns]
        py2 = cloud_np [i + ns]

      # Last point in slice
      elif i == s * ns + (ns - 1):
        px1 = cloud_np [i - 1]
        px2 = cloud_np [i - (ns - 1)]
        py1 = cloud_np [i - ns]
        py2 = cloud_np [i + ns]

      # Other points in slice
      else:
        px1 = cloud_np [i - 1]
        px2 = cloud_np [i + 1]
        py1 = cloud_np [i - ns]
        py2 = cloud_np [i + ns]

    tanx = px2 - px1
    tany = py2 - py1

    n = np.cross (tanx, tany)
    n = n / np.linalg.norm (n)


    # TODO: resolve ambiguity of +/- direction of cross product result.
    #   Don't need to code this yet, too complicated code. Just test this
    #   simple code as it is. Progressive testing.
    # For the purposes of Pottmann estimation of center axis, this is not
    #   needed. Because both positive and negative normal will intersect
    #   center axis! Same thing for the estimation.


    # Populate ret val
    normals.append ([cloud_np [i], n])

    endpt = cloud_np [i] + n * LEN

    marker.points.append (cloud.points [i])
    marker.points.append (Point (endpt[0], endpt[1], endpt[2]))
 

  return normals


if __name__ == '__main__':

  rospy.init_node ('spin_seam', anonymous=True)

  thisNode = SpinSeam ()
  rospy.Subscriber ('/cloud1_pcd', PointCloud, thisNode.pcdCB)

  wait_rate = rospy.Rate (10)
  while not rospy.is_shutdown ():

    #thisNode.make_seam ()

    thisNode.spin_cloud_hardcode ()

    try:
      wait_rate.sleep ()
    except rospy.exceptions.ROSInterruptException, err:
      break

