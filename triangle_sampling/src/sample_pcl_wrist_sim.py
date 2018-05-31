#!/usr/bin/env python

# Mabel Zhang
# 6 Jul 2016
#
# Used with PCL triangle sampling (sample_pcl.cpp), to create fake wrist
#   poses for each triangle. The wrist poses are used in probabilities data
#   (written by io_probs.py) for active touch.
#
# Subscribes from message published by sample_pcl.cpp, at the end of each
#   object. The message contains all triangles sampled from the object, usually
#   thousands of triangles. For each triangle, compute a fake wrist pose.
#

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import numpy as np

# My packages
from util.quat_utils import get_relative_rotation_v
from tactile_map.create_marker import create_marker
from triangle_sampling.sample_gazebo_utils import SampleGazeboConfigs


# Calculate a fake wrist pose that could have generated a triangle of the
#   three specified points of contact.
# Parameters:
#   p0, p1, p2: nTriangles x 3 NumPy array, each row is (x, y, z).
#     Each element is one vertex on a triangle.
#   obj_center: (3,) NumPy array. Center of object point cloud loaded into the
#     world
def simulate_wrist_poses (pts0, pts1, pts2, obj_center, obj_radii,
  vis_arr_pub):

  # User adjust parameters

  doRViz = False

  # If True, wrist orientation is defined as a vector normal to the plane
  #   containing the triangle of contacts. This results in thousands of 
  #   unique wrist poses for even a 3 cm radius cube. 400 MB nxn probs file.
  #   Not manageable.
  # If False, wrist orientation is defined as a vector parallel to the vector
  #   from object center to the triangle center. In effect, the wrist is
  #   always facing object center. This is closer to the actual Gazebo training,
  #   where the wrist always facing object center, plus an addition -10 deg
  #   rotation wrt wrist y axis. Here, we won't have the -10 deg wrt y case,
  #   it's simplified, but hopefully since PCL has more contact points than
  #   Gazebo, it'll still be a good approximation. The triangle histogram
  #   distribution will probably certainly be different btw PCL and Gazebo
  #   probs training still.
  wrist_normal_to_tri = False


  # Main loop

  nTriangles = len (pts0)

  # Position and quaternion of wrist for each triangle sampled in the object
  ps_wrist = np.zeros ((nTriangles, 3))
  qs_wrist = np.zeros ((nTriangles, 4))


  for i in range (0, nTriangles):

    # Calc cross product of two vectors on the triangle, to get a vector
    #   perpendicular to the traingle plane
    # Remember that p0 or p1 is NOT a side of the triangle! It's simply a
    #   vector from origin 0. To get an actual side as a vector, need to do
    #   p0-p1, p1-p2, p2-p0, etc.
    #   Look at s10, s12 etc lines below.
    s10 = pts0[i] - pts1[i]
    s12 = pts2[i] - pts1[i]
 
    # A vector perpendicular to two sides of the triangle (cross prod of them),
    #   therefore normal to the plane containing the triangle.
    tri_n = np.cross (s10, s12)
 
    # Normalize the vector perpendicular to triangle
    tri_n /= np.linalg.norm (tri_n)
 
 
    #
    # Fake wrist orientation.
    # Pick the direction of the normal vector that points to inside of object.
    #
 
    # Triangle center
    tri_center = (pts0[i] + pts1[i] + pts2[i]) / 3.0;
 
    # A vector pointing from object center to triangle center.
    #   This indicates which direction is pointing "outside" of the object.
    v_out = tri_center - obj_center
 
    # Find vector representing the direction wrist is pointing
    # If want wrist z direction to be normal to the plane containing contact
    #   triangle
    if wrist_normal_to_tri:

      # Find a vector normal to the plane containing the contact triangle
      #   (there are 2 solutions, +/- dirs). Pick the direction pointing to
      #   "inside" of object.
      #   This simulates the wrist orientation. The vector is the wrist z axis,
      #   pointing from wrist to object.
      # Dot product is cos, 1 at 0 degs, 0 at 90 or 270 degs, and -1 at 180 deg.
      #   Vector pointing from "inside" of object should be in general direction
      #   of -180 degs from v_out. Thus just take the direction of normal that
      #   has a smaller dot product (with v_out), the product closer to -1.
      # This is translation-invariant. To visualize it, just add it to triangle
      #   center.
      if np.dot (-tri_n, v_out) < np.dot (tri_n, v_out):
        v_wrist = -tri_n
      else:
        v_wrist = tri_n

    # If want wrist z to simply point at object center, passing through the
    #   contact triangle center
    else:
      # Simply parallel to and in direction opposite to the "outside" vector
      v_wrist = -v_out
      v_wrist /= np.linalg.norm (v_wrist)
 
    # Calculate Quaternion that represents the orientation vector
    v_ref = (1, 0, 0)
    q_wrist, _ = get_relative_rotation_v (v_ref, v_wrist)
 
 
    #
    # Fake wrist position.
    # Along the triangle normal, offsetted from the triangle center by
    #   PALM_THICKNESS outside the object. "Outside" of object is determined
    #   above by calculation for wrist vector.
    #
 
    # 0.1
    PALM_THICKNESS = SampleGazeboConfigs.ELL_PALM_THICKNESS

    # Approximate distance btw object center and where the ray intersects the
    #   object model. In OpenSceneGraph, you'd be able to get this exact
    #   distance, given an object mesh and a ray. Here uses a heuristic.
    # The model radii of the dimension that has the greatest magnitude in
    #   triangle center, is used.
    max_dim_idx = np.argmax (np.fabs (tri_center))
    # With radii alone, points still too close to object! Add diameter then.
    #   Diameter makes hand dangerously close to 3 cm cube, but never colliding.
    #   Safer to just add a factor * PALM_THICKNESS.
    approx_radii = obj_radii [max_dim_idx]
 
    # Scale wrist direction (triangle normal pointing to inside of object)
    #   by palm thicknesses.
    # 2 * PALM_THICKNESS is too far from object.
    v_wrist_scaled = (1.5 * PALM_THICKNESS + approx_radii) * v_wrist
 
    # Wrist position is computed by a vector, starting from the triangle center,
    #   magnitude of PALM_THICKNESS, direction along triangle normal that points
    #   to outside of object.
    # Since v_wrist points to object inside, take the negative to point outside.
    p_wrist = tri_center - v_wrist_scaled
 

    ps_wrist [i, :] = p_wrist
    qs_wrist [i, :] = q_wrist


    # It's a bummer I can't simulate wrists in the C++ file, where everything
    #   else is visualized!
    #   So this will be a separate visualization... basically playing the whole
    #   thing all over again, but in Python - -.
    if doRViz:
      visualize_wrist_pose (p_wrist, q_wrist, tri_center, obj_center,
        pts0[i], pts1[i], pts2[i], i, vis_arr_pub)

    if rospy.is_shutdown ():
      break

  return ps_wrist, qs_wrist



# Visualize simulated wrist pose in RViz Markers
# Parameters:
#   p_wrist: position of wrist
#   q_wrist: orientation of wrist, in Quaternion
def visualize_wrist_pose (p_wrist, q_wrist, tri_center, obj_center,
  pt0, pt1, pt2, tri_idx, vis_arr_pub):

  # Marker code copied from geo_ellipsoid.py, where I also visualize a
  #   quaternion with an arrow! That has the correct code.

  marker_arr = MarkerArray ()

  frame_id = '/base'
  alpha = 0.8
  duration = 0

  # Plot a cyan square at wrist position
  marker_p = Marker ()
  create_marker (Marker.POINTS, 'wrist_pos', frame_id, 0,
    0, 0, 0, 0, 1, 1, alpha, 0.005, 0.005, 0.005,
    marker_p, duration)  # Use 0 duration for forever
  marker_p.points.append (Point (p_wrist[0], p_wrist[1], p_wrist[2]))
  marker_arr.markers.append (marker_p)

  # Add a copy to cumulative markers
  marker_p = Marker ()
  create_marker (Marker.POINTS, 'wrist_pos_cumu', frame_id, tri_idx,
    0, 0, 0, 0, 1, 1, alpha, 0.002, 0.002, 0.002,
    marker_p, duration)  # Use 0 duration for forever
  marker_p.points.append (Point (p_wrist[0], p_wrist[1], p_wrist[2]))
  marker_arr.markers.append (marker_p)


  # Plot a cyan arrow for orientation. Arrow starts at wrist orientation, ends
  #   at average center of current triangle.
  marker_q = Marker ()
  create_marker (Marker.ARROW, 'wrist_quat', frame_id, 0,
    p_wrist[0], p_wrist[1], p_wrist[2], 0, 1, 1, alpha,
    # scale.x is length, scale.y is arrow width, scale.z is arrow height
    np.linalg.norm (tri_center - p_wrist), 0.002, 0,
    marker_q, duration,  # Use 0 duration for forever
    qw=q_wrist[3], qx=q_wrist[0], qy=q_wrist[1], qz=q_wrist[2])
  marker_arr.markers.append (marker_q)


  # Plot a cyan arrow from average center of current triangle, to average
  #   center of object.
  marker_out = Marker ()
  create_marker (Marker.ARROW, 'tri_normal', frame_id, 0,
    0, 0, 0, 0, 1, 1, alpha,
    # scale.x is shaft diameter, scale.y is arrowhead diameter, scale.z is
    #   arrowhead length if non-zero
    0.001, 0.002, 0,
    marker_out, duration)  # Use 0 duration for forever
  marker_out.points.append (Point (obj_center [0], obj_center [1],
    obj_center [2]))
  marker_out.points.append (Point (tri_center [0], tri_center [1],
    tri_center [2]))
  marker_arr.markers.append (marker_out)


  # Plot current triangle, red
  marker_tri_l = Marker ()
  create_marker (Marker.LINE_STRIP, 'sample_tri', frame_id, 0,
    # scale.x is width of line
    0, 0, 0, 1, 0, 0, alpha, 0.001, 0, 0,
    marker_tri_l, duration)  # Use 0 duration for forever
  marker_tri_l.points.append (Point (pt0[0], pt0[1], pt0[2]))
  marker_tri_l.points.append (Point (pt1[0], pt1[1], pt1[2]))
  marker_tri_l.points.append (Point (pt2[0], pt2[1], pt2[2]))
  marker_tri_l.points.append (Point (pt0[0], pt0[1], pt0[2]))
  marker_arr.markers.append (marker_tri_l)

  marker_tri_p = Marker ()
  create_marker (Marker.POINTS, 'sample_pts', frame_id, 0,
    0, 0, 0, 1, 0, 0, alpha, 0.005, 0.005, 0.005,
    marker_tri_p, duration)  # Use 0 duration for forever
  marker_tri_p.points.append (Point (pt0[0], pt0[1], pt0[2]))
  marker_tri_p.points.append (Point (pt1[0], pt1[1], pt1[2]))
  marker_tri_p.points.append (Point (pt2[0], pt2[1], pt2[2]))
  marker_arr.markers.append (marker_tri_p)


  vis_arr_pub.publish (marker_arr)
  # Pause a bit, to let msg get pushed through
  rospy.sleep (0.1)
  # Pause longer, for user to see
  #rospy.sleep (0.2)


