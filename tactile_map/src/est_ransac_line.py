#!/usr/bin/env python

# Mabel Zhang
# 6 Apr 2015
#
# Use RANSAC to estimate a line from a set of given 3D points.
#
# Called by est_center_axis_ransac.py
#


# ROS
from geometry_msgs.msg import Point

# OpenCV, for fitLine()
import cv2

# numpy, for fitLine ()
import numpy as np

# Python
import math
import random


# Centroid of all points.
#   The "average center" IS the centroid, by definition, of a set of points in
#     space.
#   This is not the same as the centroid for a polygon (i.e. all points are
#     coplanar), for which you actually have to sort the points in CCW or CW.
#   For a set of points in space, the average center is the only formula.
#   Ref: http://stackoverflow.com/questions/77936/whats-the-best-way-to-calculate-a-3d-or-n-d-centroid
#
# Disadvantage: If you have 3 coplanar points, then the polygon centroid
#   might be better. Imagine having 3 points, two on fore-fingers, one on
#   thumb. Then the average will bias toward the fore-fingers' side, because
#   there are more points in that direction.
#   For non-coplanar points though, which is more often the case, this is the
#   real 3D centroid.
#
# Parameters:
#   pts: geometry_msgs/Point[]. A list of >= 3 points.
# Returns geometry_msgs/Point
#
# To test this fn in Python command line, execute these in order:
#   from est_ransac_line import calc_3d_centroid
#   from geometry_msgs.msg import Point
#   calc_3d_centroid ([Point(1,2,3), Point(3,2,1), Point(4,4,20)])
def calc_3d_centroid (pts):

  xsum = float (sum ([pts[i].x for i in range (0, len (pts))]))
  ysum = float (sum ([pts[i].y for i in range (0, len (pts))]))
  zsum = float (sum ([pts[i].z for i in range (0, len (pts))]))

  return Point (xsum / len (pts), ysum / len (pts), zsum / len (pts))


'''
# Parameters:
#   pts: geometry_msgs/Point[]. A list of >= 3 points, which form a polygon
# Returns geometry_msgs/Point
# Ref polygon centroid:
#   Polygon: http://en.wikipedia.org/wiki/Centroid#Centroid_of_polygon
#   Quadrilateral: http://mathworld.wolfram.com/Quadrilateral.html
#   Triangle: https://www.easycalculation.com/analytical/learn-centroid.php
#
#   If use polygon formula, need to sort points.
#   Sort points in CW: http://stackoverflow.com/questions/6989100/sort-points-in-clockwise-order
def calc_2d_centroid (pts):

  # TODO wait wait wait.... what if the n points aren't even all in the same
  #   plane??? Is centroid even possible to do... omg wouldn't it have to
  #   be centroid of a poly-volume thing??? How do you even do that.


  ## Sort points in counter clockwise order
  #    Ref: http://stackoverflow.com/questions/6989100/sort-points-in-clockwise-order


  ## Calculate centroid
  #  Ref: http://en.wikipedia.org/wiki/Centroid#Centroid_of_polygon
'''



# Calls OpenCV fitLine ()
# Parameters:
#   pts: geometry_msgs/Point[]
# Returns normalized vector.
#   [vx, vy, vz] is a normalized vector colinear to fitted line;
#   [x0, y0, z0] is a point on the line.
def est_least_sqr_line (pts):

  # Convert points to numpy float32 format, which OpenCV takes
  #   Ref: np dtype 'f' http://docs.scipy.org/doc/numpy/user/basics.types.html
  pts_np = np.asarray (zip ( \
    [pts[i].x for i in range (0, len (pts))],
    [pts[i].y for i in range (0, len (pts))],
    [pts[i].z for i in range (0, len (pts))]), dtype = 'f')

  # Ref: 
  #   Example: http://stackoverflow.com/questions/14184147/detect-lines-opencv-in-object
  #   API: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
  # fitLine() returns a ndarray of ndarrays of length 1, weird. Why doesn't it
  #   just return a ndarray of floats??
  [vx, vy, vz, x0, y0, z0] = cv2.fitLine (pts_np,
    cv2.cv.CV_DIST_L2, 0, 0.01, 0.01)

  # Return the raw floats, not the length-1 ndarrays
  return [vx[0], vy[0], vz[0], x0[0], y0[0], z0[0]]



# Calculate perpendicular distance from each point in pts to the given line.
# Parameters:
#   pt1, pt2: 1 x 3 Numpy array. (pt2 - pt1) is a vector parallel to a line.
#     +/- direction of vector doesn't matter.
#   pts: A set of points. List of 1 x 3 Numpy arrays
# Ref: http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
def calc_pt_to_line_dist (pt1, pt2, pts):

  # Calc magnitude of vector parallel to line
  line = (pt2 - pt1)
  line_norm = np.linalg.norm (line)


  # Magnitude of cross product is equal to the area of the parallelogram
  #   formed by the two vectors. Area of parallelogram is twice the area of
  #   the triangle formed by the two vectors.
  #     A_para = |a x b|
  #     A_tri =  A_para / 2
  # Type: list of numpy arrays = list of numpy arrays - numpy array
  vec1 = pts - pt1  #pts_unchosen - pt1
  vec2 = pts - pt2  #pts_unchosen - pt2
  # Element-wise. ith elt in vec1 cross ith elt in vec2
  # Type: Numpy array. 1 x (# pts)
  area_paral = np.linalg.norm (np.cross (vec1, vec2), axis=1)

  # Triangle area A_tri = 1/2 * b * h
  #                   h = 2 A_tri / b
  #                   h = A_para  / b
  #   where b is base of triangle. It can be any of the 3 sides of the
  #     triangle, because the areas of the 3 (unique?) parallelograms formed
  #     by doubling the triangle towards any of the 3 legs are equal.
  #     So any b * h will yield A_para, b can be any side of the triangle.
  #     h corresponds to the height that sits on the chosen b.
  #   We want the b that's sitting on the chosen line. Then h gives us the
  #     height sitting on b, i.e. the perpendicular distance from the chosen
  #     line to the point!
  # Type: Numpy array. 1 x (# pts)
  dist = area_paral / line_norm

  return dist


# Use RANSAC to estimate a line that fits through the given points.
#   Need a minimum of 2 points to get a line.
# Parameters:
#   pts: geometry_msgs/Point[]. A list of points roughly along a line
#     (hopefully).
# Returns:
#   Highest voted line, in form of indices of the two points [i1, i2] chosen
#     from parameter to get this line.
#   Inliers, in form of indices of points that voted for the highest vote line.
#     This includes the two points that were voted highest.
#   #Perpendicular distances from all points to the highest voted line.
'''
# To test the calculation in this and calc_pt_to_line_dist() fn in python shell:

pt1=np.asarray([2,2,5])
pt2=np.asarray([8,4,1])

line = (pt2 - pt1) 
line_norm = np.linalg.norm (line)

pt_unchosen = [[2,4,5],[7,4,1],[5,1,5],[2,5,4]]
pt_unchosen = [np.asarray(p) for p in pt_unchosen]
vec1 = pt_unchosen - pt1
vec2 = pt_unchosen - pt2
area_paral = np.linalg.norm (np.cross (vec1, vec2), axis=1)
dist = area_paral / line_norm

dist_thresh = 1
dist < dist_thresh
'''
def est_ransac_line (pts_msg):

  ## Convert pts geometry_msgs/Point to numpy format, so can do math operations
  #  Result: a list of numpy arrays, [array[x1, y1, z1], ... array[xn, yn, zn]]
  pts = [np.asarray ([p.x, p.y, p.z]) for p in pts_msg]


  ## Calculate number of iterations to run

  # Ref:
  #   p.43-47:
  #   http://6.869.csail.mit.edu/fa12/lectures/lecture13ransac/lecture13ransac.pdf
  #   p.3-4:
  #   https://engineering.purdue.edu/kak/computervision/ECE661Folder/Lecture11.pdf

  # P. Minimum # of points
  nPtsToSel = 2

  # G. Percentage of inliers required
  #   Adjust this to change number of iterations to run. Lower = more iters
  pcInliers = 0.2

  # S. Adjust this to change number of iterations to run. Higher = more iters
  probSelectedAllInliersSet = 0.99

  # N = log_base(1-G^P)_of(1 - S)
  #   Use change of base to do the log
  maxRuns = int (math.ceil (math.log (1 - probSelectedAllInliersSet) /
    math.log (1 - pcInliers ** nPtsToSel)))
  #print ('Need %d iterations for RANSAC' % (maxRuns))


  ## Main loop

  # Threshold. Margin of distance from the guessed line
  dist_thresh = 0.02

  # Highest voted pair of points that form a line, i.e. pair of points whose
  #   connecting line has the most # of inlier points.
  max_n_inliers = 0
  # List of 2 integers, indexes pts array in parameter
  highest_voted_set_idx = []
  # Inlier indices
  pts_voted_for_highest = []
  # Distance of all points to the highest voted line
  #dists_to_voted_line = []


  for nRuns in range (0, maxRuns):

    # Choose a pair of points randomly
    idx = random.sample (range (0, len (pts)), 2)

    # 1 x 3 numpy arrays
    pt1 = pts [idx [0]]
    pt2 = pts [idx [1]]


    # Just take all points. You don't need to distinguish between chosen and
    #   unchosen. Since all samples have size 2, at the end, the max voted
    #   still has max votes. If you extract the unchosen ones, they won't have
    #   same index as the paramter pts_msg! Then you have to account for it and
    #   offset the indices you return!! That's way too much work than nec!
    #   So just use all the points.
    # X|
    # Take all the unchosen points.
    #   Ref: http://stackoverflow.com/questions/11791568/what-is-the-most-pythonic-way-to-exclude-elements-of-a-list-that-start-with-a-sp
    # Type: list of numpy arrays
    # np array doesn't work with list comprehension very well, have to convert
    #   to Python list using tolist().
    #pts_unchosen = [p for p in pts if p.tolist() not in [pt1.tolist(), pt2.tolist()]]

    # Compute perpendicular distance of each point to the chosen line
    # Note: Don't distinguish btw chosen and unchosen points. If you do, you'll
    #   have to adjust the indices accordingly, so the indices you return still
    #   correspond to the orig param array, not a new array with 2 fewer elts!
    #   Chosen pts will just have dist 0 to line and vote for themselves.
    #   That's fine, since all pairs will do it consistently across the board.
    dist = calc_pt_to_line_dist (pt1, pt2, pts)


    # Count inliers
    inliers_mask = dist < dist_thresh
    n_inliers = sum (i == True for i in inliers_mask)

    # Update highest voted line
    if n_inliers > max_n_inliers:
      max_n_inliers = n_inliers

      # Indices of the 2 chosen points for the line
      highest_voted_set_idx = idx

      # Indices of inliers
      pts_voted_for_highest = [i for i, x in enumerate(inliers_mask) if x == True]

      # Distance of all points to the highest voted line
      #dists_to_voted_line = dist


  #print ('ransac result: ')
  #print (highest_voted_set_idx)

  print ('RANSAC: Total %d points. %d inliers. Percent inlier: %f' \
    % (len (pts), len (pts_voted_for_highest), \
       float (len (pts_voted_for_highest)) / float (len (pts)))) 

  return (highest_voted_set_idx, pts_voted_for_highest)



# Project a set of points onto a given line.
# Parameters:
#   pt1, pt2: 1 x 3 Numpy array. (pt2 - pt1) is a vector parallel to a line.
#   pts: A set of points. List of 1 x 3 Numpy arrays
# Returns 1 x nPts Numpy array. 
#   Values mean distance of projection from pt1.
#   Positive value means projected onto (pt2 - pt1), i.e. in between points pt1
#     and pt2;
#   negative value means projected onto -(pt2 - pt1).
'''
# To test this fn in python prompt:
a = [[2,3,5],[4,12,34],[2,7,3]]
a = [np.asarray(i) for i in a]
l = np.asarray ([1,0,0])
np.dot (a, l)
'''
def project_pts_onto_line (pt1, pt2, pts):

  # Projection of a point onto a vector is equal to the dot product of the point
  #   and the unit vector of the vector.
  #   Note the point and the vector must have the same starting point (see pic
  #   in link), the starting point of the vector. So if the point is in terms
  #   of origin, it needs to be in terms of the starting point (pt1) of the
  #   vector.
  #   
  #   Ref: http://en.wikipedia.org/wiki/Vector_projection

  # Calc unit vector parallel to line
  line = (pt2 - pt1)
  line_norm = np.linalg.norm (line)
  line_unit = line / line_norm

  # 1 x nPts Numpy array. Distance of projection from pt1.
  # Value is positive if point projects onto a point in between pt2 - pt1, i.e.
  #   in positive direction of the vector pt2 - pt1. Value is negative if point
  #   projects onto a point on negative direction of (pt2 - pt1).
  # Subtract pts by pt1, to put points wrt starting point of vector. Otherwise,
  #   since points are wrt origin [0 0 0], the result projection height will
  #   be in terms of origin, i.e. projection onto (pt2 - [0 0 0]), which
  #   isn't what I want.
  projection = np.dot (pts - pt1, line_unit)

  return projection


