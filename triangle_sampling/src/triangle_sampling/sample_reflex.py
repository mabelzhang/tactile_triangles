#!/usr/bin/env python

# Mabel Zhang
# 28 Aug 2015
#
# Refactored from tactile_collect triangles_collect.py
# Written by copying from sample_pcl.cpp
#
# Called by tactile_collect triangles_collect.py record_tri().
#


# Python
import itertools  # For exhaustive sampling

# Numpy
import numpy as np
# For n choose k
from scipy.misc import comb


# Parameters:
#   pts: geometry_msgs/Point[]
#   nTriParams: Number of points (3) to sample, to make up a shape (triangle)
# Returns:
#   triangles: geometry_msgs/Point[]. XYZ positions of points selected to
#     form triangles. Array size is (3 * number of triangles sampled).
#     Linear array. Index with i * 3 + 0, i * 3 + 1, i * 3 + 2 to get 3 pts
#     of a triangle.
#   l0, l1, l2, a0, a1, a2: 3 side lengths and 3 angles (in radians).
#     Lists sizes are number of triangles sampled.
#   vis_text: The first two sampled side lengths, and the angle btw them.
#     Not sorted in any order, other than they're randomly picked first.
#
# To test this function in bare Python shell:
'''
from triangle_sampling import sample_reflex
from geometry_msgs.msg import Point
# This is 5 points. Check that script returns 5 sampled triangles, as defined
#   by hand below, nSamples=5 if input is 5 points.
pts = [Point(1,0,0), Point(0,1,0), Point(0,0,1), Point(1,1,0), Point(1,0,1)]
(triangles, l0, l1, l2, a0, a1, a2, vis_text) = sample_reflex.sample_tris (pts, 3)

# If need to reload, use:
reload (sample_reflex)

'''
def sample_tris (pts, nTriParams, exhaustive=True):

  # Number of points to pick from
  nPts = len (pts)

  # geometry_msgs/Point[]. 3 elements make a triangle.
  #   Sampled triangles. Size nSamples * 3.
  #   Linear index. When want to get ith triangle, use this to get the 3 pts::
  #     i * 3 + 0, i * 3 + 1, i * 3 + 2
  triangles = []

  # Triangle parameters. Size nSamples. List of floats
  l0 = []
  l1 = []
  l2 = []
  a0 = []
  a1 = []
  a2 = []

  # First two sampled sides, second sampled angle, for displaying text in RViz.
  # List of list of 3 floats. Size nSamples, inner lists size 3.
  vis_text = []


  #####
  # Sample triangles from contact points in this iteration.
  #   Translated from triangle_sampling sample_pcl.cpp
  #####

  nChoose3 = int (round (comb (nPts, nTriParams)))

  if not exhaustive:
    # Sample a subset of (nPts choose 3) possible combinations of 3 points
    #   4C3 = 4
    #   5C3 = 10
    #   6C3 = 20
    #   7C3 = 35
    #   8C3 = 56
    # You'd probably never see 8, probably never even see 7.

    nSamples = 1
    if nPts == 3:
      nSamples = 1
    elif nPts == 4:
      nSamples = 2
    elif nPts == 5:
      nSamples = 5
    elif nPts == 6:
      nSamples = 6
    elif nPts > 6:
      nSamples = 8

  # If exhaustive, sample all possible sets
  else:
    nSamples = nChoose3
    # TEMPORARY, to train 3 cm cube, `.` it gets too many triangles, pkl probs
    #   file never finishes saving!! The 0.4 ratio lets it finish saving!
    #nSamples = int (nChoose3 * 0.4)

    # Test on bare python shell:
    #   itertools.combinations ([[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]], 3)
    sample_indices = itertools.combinations (range (0, nPts), nTriParams)

  print ('Got %d points. Will try to sample %d triangles' % (nPts, nSamples))

  # List of unordered sets of three points
  #sampled_tris_idx = []
  # sets are faster than lists for the "in" operation
  sampled_tris_idx = set ()

  # List of unordered sets that have been attempted in sampling
  #pass_attempted_sets = []
  # sets are faster than lists for the "in" operation
  pass_attempted_sets = set ()


  # Using while-loop instead of for-loop, `.` index may be changed inside the
  #   loop, and Python doesn't adjust the loop counter live.
  # Note that compared to sample_pcl.cpp's for-loop, since this is a
  #   while-loop, the counter is not incremented automatically at the
  #   beginning of each iteration. So there is no need to do sample_count-=1
  #   when samples are invalid and we need to "continue" to next iter to
  #   re-sample.
  sample_count = 0
  while sample_count < nSamples:

    # Sanity check
    # n choose 3 is the max possible number of combinations you can sample, out
    #   of the given set of points.
    # If exceed this bound, no more hope. Just return what we have.
    # Don't use actual n choose 3, takes forever. Take 80%.
    # Test for >=, not >, because pass_attempted_sets size is 
    #   incremented in previous iteration. So placing this check at
    #   beginning of loop would check whether previous samples have
    #   reached nChoose3. You actually never go >, so checking for >
    #   would do nothing! > is just a safety measure in case you 
    #   skipped the == nChoose3 for some reason. So you're really
    #   checking for ==.
    if len (pass_attempted_sets) >= nChoose3 * .8:
      print ('Exhausted %d out of %d possible combinations, still did not sample enough points. Giving up and returning what we have (%d triangles)...' % \
        (len (pass_attempted_sets), nChoose3, len (sampled_tris_idx)))
      break


    #print ('Sampling %dth triangle... ' % (sample_count + 1)),

    #####
    # Sample a set of 3 indices. These index "poses" param
    #####

    if not exhaustive:
      # Uniquely sample a list of 3 integer indices.
      #   Set replace=False, for unique samples.
      sample_idx = np.random.choice (len (pts), size=nTriParams, replace=False)
      print ('Got indices %d %d %d' % (sample_idx[0], sample_idx[1],
        sample_idx[2]))

    else:
      # Pick up the next one from the combination list of n choose 3 items
      sample_idx = sample_indices.next ()

    #sample_idx_set = set (sample_idx)
    # set needs hashable type, so sets and lists don't work. Need tuples.
    #   Always sort the 3 indices, then the tuples will match future ones.
    #   Don't need to store sets of 3 indices as unordered set!
    sample_idx_set = tuple (np.sort (sample_idx))

    # Add this sample to the list of attempted samples
    if sample_idx_set not in pass_attempted_sets:
      #pass_attempted_sets.append (sample_idx_set)
      pass_attempted_sets.add (sample_idx_set)
    else:
      print ('This set has been attempted before and invalid. Re-sampling...')
      continue

    # If this set has been sampled before, resample
    #   Ref: https://docs.python.org/2/tutorial/datastructures.html#sets
    if sample_idx_set in sampled_tris_idx:
      print ('This set of 3 points already been sampled before. Re-sampling...')
      continue


    #####
    # Calculate the 6 triangle parameters
    #####

    # Add the last 3 points, to triangles list     
    pt0 = np.array ([pts [sample_idx [0]].x, pts [sample_idx [0]].y,
      pts [sample_idx [0]].z])
    pt1 = np.array ([pts [sample_idx [1]].x, pts [sample_idx [1]].y,
      pts [sample_idx [1]].z])
    pt2 = np.array ([pts [sample_idx [2]].x, pts [sample_idx [2]].y,
      pts [sample_idx [2]].z])
   
   
    # Pick the first two sides
    # They must subtract by same point, in order for dot product to
    #   compute correctly! Else your sign may be wrong. Subtracting by
    #   same point puts the two vectors at same origin, w hich is what
    #   dot product requires, in order to give you correct theta btw
    #   the two vectors!
   
    s10 = pt0 - pt1
    s12 = pt2 - pt1
   
    s01 = pt1 - pt0
    s02 = pt2 - pt0
   
    s20 = pt0 - pt2
    s21 = pt1 - pt2
   
   
    # Two side lengths
    len_s10 = np.linalg.norm (s10)
    len_s12 = np.linalg.norm (s12)
   
    len_s01 = np.linalg.norm (s01)
    len_s02 = np.linalg.norm (s02)
   
    len_s20 = np.linalg.norm (s20)
    len_s21 = np.linalg.norm (s21)
   
   
    angle1 = 0
    # Angle btw two vectors.
    #   Dot product is dot(a,b) = |a||b| cos (theta).
    #       dot(a,b) / (|a||b|) = cos(theta)
    #                     theta = acos (dot(a,b) / (|a||b|))
    if len_s10 * len_s12 != 0:
      angle1 = np.arccos (np.dot (s10, s12) / (len_s10 * len_s12))

      #print (str (np.dot (s10, s12)))
      #print (str (len_s10 * len_s12))
      #print (str (np.arccos (np.dot (s10, s12) / (len_s10 * len_s12))))

      if np.isnan (angle1):
        # On real robot, can't just set to 0 and ignore.
        print ('angle1 is nan. Resampling...')
        continue

      # Cover the straight-line case. This happens when the sampled sensors
      #   are on one segment of one finger. Then resulting triangle is a line.
      elif np.fabs (angle1 - np.pi) < 1e-5:
        print ('angle1 is pi. This is a straight-line triangle. Resampling...')
        continue
   
    # Exception case: length is 0. Re-select the triangle.
    else:
      print ('Triangle had 0-length side 1. Resampling...');
      continue


    angle0 = 0
    if len_s01 * len_s02 != 0:
      angle0 = np.arccos (np.dot (s01, s02) / (len_s01 * len_s02))

      #print (str (np.dot (s01, s01)))
      #print (str (len_s01 * len_s01))
      #print (str (np.arccos (np.dot (s01, s02) / (len_s01 * len_s02))))

      if np.isnan (angle0):
        print ('angle0 is nan. Resampling...')
        continue
      elif np.fabs (angle0 - np.pi) < 1e-5:
        print ('angle0 is pi. This is a straight-line triangle. Resampling...')
        continue
   
    # Exception case: length is 0. Re-select the triangle.
    else:
      print ('Triangle had 0-length side 0. Resampling...');
      continue


    angle2 = 0
    if len_s20 * len_s21 != 0:
      angle2 = np.arccos (np.dot (s20, s21) / (len_s20 * len_s21))

      #print (str (np.dot (s20, s21)))
      #print (str (len_s20 * len_s21))
      #print (str (np.arccos (np.dot (s20, s21) / (len_s20 * len_s21))))

      if np.isnan (angle2):
        print ('angle2 is nan. Resampling...')
        continue
      elif np.fabs (angle2 - np.pi) < 1e-5:
        print ('angle2 is pi. This is a straight-line triangle. Resampling...')
        continue
   
    # Exception case: length is 0. Re-select the triangle.
    else:
      print ('Triangle had 0-length side 2. Resampling...');
      continue


    try:
      assert (np.fabs (angle0 + angle1 + angle2 - np.pi) < 1e-3)
    except AssertionError, e:
      print ('Sum (%f) of 3 angles of sampled triangle did not add up to 180.' +
        ' Resampling...' % (angle0 + angle1 + angle2))
      continue

    # Store the first two sampled sides, and the angle btw them, for visualize
    vis_text.append ([len_s10, len_s12, angle1])
  

    #####
    # Store in sorted order, so that parameters are consistent across
    #   different triangles, i.e. l0 is the longest side, l1 is medium
    #   side, l2 is shortest side; a0 is largest angle, a1 is medium
    #   angle, a2 is smallest angle.
    # I don't know what the most efficient way to do this is. I want
    #   code to be short, so I did vector max min instead of if-stmts.
    #####

    # Sort from small to big (`.` numpy default), store from big to small

    # This is in order of which point was picked first
    tmp_lens = np.array ([len_s02, len_s10, len_s12])
    tmp_lens = np.sort (tmp_lens)
   
    l0.append (tmp_lens [2])
    l1.append (tmp_lens [1])
    l2.append (tmp_lens [0])
   
    # This is in order of which point was picked first
    tmp_angs = np.array ([angle0, angle1, angle2])
    tmp_angs = np.sort (tmp_angs)

    a0.append (tmp_angs [2])
    a1.append (tmp_angs [1])
    a2.append (tmp_angs [0])


    #####
    # At this point, there should be no more errors causing a need to
    #   resample. This sample is final. Store to vectors.
    #####

    # Add to set of sampled points, for checking duplicates
    #sampled_tris_idx.append (sample_idx_set)
    # sets are faster than lists for the "in" operation
    sampled_tris_idx.add (sample_idx_set)

    # Collect the samples into ret val
    for i in range (0, len (sample_idx)):
      triangles.append (pts [sample_idx [i]])


    # Increment loop counter. This is "tri" in sample_pcl.cpp
    sample_count += 1

  # end while sample_count < nSamples


  # This prints a LOT of lines if you get a lot of contacts. Comment out if
  #   you want to see other debug info.
  #print ('l0, l1, a0:')
  #for i in range (0, len (l0)):
  #  print ('%.2f %.2f %.2f' % (l0[i], l1[i], a0[i]))

  return (triangles, l0, l1, l2, a0, a1, a2, vis_text)

