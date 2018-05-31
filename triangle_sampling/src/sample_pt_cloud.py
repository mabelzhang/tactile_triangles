#!/usr/bin/env python

# Mabel Zhang
# 27 Jul 2015
#
# Note: Not in use anymore, because this simply samples randomly from entire
#   object's point cloud. Not realistic enough.
#   Replacement is sample_pcl.cpp, which uses PCL to partition point cloud
#   into voxels the size of the hand, and samples by voxel. That is more
#   realistic.
#   Could not do that here, because PCL Python interface is lame and can't do
#   anything.
#
#
# Randomly samples sets of 3 points from a point cloud.
#
# To run:
#   $ rosrun tactile_map keyboard_interface.py
#   $ python sample_pt_cloud.py
# To see visualization:
#   $ rosrun rviz rviz
#   Select point cloud Size (m) = 0.001
#   Enable Marker.
#     For per-iteration, only enable namespaces without *_cumu suffix.
#     For cumulative, enable *_cumu suffix.
#

# ROS
import rospy
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point, Point32
from visualization_msgs.msg import Marker
from std_msgs.msg import String

# Python
from operator import mul    # or mul=lambda x,y:x*y
from fractions import Fraction
import random
import os

# Numpy
import numpy as np

# Third-party
# Requires installation of pywavefront. This is used to read OBJ files. You can
#   install this, or write your own OBJ reader. It's really easy. I would have
#   done that in shorter time than the time I took to change pywavefront to
#   work...
import pywavefront

# Local
from tactile_map.create_marker import create_marker
from plot_hist_dd import create_subplot, plot_hist_dd, show_plot


class SampleObj:

  def __init__ (self):

    rospy.Subscriber ('/keyboard_interface/key', String, self.keyCB)

    self.doPause = False
    self.doSkip = False
    self.doTerminate = False

  def keyCB (self, msg):

    if msg.data == ' ':
      print 'Got user signal to toggle pause / resume...'
      self.doPause = not self.doPause
    elif msg.data.lower () == 's':
      print 'Got user signal to skip rest of this object model, skipping...'
      self.doSkip = True
    elif msg.data.lower () == 'q':
      print 'Got user signal to terminate program, terminating...'
      self.doTerminate = True


# From
#   http://stackoverflow.com/questions/3025162/statistics-combinations-in-python
def nCk(n,k): 
  return int( reduce(mul, (Fraction(n-i, i+1) for i in range(k)), 1) )


def main ():

  #####
  # User adjust params
  #####

  # True means 10 Hz pause btw each visualization. Set to False if you just want
  #   to see the histogram plots quickly.
  doRViz = False


  #####
  # Init ROS
  #####

  rospy.init_node ('sample_obj', anonymous=True)

  thisNode = SampleObj ()

  # Prompt to display at keyboard_interface.py prompt
  prompt_pub = rospy.Publisher ('keyboard_interface/prompt', String)
  prompt_msg = String ()
  prompt_msg.data = 'Press space to pause, '

  vis_pub = rospy.Publisher ('/visualization_marker', Marker)

  cloud_pub = rospy.Publisher ('/sample_obj/cloud', PointCloud)

  wait_rate = rospy.Rate (10)

  print ('sample_obj node initialized...')


  # All model files
  model_path = '/Users/master/courses/15summer/MenglongModels/'
  model_name = ['axe1_c.obj', 'bigbox_c.obj', 'bottle_c.obj', 'broom_c.obj',
    'brush_c.obj', 'flowerspray_c.obj', 'gastank_c.obj', 'handlebottle_c.obj',
    'heavyranch1_c.obj', 'pan_c.obj', 'pipe_c.obj', 'shovel_c.obj',
    'spadefork_c.obj', 'spraybottle_c.obj', 'watercan_c.obj',
    'wreckbar1_c.obj']
  #model_name = ['broom_c.obj', 'watercan_c.obj']

  # For plotting histograms
  ylbls = ['len1', 'len2', 'alpha']

  figs = [None] * 3
  axes = [None] * 3
  # For each triangle parameter, make a 4 x 4 subplot grid
  for i in range (0, 3):
    figs[i], axes[i] = create_subplot ([4, 4])
 


  # Loop through each obj file
  for obj_idx in range (0, len (model_name)):

    if rospy.is_shutdown ():
      break

 
    #####
    # Load OBJ file
    #####
 
    # See ObjParser type here, and all the fields it contains (vertices,
    #   normals, tex_coords), on my MacBook Air:
    #   /usr/local/lib/python2.7/site-packages/pywavefront/__init__.py
    # pywavefront module for Python: https://pypi.python.org/pypi/PyWavefront
    model = pywavefront.Wavefront (os.path.join (model_path,
      model_name [obj_idx]))
 
    nPts = len (model.parser.vertices)
    print ('%d vertices' % nPts)
 
    # Convert to numpy array so I can index using a list of indices below
    # Shrink model, because orig points are in hundreds scale, too big for
    #   meters. I'm guessing they are in milimeters?
    verts = 0.001 * np.asarray (model.parser.vertices)
 
 
    #####
    # Downsample cloud to 1 point every 8 mm (0.008 m)
    #####
 
    # TODO now that I don't use pcl, i'll have to do this myself. Sort point
    #   by euclidean somehow, and then skip every so and so points. I don't know.
    #   Or just bin them into a grid, and only keep 1 point per bin. That might
    #   be easier.
 
    # Naive approach of downsampling (because there's toooo many points, I need
    #   to quickly test a downsampled model, before I have time to implement a
    #   real downsampling).
    # Just randomly pick 10% of the points
    down_ratio = 0.1
    down_idx = random.sample (range (0, len (model.parser.vertices)),
      int (np.floor (nPts * down_ratio)))
 
    verts_down = verts [down_idx]
    nPts_down = len (verts_down)
 
    print ('%d vertices after RANDOM downsampling' % nPts_down)
 
 
    #####
    # Sample a set of 3 points, many sets.
    #####
 
    nTri_exhaust = nCk (nPts, 3)
    print ('%d triangles needed to be exhaustive (%d choose 3)' % (nTri_exhaust,
      nPts))
 
    nTri_down_exhaust = nCk (nPts_down, 3)
    print ('%d triangles needed for downsampled model' % (nTri_down_exhaust))
 
    # Pick a realistic number. Note since sampling is not enforced to be unique
    #   (for now), this has to be bigger than what you think, to account for
    #   triangles that were sampled more than once.
    nTri_down = 200 #len (verts_down)
 
    # List of 3-point numpy arrays. Numpy arrays are
    #   array([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
    samples = []
 
    # List of triangle parameters
    l1 = []
    l2 = []
    alpha = []
 
 
    for i in range (0, nTri_down):
 
      # Do this in a loop, instead of just sample 3 * nTri_down points. `.`
      #   sample() returns unique numbers.
      sample_idx = random.sample (range (0, len (verts_down)), 3)
 
      # Append a list, to group these 3 pts together
      samples.append (verts_down [sample_idx])
 
 
      #####
      # Compute triangle connected by the 3 sampled pts, represented by 3 params
      # TODO: decide what params to use, any combination (except for 3 angles) of
      #   3 angles and 3 sides will work:
      #   Ref: https://www.mathsisfun.com/algebra/trig-solving-triangles.html
      #   SAS looks easiest to compute. But... what's the best for a distinctive
      #     3D object descriptor though? I'd think 2 angles + 1 side... Is it?
      #     but that doesn't tell you as much about the length... length tells you
      #     about size obj. Angles are kind of arbitrary and don't tell you much
      #     about the object, since the 3 points are randomly sampled! You may
      #     very well end up with two triangles with 2 same angles, on very diff
      #     objs, because the location of the 3 sample points!
      #   This choice might also have to be empirical, in that we do all 5 cases,
      #     compute the distance btw histograms of all objs, and pick the 
      #     parameterization that gives the greatest distance!
      #####
  
      # I will use SAS for now. It tells 2 side lengths, which can be informative
      #   about object size. 2 anglees + 1 side seems more arbitrary and not as
      #   distinctive, since points are randomly chosen, so angles are random too
      #   and can be same btw differnet objs.
 
      # Pick the first two sides
      # They must subtract by same point, in order for dot product to compute
      #   correctly! Else your sign may be wrong. Subtracting by same point
      #   puts the two vectors at same origin, w hich is what dot product
      #   requires, in order to give you correct theta btw the two vectors!
      s1 = verts_down [sample_idx] [0] - verts_down [sample_idx] [1]
      s2 = verts_down [sample_idx] [2] - verts_down [sample_idx] [1]
 
      # Two side lengths
      len_s1 = np.linalg.norm (s1)
      len_s2 = np.linalg.norm (s2)
 
      # Angle btw two vectors.
      #   Dot product is dot(a,b) = |a||b| cos (theta).
      #       dot(a,b) / (|a||b|) = cos(theta)
      #                     theta = acos (dot(a,b) / (|a||b|))
      angle = np.arccos (np.dot (s1, s2) / (len_s1 * len_s2))
      # Exception case: length is 0. In this case, should really re-select the
      #   triangle. But for time being just let it go. TODO fix this
      if np.isnan (angle) or np.isinf (angle):
        print ('angle was NaN or Inf. Replacing with 0')
        angle = 0

      if np.isnan (len_s1) or np.isinf (len_s1):
        print ('len_s1 was NaN or Inf. Replacing with 0')
        len_s1 = 0
      if np.isnan (len_s2) or np.isinf (len_s2):
        print ('len_s2 was NaN or Inf. Replacing with 0')
        len_s2 = 0
 
      # Add to lists for histogram descriptor calculation
      l1.append (len_s1)
      l2.append (len_s2)
      alpha.append (angle)
 
 
    #####
    # Visualize sampled points and the triangle connecting them, in sequence
    #   in RViz
    #####

    cloud = PointCloud ()
    cloud.header.stamp = rospy.Time.now ()
    cloud.header.frame_id = '/'
 
    # Create a PointCloud type, from all points in downsampled model
    for i in range (0, len (verts_down)):
 
      # PointCloud.points is Point32[] type
      cloud.points.append (Point32 (verts_down [i][0], verts_down [i][1],
        verts_down [i][2]))
 
 
    #####
    # Create histogram from the 3 params, for this entire obj, i.e. for all sets
    #   of 3 points sampled.
    # This is the final descriptor for this object.
    #####
 
    # Convert list of lists to np.array, this gives a 3 x n array. Transpose to
    #   get n x 3, which histogramdd() wants.
    tri_params = np.array ([l1, l2, alpha]).T 
    print (tri_params)
 
    # Now done in plot_hist_dd.py. Delete from here when that works.
    '''
    # Normalize the histogram, because different objects have different number of
    #   points. Then histogram will have higher numbers for big object model.
    #   Bias is not good.
    hist, edges = np.histogramdd (tri_params, bins=[10,10,18], normed=True)
 
    #print ('Shape of 3D histogram:')
    #print (np.shape (hist))
    #print (np.shape (hist[0]))
    #print (hist)
 
    #print ('Shape of edges:')
    #print (np.shape (edges))
    #print (edges)
 
 
    #####
    # Plot the histogram in matplotlib
    #####
 
    # Every parameter is 3 by something, other than obj_idx, which is a scalar,
    #   indicating which subplot to use.
    plot_hist_dd (hist, edges, figs, axes, obj_idx, ylbls)
    '''

    # Every parameter is 3 by something, other than tri_params which is n x 3,
    #   and obj_idx, which is a scalar, indicating which subplot to use.
    plot_hist_dd (tri_params, [10, 10, 18], figs, axes, obj_idx, ylbls)
 
 
    #####
    # Visualize things in RViz, using a ROS loop
    #####
 
    seq = 0
 
    while doRViz and not rospy.is_shutdown ():
 
      prompt_pub.publish (prompt_msg)
 
      if not thisNode.doPause:
 
        # Grab the next 3 sampled poseqnts
        p0 = Point ( \
          samples [seq] [0] [0], samples [seq] [0] [1], samples [seq] [0] [2])
        p1 = Point (\
          samples [seq] [1] [0], samples [seq] [1] [1], samples [seq] [1] [2])
        p2 = Point (\
          samples [seq] [2] [0], samples [seq] [2] [1], samples [seq] [2] [2])
       
        # Create a marker of 3 points
        marker_sample = Marker ()
        create_marker (Marker.POINTS, 'sample_pts', '/', 0,
          0, 0, 0, 1, 0, 0, 0.8, 0.01, 0.01, 0.01,
          marker_sample, 0)  # Use 0 duration for forever
        marker_sample.points.append (p0)
        marker_sample.points.append (p1)
        marker_sample.points.append (p2)
       
        # Make a copy of the marker of 3 pts, to publish in cumulative namespace.
        # This lets us see how well the sampling of triangle was - did it sample
        #   enough triangles to cover the entire object.
        marker_sample_cumu = Marker ()
        create_marker (Marker.POINTS, 'sample_pts_cumu', '/', seq,
          0, 0, 0, 1, 1, 0, 0.8, 0.01, 0.01, 0.01,
          marker_sample_cumu, 0)  # Use 0 duration for forever
        marker_sample_cumu.points = marker_sample.points [:]
       
       
        # Create a LINE_LIST Marker for the triangle
        # Simply connect the 3 points to visualize the triangle
        marker_tri = Marker ()
        create_marker (Marker.LINE_LIST, 'sample_tri', '/', 0,
          0, 0, 0, 1, 0, 0, 0.8, 0.001, 0, 0,
          marker_tri, 0)  # Use 0 duration for forever
        marker_tri.points.append (p0)
        marker_tri.points.append (p1)
        marker_tri.points.append (p1)
        marker_tri.points.append (p2)
        marker_tri.points.append (p2)
        marker_tri.points.append (p0)
       
        # Make a copy for cumulative namespace
        marker_tri_cumu = Marker ()
        create_marker (Marker.LINE_LIST, 'sample_tri_cumu', '/', seq,
          0, 0, 0, 1, 0.5, 0, 0.8, 0.001, 0, 0,
          marker_tri_cumu, 0)  # Use 0 duration for forever
        marker_tri_cumu.points = marker_tri.points [:]
       
       
        # Create text labels for sides and angle, to visually see if I calculated
        #   correctly.
        # Ref: http://stackoverflow.com/questions/5309978/sprintf-like-functionality-in-python
       
        # Draw text at midpoint of side
        # NOTE: l1 is currently side btw pts [0] and [1]. Change if that changes.
        marker_l1 = Marker ()
        create_marker (Marker.TEXT_VIEW_FACING, 'text', '/', 0,
          (p0.x + p1.x) * 0.5, (p0.y + p1.y) * 0.5, (p0.z + p1.z) * 0.5,
          1, 0, 0, 0.8, 0, 0, 0.02,
          marker_l1, 0)
        marker_l1.text = '%.2f' % l1 [seq]
       
        # Draw text at midpoint of side
        # NOTE: l2 is currently side btw pts [1] and [2]. Change if that changes.
        marker_l2 = Marker ()
        create_marker (Marker.TEXT_VIEW_FACING, 'text', '/', 1,
          (p1.x + p2.x) * 0.5, (p1.y + p2.y) * 0.5, (p1.z + p2.z) * 0.5,
          1, 0, 0, 0.8, 0, 0, 0.02,
          marker_l2, 0)
        marker_l2.text = '%.2f' % l2 [seq]
       
        # NOTE: Angle currently is btw the sides [0][1] and [1][2], so plot angle at
        #   point [1]. If change definition of angle, need to change this too!
        marker_alpha = Marker ()
        create_marker (Marker.TEXT_VIEW_FACING, 'text', '/', 2,
          p1.x, p1.y, p1.z, 1, 0, 0, 0.8, 0, 0, 0.02,
          marker_alpha, 0)
        marker_alpha.text = '%.2f' % (alpha [seq] * 180.0 / np.pi)
       
       
        # Publish .obj model cloud
        cloud_pub.publish (cloud)
       
        # Publish sampled points and triangle.
        # To see current selected sample, only enabe namespaces without the _cumu
        #   suffix, i.e. enable sample_pts and sample_tri in RViz.
        vis_pub.publish (marker_sample)
        vis_pub.publish (marker_sample_cumu)
        vis_pub.publish (marker_tri)
        vis_pub.publish (marker_tri_cumu)
       
        # Text labels
        vis_pub.publish (marker_l1)
        vis_pub.publish (marker_l2)
        vis_pub.publish (marker_alpha)
       
       
        # Update book-keeping for next iter
        seq += 1
        if seq >= len (samples):
          #seq = 0
          break
 

      # Don't flip flag yet, flip it in outer loop
      if thisNode.doSkip:
        break
 
      if thisNode.doTerminate:
        break
 
      # ROS loop control
      try:
        wait_rate.sleep ()
      except rospy.exceptions.ROSInterruptException, err:
        break


    # Check termination again in outer loop

    if thisNode.doSkip:
      thisNode.doSkip = False
      continue

    if thisNode.doTerminate:
      break

    # ROS loop control
    try:
      wait_rate.sleep ()
    except rospy.exceptions.ROSInterruptException, err:
      break


  # Show matplotlib plot and save figure to file
  show_plot (figs, ylbls)


if __name__ == '__main__':

  main ()

