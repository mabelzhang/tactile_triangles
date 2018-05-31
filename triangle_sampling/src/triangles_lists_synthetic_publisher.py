#!/usr/bin/env python

# Mabel Zhang
# 3 Sep 2015
#
# Reads triangle lists saved from sample_pcl_calc_hist.py
#   previously, and publish them, so that sample_pcl_calc_hist.py
#   can subscribe and re-compute histograms.
# This is solely to speed up the experiment process so that histograms
#   can be recomputed using different parameters, on the same set of sampled
#   triangles.


# ROS
import rospy
from std_msgs.msg import Int32, Float32

# Python
import csv
import os

# My packages
from triangle_sampling_msgs.msg import TriangleParams
from tactile_collect import tactile_config
from triangle_sampling.config_hist_params import TriangleHistogramParams as \
  HistP


# To test on bare Python shell, opened in this directory:
'''
import triangles_lists_synthetic_publisher
# To reload, use:
#reload( triangles_lists_synthetic_publisher)
node = triangles_lists_synthetic_publisher.TrianglesListsSyntheticPublisher()
node.read_config ()
node.PR   # Check this is expected, no endline chars

'''
class TrianglesListsSyntheticPublisher:

  # Parameters:
  #   sampling_subpath: Name of subdirectory to access. String returned from
  #     config_paths.py get_sampling_subpath(), when
  #     passed in nSamples and nSamplesRatio.
  def __init__ (self, sampling_subpath, nSamples, nSamplesRatio):

    # Triangles path
    # Synthetic 3D model data, saved from running sample_pcl.cpp and
    #   sample_pcl_calc_hist.py
    self.tri_subpath = 'triangle_sampling/csv_tri_lists/'
    self.tri_path = tactile_config.config_paths ('custom',
      os.path.join (self.tri_subpath, sampling_subpath))

    self.tri_conf_name = os.path.join (self.tri_path, 'tri_conf.csv')

    self.tri_pub = rospy.Publisher ('/sample_pcl/triangle_params',
      TriangleParams, queue_size=5)

    # To determine what folder the data will be saved in, by an external node
    #   (sample_pcl_calc_hist.py) that subscribes to this node.
    self.nSamples_pub = rospy.Publisher ('/sample_pcl/nSamples',
      Int32, queue_size=1);
    self.nSamplesRatio_pub = rospy.Publisher ('/sample_pcl/nSamplesRatio',
      Float32, queue_size=1);
    self.nSamples_msg = Int32 ()
    self.nSamples_msg.data = nSamples
    self.nSamplesRatio_msg = Float32 ()
    self.nSamplesRatio_msg.data = nSamplesRatio

    self.tri_msg = None
    self.obj_idx = -1

    self.PR = []

    self.L0 = HistP.L0
    self.L1 = HistP.L1
    self.L2 = HistP.L2
    self.A0 = HistP.A0
    self.A1 = HistP.A1
    self.A2 = HistP.A2


    # Book-keeping

    self.configured = False


  def read_config (self):

    #####
    # Load triangle config data, in tri_conf.csv, saved by
    #   sample_pcl_calc_hist.py
    #####

    # There are 3 lines in file, e.g.:
    #   l0
    #   l1
    #   a0
    # They tell you what the 3 lines stored in the triangle .csv files are.
    with open (self.tri_conf_name, 'rb') as tri_conf_file:

      for line in tri_conf_file:
        # e.g. 'l0'
        #line = tri_conf_file.readline ()

        # Strip the white spaces (mainly the \r\n at the end)
        self.PR.append (line.rstrip ())

    self.configured = True


  # Read one csv_tri_lists file
  # Parameters:
  #   csv_base: Base name to CSV file to read
  #   csv_subdir: Immediate directory that csv_base file is in. e.g. object
  #     category folder.
  #   full_path: If True, then csv_base is the full path, instead of just
  #     basename.
  # Returns number of triangles in this file
  def read_triangles (self, csv_base, csv_subdir='', full_path=False):

    if not self.configured:
      self.read_config ()


    self.tri_msg = TriangleParams ()

    if not full_path:
      tri_name = os.path.join (self.tri_path, csv_subdir, csv_base)
      if not os.path.exists (os.path.dirname (tri_name)):
        os.makedirs (os.path.dirname (tri_name))

    else:
      tri_name = csv_base

    # sample_pcl_calc_hist.py wants basename and category name
    #   in the path
    self.tri_msg.obj_name = tri_name

    print ('Reading triangles from %s' % tri_name)
    tri_file = open (tri_name, 'rb')
    tri_reader = csv.reader (tri_file)

    row_num = 0

    nTriangles = 0

    for row in tri_reader:

      # Convert strings to floats
      row_float = [float (row [i]) for i in range (0, len (row))]

      # Each row has the values of all triangles, for ONE triangle parameter
      # See which triangle parameter the data in this row is for
      if self.PR [row_num] == self.L0:
        self.tri_msg.l0.extend (row_float)
      elif self.PR [row_num] == self.L1:
        self.tri_msg.l1.extend (row_float)
      elif self.PR [row_num] == self.L2:
        self.tri_msg.l2.extend (row_float)
      elif self.PR [row_num] == self.A0:
        self.tri_msg.a0.extend (row_float)
      elif self.PR [row_num] == self.A1:
        self.tri_msg.a1.extend (row_float)
      elif self.PR [row_num] == self.A2:
        self.tri_msg.a2.extend (row_float)

      row_num += 1

      # Populate return value. Each row has same number of triangles, so
      #   doesn't matter if overwrite btw rows.
      nTriangles = len (row)


    # Upon reading successfully, increment counter
    self.obj_idx += 1
    # Set object sequence number on message
    self.tri_msg.obj_seq = self.obj_idx

    return nTriangles


  def get_need_to_publish (self):

    return (self.tri_msg is not None)


  def publish (self):

    if not self.tri_msg:
      return

    self.tri_pub.publish (self.tri_msg)

    self.nSamples_pub.publish (self.nSamples_msg)
    self.nSamplesRatio_pub.publish (self.nSamplesRatio_msg)

    #print ('Published triangles for object %d' % self.tri_msg.obj_seq)


if __name__ == '__main__':
  main ()

