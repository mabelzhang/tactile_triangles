#!/usr/bin/env python

# Mabel Zhang
# 29 Jan 2016
#
# Convert csv_tri_lists file format to csv_tri file format, to easily use
#   scripts written for csv_tri files on PCL-trained triangles.
#
# csv_tri_lists are for PCL-trained point cloud data, generated by
#   sample_pcl.cpp. 6 lines total in csv, each line is one triangle
#   parameter, contains nTriangles numbers for the entire object.
# csv_tri files are for real-robot and Gazebo-trained data, generated by
#   triangles_collect.py. nTriangles lines total in csv, each line is one
#   triangle, 6 numbers.
#

# ROS
import rospkg

# Python
import os
import csv
import argparse

# My pkgs
from tactile_collect import tactile_config
from triangle_sampling.parse_models_list import read_meta_file, \
  get_meta_cat_name
from triangle_sampling.config_paths import get_sampling_subpath, \
  get_robot_tri_path
from triangle_sampling.config_hist_params import TriangleHistogramParams as \
  HistP

# Local
from triangles_lists_synthetic_publisher import TrianglesListsSyntheticPublisher


def main ():

  #####
  # Parse cmd line args
  #####

  arg_parser = argparse.ArgumentParser ()

  arg_parser.add_argument ('nSamples', type=str,
    help='Used to create directory name to read from. ' + \
      'For point cloud, nSamples used when triangles were sampled.')
  arg_parser.add_argument ('nSamplesRatio', type=str,
    help='Used to create directory name to read from. ' + \
      'nSamplesRatio used when triangles were sampled.')

  arg_parser.add_argument ('--meta', type=str, default='models_simple.txt',
    help='String. Base name of meta list file in triangle_sampling/config directory')

  arg_parser.add_argument ('--out_tri_subdir', type=str, default='',
    help='String. Subdirectory name of output directory under csv_tri/. e.g. nSamples10_ratio095')

  args = arg_parser.parse_args ()


  # Construct input subdir name
  nSamples = int (args.nSamples)
  nSamplesRatio = float (args.nSamplesRatio)
  sampling_subpath = get_sampling_subpath (nSamples, nSamplesRatio)

  # Instantiate reader for csv_tri_lists file format
  readerNode = TrianglesListsSyntheticPublisher (sampling_subpath, nSamples,
    nSamplesRatio)
  readerNode.read_config ()


  # Read meta file, to get input file names
  rospack = rospkg.RosPack ()
  meta_list_path = os.path.join (rospack.get_path ('triangle_sampling'),
    'config')
  meta_list_name = os.path.join (meta_list_path, args.meta)
  meta_list = read_meta_file (meta_list_name)


  # Construct output dir name
  tri_path = get_robot_tri_path ('')  # triangle_sampling/csv_tri
  if args.out_tri_subdir:
    tri_path = os.path.join (tri_path, args.out_tri_subdir)
  if not os.path.exists (tri_path):
    os.makedirs (tri_path)


  for line in meta_list:

    line = line.strip ()

    # Construct input csv_tri_lists file base name
    base = os.path.basename (line)
    tri_lists_name = os.path.splitext (base) [0] + '.csv'

    # Get category name
    cat_name = get_meta_cat_name (line)

    # Read triangles, which are stored in readerNode.tri_msg, a
    #   triangle_sampling_msgs/TriParams.msg type
    readerNode.read_triangles (tri_lists_name, csv_subdir=cat_name)

    print ('Input csv_tri_lists triangle lists file: %s' % ( \
      os.path.join (readerNode.tri_path, tri_lists_name)))


    # Construct output csv_tri file name
    tri_name = os.path.join (tri_path, cat_name, tri_lists_name)

    if not os.path.exists (os.path.dirname (tri_name)):
      os.makedirs (os.path.dirname (tri_name))

    # Output to csv_tri file
    with open (tri_name, 'wb') as tri_file:

      tri_writer = csv.DictWriter (tri_file, fieldnames=HistP.TRI_PARAMS)
      tri_writer.writeheader ()

      # Loop through each triangle, grab all 6 params, write 6 params of
      #   a triangle to 1 line in csv.
      for i in range (0, len (readerNode.tri_msg.l0)):

        row = dict ()
        row [HistP.L0] = readerNode.tri_msg.l0 [i]
        row [HistP.L1] = readerNode.tri_msg.l1 [i]
        row [HistP.L2] = readerNode.tri_msg.l2 [i]
        row [HistP.A0] = readerNode.tri_msg.a0 [i]
        row [HistP.A1] = readerNode.tri_msg.a1 [i]
        row [HistP.A2] = readerNode.tri_msg.a2 [i]

        tri_writer.writerow (row)

    print ('Output csv_tri triangles file outputted to %s' % tri_name)
    print ('')


if __name__ == '__main__':
  main ()

