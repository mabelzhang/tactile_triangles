#!/usr/bin/env python

# Mabel Zhang
# 3 Sep 2015
#
# Reads models.txt for object paths, passes the base name to
# either
#   triangles_on_robot_to_hists.py, to calculate histograms for real-robot
#     (--real) or Gazebo (--gazebo) collected individual triangles,
# or
#   triangles_publisher.py, to run in parallel with
#     sample_pcl_calc_hist.py, to calculate histograms for
#     synthetic (--pcd) data, whose triangle lists were saved previously from
#     sample_pcl_calc_hist.py.
#
#     This option is a sped up way to run experiments with varied histogram
#     parameters, while sampling parameters (set before sampling in
#     sample_pcl.cpp) are unchanged.
#
# This script also saves stats of number of triangles to csv_*tri/_stats/,
#   e.g. csv_gz_tri/_stats/num_triangles_triParams_l0l1l2_nbins6_6_6.csv.
#
#
# These are two separate pipelines:
#
#   1. input PCD 3D object point cloud
#     -> sample_pcl.cpp -> sample_pcl_calc_hist.py
#       -> histogram CSV files, and (optionally) raw triangle list files
#
#   In pipeline above, if want to quickly change histogram parameters and
#     re-calculate histograms from the triangles, it requires running
#     sample_pcl.cpp and resampling all the objects, even if all the
#     triangles are unchanged. That is a waste of time, since now it takes
#     20 minutes to sample 192 objects, with nSamples=300, nSamplesRatio=0.95.
#
#   Having a reader to read those raw triangles files, and publishing to
#     sample_pcl_calc_hist.py would eliminate this inconvenience
#     and save a lot of time.
#
#   Hence this file, to read models.txt line-by-line, load the triangles for
#     the corresponding object on each line, and publish it just like
#     sample_pcl.cpp would.
#
#   2. input real robot tactile contacts
#     -> triangles_collect.py
#       -> raw individual triangles files
#          No histogram here!
#
#   In pipeline above, histograms need to be computed from the raw triangle
#     files, in order to run recognition on the histogram descriptors.
#
#   Hence this file, to read real.txt line-by-line, load the triangles
#     for the corresponding object for each line, and pass to
#     triangles_on_robot_to_hists.py to calculate and save histograms.
#
#   3. PCL and Gazebo data, MIXED_DATA mode
#      See 1 and 2 above.
#
#      You'd probably want to run sample_pcl_calc_hist.py, if
#        you didn't already save histograms from that file. It will compute
#        and save PCL histograms.
#
#

# ROS
import rospy
import rospkg

# Python
import os
import sys
import time
import csv
import argparse

# My packages
from util.ansi_colors import ansi_colors
from triangle_sampling_msgs.msg import TriangleParams
from tactile_collect import tactile_config
from triangle_sampling.config_paths import get_sampling_subpath, \
  get_triparams_nbins_subpath
from triangle_sampling.triangles_on_robot_to_hists import \
  TrianglesOnRobotToHists

# Local
from triangles_lists_synthetic_publisher import TrianglesListsSyntheticPublisher


def write_num_triangles (tri_path, sampling_subpath, n_tris):

  # Remove the '/' at the end
  if sampling_subpath.endswith ('/'):
    sampling_subpath = os.path.split (sampling_subpath) [0]

  # If have a suffix
  underscore = ''
  if sampling_subpath:
    underscore = '_'

  tri_name = os.path.join (tri_path, 'num_triangles' + underscore + \
    sampling_subpath + '.csv')

  with open (tri_name, 'wb') as tri_file:

    tri_writer = csv.writer (tri_file)

    # Just one row, triangles of each object, for all the objects
    # Number of columns == number of objects
    tri_writer.writerow (n_tris)

  print ('%sNumber of triangles for each object written to %s%s' % ( \
    ansi_colors.OKCYAN, tri_name, ansi_colors.ENDC))


def main ():

  rospy.init_node ('triangles_reader', anonymous=True)

  rospack = rospkg.RosPack ()
  model_list_path = os.path.join (rospack.get_path ('triangle_sampling'),
    'config')

  PCD_CLOUD_DATA = 0
  REAL_ROBOT_DATA = 1
  GAZEBO_HAND_DATA = 2
  MIXED_DATA = 3
  data_mode = -1

  wait_rate = rospy.Rate (10)


  #####
  # Parse cmd line args
  #####

  arg_parser = argparse.ArgumentParser ()

  arg_parser.add_argument ('histSubdirParam1', type=str,
    help='Used to create directory name to read from.\n' + \
      'For point cloud, nSamples used when triangles were sampled.\n' + \
      'For real robot data, specify the sampling density you want to classify real objects with, e.g. 10, will be used to load histogram bin configs.\n' + \
      'For Gazebo, triangle params desired, with no spaces, e.g. l0,l1,a0\n' + \
      'For mixed, enter 2 point cloud params first, then 2 Gazebo params, 4 total')
  arg_parser.add_argument ('histSubdirParam2', type=str, nargs='+',
    help='Used to create directory name to read from.\n' + \
      'For point cloud, nSamplesRatio used when triangles were sampled.\n' + \
      'For real robot data, specify the sampling density you want to classify real objects with, e.g. 0.95, will be used to load histogram bin configs.\n' + \
      'For Gazebo, number of bins in 3D histogram, with no spaces, e.g. 10,10,10.\n' + \
      'For mixed, enter 2 point cloud params first, then 2 Gazebo params, 4 total.')

  # Ref: Boolean (Ctrl+F "flag") https://docs.python.org/2/howto/argparse.html
  arg_parser.add_argument ('--pcd', action='store_true', default=False,
    help='Boolean flag, no args. Run on synthetic data in csv_tri_lists/ from point cloud')
  arg_parser.add_argument ('--real', action='store_true', default=False,
    help='Boolean flag, no args. Run on real robot data in csv_bx_tri/')
  arg_parser.add_argument ('--gazebo', action='store_true', default=False,
    help='Boolean flag, no args. Run on synthetic data in csv_gz_tri/ from Gazebo. nSamples and nSamplesRatio do not make sense currently, so just always enter same thing so data gets saved to same folder, like 0 0')
  arg_parser.add_argument ('--mixed', action='store_true', default=False,
    help='Boolean flag, no args. Run on a mix of PCL, Gazebo, etc data. This assumes you will specify long csv paths (--long_csv_path in some other scripts) in the meta file! We will read starting from train/ directory, everything after must be specified in meta file. If a line is for csv_gz_tri, histogram will be outputted to csv_gz_hists; csv_pcl_tri would get output in csv_pcl_hists.')

  arg_parser.add_argument ('--pcdtest', action='store_true', default=False,
    help='Boolean flag, no args. Use models_test.txt as opposed to models.txt.')
  arg_parser.add_argument ('--realtest', action='store_true', default=False,
    help='Boolean flag, no args. Run Kernel Density Estimate 2D histogram slices plots.')
  # No test mode for gazebo yet, don't need it yet

  arg_parser.add_argument ('--meta', type=str, default='',
    help='String. Base name of meta list file in triangle_sampling/config directory')

  args = arg_parser.parse_args ()

  pcdTest = args.pcdtest
  realTest = args.realtest

  # More than one Boolean is True
  if args.pcd + args.real + args.gazebo + args.mixed > 1:
    print ('ERROR: More than one of --pcd, --real, --gazebo, and --mixed were specified. You must choose one. Terminating...')
    return
  elif args.pcd + args.real + args.gazebo + args.mixed == 0:
    print ('%sERROR: Neither --pcd, --real, or --gazebo were specified. You must choose one. Terminating...%s' ( \
      ansi_colors.FAIL, ansi_colors.ENDC))
    return

  # Point clouds PCD data sampled by PCL
  if args.pcd or args.pcdtest:
    data_mode = PCD_CLOUD_DATA
  elif args.real or args.realtest:
    data_mode = REAL_ROBOT_DATA
  # Mesh model COLLADA data sampled by Gazebo physics
  elif args.gazebo:
    data_mode = GAZEBO_HAND_DATA
  elif args.mixed:
    data_mode = MIXED_DATA

  print ('%sdata_mode set to %s %s' % (ansi_colors.OKCYAN, data_mode,
    ansi_colors.ENDC))
  print ('  (PCD_CLOUD_DATA 0, REAL_ROBOT_DATA 1, GAZEBO_HAND_DATA 2, MIXED_DATA 3)')


  if args.pcd:
    if len (args.histSubdirParam2) != 1:
      print ('%sERROR: Expect histSubdirParam2 to only have one element, for --pcd mode. Check your args and retry.%s' % ( \
        ansi_colors.FAIL, ansi_colors.ENDC))
      return

    nSamples = int (args.histSubdirParam1)
    nSamplesRatio = float (args.histSubdirParam2 [0])
    print ('%sAccessing directory with nSamples %d, nSamplesRatio %f%s' % \
      (ansi_colors.OKCYAN, nSamples, nSamplesRatio, ansi_colors.ENDC))

    # Sampling subpath to save different number of samples, for quick accessing
    #   without having to rerun sample_pcl.cpp.
    sampling_subpath = get_sampling_subpath (nSamples, nSamplesRatio)

  elif args.real:
    if len (args.histSubdirParam2) != 1:
      print ('%sERROR: Expect histSubdirParam2 to only have one element, for --gazebo mode. Check your args and retry.%s' % ( \
        ansi_colors.FAIL, ansi_colors.ENDC))
      return

    sampling_subpath, _ = get_triparams_nbins_subpath (args.histSubdirParam1,
      args.histSubdirParam2 [0])

  elif args.gazebo:
    if len (args.histSubdirParam2) != 1:
      print ('%sERROR: Expect histSubdirParam2 to only have one element, for --gazebo mode. Check your args and retry.%s' % ( \
        ansi_colors.FAIL, ansi_colors.ENDC))
      return

    sampling_subpath, _ = get_triparams_nbins_subpath (args.histSubdirParam1,
      args.histSubdirParam2 [0])

  # For mixed data, need to specify both kinds of histSubdirParams, a pair for
  #   PCL data's sampling_subpath, a pair for Gazebo data's sampling_subpath.
  elif args.mixed:
    if len (args.histSubdirParam2) != 3:
      print ('%sERROR: Expect histSubdirParam2 to have three elements, for --mixed mode. Check your args and retry.%s' % ( \
        ansi_colors.FAIL, ansi_colors.ENDC))
      return

    nSamples = int (args.histSubdirParam1)
    nSamplesRatio = float (args.histSubdirParam2 [0])
    sampling_subpath = get_sampling_subpath (nSamples, nSamplesRatio)

    tri_nbins_subpath, _ = get_triparams_nbins_subpath (
      args.histSubdirParam2 [1], args.histSubdirParam2 [2])


  #####
  # Init the correct node for the task
  #####

  if (data_mode == PCD_CLOUD_DATA) or pcdTest:
    pcdNode = TrianglesListsSyntheticPublisher (sampling_subpath, nSamples,
      nSamplesRatio)
    pcdNode.read_config ()

    if args.meta:
      model_list_name = os.path.join (model_list_path, args.meta)
    else:
      if pcdTest:
        model_list_name = os.path.join (model_list_path, 'models_test.txt')
      else:
        model_list_name = os.path.join (model_list_path, 'models.txt')

  # For real data, use the histogram bins saved in config when training on
  #   synthetic data. This is to make sure real data and synthetic data fall
  #   in the same bins, so that they can be used in recognition. Train
  #   synthetic test real.
  elif (data_mode == REAL_ROBOT_DATA) or realTest:
    realNode = TrianglesOnRobotToHists (sampling_subpath, testKDE=realTest,
      csv_suffix='bx_')
    realNode.read_config ()

    if args.meta:
      model_list_name = os.path.join (model_list_path, args.meta)
    else:
      if realTest:
        model_list_name = os.path.join (model_list_path, 'models_test.txt')
      else:
        model_list_name = os.path.join (model_list_path, 'real.txt')

  # For Gazebo hand data
  elif data_mode == GAZEBO_HAND_DATA:
    # Not setting testKDE to True, don't need it yet
    gazeboNode = TrianglesOnRobotToHists (sampling_subpath,
      csv_suffix='gz_')
    gazeboNode.read_config ()

    if args.meta:
      model_list_name = os.path.join (model_list_path, args.meta)
    else:
      #model_list_name = os.path.join (model_list_path, 'models_active.txt')
      # Use models_test.txt, so don't need to uncomment and recomment btw
      #   running this file and sample_gazebo.py, because Gazebo can't handle
      #   more than one object file per run of sample_gazebo.py, because hand
      #   gets broken and can't be reloaded normally - -
      #model_list_name = os.path.join (model_list_path, 'models_active_test.txt')
      model_list_name = os.path.join (model_list_path, 'models_gazebo_csv.txt')

  elif data_mode == MIXED_DATA:

    # Need to initialize both the TrianglesListsSyntheticPublisher AND
    #   the TrianglesOnRobotToHists nodes.

    # This should give the train/ directory. Used for prepending to long csv
    #   paths in meta file.
    train_path = tactile_config.config_paths ('custom', '')

    # One node for PCL data
    pcdNode = TrianglesListsSyntheticPublisher (sampling_subpath, nSamples,
      nSamplesRatio)
    pcdNode.read_config ()

    # One node for Gazebo data
    #   Use the PCL data's hist_conf.csv, `.` training is done on PCL data,
    #     need Gazebo histograms to be in same range, for descriptors to make
    #     sense!
    hist_conf_path = os.path.join (train_path, 'triangle_sampling',
      'csv_hists', sampling_subpath, tri_nbins_subpath, 'hist_conf.csv')
    # Use the Gazebo histograms in gz_pclrange_ with PCL hist_conf.csv range,
    #   not the pure Gazebo data in csv_gz_hists using Gazebo hist_conf.csv
    #   range.
    gazeboNode = TrianglesOnRobotToHists (tri_nbins_subpath,
      csv_suffix='gz_pclrange_', custom_hist_conf_name=hist_conf_path)
    # This will read the custom hist_conf_name passed in to ctor above. That
    #   makes Gazebo histograms use same range as the PCL training data.
    gazeboNode.read_config (convert_to_decimeter=False)

    if args.meta:
      model_list_name = os.path.join (model_list_path, args.meta)
    else:
      # This meta file contains long csv paths starting from triangle_sampling
      model_list_name = os.path.join (model_list_path, 'models_gazebo_pclrange_test.txt')
      # If you're lazy to uncomment everything in models_gazebo_csv.txt every
      #   time a new object is trained, just put one csv in a temp file and
      #   run this file on just one object...
      #model_list_name = os.path.join (model_list_path, 'tmp.txt')


  #####
  # Get triangle file names, pass to appropriate node
  #####

  model_list_file = open (model_list_name)

  nIters = 6

  nObjs = 0

  # Record number of triangles per object
  nTriangles = []

  # Seconds
  start_time = time.time ()

  ended_immature = False

  # Read model list file line by line
  for line in model_list_file:

    if rospy.is_shutdown ():
      break

    # Skip empty lines
    if not line:
      continue

    # Strip endline char
    # Some path, doesn't matter what. Nodes only need the base name. The data
    #   parent paths are defined within each node.
    line = line.rstrip ()

    # Skip comment lines
    if line.startswith ('#') or line == '':
      continue


    print ('\n%s' % line)

    # Increment # objects seen
    nObjs += 1

    # Find base name
    base = os.path.basename (line)

    # Find category name
    #   Copied from triangle_sampling/parse_models_list.py
    # Drop base name
    cat_name, _ = os.path.split (line)
    # Grab last dir name. This won't contain a slash, guaranteed by split()
    _, cat_name = os.path.split (cat_name)

    if data_mode == PCD_CLOUD_DATA:
      # Synthetic model list files have .pcd extension. Chop it off, append .csv
      #   extension.
      csv_name = os.path.splitext (base) [0] + '.csv'

      # Get a ROS msg with triangles, publish them later
      nTriangles.append (pcdNode.read_triangles (csv_name, cat_name))

    elif data_mode == REAL_ROBOT_DATA:
      # Read the triangles, compute histogram, and save histogram file.
      #   This node doesn't care about full path being in real.txt, it
      #   just reads base name from csv_tri/ directory.
      # Real-robot collected list files already have the correct .csv extension
      nTriangles.append (realNode.read_triangles (base, cat_name))

    elif data_mode == GAZEBO_HAND_DATA:
      # Synthetic model list files have .dae extension. Chop it off, append .csv
      #   extension.
      #csv_name = os.path.splitext (base) [0] + '_hand.csv'
      # models_gazebo_csv.txt has long paths.
      csv_name = os.path.splitext (base) [0] + '.csv'

      # Read the triangles, compute histogram, and save histogram file.
      nTriangles.append (gazeboNode.read_triangles (csv_name, cat_name))

    elif data_mode == MIXED_DATA:

      # Use full path
      csv_name = os.path.join (train_path, line)

      # Find the csv directory in this line, to decide which instance to call
      if 'csv_tri' in line:
        # Get a ROS msg with triangles, publish them later
        nTriangles.append (pcdNode.read_triangles (csv_name, full_path=True))

      elif 'csv_gz_tri' in line:
        # Read the triangles, compute histogram, and save histogram file.
        nTriangles.append (gazeboNode.read_triangles (csv_name, cat_name,
          full_path=True))

      else:
        print ('%sDid not find csv_tri or csv_gz_tri keywords in this line of meta file. Did you specify the full path of csv files? Offending line: %s%s' % ( \
          ansi_colors.WARNING, line, ansi_colors.ENDC))
        continue


    if data_mode == PCD_CLOUD_DATA or data_mode == MIXED_DATA:

      # If there are any triangles loaded
      if pcdNode.get_need_to_publish ():

        seq = 0
        while not rospy.is_shutdown ():
       
          # Publish to sample_pcl_calc_hist.py, which will write
          #   histograms to file.
          pcdNode.publish ()
       
          seq += 1
          if seq >= nIters:
            break
       
          try:
            wait_rate.sleep ()
          except rospy.exceptions.ROSInterruptException, err:
            ended_immature = True
            break


  # end for line

  model_list_file.close ()



  # Copied from sample_pcl.cpp
  # Print out running time
  # Seconds
  end_time = time.time ()
  print ('Total time for %d objects: %f seconds.' % \
    (nObjs, end_time - start_time))
  if nObjs != 0:
    print ('Average %f seconds per object.\n' % ((end_time - start_time) / nObjs))


  # Copied from sample_pcl.cpp
  # Publish a last msg to tell Python node we're done, so that they can show
  #   matplotlib plot images and save them
  # I think this only works if program shuts down normally. If ros::ok() is
  #   false, then these msgs don't get published.
  if data_mode == PCD_CLOUD_DATA:
    print ('Publishing last %d messages to tell subscriber we are terminating...' %\
      nIters)

    tri_pub = rospy.Publisher ('/sample_pcl/triangle_params',
      TriangleParams, queue_size=5)

    tri_msg = TriangleParams ()
    tri_msg.obj_seq = -1;

    for i in range (0, nIters):
      tri_pub.publish (tri_msg);
      wait_rate.sleep ();


  #####
  # Write number of triangles to file
  #   Best place for this code to be is really
  #     sample_pcl_calc_hist.py, but since I'm not running
  #     sampling any time soon, that file will not have access to this.
  #     So putting it here for now, move there the next time I run
  #     sample_pcl.cpp
  #####

  # Don't record number of triangles if ended midway! You could overwrite
  #   existing good ones.
  if not ended_immature and not pcdTest and not realTest:

    if data_mode == PCD_CLOUD_DATA:
      # Split off the sampling_subpath directory
      stats_dir = tactile_config.config_paths ('custom',
        os.path.split (pcdNode.tri_subpath) [0] + '/_stats/')
      write_num_triangles (stats_dir, sampling_subpath, nTriangles)

    elif data_mode == REAL_ROBOT_DATA:
      # No sampling_subpath dir to split off
      stats_dir = tactile_config.config_paths ('custom',
        realNode.tri_subpath + '/_stats/')
      write_num_triangles (stats_dir, sampling_subpath, nTriangles)

    elif data_mode == GAZEBO_HAND_DATA:
      stats_dir = tactile_config.config_paths ('custom',
        gazeboNode.tri_subpath + '/_stats/')
      # No sampling_subpath necessary here. Number of triangles doesn't differ
      #   btw different triangle parameter choices or hist bin choices. Those
      #   are for histogram only and are further down the pipeline.
      write_num_triangles (stats_dir, '', nTriangles)

  
if __name__ == '__main__':
  main ()


