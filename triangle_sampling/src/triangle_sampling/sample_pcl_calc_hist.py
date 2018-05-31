#!/usr/bin/env python

# Mabel Zhang
# 4 Aug 2015
#
# This node subscribes to the sample_pcl.cpp node's triangles data, and
#   plots histogram intersection in matplotlib.
#
# Purpose of plotting different objects together is that you should observe
#   that objects that look similar should have similar histograms. If not, then
#   the descriptor doesn't really work - it does not describe similar objs
#   the same way, then it's not good for recognition.
#
# To run for plotting:
#   In this file, set self.doPlot = True, self.plotMinus = True .
#   In sample_pcl.cpp, uncomment files in model_name array that you want to
#     plot in a confusion matrix.
#:
#   $ rosrun triangle_sampling sample_pcl_calc_hist.py
#   Wait till it says initialized, before running next one (otherwise some msgs
#     published by C++ node might not be received in time.
#
#   $ rosrun triangle_sampling sample_pcl
#
# To run for saving 3D histograms to file:
#   In this file, set self.doPlot = False, self.plotMinus = False .
#   In sample_pcl.cpp, uncomment ALL files in model_name array.
#     `.` in each run, sampling is random, therefor histogram bin ranges are
#     different, and therefore bin edges are different. Then the meaning of
#     each bin changes across runs. So you cannot use histogram files from
#     different runs together!! They will be garbage information!!! You MUST
#     run ALL files you intend to use histograms for training, for MDS,
#     whatever, in ONE run of sample_pcl.cpp .
#
#   Then run the same commands as above.
#

# ROS
import rospy
from std_msgs.msg import Int32, Float32
from visualization_msgs.msg import MarkerArray

# Python
import csv, os
import argparse
import time, datetime
from copy import deepcopy

# Numpy
import numpy as np

# Matplotlib
import matplotlib as mpl

# My packages
# Compile this msg first, then remove its lines in CMakeLists.txt and 
#   package.xml. When the lines are in there, pcl_conversions.h reports lots
#   errors. I don't know why. Something wrong with that file. It doesn't
#   compile with pcl_ros either.
from triangle_sampling_msgs.msg import TriangleParams
from tactile_collect import tactile_config
from util.ansi_colors import ansi_colors
from triangle_sampling.parse_models_list import get_meta_cat_name
from triangle_sampling.config_paths import get_sampling_subpath, \
  get_triparams_nbins_subpath, get_ints_from_comma_string
# Shorten a name http://stackoverflow.com/questions/840969/how-do-you-alias-a-python-class-to-have-another-name-without-using-inheritance
from triangle_sampling.config_hist_params import TriangleHistogramParams as \
  HistP

# Local
from plot_hist_dd import show_plot
from write_hist_3d import write_hist_3d_csv
from find_data_range_for_hist import find_data_range_for_hist
from plot_hist_intersection import plot_conf_mat_hist_inter
from sample_pcl_wrist_sim import simulate_wrist_poses
from io_probs import IOProbs


class SampleObjPCLPlotterHistInter:

  # hist3d, kde: mutually exclusive, so that when you run 1 file KDE, you don't
  #   overwrite hist_conf.csv by mistake, replacing the correct one with
  #   bin range computed from entire dataset - it is different from the bin
  #   range of just a few objects, typically wider.
  def __init__ (self, nSamples, nSamplesRatio,
    hist3d=True, kde=False, kdeTest=False, raw=False,
    prs=[HistP.PR1, HistP.PR2, HistP.PR3],
    nbins=[HistP.bins3D[0], HistP.bins3D[1], HistP.bins3D[2]]):

    #####
    # User adjust params
    #####

    # User adjust param
    # Plot only if have 4 or fewer objects. Else plot too small to see anyway,
    #   and takes forever to plot so many subplots. There are nObjs x nObjs
    #   subplots per figure, and there are 6 figures, for the 6 triangle
    #   params!
    self.doPlot = False
    # Whether to also plot orig histogram minus histogram intersection.
    #   Adds another 6 figures.
    self.plotMinus = False

    # Don't set this to True. I don't use this anymore. Don't waste time.
    # Save 1D histograms
    self.save1D = False

    self.decimeter = HistP.decimeter
    print ('%sMeter to DECIMETER conversion set to %s%s' % \
      (ansi_colors.OKCYAN, self.decimeter, ansi_colors.ENDC))

    # Can save time from sampling if save triangles
    # Only set this to true if you're running real sampling from
    #   sample_pcl.cpp.
    #   If you're running triangles_reader.py, then the triangles you're
    #   inputting are already from saved raw triangles! Don't save over them!
    #   They'll be the same, but you may have problem with file I/O locks!
    self.saveRaw = False
    if raw:
      self.saveRaw = True

    self.testKDE = False
    self.saveKDE = False
    self.save3D = False
    if kdeTest:
      self.testKDE = True

    elif kde:
      # Perform kernel density estimation (KDE), instead of histograms.
      #   KDE produces smoother versions of histograms, that are no longer
      #   called histograms.
      self.saveKDE = True

    elif hist3d:

      # Save 3D histograms. This is the main purpose of this script!!
      #   If you don't want to overwrite existing 3D histograms while running
      #   this script for something else, flip this to False.
      self.save3D = True


    self.nSamples = nSamples
    self.nSamplesRatio = nSamplesRatio

    self.nbins = nbins
    self.prs = prs


    #####
    # Data and writing
    #####

    self.nDims = 6

    rospy.Subscriber ('/sample_pcl/triangle_params', TriangleParams,
      self.dataCB)

    rospy.Subscriber ('/sample_pcl/nSamples', Int32,
      self.nSamplesCB)
    rospy.Subscriber ('/sample_pcl/nSamplesRatio', Float32,
      self.nSamplesRatioCB)

    self.obj_names = []
    self.obj_cats = []

    self.tri_params = []

    self.tri_pts0 = []
    self.tri_pts1 = []
    self.tri_pts2 = []

    # ['l0', 'l1', 'l2', 'a0', 'a1', 'a2']
    self.param_names = deepcopy (HistP.TRI_PARAMS)


    # TODO: Testing whether taking mean quaternion at each wrist position
    #   would decrease number of unique poses.
    self.mean_quaternion = True

    self.vis_arr_pub = rospy.Publisher ('/visualization_marker_array',
      MarkerArray, queue_size=2)


    #####
    # Constants
    #####

    # Shorthand for constants
    #   Which params you're using for histogram
    # ATTENTION: If change these, must check the saveKDE condition below,
    #   to make sure you're still scaling the lengths, not the angles.
    #   TODO If this works, will do the multiplication * 10 in
    #   sample_pcl.cpp, then don't have to artificially rescale here.
    #   > (21 Jan 2016) Don't do that. Store triangles in meters. This keeps
    #     point cloud data, real robot data, and Gazebo data uniform. Just
    #     rescale when make histogram, if necessary. Don't rescale orig data.
    #     Keep orig data as normal and original as possible!
    self.pr1 = prs [0] #HistP.PR1
    self.pr2 = prs [1] #HistP.PR2
    self.pr3 = prs [2] #HistP.PR3

    self.pr1_idx = HistP.TRI_PARAMS.index (self.pr1) #HistP.PR1_IDX
    self.pr2_idx = HistP.TRI_PARAMS.index (self.pr2) #HistP.PR2_IDX
    self.pr3_idx = HistP.TRI_PARAMS.index (self.pr3) #HistP.PR3_IDX

    self.suptitles = deepcopy (self.param_names)
    self.file_suff = deepcopy (self.param_names)
    if self.plotMinus:
      self.suptitles.extend (self.suptitles)
      self.file_suff.extend ([i + '_minusHist' for i in self.file_suff])
  
    # Size is nDims
    # For linear (flattened 1D) histogram plots
    self.bins = [30, 30, 30, 36, 36, 36]
    # This is for raw triangles file in csv_tri_lists/, first created for sending to Ani
    #self.TRI_TITLES = [self.pr1, self.pr2, self.pr3]
    # Now want all 6 params, so can convert csv_tri_lists files to csv_tri files, for
    #   easy comparison btw Gazebo and PCL trained data, and for plotting accuracy graph
    #   vs. set of triangle params chosen.
    self.TRI_TITLES = HistP.TRI_PARAMS

    self.END_CONDITION = -1
    self.INIT_VAL = -2
    self.obj_seq = self.INIT_VAL

    self.plot_shown = False
    self.doTerminate = False

    self.HIST3D = 0
    self.TRI = 1
    self.HIST1D = 2
    self.KDE = 3

    print ('%sTriangle parameters set to %s,%s,%s. Histogram number of bins set to %d,%d,%d%s' % ( \
      ansi_colors.OKCYAN,
      HistP.TRI_PARAMS [self.pr1_idx],
      HistP.TRI_PARAMS [self.pr2_idx],
      HistP.TRI_PARAMS [self.pr3_idx],
      self.nbins[0], self.nbins[1], self.nbins[2], ansi_colors.ENDC))



  def dataCB (self, msg):

    #rospy.logerr ('Received obj_seq ' + str(self.obj_seq))

    # If we already received this object's data, don't need to look at it again
    if msg.obj_seq == self.obj_seq:
      return

    # If this is the terminating msg, it contains no data, don't need to look
    #   at it
    if msg.obj_seq == self.END_CONDITION:
      self.obj_seq = msg.obj_seq
      return

    # int
    self.obj_seq = msg.obj_seq

    # string. Some kind of path straight from the models.txt, we'll only
    #   access the base name, and the immediate category subdir that the file
    #   is in.
    self.obj_names.append (msg.obj_name)

    # string
    self.obj_cats.append (msg.obj_cat)


    nTriangles = len (msg.l0)
 
    # Make space for this object's data, nTriangles x nDims matrix. nDims = 6
    self.tri_params.append (np.zeros ([nTriangles, self.nDims]))
    endi = len (self.tri_params) - 1

    print ('Received data for %dst object (seq %d)' % (endi+1, self.obj_seq))

    #print ('nTriangles: %d, ' % (nTriangles))
    #print (np.shape(self.tri_params [endi]))
 
    # float32[]
    # tri_params is a Python list of NumPy 2D arrays.
    #   tri_params [i] [:, dim] gives data for object i, dimension dim.
    # endi indexes the newest added object
    if msg.l0:
      self.tri_params [endi] [:, HistP.L0_IDX] = np.array ([i for i in msg.l0])
    if msg.l1:
      self.tri_params [endi] [:, HistP.L1_IDX] = np.array ([i for i in msg.l1])
    if msg.l2:
      self.tri_params [endi] [:, HistP.L2_IDX] = np.array ([i for i in msg.l2])
    if msg.a0:
      self.tri_params [endi] [:, HistP.A0_IDX] = np.array ([i for i in msg.a0])
    if msg.a1:
      self.tri_params [endi] [:, HistP.A1_IDX] = np.array ([i for i in msg.a1])
    if msg.a2:
      self.tri_params [endi] [:, HistP.A2_IDX] = np.array ([i for i in msg.a2])

    # nTriangles x 3 numpy arrays
    # List of 3 xyz points on the triangles
    #   Convert to a list of 3-tuples. Have 3 lists, for the 3 pts on triangle.
    #   Ordering of the 3 lists and 3 points doesn't matter.
    # Note: msg.pt0, pt1, pt2 must have the same size (nTriangles)!
    #   msg.pt0 pt1 pt2 are geometry_msgs/Point[]
    self.tri_pts0.append (np.array ([(p.x, p.y, p.z) for p in msg.pt0]))
    self.tri_pts1.append (np.array ([(p.x, p.y, p.z) for p in msg.pt1]))
    self.tri_pts2.append (np.array ([(p.x, p.y, p.z) for p in msg.pt2]))

    self.obj_center = msg.obj_center
    self.obj_radii = msg.obj_radii


  def nSamplesCB (self, msg):
    self.nSamples = msg.data

  def nSamplesRatioCB (self, msg):
    self.nSamplesRatio = msg.data


  def plot (self):

    # If haven't received any data to plot, nothing to do
    # Check obj_seq, don't check l1 is None, because the terminating msg -1
    #   will contain l1 = None! Then this makes it return, and no plot will
    #   ever be saved or shown!
    if self.obj_seq == self.INIT_VAL:
      return

    # If haven't received data for ALL objects yet, do not plot yet. Cannot
    #   plot confusion matrix style plot without having all objects!
    if self.obj_seq != self.END_CONDITION:
      return

    # Publisher signaling to us that it's done
    # This conditional must be before the two below, else self.l1 gets reverted
    #   to 0
    # If already shown, nothing to do.
    if self.plot_shown:
      self.doTerminate = True
      return


    #####
    # Print for user to make sure we got correct info from sample_pcl.cpp
    #####

    print ('%sSampling density info received:%s' % (ansi_colors.OKCYAN,
      ansi_colors.ENDC))
    print ('%snSamples: %d, nSamplesRatio: %f%s' % (ansi_colors.OKCYAN,
      self.nSamples, self.nSamplesRatio, ansi_colors.ENDC))


    #####
    # Find range for histogram bins, so that the same range can be used for
    #   ALL objects. This is a requirement for computing histogram
    #   intersections btw each possible pair of objects.
    # Default range by np.histogram() is simply (min(data), max(data)). So we
    #   just need to find min and max of data from ALL objects.
    #####

    nObjs = len (self.tri_params)

    print ('[%d, %d, %d] bins for the 3 dimensions' % ( \
      #HistP.bins3D[0], HistP.bins3D[1], HistP.bins3D[2]))
      self.nbins[0], self.nbins[1], self.nbins[2]))

    bin_range, bin_range3D, header, row = find_data_range_for_hist ( \
      self.tri_params, self.decimeter, self.nbins,
      (self.pr1_idx, self.pr2_idx, self.pr3_idx))


    #####
    # Save configs of 3D histogram to a separate .csv file
    #####

    if self.save3D:

      # Write just one row, with headers to say what each column is

      (conf_outfile_name, conf_outfile, conf_writer, _) = \
        self.create_output_file ('hist_conf', self.HIST3D, header)
     
      conf_writer.writerow (dict (zip (header, row)))
      conf_outfile.close ()
     
      print ('Outputted histogram configs to ' + conf_outfile_name)


    #####
    # Save configs of raw triangles file to a separate .csv file
    #####

    # Write 3 rows. Each row is a string saying what the raw triangle .csv
    #   file's corresponding row is.

    if self.saveRaw:

      (tri_conf_outfile_name, tri_conf_outfile, tri_conf_writer, _) = \
        self.create_output_file ('tri_conf', self.TRI)

      # Each row is a single string
      for i in range (0, len (self.TRI_TITLES)):
        tri_conf_writer.writerow ([self.TRI_TITLES [i]])

      tri_conf_outfile.close ()
      print ('Outputted raw triangles configs to ' + tri_conf_outfile_name)

 

    #####
    # Plot confusion matrix style graphs -
    #   graph in axes (i, i) is histogram intersection of histogram[i] with
    #   itself, so it's jsut the histogram itself (100% intersection).
    #####

    if self.doPlot:

      xlbl_suffix = ' (Meters)'
      if self.decimeter:
        xlbl_suffix = ' (Decimeters)'

      figs, success = plot_conf_mat_hist_inter (self.tri_params, self.bins,
        bin_range, self.nDims, nObjs, self.plotMinus, self.suptitles,
        self.obj_names, xlbl_suffix=xlbl_suffix)
     
      if not success:
        self.doTerminate = True
        return

      else:
        # Show the plot
        print ('Showing figure...')
        show_plot (self.figs, self.file_suff, 1)



    #####
    # Save data to files
    #####

    firstLoop = True

    # Seconds
    # Time how long real processing takes.
    start_time = time.time ()


    # Loop through each object and save files
    for i in range (0, nObjs):

      print ('Calculating for object %d' % i)

      # File base name without extension
      file_name = os.path.splitext (os.path.basename (self.obj_names [i])) [0]

      # Immediate subdir that the file is in, the category name
      cat_name = get_meta_cat_name (self.obj_names [i])


      # Create the 3D data
      if self.save3D or self.saveKDE or self.testKDE:

        #print (self.tri_params [0].shape)

        # n x 3
        tri_params_3D = np.asarray ([
          self.tri_params [i] [:, self.pr1_idx],
          self.tri_params [i] [:, self.pr2_idx],
          self.tri_params [i] [:, self.pr3_idx]]).T
        #print (tri_params_3D.shape)


      # Rescale the lengths by *10, to use decimeters, so KDE smoothing can
      #   work better.
      if self.decimeter:

        # NOTE ATTENTION: Must make sure you are rescaling the LENGTHS, NOT
        #   the angles!! So if you change PR#, must check here

        if self.pr1 == HistP.L0 or self.pr1 == HistP.L1 or self.pr1 == HistP.L2:
          tri_params_3D [:, 0] *= 10

        if self.pr2 == HistP.L0 or self.pr2 == HistP.L1 or self.pr2 == HistP.L2:
          tri_params_3D [:, 1] *= 10

        if self.pr3 == HistP.L0 or self.pr3 == HistP.L1 or self.pr3 == HistP.L2:
          tri_params_3D [:, 2] *= 10


      #####
      # Save 3D histograms to .csv file
      #####

      if self.save3D:

        (outfile_name, outfile, writer, _) = self.create_output_file (file_name,
          self.HIST3D, cat_name=cat_name)
       
        # histdd is nbins[0] x nbins[1] x nbins[2] 3D matrix
        histdd, _, hist_linear = write_hist_3d_csv (writer, tri_params_3D,
          #bins=HistP.bins3D, bin_range=bin_range3D, normed=True)
          bins=self.nbins, bin_range=bin_range3D, normed=True)

        outfile.close ()

        print ('Histogram sum should be same for all objects this run: %f' % \
          np.sum (hist_linear))
 
        print ('%d nonzero values' % len (hist_linear [np.nonzero (hist_linear)]))
       
        print ('Outputted 3D histogram of ' + self.obj_names [i] + \
          ' to ' + outfile_name)


      #####
      # Save concatenated 1D histograms to file
      #####

      if self.save1D:

        hist1d_pr1, _ = np.histogram (self.tri_params [i] [:, self.pr1_idx],
          bins=self.bins[self.pr1_idx], range=bin_range[self.pr1_idx],
          normed=True)
        hist1d_pr2, _ = np.histogram (self.tri_params [i] [:, self.pr2_idx],
          bins=self.bins[self.pr2_idx], range=bin_range[self.pr2_idx],
          normed=True)
        hist1d_pr3, _ = np.histogram (self.tri_params [i] [:, self.pr3_idx],
          bins=self.bins[self.pr3_idx], range=bin_range[self.pr3_idx],
          normed=True)
       
        # Concatenate the three 1D histograms
        row = []
        row.extend (hist1d_pr1.tolist())
        row.extend (hist1d_pr2.tolist())
        row.extend (hist1d_pr3.tolist())
       
        (outfile_name_1d, outfile_1d, writer_1d, _) = self.create_output_file (\
          file_name, self.HIST1D, cat_name=cat_name)
          #file_name + '_1d', self.HIST1D)
        writer_1d.writerow (row)
       
        outfile_1d.close ()
       
        print ('Outputted 1D histogram of ' + self.obj_names [i] + \
          ' to ' + outfile_name_1d)


      #####
      # Save raw triangle 3 params' data, to give to Ani
      #####

      if self.saveRaw:

        (raw_outfile_name, raw_outfile, raw_writer, _) = \
          self.create_output_file (file_name, self.TRI, cat_name=cat_name)

        # Write the 3 chosen triangle parameters. One parameter per row
        #raw_writer.writerow (self.tri_params [i] [:, self.pr1_idx].tolist ())
        #raw_writer.writerow (self.tri_params [i] [:, self.pr2_idx].tolist ())
        #raw_writer.writerow (self.tri_params [i] [:, self.pr3_idx].tolist ())

        # Write all 6 triangle parameters. One parameter per row
        # Ordering in TRI_PARAMS_IDX is same as TRI_PARAMS, which is TRI_TITLES we wrote
        #   to tri_conf.csv. They need to correspond, so reader knows what each row is!
        raw_writer.writerow (self.tri_params [i] [:, HistP.TRI_PARAMS_IDX [0]].tolist ())
        raw_writer.writerow (self.tri_params [i] [:, HistP.TRI_PARAMS_IDX [1]].tolist ())
        raw_writer.writerow (self.tri_params [i] [:, HistP.TRI_PARAMS_IDX [2]].tolist ())
        raw_writer.writerow (self.tri_params [i] [:, HistP.TRI_PARAMS_IDX [3]].tolist ())
        raw_writer.writerow (self.tri_params [i] [:, HistP.TRI_PARAMS_IDX [4]].tolist ())
        raw_writer.writerow (self.tri_params [i] [:, HistP.TRI_PARAMS_IDX [5]].tolist ())

        raw_outfile.close ()

        print ('Outputted raw triangles of ' + self.obj_names [i] + \
          ' to ' + raw_outfile_name)


      #####
      # Save kernel density estimation (KDE) smoothed histograms to file
      #####

      if self.saveKDE or self.testKDE:

        (kde_name, kde_file, kde_writer, datapath) = self.create_output_file (\
          file_name, self.KDE, cat_name=cat_name)

        # If first loop, save kde config file
        config_path = None
        if firstLoop:
          config_path = datapath

        histdd, edgesdd, density_linear = write_hist_3d_csv ( \
          kde_writer, tri_params_3D,
          #bins=HistP.bins3D, bin_range=bin_range3D, normed=True, kde=True,
          bins=self.nbins, bin_range=bin_range3D, normed=True, kde=True,
          obj_name=file_name, debug=self.testKDE, config_path=config_path)

        kde_file.close ()

        print ('Outputted 3D kernel density estimated (KDE) histogram of ' + \
          self.obj_names [i] + ' to ' + kde_name)


      #####
      # Save probabilities data
      #####

      # Create a new IOProbs object for each object, so its data can be 
      #   cleared out each time, and don't have to worry about resetting it.
      #   `.` each object will get saved to a separate file.

      # Discretize params copied from sample_gazebo.py.
      # This results in 2389 absolute poses, which n x n is 4 million entries!
      #   Too big. Rounding more.
      #self.io_probs = IOProbs ('_pcd', discretize_m_q_tri=(2, 3, 2))
      # (0.08, 0.3, 2) gets 172 poses, which is more right. But the numbers look
      #   too rounded. Most positions are some combination of 0.08 and 0... how's
      #   that going to capture any important information... Quaternion is way
      #   too rough too.
      # (0.1, 0.5, 2) gets 42 poses, this number looks more right, but the
      #   values in abs_poses matrix are junk... All positions are 0.1, all quats
      #   are 0.5... Just different combos of 0.1, 0.5, 0. This is so useless!
      # (0.1, 0.5, 0.05) gets 42 poses. Now file is smaller though since I made
      #   the observations' resolution coarser, 102 MB.
      # TODO: I should visualize the abs_poses that end up getting collected in
      #   io_probs.py. See if they're any good at all. If they're too sparse,
      #   then the data is useless!
      # (0.06, 0.4, 0.05) with mean_quaternion=True, gets 32 poses, 5599
      #   triangles. Wrist poses looks good in RViz with cube object.
      self.io_probs = IOProbs ('_pcd', obj_center=(0.0, 0.0, 0.0),
        discretize_m_q_tri=(0.06, 0.05, 0.08))

      # Set file name for costs and probs data. Else file won't be saved!
      # Ref: http://stackoverflow.com/questions/13890935/timestamp-python
      timestamp = time.time ()
      timestring = datetime.datetime.fromtimestamp (timestamp).strftime (
        '%Y-%m-%d-%H-%M-%S')
      self.io_probs.set_costs_probs_filenames (timestring)

      # Simulate a fake wrist pose for each triangle sampled, to train
      #   probabilities data.
      # Ret val is a list of 3-tuple wrist positions, and a list of 4-tuple
      #   wrist orientations.
      wrist_ps, wrist_qs = simulate_wrist_poses (
        self.tri_pts0 [i], self.tri_pts1 [i], self.tri_pts2 [i],
        np.array ((self.obj_center.x, self.obj_center.y, self.obj_center.z)),
        (self.obj_radii.x, self.obj_radii.y, self.obj_radii.z),
        self.vis_arr_pub)

      if rospy.is_shutdown ():
        break


      # Use of io_probs.py functions copied from sample_gazebo.py

      # For each sampled triangle, add it to probs matrix
      for tri_i in range (0, self.tri_pts0 [i].shape [0]):

        # Concatenate p and q to make a 7-tuple.
        #   There's only 1 triangle observed. This is by nature of pt cloud
        #   sampling, since I only sample 3 pts at a time.
        self.io_probs.add_abs_pose_and_obs (
          wrist_ps [tri_i, :].tolist () + wrist_qs [tri_i, :].tolist (),
          # Reshape row into 2D array `.` fn wants 2D. np.reshape (, (1, -1))
          #   http://stackoverflow.com/questions/12575421/convert-a-1d-array-to-a-2d-array-in-numpy
          np.reshape (self.tri_params [i] [tri_i, :], (1, -1)))

        #uinput = raw_input ('[DEBUG] Press enter or q: ')
        #if uinput == 'q':
        if rospy.is_shutdown ():
          break

      # Take mean of quaternions, to reduce number of poses
      if self.mean_quaternion:
        self.io_probs.take_mean_quaternion ()

      self.io_probs.visualize_abs_poses (self.vis_arr_pub)

      # Write probs and costs to file
      # TODO: Does object name really need to be full path? This is partial
      #   path right now. If something doesn't work, then need to make
      #   sample_pcl.cpp publish the full file path in obj_name field of msg.
      self.io_probs.compute_costs_probs (self.obj_cats [i], self.obj_names [i],
        np.array ([self.obj_center.x, self.obj_center.y, self.obj_center.z]))
      self.io_probs.write_costs_probs ()

      # TODO: Check if time delay btw 2 triangles in sample_pcl.cpp is enough
      #   for this probs I/O to finish! Maybe it's too short, then will need
      #   to add manual wait time in sample_pcl.cpp. Unless you want to do it
      #   the ACK way, which is more messy - I don't recommend it! Waste of
      #   time. I spent forever on that for the GSK GUI / state machine / C++
      #   ACKing!


      # For next iteration
      firstLoop = False

    # end for nObjs


    # Copied from triangles_reader.py
    # Print out running time in seconds
    end_time = time.time ()
    print ('Total time for %d objects: %f seconds.' % \
      (nObjs, end_time - start_time))
    print ('Average %f seconds per object.\n' % \
      ((end_time - start_time) / nObjs))


    #####
    # Tell main() thread we're done
    #####

    self.plot_shown = True
    self.doTerminate = True


  # Parameters:
  #   file_name: Base name of .csv file to write
  #   dictTitles: If want a csv DictWriter, pass in the column titles in a list
  #     of strings, Else pass in None.
  #   cat_name: Immediate directory to put output file name, inside the
  #     parent csv_tri_lists and csv_hists directories. e.g. object category
  #     subfolder.
  def create_output_file (self, file_name, mode, dictTitles=None,
    cat_name=''):

    sampling_subpath = get_sampling_subpath (self.nSamples, self.nSamplesRatio)

    prsStr = '%s,%s,%s' % (self.prs[0], self.prs[1], self.prs[2])
    nbinsStr = '%d,%d,%d' % (self.nbins[0], self.nbins[1], self.nbins[2])

    # e.g. triParams_l0l1a0_nbins10_10_10
    # This only pertains to histogram files. Triangle files don't need this,
    #   `.` triangle files save all 6 params, and histogram bin choice is
    #   irrelevant to triangle files.
    tri_nbins_subpath, _ = get_triparams_nbins_subpath (prsStr, nbinsStr,
      endSlash=False)

    if mode == self.HIST3D:
      datapath = tactile_config.config_paths ('custom',
        os.path.join ('triangle_sampling/csv_hists/', sampling_subpath,
          tri_nbins_subpath))
    elif mode == self.TRI:
      datapath = tactile_config.config_paths ('custom',
        os.path.join ('triangle_sampling/csv_tri_lists/', sampling_subpath))
    elif mode == self.HIST1D:
      datapath = tactile_config.config_paths ('custom',
        os.path.join ('triangle_sampling/csv_hists_1d/', sampling_subpath,
          tri_nbins_subpath))
    elif mode == self.KDE:
      datapath = tactile_config.config_paths ('custom',
        os.path.join ('triangle_sampling/csv_kde/', sampling_subpath,
          tri_nbins_subpath))
    #print ('Training data will be outputted to ' + datapath)

    # Create output file
    outfile_name = os.path.join (datapath, cat_name,
      file_name + '.csv')
    if not os.path.exists (os.path.dirname (outfile_name)):
      os.makedirs (os.path.dirname (outfile_name))
    outfile = open (outfile_name, 'wb')

    # If a field is non-existent, output '-1'
    if dictTitles:
      writer = csv.DictWriter (outfile, fieldnames=dictTitles)
      writer.writeheader ()
    else:
      writer = csv.writer (outfile)

    #print ('Data will be outputted to %s' % outfile_name)

    return (outfile_name, outfile, writer, datapath)



def main ():

  rospy.init_node ('sample_pcl_calc_hist', anonymous=True)


  #####
  # Parse command line args
  #####

  arg_parser = argparse.ArgumentParser ()
  # Ref: Boolean (Ctrl+F "flag") https://docs.python.org/2/howto/argparse.html
  arg_parser.add_argument ('--hist3d', action='store_true', default=True,
    help='Boolean flag, no args.')
  arg_parser.add_argument ('--kde', action='store_true', default=False,
    help='Boolean flag, no args. Overrides --hist3d.')
  arg_parser.add_argument ('--kdetest', action='store_true', default=False,
    help='Boolean flag, no args. Overrides --hist3d.')
  arg_parser.add_argument ('--raw', action='store_true', default=False,
    help='Boolean flag, no args.')

  arg_parser.add_argument ('--nSamples', type=int, default=0,
    help='nSamples used when triangles were sampled, used to create directory name to read from.')
  arg_parser.add_argument ('--nSamplesRatio', type=float, default=0,
    help='nSamplesRatio used when triangles were sampled, used to create directory name to read from.')

  # Copied from hist_conf_writer.py
  arg_parser.add_argument ('--prs', type=str,
    default='%s,%s,%s' % (HistP.PR1, HistP.PR2, HistP.PR3),
    help='Triangle parameters to use for 3D histogram, e.g. l0,l1,a0, no spaces.')
  # Number of histogram bins. Used to systematically test a range of different
  #   number of bins, to plot a graph of how number of bins affect SVM
  #   classification accuracy. For paper.
  arg_parser.add_argument ('--nbins', type=str,
    default='%d,%d,%d' % (HistP.bins3D[0], HistP.bins3D[1], HistP.bins3D[2]),
    help='Number of histogram bins. Same number for all 3 triangle parameter dimensions. This will be outputted to hist_conf.csv for all subsequent files in classification to use.')

  args = arg_parser.parse_args ()

  hist3d = args.hist3d
  kde = args.kde
  kdeTest = args.kdetest
  raw = args.raw

  nSamples = args.nSamples
  nSamplesRatio = args.nSamplesRatio
  print ('Initializing to nSamples %d, nSamplesRatio %f. May be overwritten upon receiving rostopic /sample_pcl/nSamples*' % \
    (nSamples, nSamplesRatio))


  #####
  # Main ROS loop
  #####

  nbins = get_ints_from_comma_string (args.nbins)
  prs = args.prs.split (',')

  thisNode = SampleObjPCLPlotterHistInter (nSamples, nSamplesRatio,
    hist3d, kde, kdeTest, raw, prs=prs, nbins=nbins)

  # ATTENTION: Make sure these printouts corresponds with what __init__()
  #   does in SampleObjPCLPlotterHistInter class!!
  # kde and hist3d options are exclusive. kdeTest takes precedence, then kde
  if kdeTest:
    print ('%sKDE-test set to %s. Will plot KDE data, will NOT save KDE data, no 3D histograms will be saved.%s' % \
      (ansi_colors.OKCYAN, kdeTest, ansi_colors.ENDC))
  elif kde:
    print ('%sKDE set to %s. Will save KDE data, no 3D histograms will be saved.%s' % \
      (ansi_colors.OKCYAN, kde, ansi_colors.ENDC))
  elif hist3d:
    print ('%shist3d set to %s. Will save 3D histograms, no KDE will be run.%s' % \
      (ansi_colors.OKCYAN, hist3d, ansi_colors.ENDC))

  if raw:
    print ('%sraw set to %s. Will save raw triangles.%s' % (ansi_colors.OKCYAN,
      raw, ansi_colors.ENDC))


  wait_rate = rospy.Rate (10)

  print ('sample_pcl_calc_hist node initialized...')


  while not rospy.is_shutdown ():

    if thisNode.obj_seq == thisNode.END_CONDITION:
      thisNode.plot ()

    if thisNode.doTerminate:
      break

    try:
      wait_rate.sleep ()
    except rospy.exceptions.ROSInterruptException, err:
      break


if __name__ == '__main__':
  main ()

