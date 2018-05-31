#!/usr/bin/env python

# Mabel Zhang
# 3 Aug 2015
#
# This node subscribes to the sample_pcl.cpp node's triangles data, and
#   plots a 3D histogram (1D per triangle parameter) in matplotlib.
#
# Purpose of plotting different objects together is that you should observe
#   that objects that look similar should have similar histograms. If not, then
#   the descriptor doesn't really work - it does not describe similar objs
#   the same way, then it's not good for recognition.
#
# To run:
#   $ rosrun triangle_sampling sample_pcl_plotter.py
#   Wait till it says initialized, before running next one (otherwise some msgs
#     published by C++ node might not be received in time.
#
#   $ rosrun triangle_sampling sample_pcl
#

# ROS
import rospy

# Numpy
import numpy as np

from triangle_sampling.config_hist_params import TriangleHistogramParams

# Local
from plot_hist_dd import create_subplot, plot_hist_dd, show_plot
# Compile this msg first, then remove its lines in CMakeLists.txt and 
#   package.xml. When the lines are in there, pcl_conversions.h reports lots
#   errors. I don't know why. Something wrong with that file. It doesn't
#   compile with pcl_ros either.
from triangle_sampling_msgs.msg import TriangleParams


class SampleObjPCLPlotter:

  def __init__ (self, figs, axes, figs1D, axes1D):

    self.figs = figs
    self.axes = axes
    self.figs1D = figs1D
    self.axes1D = axes1D

    rospy.Subscriber ('/sample_pcl/triangle_params', TriangleParams,
      self.dataCB)

    self.END_CONDITION = -1
    self.INIT_VAL = -2

    self.l1 = None
    self.l2 = None
    self.a1 = None
    self.l0 = None
    self.s0 = None
    self.s2 = None
    self.obj_seq = self.INIT_VAL
    self.obj_name = ''

    self.prev_plotted_seq = self.INIT_VAL

    self.ylbls = ['len1', 'len2', 'angle1']
    # ['l0', 'l1', 'l2', 'a0', 'a1', 'a2']
    self.ylbls1D = [TriangleHistogramParams.L0,
      TriangleHistogramParams.L1,
      TriangleHistogramParams.L2,
      TriangleHistogramParams.A0,
      TriangleHistogramParams.A1,
      TriangleHistogramParams.A2]

    self.bins = [30, 30, 36]
    self.bins1D = [30, 30, 30, 36, 36, 36]

    self.plot_shown = False
    self.doTerminate = False


  def dataCB (self, msg):

    # float32[]
    self.l0 = msg.l0
    self.l1 = msg.l1
    self.l2 = msg.l2
    self.a0 = msg.a0
    self.a1 = msg.a1
    self.a2 = msg.a2

    # int
    self.obj_seq = msg.obj_seq

    # string
    self.obj_name = msg.obj_name

    #rospy.logerr ('Received obj_seq ' + str(self.obj_seq))


  def plot (self):

    # If haven't received any data to plot, nothing to do
    # Check obj_seq, don't check l1 is None, because the terminating msg -1
    #   will contain l1 = None! Then this makes it return, and no plot will
    #   ever be saved or shown!
    if self.obj_seq == self.INIT_VAL:
      return

    # Publisher signaling to us that it's done
    # This conditional must be before the two below, else self.l1 gets reverted
    #   to 0
    if self.obj_seq == self.END_CONDITION:

      # If already shown, nothing to do.
      if self.plot_shown:
        self.doTerminate = True
        return

      self.plot_shown = True
      self.doTerminate = True

      # Show the plot
      print ('Showing figure...')
      show_plot (self.figs, self.ylbls, 3)
      show_plot (self.figs1D, self.ylbls1D, 1)
      return

    # If already plotted the msg in store, nothing new to plot
    elif self.prev_plotted_seq >= self.obj_seq:
      return
    else:
      self.prev_plotted_seq = self.obj_seq


    #####
    # Plot 3D histograms
    # NOTE Flaw in this file: Unlike sample_pcl_calc_hist.py , this
    #   file did not find a common bin range for all objects. To be able to do
    #   that, you have to save all objects' histograms in memory, until end of
    #   all objects, then find min and max across all histograms. At the time
    #   this file was written, I only needed to plot the hists, not do any
    #   real calculation (like saving hists to .csv file for real training,
    #   where the meaning of a bin should be SAME across all objects).
    #
    #   So this file's 3D histograms are only good for viewing purposes.
    #   To perform calculations on 3D histograms, use
    #   sample_pcl_calc_hist.py .
    #####

    # Convert list of lists to np.array, this gives a 3 x n array. Transpose to
    #   get n x 3, which histogramdd() wants.
    tri_params = np.asarray ( [\
      [i for i in self.l1],
      [i for i in self.l2],
      [i for i in self.a1]]).T

    hist, _ = plot_hist_dd (tri_params, self.bins, self.figs, self.axes,
      self.obj_seq, self.ylbls)


    #####
    # Plot 1D histograms for debugging
    #####

    single_param = np.zeros ([len (self.l0), 6])
    single_param [:, 0] = np.asarray ([i for i in self.l0])
    single_param [:, 1] = np.asarray ([i for i in self.l1])
    single_param [:, 2] = np.asarray ([i for i in self.l2])
    single_param [:, 3] = np.asarray ([i for i in self.a0])
    single_param [:, 4] = np.asarray ([i for i in self.a1])
    single_param [:, 5] = np.asarray ([i for i in self.a2])

    for i in range (0, 6):
      plot_hist_dd (single_param [:, i], [self.bins1D [i]], [self.figs1D [i]],
        [self.axes1D [i]], self.obj_seq, [self.ylbls1D [i]])

    print ('Plotted for obj_seq %d' % self.obj_seq)



def main ():

  rospy.init_node ('sample_pcl_plotter', anonymous=True)


  # 3D histograms
  figs = [None] * 3
  axes = [None] * 3
  # For each triangle parameter, make a 4 x 4 subplot grid
  for i in range (0, len (figs)):
    figs[i], axes[i] = create_subplot ([2, 2])

  # 1D histograms for debugging
  figs1D = [None] * 6
  axes1D = [None] * 6
  for i in range (0, len (figs1D)):
    figs1D[i], axes1D[i] = create_subplot ([2, 2])

  thisNode = SampleObjPCLPlotter (figs, axes, figs1D, axes1D)


  wait_rate = rospy.Rate (10)

  print ('sample_pcl_plotter node initialized...')


  while not rospy.is_shutdown ():

    thisNode.plot ()

    if thisNode.doTerminate:
      break

    try:
      wait_rate.sleep ()
    except rospy.exceptions.ROSInterruptException, err:
      break


if __name__ == '__main__':
  main ()

