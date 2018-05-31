#!/usr/bin/env python

# Mabel Zhang
# 19 Sep 2015
#
# Load linearized histograms in meta list (by calling load_hists.py), convert
#   to 3D histograms. For each slice in a chosen dimension of the 3D histogram,
#   draw the values in the two other dimensions as a heat map. Save all slices
#   into a video (by calling make_movie.py).
#
# Usage:
#   $ rosrun triangle_sampling draw_2d_slices.py 10 0.95 --meta models_icra2016_video_syn.txt
#


import rospkg

import os

import numpy as np
import argparse

import matplotlib.pyplot as plt
# For imshow etc
#from matplotlib.pylab import *

# My packages
from tactile_collect import tactile_config
from util.ansi_colors import ansi_colors
from triangle_sampling.load_hists import load_hists, read_hist_config
from triangle_sampling.config_paths import get_sampling_subpath
from util.make_movie import MakeMovie, create_video_from_numpy_list

# Local
from write_hist_3d import flat_to_3d_hist


def main ():

  #####
  # Parse command line args
  #   Ref: Tutorial https://docs.python.org/2/howto/argparse.html
  #        Full API https://docs.python.org/dev/library/argparse.html
  #####

  arg_parser = argparse.ArgumentParser ()

  arg_parser.add_argument ('--meta', type=str, default='models_test.txt',
    help='String. Base name of meta list file in triangle_sampling/config directory. For this script, we suggest putting only one file in models.txt (or whatever meta file you supply with --meta), because of video size, and you probably only need one video.')

  arg_parser.add_argument ('nSamples', type=int,
    help='nSamples used when triangles were sampled, used to create directory name to read from.')
  arg_parser.add_argument ('nSamplesRatio', type=float,
    help='nSamplesRatio used when triangles were sampled, used to create directory name to read from.')

  args = arg_parser.parse_args ()

  metafile_name = args.meta


  #####
  # Load histogram
  #####

  # Copied from triangles_nn.py
  rospack = rospkg.RosPack ()
  pkg_path = rospack.get_path ('triangle_sampling')
  model_metafile_path = os.path.join (pkg_path, 'config/', metafile_name)

  print ('%sAccessing directory with nSamples %d, nSamplesRatio %f%s' % \
    (ansi_colors.OKCYAN, args.nSamples, args.nSamplesRatio, ansi_colors.ENDC))
  sampling_subpath = get_sampling_subpath (args.nSamples, args.nSamplesRatio,
    endSlash=False)

  # Input and output path
  # This is same as sample_pcl_calc_hist.py
  hist_subpath = 'csv_hists'
  hist_path = tactile_config.config_paths ('custom',
    os.path.join ('triangle_sampling', hist_subpath, sampling_subpath) + '/')

  hist_conf_name = os.path.join (hist_path, 'hist_conf.csv')

  # Read histogram config file
  (param_names, nbins, bin_range) = read_hist_config (hist_conf_name)

  # Load all histograms in meta list
  print ('Loading descriptor data from %s' % hist_path)
  [hists, lbls, catnames, _, catids, sample_names] = load_hists ( \
    model_metafile_path, hist_path)

  sample_bases = [os.path.basename (n) for n in sample_names]


  #####
  # Plot 2D slices
  #####

  slices_path = tactile_config.config_paths ('custom',
    os.path.join ('triangle_sampling/imgs/slices', sampling_subpath) + '/')


  # Pick a dimension, 0, 1, or 2
  pick_dim = 2

  nSamples = np.shape (hists) [0]

  # Turn off interaction
  plt.ioff ()

  # Ref no border: http://stackoverflow.com/questions/8218608/scipy-savefig-without-frames-axes-only-content/8218887#8218887
  fig = plt.figure (frameon = False)
  #ax = plt.Axes (fig, [0, 0, 1, 1])

  videoNode = MakeMovie ()


  # For each histogram
  for i in range (0, nSamples):

    # Convert linear histogram to 3D
    hist_3d = flat_to_3d_hist (hists [i, :], nbins)

    # Loop through all slices of the picked dimension
    for s in range (0, np.shape (hist_3d) [pick_dim]):

      print ('Slice %d' % s)

      # Copied from write_hist_3d.py

      if pick_dim == 0:
        curr_slice = hist_3d [s, :, :]
        other_dim1 = 1
        other_dim2 = 2
      elif pick_dim == 1:
        curr_slice = hist_3d [:, s, :]
        other_dim1 = 0
        other_dim2 = 2
      elif pick_dim == 2:
        curr_slice = hist_3d [:, :, s]
        other_dim1 = 0
        other_dim2 = 1

      ax = plt.gca ()
      ax.set_xlabel (param_names [other_dim1])
      ax.set_ylabel (param_names [other_dim2])


      # Set ticks to bin centers

      ax.set_xticklabels ([format ('%.1f' % bi) for bi in \
        np.linspace (bin_range [other_dim1] [0],
          bin_range [other_dim1] [1], nbins [other_dim1])], rotation='vertical')
      ax.set_yticklabels ([format ('%.1f' % bi) for bi in \
        np.linspace (bin_range [other_dim2] [0],
          bin_range [other_dim2] [1], nbins [other_dim2])])

      plt.xticks (np.arange (0, np.shape (curr_slice) [0], 1.0))
      plt.yticks (np.arange (0, np.shape (curr_slice) [1], 1.0))


      #ax.set_axis_off ()
      #fig.add_axes (ax)
  
      #ax.get_xaxis ().set_visible (False)
      #ax.get_yaxis ().set_visible (False)


      im = plt.imshow (curr_slice, interpolation='nearest')

      fig.savefig (os.path.join (slices_path, 'tmp_slice.eps'),
        bbox_inches='tight', pad_inches=0)

      plt.clf ()


      # Load the saved image, create one frame of video out of the image
      _ = videoNode.load_eps_to_numpy (os.path.join (slices_path, 'tmp_slice.eps'))

    # Save the image sequence to video file
    create_video_from_numpy_list (videoNode.imgs, videoNode.max_h, videoNode.max_w,
      os.path.join (slices_path, os.path.splitext (sample_bases[i]) [0] + '_' + \
      param_names[pick_dim] + '.mp4'), frame_duration=0.5)

    videoNode.clear_sequence ()
 


if __name__ == '__main__':
  main ()

