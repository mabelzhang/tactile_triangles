#!/usr/bin/env python

# Mabel Zhang
# 20 Jan 2016
#
# Don't prune anymore, not necessary after I discovered there is no need
#   to prune out noise, if I just lower the duplicate threshold in
#   triangles_collect.py. I was throwing away too many good triangles and
#   keeping all the noise, by having a large threshold, that is why I
#   wrote this file.
# Now this file is just used for generating hist_conf.csv and plotting
#   1d histogram intersection.
# Pass in a large --thresh_l so that nothing useful is pruned, like 0.5 m.
#
#
# Read triangle csv files in csv*_tri, and prune lines with parameters
#   exceeding a specified threshold.
# New files are saved to csv*_tri_pruned. Original files not replaced.
#
# Usage:
#   $ rosrun triangle_sampling hist_conf_writer.py 0 0 --gazebo
#
#   To plot flattened 1D histogram intersections, in confusion matrix style
#     plot, for a FEW objects (make sure meta file only has < 6 objects!):
#   $ rosrun triangle_sampling hist_conf_writer.py 0 0 --plot_hist_inter --meta models_hist_inter.txt
#

# ROS
import rospkg

# Python
import os
import csv
import argparse
from copy import deepcopy

import numpy as np

import matplotlib

# My packages
from tactile_collect import tactile_config
from util.ansi_colors import ansi_colors
from triangle_sampling.parse_models_list import read_meta_file
from triangle_sampling.config_paths import get_sampling_subpath, \
  get_triparams_nbins_subpath, get_ints_from_comma_string, \
  get_recog_meta_path
from triangle_sampling.config_hist_params import TriangleHistogramParams as \
  HistP

# Local
from triangle_sampling.triangles_on_robot_to_hists import \
  TrianglesOnRobotToHists, scale_tris_to_decimeters
from find_data_range_for_hist import find_data_range_in_one_obj, \
  make_bin_ranges_from_min_max
from triangle_sampling.load_hists import scale_bin_range_to_decimeter
from plot_hist_intersection import plot_conf_mat_hist_inter
from plot_hist_dd import show_plot, save_individual_subplots, \
  plot_hists_side_by_side


# Parameters:
#   tris: 2D NumPy array of 6 x n. One row for each triangle parameter. n is
#     number of triangles.
#   param_names: (pr1, pr2, pr3), where PR# is a string of the parameter, e.g.
#     'l0', 'l1', 'a0'.
#   thresh_l: Meters. Threshold for lengths. If decimeters, will mult by 10.
#     If your file's units are not same as the constant file's setting for
#     decimeter,s you may need to change code, and manually pass in the right
#     thresh_l, instead of checking decimeters here. 
#     Units of file can be checked in whatever file that created the csv files:
#     For PCD's, it's sample_pcl_calc_hist.py;
#     For real robot hand collection, it's triangles_on_robot_to_hists.py.
#     For Gazebo hand training, it's triangles_on_robot_to_hists.py.
# Returns pruned triangles, a list of 6 lists of nTriangles floats, now
#   converted to a NumPy array of 6 x nTriangles.
def prune (tris, param_names, thresh_l):

  # Sanity checks
  #if len (tris) == 0:
  #  return tris
  #elif len (tris [0]) == 0:
  #  return tris
  if np.shape (tris) [0] == 0 or np.shape (tris) [1] == 0:
    return tris

  #if len (tris) < 3:
  if np.shape (tris) [0] < 3:
    print ('ERROR (hist_conf_writer.py prune()): tris has less than 3 triangle parameters per triangle. This is unexpected. Check what is wrong, perhaps in triangles csv file?')
    return tris


  param_is_length = []

  # Check which parameters are length. Only prune lengths, don't need to prune
  #   angles (because I don't know what angle is good indicator of outlier...)
  for i in range (0, len (param_names)):
    # Parameters starting with 'l' are lengths. ('a' are angles)
    if param_names [i].startswith ('l'):
      param_is_length.append (True)
    else:
      param_is_length.append (False)


  pruned = []
  # Populate with 6 empty lists, one for each triangle param
  for i in range (0, len (param_names)):
    pruned.append ([])

  #nTris = len (tris [0])
  nTris = np.shape (tris) [1]
  print ('Original number of triangles: %d' % nTris)

  # Stores the indices of triangles that we've added to pruned list, so we
  #   don't add it again.
  triangle_idx_prune = []

  # Loop through each triangle parameter that's a length
  for p_idx in range (0, len (param_names)):

    # If this param is not a length, don't need to prune. Go to next tri param.
    if not param_is_length [p_idx]:
      continue

    # Loop through each triangle
    for t_idx in range (0, nTris):

      # If this triangle has already been recorded to remove, skip it
      if t_idx in triangle_idx_prune:
        continue

      # If ANY length parameter of the triangle exceeds threshold, remove it
      #   (Want ANY, not ALL, therefore recording a list to remove, not a list
      #   to keep. Recording list to keep would require checking all 3 length
      #   params, too much unncessary work.)
      #if tris [p_idx] [t_idx] >= thresh_l:
      if tris [p_idx, t_idx] >= thresh_l:
        # Record we are removing this triangle
        triangle_idx_prune.append (t_idx)


  # Add the unpruned triangles into the new list, this means all parameters of
  #   a triangle.
  for t_idx in range (0, nTris):
    # If not removing this triangle, add it
    if t_idx not in triangle_idx_prune:

      for p_idx in range (0, len (param_names)):
        pruned [p_idx].append (tris [p_idx, t_idx])

  print ('Triangles kept: %d' % (nTris - len (triangle_idx_prune)))
  print ('Triangles actually kept: %d (should match above)' % len (pruned [0]))
 
  # Convert to NumPy array. 6 x nTriangles
  pruned = np.asarray (pruned)

  return pruned


def main ():

  #####
  # Parse cmd line args
  #####

  arg_parser = argparse.ArgumentParser ()

  arg_parser.add_argument ('histSubdirParam1', type=str,
    help='Used to create directory name to read from.\n' + \
      'For point cloud, nSamples used when triangles were sampled.\n' + \
      'For real robot data, specify the sampling density you want to classify real objects with, e.g. 10, will be used to load histogram bin configs.\n' + \
      'For Gazebo, omit this (pass in like 0) and specify --prs instead. Triangle params to use for 3D histogram, with no spaces, e.g. l0,l1,a0')
  arg_parser.add_argument ('histSubdirParam2', type=str,
    help='Used to create directory name to read from.\n' + \
      'For point cloud, nSamplesRatio used when triangles were sampled.\n' + \
      'For real robot data, specify the sampling density you want to classify real objects with, e.g. 0.95, will be used to load histogram bin configs.\n' + \
      'For Gazebo, omit this (pass in like 0) and specify --nbins instead. Number of bins in 3D histogram, with no spaces, e.g. 10,10,10. This will be outputted to hist_conf.csv for all subsequent files in classification to use.')

  arg_parser.add_argument ('--pcd', action='store_true', default=False,
    help='Boolean flag, no args. Run on synthetic data in csv_tri_lists/ from point cloud')
  arg_parser.add_argument ('--gazebo', action='store_true', default=False,
    help='Boolean flag, no args. Run on synthetic data in csv_tri/ from Gazebo. nSamples and nSamplesRatio do not make sense currently, so just always enter same thing so data gets saved to same folder, like 0 0')
  arg_parser.add_argument ('--real', action='store_true', default=False,
    help=format ('Boolean flag. Run on real-robot data.'))

  # Custom subdir under csv_tri, that doesn't have nSamples_nSamplesRatio
  #   style, nor triparams_nbins style.
  # Used for plotting 1D hist intersection plots for simple known shapes.
  arg_parser.add_argument ('--out_tri_subdir', type=str, default='',
    help='String. Subdirectory name of output directory under csv_tri/. This overwrites --gazebo flag. Do not specify both.')
  arg_parser.add_argument ('--long_csv_path', action='store_true',
    default=False,
    help='Boolean flag, no args. Specify it to read the full path in config file, as opposed to just using the base name in config file to read csv file from csv_tri or csv_gz_tri. Useful for comparing pcl and Gazebo data, which are outputted to different csv_*_tri paths.')

  # Meters. Though histograms use decimeters, raw triangles recorded from
  #   PCL, real robot, and Gazebo are all in meters
  arg_parser.add_argument ('--thresh_l', type=float, default=0.5,
    help='Length threshold in meters, above which to throw away a triangle.')
  arg_parser.add_argument ('--no_prune', action='store_true', default=False,
    help='Specify this if you want no pruning to occur, i.e. keep all triangles.')

  # Number of histogram bins. Used to systematically test a range of different
  #   number of bins, to plot a graph of how number of bins affect SVM
  #   classification accuracy. For paper.
  arg_parser.add_argument ('--nbins', type=str,
    default='%d,%d,%d' % (HistP.bins3D[0], HistP.bins3D[1], HistP.bins3D[2]),
    help='Number of histogram bins. Same number for all 3 triangle parameter dimensions. This will be outputted to hist_conf.csv for all subsequent files in classification to use.')
  arg_parser.add_argument ('--prs', type=str,
    default='%s,%s,%s' % (HistP.PR1, HistP.PR2, HistP.PR3),
    help='Triangle parameters to use for 3D histogram, e.g. l0,l1,a0, no spaces.')

  arg_parser.add_argument ('--plot_hist_inter', action='store_true',
    default=False, help='Plot confusion matrix style histogram minus histogram intersection (only enable if have very few objects in meta list!!!! Plot is nObjs x nObjs big!)')
  # This looks ugly. I'm not using this. Would need to look up how to resize
  #   subplots to like 4 x 3 or something. Now it's all stretched out. Also
  #   not as useful as plot_hist_inter to spot differences.
  arg_parser.add_argument ('--plot_hists', action='store_true',
    default=False, help='Plot histograms of all objects side by side, in one row. Only enable if you have very few objects in meta list! Else plots will be very small to fit on screen.')
  arg_parser.add_argument ('--save_ind_subplots', action='store_true',
    default=False, help='Save each histogram intersection subplot to an individual file. Only in effect if --plot_hist_inter or --plot_hists is specified.')

  #arg_parser.add_argument ('--meta', type=str, default='models_active_test.txt',
  arg_parser.add_argument ('--meta', type=str, default='models_gazebo_csv.txt',
    help='String. Base name of meta list file in triangle_sampling/config directory')

  # Set to True to upload to ICRA. (You can't view the plot in OS X Preview)
  # Set to False if want to see the plot for debugging.
  arg_parser.add_argument ('--truetype', action='store_true', default=False,
    help='Tell matplotlib to generate TrueType 42 font, instead of rasterized Type 3 font. Specify this flag for uploading to ICRA.')

  arg_parser.add_argument ('--black_bg', action='store_true', default=False,
    help='Boolean flag. Plot with black background, useful for black presentation slides.')


  args = arg_parser.parse_args ()

  if args.out_tri_subdir and args.long_csv_path:
    print ('Both --out_tri_subdir and --long_csv_path are specified. These are used to construct subdirectory name under csv_tri/<out_tri_subdir> or an explicit path in config file. Cannot have both paths as input. Pick only one, and try again.')
    return

  # Construct input dir name
  out_tri_subdir = args.out_tri_subdir
  long_csv_path = args.long_csv_path

  csv_suffix = ''
  if args.gazebo:
    sampling_subpath, bins3D = get_triparams_nbins_subpath (args.prs,
      args.nbins)
    csv_suffix = 'gz_'

  elif args.real:
    sampling_subpath, bins3D = get_triparams_nbins_subpath (args.prs,
      args.nbins)
    csv_suffix = 'bx_'

  # default to pcd mode
  # though I haven't tried this since added the other modes...
  else:
    # Sampling subpath to save different number of samples, for quick accessing
    #   without having to rerun sample_pcl.cpp.
    nSamples = int (args.histSubdirParam1)
    nSamplesRatio = float (args.histSubdirParam2)
    sampling_subpath = get_sampling_subpath (nSamples, nSamplesRatio)
    bins3D = get_ints_from_comma_string (args.nbins)

  if args.no_prune:
    print ('%sNo pruning%s' % (ansi_colors.OKCYAN, ansi_colors.ENDC))
  else:
    thresh_l = args.thresh_l
    print ('Length threshold (meters) above which to throw away: %g' % thresh_l)
    # No decimeters when pruning! All files should be saved in meters, so that
    #   when run triangles_reader(), which calls read_triangles(), the decimeters
    #   isn't double counted!!! All data file on disk should be in meters!!
    #if HistP.decimeter:
    #  thresh_l *= 10

  # Parse the chosen parameters, to get a list of strings
  #   e.g. ['l0', 'l1', 'a0']
  prs = args.prs.split (',')
  # Figure out the index
  prs_idx = []
  for i in range (0, len (prs)):
    if prs [i] == HistP.A0:
      prs_idx.append (HistP.A0_IDX)
    elif prs [i] == HistP.A1:
      prs_idx.append (HistP.A1_IDX)
    elif prs [i] == HistP.A2:
      prs_idx.append (HistP.A2_IDX)
    elif prs [i] == HistP.L0:
      prs_idx.append (HistP.L0_IDX)
    elif prs [i] == HistP.L1:
      prs_idx.append (HistP.L1_IDX)
    elif prs [i] == HistP.L2:
      prs_idx.append (HistP.L2_IDX)

  plot_hist_inter = args.plot_hist_inter
  plot_hists = args.plot_hists
  save_ind_subplots = False
  if plot_hist_inter:
    print ('%splot_hist_inter is set to true. Make sure your meta file has no more than 6 objects uncommented!!! Else you may run out of memory, trying to plot nObjs x nObjs plots at the end.%s' % ( \
      ansi_colors.OKCYAN, ansi_colors.ENDC))

  if plot_hist_inter or plot_hists:
    # Only in effect if --plot_hist_inter or --plot_hists is specified, else
    #  ignore it
    save_ind_subplots = args.save_ind_subplots

  # Background color of bar graph. If black, text will be set to white via
  #    matplotlib_util.py black_background in call from plot_hist_dd.py.
  if args.black_bg:
    bg_color = 'black'
    fg_color = (0.0, 1.0, 1.0)
  else:
    bg_color = 'white'
    fg_color = 'g'


  # Init node to read triangle csv file
  triReaderNode = TrianglesOnRobotToHists (sampling_subpath,
    csv_suffix=csv_suffix)
  triReaderNode.default_config ()

  # Read meta list file
  rospack = rospkg.RosPack ()
  meta_name = os.path.join (get_recog_meta_path (), args.meta)
  meta_list = read_meta_file (meta_name)
  print ('%sReading meta file from %s%s' % ( \
    ansi_colors.OKCYAN, meta_name, ansi_colors.ENDC))
 
  # Init output dir
  if not args.no_prune:
    pruned_dir = triReaderNode.tri_path.replace ('_tri', '_pruned_tri')
    print ('Pruned files will be outputted to %s' % pruned_dir)
    if not os.path.exists (pruned_dir):
      os.makedirs (pruned_dir)

  if long_csv_path:
    # This should give the train/ directory
    train_path = tactile_config.config_paths ('custom', '')


  # For ICRA PDF font compliance. No Type 3 font (rasterized) allowed
  #   Ref: http://phyletica.org/matplotlib-fonts/
  # You can do this in code, or edit matplotlibrc. But problem with matplotlibrc
  #   is that it's permanent. When you export EPS using TrueType (42), Mac OS X
  #   cannot convert to PDF. So you won't be able to view the file you
  #   outputted! Better to do it in code therefore.
  #   >>> import matplotlib
  #   >>> print matplotlib.matplotlib_fname()
  #   Ref: http://matplotlib.1069221.n5.nabble.com/Location-matplotlibrc-file-on-my-Mac-td24960.html
  if args.truetype:
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42



  nDims = -1
  min_vals = None
  max_vals = None

  # Copied from sample_pcl_calc_hist.py
  # tri_params is a Python list of NumPy 2D arrays.
  #   tri_params [i] [:, dim] gives data for object i, dimension dim.
  tri_params = []
  obj_names = []
 

  #####
  # Prune outlier triangles (NOT IN USE ANYMORE. Do NOT prune! It was a
  #   wrong solution for a bug I later found the real solution to (duplicate
  #   threshold in triangles_collect.py was too big).)
  #####

  for line in meta_list:

    #####
    # Read triangle .csv file
    #####

    line = line.strip ()

    # Full path to triangle csv file
    #   Construct csv file name from object model base name in meta file
    base = os.path.basename (line)
    if out_tri_subdir:
      base_tri = os.path.splitext (base) [0] + '_robo.csv'
      tri_name = os.path.join (triReaderNode.tri_path, out_tri_subdir, base_tri)
    elif long_csv_path:
      # Read path from csv file, instead of just the base name
      base_tri = base
      tri_name = os.path.join (train_path, line)
      if not tri_name.endswith ('.csv'):
        print ('%sLine in meta file does not end with .csv. Fix it and try again: %s%s' % (
          ansi_colors.FAIL, line, ansi_colors.ENDC))
        return
    else:
      base_tri = os.path.splitext (base) [0] + '_robo.csv'
      tri_name = os.path.join (triReaderNode.tri_path, base_tri)

    print (tri_name)
 
    # Ret val is a Python list of 3 n-item lists, each list storing one
    #   parameter for all n triangles in the file.
    tris, param_names = triReaderNode.read_tri_csv (tri_name,
      read_all_params=True)
    if tris is None:
      print ('%sread_tri_csv() encountered error. Terminating...%s' % (\
        ansi_colors.FAIL, ansi_colors.ENDC))
      return

    # Only true in first iteration. Initialize nDims and arrays
    if nDims < 0:
      nDims = len (param_names)
      min_vals = [1000] * nDims
      max_vals = [-1000] * nDims
 
 
    #####
    # Prune triangles
    #####
 
    # Ret val is a list of 6 lists of nTriangles floats, now converted to a
    #   NumPy array of 6 x nTriangles.
    if not args.no_prune:
      pruned = prune (tris, param_names, thresh_l)
    else:
      pruned = tris


    # Reshape ret val from prune() to a format accepted by
    #   plot_conf_mat_hist_inter(), for later plotting.
    # Do a sanity check, for if there are any triangles at all
    if (plot_hist_inter or plot_hists) and np.shape (pruned) [0] > 0:

      nTriangles = np.shape (pruned) [1]

      # tri_params is a Python list of 6 NumPy 2D arrays.
      #   tri_params [i] [:, dim] gives data for object i, dimension dim.
      tri_params.append (np.zeros ([nTriangles, nDims]))

      if HistP.decimeter:
        pruned_deci = scale_tris_to_decimeters (pruned, True)
      else:
        pruned_deci = pruned

      for dim in range (0, nDims):
        tri_params [len (tri_params) - 1] [:, dim] = pruned_deci [dim, :]

      obj_names.append (os.path.splitext (base) [0])

 
    #####
    # Write triangles to CSV file
    #   Code copied from triangles_collect.py. Decided not to refactor that
    #     file, `.` that file needs to run fast.
    #####

    if not args.no_prune:

      pruned_name = os.path.join (pruned_dir, base_tri)
      print ('Pruned triangle data will be outputted to %s' % (pruned_name))
  
      pruned_file = open (pruned_name, 'wb')
  
      # Column names (triangle parameters) are defned in param_names, a list of 6
      #   strings.
      pruned_writer = csv.DictWriter (pruned_file,
        fieldnames = param_names, restval='-1')
      pruned_writer.writeheader ()
  
      # Write each row. Each row is a triangle, represented by 6 floats
      for t_idx in range (0, np.shape (pruned) [1]):
  
        # Each row is a dictionary. Keys are column titles (triangle parameters in
        #   strings), values are floats.
        row = dict ()
  
        for p_idx in range (0, len (param_names)):
          row [param_names [p_idx]] = pruned [p_idx, t_idx]
         
        pruned_writer.writerow (row)


    #####
    # Find min and max in triangle data, for histogram min max edges
    #   Like find_data_range_for_hist() in find_data_range_for_hist.py, but we
    #   don't want to store all objects' data, so we don't call that function,
    #   just call each of the two funcitons separately.
    #####

    min_vals, max_vals = find_data_range_in_one_obj (pruned.T,
      min_vals, max_vals)

  # end for each line in meta list file

  print ('')

  # If meta file was empty (or all commented out)
  if (plot_hist_inter or plot_hists) and not obj_names:
    print ('%sNothing was loaded from meta file. Did you specify the correct one? Did you uncomment at least one line? Terminating...%s' % ( \
      ansi_colors.FAIL, ansi_colors.ENDC))
    return

  # Debug
  #print ('min and max vals:')
  #print (min_vals)
  #print (max_vals)
 

  #####
  # Save hist_conf.csv using min max from all objects
  #   Copied from sample_pcl_calc_hist.py.
  #####

  # Pass in decimeter=False, `.` above, when read_tri_csv, already read the
  #   triangles data in as decimeters! Don't need to do another multiplication
  #   by 10 here, it'd be doing it twice!! Then it'd become centimeter scale!
  # Pass in bins3D to pick bin sizes to write to hist_conf.csv.
  # Pass in prs_idx to pick the 3 triangle parameters to write to header string
  #   of hist_conf.csv, and pick the appropriate bin ranges out of 6.
  bin_range, bin_range3D, header, row = make_bin_ranges_from_min_max ( \
    min_vals, max_vals, decimeter=False,
    # Parameters set at top
    bins3D=bins3D, prs_idx=prs_idx)


  if not args.no_prune:
    conf_path = tactile_config.config_paths ('custom',
      os.path.join ('triangle_sampling', 'csv_' + csv_suffix + 'pruned_hists/',
      sampling_subpath))
  else:
    # eh... just use csv_gz_pruned_hists, I don't want it to mess up my good
    #   files in csv_gz_hists!!!
    conf_path = tactile_config.config_paths ('custom',
      os.path.join ('triangle_sampling', 'csv_' + csv_suffix + 'hists/',
      sampling_subpath))

  # Create output file
  conf_outfile_name = os.path.join (conf_path, 'hist_conf.csv')
  conf_outfile = open (conf_outfile_name, 'wb')

  conf_writer = csv.DictWriter (conf_outfile, fieldnames=header)
  conf_writer.writeheader ()

  conf_writer.writerow (dict (zip (header, row)))
  conf_outfile.close ()
  
  print ('Outputted histogram configs to ' + conf_outfile_name)


  #####
  # Plot confusion matrix style histogram minus histogram intersection,
  #   for debugging.
  #####

  tick_rot = 0

  if plot_hist_inter or plot_hists:

    nObjs = len (tri_params)

    #print ('nDims %d, nObjs %d' % (nDims, nObjs))

    xlbl_suffix = ' (Meters)'
    # When plotting, do mind decimeter mode, because classification will take
    #   this mode into account. Runtime = heed decimeter; files on disk = no
    #   decimeters.
    if HistP.decimeter:

      xlbl_suffix = ' (Decimeters)'

      # Scale bin ranges to decimeters too
      bin_range_deci, _ = scale_bin_range_to_decimeter (bin_range, 
        bin_range3D)
    else:
      bin_range_deci = deepcopy (bin_range)


    if plot_hist_inter:

      tick_rot = 45

      # Copied from sample_pcl_calc_hist.py
      # This gives you a detailed look
      #bins = [30, 30, 30, 30, 30, 30]
      # These are bins that are actually passed to classifier
      bins = [10, 10, 10, 10, 10, 10]
      plotMinus = True

      suptitles = deepcopy (HistP.TRI_PARAMS)
      file_suff = deepcopy (HistP.TRI_PARAMS)
      if plotMinus:
        suptitles.extend (suptitles)
        file_suff.extend ([i + '_minusHist' for i in file_suff])

      # Make histogram intersection confusion matrix plots
      figs, success = plot_conf_mat_hist_inter (tri_params, bins,
        bin_range_deci,
        nDims, nObjs, plotMinus, suptitles, obj_names, xlbl_suffix=xlbl_suffix,
        bg_color=bg_color, fg_color=fg_color, tick_rot=tick_rot)

      nRows = nObjs
      nCols = nObjs

    elif plot_hists:

      tick_rot = 45

      bins = [10, 10, 10]

      suptitles = (HistP.TRI_PARAMS [prs_idx[0]],
        HistP.TRI_PARAMS [prs_idx[1]],
        HistP.TRI_PARAMS [prs_idx[2]])
      file_suff = deepcopy (suptitles)

      tri_params_3only = []
      for i in range (0, len (tri_params)):
        # Append columns of the 3 chosen parameters
        tri_params_3only.append (tri_params [i] [:, 
          (prs_idx[0], prs_idx[1], prs_idx[2])])

      # For each object, make 3 titles
      # List of nHists (nObjs) lists of nDims (3) strings
      xlbls = []
      #for i in range (0, len (obj_names)):
        # Include triangle parameter name
        #xlbls.append ([])
        #xlbls [len (xlbls) - 1].append (suptitles[0] + ' in ' + obj_names [i])
        #xlbls [len (xlbls) - 1].append (suptitles[1] + ' in ' + obj_names [i])
        #xlbls [len (xlbls) - 1].append (suptitles[2] + ' in ' + obj_names [i])

        # Just object name
        #xlbls.append ([obj_names [i] [0 : min(20, len(obj_names[i]))]] * 3)

      # Pass in explicit edges, so all 4 hists have same edges! Else
      #   np.histogramdd() finds a different range for each object!
      #   bin[i]+1, the +1 is `.` there are n+1 edges for n bins.
      edges = []
      edges.append (np.linspace (bin_range3D[0][0], bin_range3D[0][1], bins[0]+1))
      edges.append (np.linspace (bin_range3D[1][0], bin_range3D[1][1], bins[1]+1))
      edges.append (np.linspace (bin_range3D[2][0], bin_range3D[2][1], bins[2]+1))

      figs, success = plot_hists_side_by_side (tri_params_3only, edges,
        suptitles, xlbls, bg_color=bg_color, fg_color=fg_color,
        tick_rot=tick_rot)

      nCols = nObjs
      nRows = 1

    if success:
      # Show the plot
      print ('Saving figures...')
      show_plot (figs, file_suff, 1, show=False, bg_color=bg_color)

      if save_ind_subplots:

        if plot_hist_inter:
          #save_individual_subplots (figs, file_suff, 'l0_minusHist',
          save_individual_subplots (figs, file_suff, 'l2_minusHist',
            nRows, nCols, scale_xmin=1.15, scale_x=1.3, scale_y=1.3, ndims=1, show=False, bg_color=bg_color,
            tick_rot=tick_rot)

        elif plot_hists:
          desired_plot = 'l0'
          save_individual_subplots (figs, file_suff, desired_plot,
            nRows, nCols, scale_xmin=1.1, scale_x=1.2, scale_y=1.15, ndims=1, show=False, bg_color=bg_color)


  # end if plot_hist_inter or plot_hists



if __name__ == '__main__':
  main ()

