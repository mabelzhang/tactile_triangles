#!/usr/bin/env python

# Mabel Zhang
# 4 Sep 2015
#
# Load 3D histograms saved to CSV files by sample_pcl_calc_hist.py
#
# Returns histogram (float Numpy matrix), and category information returned by
#   parse_meta_one_line.py.
#
# Calls parse_meta_one_line.py to parse models.txt file.
#

# Python
import os
import csv
from copy import deepcopy

# Numpy
import numpy as np

# My package
from triangle_sampling.parse_models_list import parse_meta_one_line
from util.ansi_colors import ansi_colors
from triangle_sampling.config_hist_params import TriangleHistogramParams as \
  HistP


# Parameters:
#   model_metafile_path: string
#   hist_path: string. Full path to folder containing hist_conf.csv.
#     Only used if mixed_paths=False.
#   custom_catids: True or False. Whether user wants to overwrite all category
#     IDs. If specified, must specify for ALL categories, not just some.
#     This is only used by concat_hists_to_train.py, but therefore
#     parse_meta_one_line.py requires it, so just supply it.
#   use_cat_subdirs: True if triangles data are not stored in root of csv_tris,
#     but are in subdirectories named by the object category, e.g. in the case
#     of data trained by Gazebo simulated hand.
#     TODO This is now here for initial Gazebo data testing, but should be
#     removed, for more uniform folder structure. csv_hists does not do this!
#     PCD and real robot don't do this! It's just extra work!)
#   tri_suffix: '' for PCD and real robot case, '_robo' or '_hand' for Gazebo
#     simulated hand trained case.
#   mixed_paths: True if csv file contains both PCL and Gazebo data, e.g.
#     models_gazebo_csv.txt, then need to modify the meta line differently to
#     get csv histogram file name, compared to models.txt.
#   sampling_subpath, tri_nbins_subpath: Used to construct subdirectory names
#     of histogram csv files. Caller should decide whether their data mode has
#     one or both of these. Pass in empty string if the data mode doesn't
#     have one or more of them.
# Returns:
#   hists: NumPy array. Histograms loaded. nSamples x nBins
#   lbls: Python list. Assigned class ID for each sample. Length = number of
#     samples
#   catnames: Length = # cats. String names of classes
#   catcounts: Length = # cats. How many samples are in each category
#   catids: Length = # cats.
#   sample_names: Length = nSamples. Full path to individual histogram files
def load_hists (model_metafile_path, hist_path, custom_catids=False,
  tri_suffix='',
  mixed_paths=False, sampling_subpath='', tri_nbins_subpath=''):

  # Histograms of all objects
  # List of lists. Each inner list is a flattened 3D histogram of an object
  #   len() of outer list is number of objects.
  hists = []
  sample_names = []

  # Size: number of categories
  catnames = []
  catcounts = []
  catids = []

  # Size: number of samples
  lbls = []

  gz_patt = 'csv_gz_tri'
  # Use the Gazebo histograms with PCL hist_conf.csv range, not the pure
  #   Gazebo data in csv_gz_hists using Gazebo hist_conf.csv range.
  #   `.` this is for mixed mode.
  gz_patt_new = 'csv_gz_pclrange_hists'

  pcl_patt = 'csv_tri'
  pcl_patt_new = 'csv_hists'

  print ('load_hists() hist_path: %s' % hist_path)


  # Read meta list file line by line
  with open (model_metafile_path, 'rb') as metafile:

    for line in metafile:

      line = line.strip ()

      # Parse line in file, for base name and category info
      parse_result = parse_meta_one_line (line, catnames, catcounts,
        catids, custom_catids)
      if not parse_result:
        continue

      basename = parse_result [0]
      cat_idx = parse_result [1]
      lbls.append (cat_idx)

      #sample_names.append (basename)

      # Construct full path to histogram .csv file
      if not mixed_paths:
        if tri_suffix:
          basename = os.path.splitext (basename) [0] + tri_suffix + \
            os.path.splitext (basename) [1]

        hist_name = os.path.join (hist_path, catnames [cat_idx], basename)

      # For models_gazebo_csv.txt, with mixed PCL and Gazebo files.
      #   Lines specify files in csv_gz_tri or csv_tri, with full csv file name.
      #   Histogram csv files will have same basename, different dir -
      #     csv_gz_hists or csv_hists.
      else:

        #   Sample Gazebo line:
        #     triangle_sampling/csv_gz_tri/mug/6faf1f04bde838e477f883dde7397db2_2016-02-21-18-14-00_hand.csv
        #     Goal:
        #     triangle_sampling/csv_gz_hists/triParams_l0l1a0_nbins10_10_10/mug/6faf1f04bde838e477f883dde7397db2_2016-02-21-18-14-00_hand.csv
        if line.find (gz_patt) != -1:

          print ('load_hists(): This is a gazebo hist')

          hist_name = line.replace (gz_patt, gz_patt_new)

          # This gets you to the '/' after 'csv_gz_tri'
          tri_path_idx = hist_name.find (gz_patt_new) + len (gz_patt_new) + 1

          # This joins: '.../csv_gz_tri/', tri_nbins_subpath, 'object_name.csv'
          hist_name = os.path.join (hist_name [0 : tri_path_idx],
            tri_nbins_subpath,
            hist_name [tri_path_idx : len (hist_name)])
          #print (hist_name)

        #   Sample PCL line:
        #     triangle_sampling/csv_tri/nSamples10_ratio095/mug/542235fc88d22e1e3406473757712946.csv
        #     Goal:
        #     triangle_sampling/csv_hists/nSamples10_ratio095/triParams_l0l1a0_nbins10_10_10/mug/6faf1f04bde838e477f883dde7397db2_2016-02-21-18-14-00_hand.csv
        elif line.find (pcl_patt) != -1:

          print ('load_hists(): This is a PCL hist')
          #print ('  sampling_subpath: %s' % sampling_subpath)

          hist_name = line.replace (pcl_patt, pcl_patt_new)

          tri_path_idx = hist_name.find (pcl_patt_new) + len (pcl_patt_new) + 1

          # This joins: '.../csv_tri/', sampling_subpath tri_nbins_subpath,
          #   'object_name.csv'
          hist_name = os.path.join (hist_name [0 : tri_path_idx],
            sampling_subpath, tri_nbins_subpath,
            hist_name [tri_path_idx : len (hist_name)])

        else:
          print ('%sDid not find %s or %s in this line in meta file, skipping it, check your meta file: %s%s' % ( \
            ansi_colors.FAIL, gz_patt, pcl_patt, line, ansi_colors.ENDC))
          continue

        hist_name = os.path.join (hist_path, hist_name)

      #print (hist_name)

      # Full histogram path
      sample_names.append (hist_name)


      if not os.path.exists (hist_name):
        print ('%sHistogram file does not exist, did you generate it yet? Returning... Offending file: %s%s' % (\
          ansi_colors.FAIL, hist_name, ansi_colors.ENDC))
        raise IOError

      # Read individual object's histogram file
      with open (hist_name, 'rb') as hist_file:
 
        # Read csv file
        hist_reader = csv.reader (hist_file)
 
        # There's only 1 row per file, the whole histogram flattened
        row = hist_reader.next ()

        # Convert strings to floats
        hists.append ([float (row [i]) for i in range (0, len (row))])


  # Numpy is row-major. Each row is an object's histogram
  hists = np.asarray (hists)


  # Print some information
  print ('%d classes:' % (len (catnames)))
  for i in range (0, len (catnames)):
    print ('%s (%d) ..... cumulative %d so far' % \
      (catnames [i], catcounts [i], np.sum (catcounts [0 : i+1])))
  print ('')


  return (hists, lbls, catnames, catcounts, catids, sample_names)


# Load a one-line linearized histogram .csv file, reshape it into 3D histogram.
# Parameters:
#   hist_name: Full path of .csv histogram file
#   nbins: 3-tuple, e.g. (10, 10, 10)
#   bin_range3D: ((min,max), (min,max), (min,max))
# Returns hist (histdd, or hist_linear if ret_linear=True), edgesdd.
def load_one_hist (hist_name, nbins, bin_range3D, ret_linear=False):

  hist_linear = []

  # Copied from triangle_sampling/load_hists.py
  # Read individual object's histogram file
  with open (hist_name, 'rb') as hist_file:

    # Read csv file
    hist_reader = csv.reader (hist_file)

    # There's only 1 row per file, the whole histogram flattened
    row = hist_reader.next ()

    # Convert strings to floats
    hist_linear.append ([float (row [i]) for i in range (0, len (row))])

  #print (hist_linear)
  hist_linear = np.asarray (hist_linear)


  # For n bins, there should be n+1 edges
  #   List of 3 lists of (number of bins) items
  # linspace() has more careful handling of floating endpoints than arange(),
  #   says NumPy arange() documentation. linspace() also includes the endpoint.
  edgesdd = []
  edgesdd.append (np.linspace (bin_range3D[0][0], bin_range3D[0][1],
    nbins[0] + 1, endpoint=True))
  edgesdd.append (np.linspace (bin_range3D[1][0], bin_range3D[1][1],
    nbins[1] + 1, endpoint=True))
  edgesdd.append (np.linspace (bin_range3D[2][0], bin_range3D[2][1],
    nbins[2] + 1, endpoint=True))

  # 6 Dec 2016: New generalization to d-D, not just 3D. Test this when run on
  #   triangles again, probably for ICRA 2017 submission comeback
  n_dims = len (nbins)
  for d in range (0, n_dims):
    edgesdd.append (np.linspace (bin_range3D[d][0], bin_range3D[d][1],
      nbins[d] + 1, endpoint=True))

  if ret_linear:
    return hist_linear, edgesdd
  else:

    # Reshape 1D hist loaded into 3D
    # This does the right thing, as long as write_hist_3d.py
    #   write_hist_3d_csv() reshapes the 3D histogram to 1D by
    #   np.reshape(histdd, (histdd.size, )), without any other fancy parameters.
    #   That will give the default reshaping, and to reshape it back, it's
    #   just np.reshape(hist_linear, histdd.shape), without any params.
    #   histdd.shape is (10,10,10), or nbins.
    histdd = np.reshape (hist_linear, (nbins))

    return histdd, edgesdd


# Reads he histogram config file.
# Parameters:
#   hist_conf_name: Full path to .csv config file
# Returns which triangle parameters are in the 3D histogram, the number of
#   bins in each dimension, and the min and max bin ranges for each dimension.
def read_hist_config (hist_conf_name):

  print ('%sReading histogram config file %s%s' % (\
    ansi_colors.OKCYAN, hist_conf_name, ansi_colors.ENDC))

  #####
  # Load histogram config data, in hist_conf.csv, saved by
  #   sample_pcl_calc_hist.py
  #####

  # First pass, just read title line as plain text, to parse and figure out
  #   which three parameters of triangle was chosen for histograms, and in
  #   what order. (The order is why we have to do this first pass. `.` Python
  #   dictionary keys are unordered. So if just do DictReader, we won't know
  #   which was param1, which was param3, in the right order! Then histogram
  #   bins plotted here won't match the ones in training descriptors!)
  with open (hist_conf_name, 'rb') as hist_conf_file:

    # Read header line to get column titles
    line = hist_conf_file.readline ()

    # Header row format:
    #   l0_nbins,l1_nbins,a0_nbins,l0_min,l0_max,l1_min,l1_max,a0_min,a0_max

    column_titles = line.split (',')

    # Triangle params of interest, for computing 3D histograms for recognition
    # Only need to parse first 3 column titles, to see what params were chosen
    #   to produce training data.
    #   (omg this is so easy with split(), can I just say I love that I used
    #   delimiters btw every word!)
    # Example: pr1 = 'l0', pr2 = 'l1', pr3 = 'a0'
    pr1 = column_titles [0].split ('_') [0]
    pr2 = column_titles [1].split ('_') [0]
    pr3 = column_titles [2].split ('_') [0]

  print ('\nhist_conf.csv lists these three params, in this order: %s %s %s' %\
    (pr1, pr2, pr3))


  # Now read the numbers in config file
  # 3-elt lists
  nbins = []
  bin_min = []
  bin_max = []
  with open (hist_conf_name, 'rb') as hist_conf_file:
    hist_conf_reader = csv.DictReader (hist_conf_file)

    # Header row format:
    #   l0_nbins,l1_nbins,a0_nbins,l0_min,l0_max,l1_min,l1_max,a0_min,a0_max

    # Only one row in file
    for row in hist_conf_reader:

      nbins.append (float (row [pr1 + '_nbins']))
      nbins.append (float (row [pr2 + '_nbins']))
      nbins.append (float (row [pr3 + '_nbins']))

      bin_min.append (float (row [pr1 + '_min']))
      bin_min.append (float (row [pr2 + '_min']))
      bin_min.append (float (row [pr3 + '_min']))

      bin_max.append (float (row [pr1 + '_max']))
      bin_max.append (float (row [pr2 + '_max']))
      bin_max.append (float (row [pr3 + '_max']))

  # Define param to pass to np.histogramdd()
  # 3 2-tuples
  #   Copied from sample_pcl_calc_hist.py
  # Don't call scale_bin_range_to_decimeter(), let caller call it themselves,
  #   `.` not every caller would want this scaled! e.g. triangles_svm.py wants
  #   it scaled, triangles_on_robot_to_hists.py doesn't, because latter
  #   loads hist_conf.csv that's already scaled, outputted from
  #   sample_pcl_calc_hist.py .
  bin_range3D = [(bin_min[i], bin_max[i]) for i in range (0, len (bin_min))]


  print ('[%d, %d, %d] bins for the 3 dimensions' % \
    (nbins[0], nbins[1], nbins[2]))
  print ('Bin mins (meters and radians):')
  print (bin_min)
  print ('Bin maxes:')
  print (bin_max)

  return ((pr1, pr2, pr3), nbins, bin_range3D)


# Parameters:
#   prs: List of 3 strings, the 3 chosen parameters. e.g. 'l0', 'l1', 'a0'.
#     If you use something different from the set defined in HistP, then pass
#     in your own.
def scale_bin_range_to_decimeter (bin_range=None, bin_range3D=None,
  prs=HistP.PRS):

  pr1, pr2, pr3 = prs

  bin_range_rv = None
  if bin_range:

    bin_range_rv = deepcopy (bin_range)
 
    # Know the L# ones are lengths
    bin_range_rv [HistP.L0_IDX] = (bin_range [HistP.L0_IDX][0] * 10, 
      bin_range [HistP.L0_IDX][1] * 10)
    bin_range_rv [HistP.L1_IDX] = (bin_range [HistP.L1_IDX][0] * 10, 
      bin_range [HistP.L1_IDX][1] * 10)
    bin_range_rv [HistP.L2_IDX] = (bin_range [HistP.L2_IDX][0] * 10, 
      bin_range [HistP.L2_IDX][1] * 10)


  bin_range3D_rv = None
  if bin_range3D:

    bin_range3D_rv = deepcopy (bin_range3D)
 
    # 1st triangle param
    if pr1 == HistP.L0 or pr1 == HistP.L1 or pr1 == HistP.L2:
      # Rescale by *10
      bin_range3D_rv [0] = (bin_range3D [0][0] * 10, bin_range3D [0][1] * 10)
 
    # 2nd triangle param
    if pr2 == HistP.L0 or pr2 == HistP.L1 or pr2 == HistP.L2:
      # Rescale by *10
      bin_range3D_rv [1] = (bin_range3D [1][0] * 10, bin_range3D [1][1] * 10)
 
    # 3rd triangle param
    if pr3 == HistP.L0 or pr3 == HistP.L1 or pr3 == HistP.L2:
      # Rescale by *10
      bin_range3D_rv [2] = (bin_range3D [2][0] * 10, bin_range3D [2][1] * 10)

  return bin_range_rv, bin_range3D_rv

