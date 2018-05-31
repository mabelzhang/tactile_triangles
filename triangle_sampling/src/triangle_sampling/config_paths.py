#!/usr/bin/env python

# Mabel Zhang
# 25 Jan 2016
#
# Unified place to define paths
#

# ROS
import rospkg

# Python
import os
import inspect
import argparse

# My packages
from tactile_collect import tactile_config
from util.ansi_colors import ansi_colors


# =================================================================== Paths ==

def get_recog_meta_path ():

  meta_path = os.path.join (rospkg.RosPack ().get_path ('triangle_sampling'),
    'config')
  return meta_path


# Used by active_touch mcts_mdp.py
def get_recog_confmat_path ():
  return tactile_config.config_paths ('custom',
    'triangle_sampling/imgs/conf_mats/')


# Used by active_touch active_predict.py
def get_active_meta_path ():

  meta_path = os.path.join (rospkg.RosPack ().get_path ('active_touch'),
    'config')
  return meta_path


# Applicable to point cloud data from sample_pcl.cpp
def get_sampling_subpath (nSamples, nSamplesRatio, endSlash=True):

  # Construct subdirectory name
  nSamplesRatioStr = format ('%.2f' % nSamplesRatio)

  # Eliminate periods in floating point number
  # Ref: http://www.tutorialspoint.com/python/string_replace.htm
  nSamplesRatioStr = nSamplesRatioStr.replace ('.', '')

  subdir_name = format ('nSamples%d_ratio%s' % (nSamples,
    nSamplesRatioStr))

  if endSlash:
    subdir_name += '/'

  return subdir_name


# PCD path used by real robot sampling
# Used by triangles_collect.py
def get_pcd_path (csv_suffix):

  pcd_path = tactile_config.config_paths ('custom',
    'triangle_sampling/pcd_' + csv_suffix + 'collected/')
  return pcd_path


# Triangles path used by real robot sampling
# Used by triangles_collect.py
def get_robot_tri_path (csv_suffix):

  tri_path = tactile_config.config_paths ('custom',
    'triangle_sampling/csv_' + csv_suffix + 'tri/')
  return tri_path


# Object parameters path used by real robot sampling
# Used by triangles_collect_semiauto.py
def get_robot_obj_params_path (csv_suffix):

  obj_params_path = tactile_config.config_paths ('custom',
    'triangle_sampling/pcd_' + csv_suffix + 'collected_params/')
  return obj_params_path


def get_robot_hists_path (csv_suffix):

  hist_path = tactile_config.config_paths ('custom',
    'triangle_sampling/csv_' + csv_suffix +  'hists/')
  return hist_path


# Parameters:
#   prs_str: String of three chosen triangle parameters, no spaces,
#     e.g. 'l0,l1,a0'
#   bins3D: String of three number of bins, no spaces, e.g. '10,10,10'
# Return configurated folder name, and list of 3 integers converted from
#   bins3D string.
def get_triparams_nbins_subpath (prs_str, bins3D, endSlash=True):

  # Get a list of the three triangle param names, from the comma-separated
  #   string
  paramsStr = prs_str.split (',')
  # Strip any spaces
  paramsStr = [p.strip() for p in paramsStr]
  # Concat list of strings into one string
  paramsStr = ''.join (paramsStr)

  # Get rid of spaces in between, in case there are any
  nbinsStr = bins3D.split (',')
  # Strip any spaces
  nbinsStr = [b.strip() for b in nbinsStr]
  # Store the list of 3 integers, for returning
  nbins = [int(b) for b in nbinsStr]
  # Concat list of strings into one string, separated by underscore
  nbinsStr = '_'.join (nbinsStr)
  
  subdir_name = 'triParams_%s_nbins%s' % (paramsStr, nbinsStr)

  if endSlash:
    subdir_name += '/'

  return subdir_name, nbins


def get_active_data_root ():
  return tactile_config.config_paths ('custom',
    'active_triangle/')


# mode_suffix: _gz for Gazebo, _pcd for PCL, _bx for Baxter.
#   (Should use csv_suffix, but I don't like that it's '' for PCL. I want to
#   start qualifying pcl folders with a _pcd suffix. So using mode_suffix
#   instead.)
def get_probs_root (mode_suffix, nxn_storage=False):

  # Get rid of the trailing underscore
  # Empty string will simply split to ['']
  #mode_suffix = csv_suffix.split ('_') [0]

  # 8 Aug 2016: I don't even use costs pkl for nxn storage anymore, just
  #   calculating action costs at run-time now. When ready, remove costs_root.
  # Only nxn storage stores costs
  if nxn_storage:
    costs_root = tactile_config.config_paths ('custom',
      'active_triangle/costs' + mode_suffix + '/')
  else:
    costs_root = ''

  probs_root = tactile_config.config_paths ('custom',
    'active_triangle/probs' + mode_suffix + '/')

  return costs_root, probs_root


# mode_suffix: _noRobot for no robot, _gz for Gazebo ReFlex teleport,
#   _bx for Baxter.
# Used by active_touch active_predict.py
def get_active_log_path (mode_suffix):
  return tactile_config.config_paths ('custom',
    'active_triangle/tree_exec' + mode_suffix + '/')

# Used by active_touch mcts_mdp.py
def get_active_img_path ():
  return tactile_config.config_paths ('custom', 'active_triangle/imgs')

# Used by active_touch fig_recog_acc.py
def get_active_img_recog_path ():
  return tactile_config.config_paths ('custom',
    'active_triangle/imgs/recog')


# Parameters:
#   bins3D: List of comma-separated integers, no spaces, e.g. '10,10,10'
# Returns a list of integers
def get_ints_from_comma_string (bins3D):

  nbinsStr = bins3D.split (',')
  # Strip any spaces
  nbinsStr = [b.strip() for b in nbinsStr]
  # Store the list of 3 integers, for returning
  nbins = [int(b) for b in nbinsStr]
 
  return nbins


# A more generic version of get_ints_from_comma_string(), can be used for
#   whatever.
# Used by plot_hist_rviz.py to take user args to shift RViz plots by some amt.
# Parameters:
#   bins3D: List of comma-separated floats, no spaces, e.g. '0.1,0.2,0.3'
# Returns a list of floats
def get_floats_from_comma_string (commaStr):

  strMod = commaStr.split (',')
  # Strip any spaces
  strMod = [s.strip() for s in strMod]
  # Store the list of floats, for returning
  flist = [float(s) for s in strMod]

  return flist


# Path to output images
def get_img_path (subdir_name=''):

  # Save to file and plot
  # Ref: http://stackoverflow.com/questions/50499/in-python-how-do-i-get-the-path-and-name-of-the-file-that-is-currently-executin
  #imgpath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
  # Ref: https://docs.python.org/2/library/os.path.html
  #imgpath = os.path.join (imgpath,
  #  './../../../../../../train/triangle_sampling/imgs/')

  rospack = rospkg.RosPack ()
  pkg_path = rospack.get_path ('triangle_sampling')
  imgpath = os.path.join (pkg_path,
    './../../../../train/triangle_sampling/imgs/')

  if subdir_name:
    imgpath = os.path.join (imgpath, subdir_name)

  imgpath = os.path.realpath (imgpath)
  # http://stackoverflow.com/questions/273192/in-python-check-if-a-directory-exists-and-create-it-if-necessary
  if not os.path.exists (imgpath):
    os.makedirs (imgpath)

  return imgpath


# Constructs csv name
# Parameters:
#   hist_path: Full path to csv_*hists folder
#   mode_suff: '_gz' or '_pcd', followed by tri_nbins_subpath or
#     sampling_subpath, respectively for Gazebo or PCL.
#   tri_params: String of 3 chosen triangle params, used for csv file suffix,
#     e.g. 'l0,l1,a0'
# Returns e.g.
#   <hist_path>/csv_gz_hists/_stats/nbins_vs_acc_l0l1a0.csv for Gazebo,
#   <hist_path>/csv_hists/nSamples300_ratio095/_stats/nbins_vs_acc_pcd_l0l1a0.csv
#     for PCL.
def get_nbins_acc_stats_name (hist_path, mode_suff, tri_params):

  suffix = ''
  if tri_params:
    suffix = tri_params.split (',')
    # Ref: Join a list into a string
    #   http://stackoverflow.com/questions/12453580/concatenate-item-in-list-to-strings-python
    suffix = ''.join (suffix)
    # Prepend underscore
    suffix = '_' + suffix

  # Put stats in subdirectory of histogram dir
  stats_path = os.path.join (hist_path, '_stats')
  # Is this a better name?
  #stats_path = os.path.join (hist_path, 'nbins_vs_acc')
  if not os.path.exists (stats_path):
    os.makedirs (stats_path)

  stats_name = os.path.join (stats_path, 'nbins_vs_acc' + mode_suff + \
    suffix + '.csv')
  return stats_name


def get_svm_model_path (hist_path, img_mode_suffix):

  svm_name = os.path.join (hist_path, 'svm' + img_mode_suffix + \
    '.joblib.pkl')
  svm_lbls_name = os.path.join (hist_path, 'svm' + img_mode_suffix + \
    '_lbls.pkl')

  return svm_name, svm_lbls_name


# =================================================== Parsing cmd line args ==

# Copied from triangles_svm.py TODO refactor prune.py to use this function
# Help parse params passed into argparse, because many files accept these
#   same params
# Call this function like this, after you get the params from argparse:
#   parse_subpath_params ([args.histSubdirParam1] + args.histSubdirParam2,
#     args.pcd, args.real, args.gazebo)
def parse_subpath_params (histSubdirParams, pcd=False, real=False, gz=False,
  mixed=False):

  # This path doesn't apply to Gazebo data
  sampling_subpath = ''
  # Cmd line args look like (see README_gazebo for latest):
  #   --pcd 10 0.95 l0,l1,a0 10,10,10
  #   --mixed 10 0.95 l0,l1,a0 10,10,10
  if pcd or mixed:
    nSamples = int (histSubdirParams [0])
    nSamplesRatio = float (histSubdirParams [1])
    sampling_subpath = get_sampling_subpath (nSamples, nSamplesRatio,
      endSlash=False)
    print ('%sAccessing directory with nSamples %d, nSamplesRatio %f%s' % \
      (ansi_colors.OKCYAN, nSamples, nSamplesRatio, ansi_colors.ENDC))

    # Index at which mandatory cmd line arg is prs_str specified
    TRI_IDX = 2
    # Index at which mandatory cmd line arg is bins3d string specified
    NBINS_IDX = 3

  # Cmd line args look like: --gazebo l0,l1,a0 10,10,10
  #   (see README_gazebo for latest)
  elif gz or real:

    TRI_IDX = 0
    NBINS_IDX = 1

  tri_nbins_subpath, _ = get_triparams_nbins_subpath (
    histSubdirParams [TRI_IDX], histSubdirParams [NBINS_IDX], endSlash=False)

  return sampling_subpath, tri_nbins_subpath, histSubdirParams [TRI_IDX]


# Returns the parsed args, and boolean for whether parse was valid, i.e. we
#   did not encounter errors.
def parse_args_for_svm (arg_parser=None):

  #####
  # Parse command line args
  #   Ref: Tutorial https://docs.python.org/2/howto/argparse.html
  #        Full API https://docs.python.org/dev/library/argparse.html
  #####

  if not arg_parser:
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
      'For Gazebo, number of bins in 3D histogram, with no spaces, e.g. 10,10,10\n' + \
      'For mixed, enter 2 point cloud params first, then 2 Gazebo params, 4 total.')

  arg_parser.add_argument ('--pcd', action='store_true', default=False,
    help='Boolean flag, no args. Run on synthetic data in csv_tri_lists/ from point cloud')
  arg_parser.add_argument ('--real', action='store_true', default=False,
    help=format ('Boolean flag. Run on real-robot data. Will hold out indices after idx_start_real_robot (0 based) from random train/test split, to use only as test data, because meta file is structured with PCD data first, then real robot data.'))
  arg_parser.add_argument ('--gazebo', action='store_true', default=False,
    help='Append category subdirectory names after csv_tris directory, to access individual object files.')
  arg_parser.add_argument ('--mixed', action='store_true', default=False,
    help='Boolean flag, no args. --meta_train and --meta_test required. Run on a mix of PCL, Gazebo, etc data. This assumes you will specify long csv paths (--long_csv_path in some other scripts) in the meta file! We will read starting from train/ directory, everything after must be specified in meta file. If a line is for csv_gz_tri, histogram will be outputted to csv_gz_pclrange_hists; csv_pcl_tri would get output in csv_pcl_hists.')

  arg_parser.add_argument ('--meta', type=str, default='models.txt',
    help='String. Base name of meta list file in triangle_sampling/config directory')
  arg_parser.add_argument ('--meta_train', type=str, default='models.txt',
    help='String. Only used if --mixed is specified; --meta is ignored. Meta file for training, e.g. PCL data.')
  arg_parser.add_argument ('--meta_test', type=str, default='models_gazebo_csv.txt',
    help='String. Only used if --mixed is specified; --meta is ignored. Meta file for testing, e.g. Gazebo data.')

  arg_parser.add_argument ('--rand_splits', action='store_true', default=False,
    help='')

  arg_parser.add_argument ('--kde', action='store_true', default=False,
    help='Boolean flag. Run on KDE smoothed histograms in csv_kde (instead of default csv_hists original histograms)')

  # Set to True to upload to ICRA. (You can't view the plot in OS X Preview)
  # Set to False if want to see the plot for debugging.
  arg_parser.add_argument ('--truetype', action='store_true', default=False,
    help='Tell matplotlib to generate TrueType 42 font, instead of rasterized Type 3 font. Specify this flag for uploading to ICRA.')
  arg_parser.add_argument ('--notitle', action='store_true', default=False,
    help='Do not plot titles, for paper figures, description should all be in caption.')

  arg_parser.add_argument ('--write_stats', action='store_true', default=False,
    help='Write average accuracy to csv file.')
  arg_parser.add_argument ('--overwrite_stats', action='store_true',
    default=False, help='Overwrite accuracy file. Only in use if --write_stats is also specified.')

  args = arg_parser.parse_args ()


  # Sanity checks
  # More than one Boolean is True
  if args.pcd + args.real + args.gazebo + args.mixed > 1:
    print ('ERROR: More than one of --pcd, --real, --gazebo, and --mixed were specified. You must choose one. Terminating...')
    return args, False

  # No Boolean is True
  elif args.pcd + args.real + args.gazebo + args.mixed == 0:
    print ('%sERROR: Neither --pcd, --real, or --gazebo were specified. You must choose one. Terminating...%s' % ( \
      ansi_colors.FAIL, ansi_colors.ENDC))
    return args, False


  return args, True


# Configures paths based on args returned from parse_args_for_svm()
# Parameters:
#   args: Returned from parse_args_for_svm()
#   mode_append: If True, img_mode_suffix will start with '_'.
#     Else, img_mode_suffix will end with '_'.
def config_hist_paths_from_args (args, mode_append=True):

  if args.pcd:

    sampling_subpath, tri_nbins_subpath, tri_paramStr = parse_subpath_params (
      [args.histSubdirParam1] + args.histSubdirParam2,
      args.pcd, args.real, args.gazebo, args.mixed)

    if args.kde:
      hist_subpath = 'csv_kde'
    else:
      hist_subpath = 'csv_hists'
    # Directory in which triParams_* folders are found
    hist_parent_path = tactile_config.config_paths ('custom',
      os.path.join ('triangle_sampling', hist_subpath, sampling_subpath))
    # Directory in which hist_conf.csv can be found
    hist_path = os.path.join (hist_parent_path, tri_nbins_subpath)

    if args.pcd:
      img_mode_suffix = '_pcd'
      tri_suffix = ''

  elif args.gazebo or args.real:

    # Doesn't exist for Gazebo files
    sampling_subpath = ''

    # e.g. triParams_l0l1a0_nbins10_10_10
    _, tri_nbins_subpath, tri_paramStr = parse_subpath_params (
      [args.histSubdirParam1] + args.histSubdirParam2,
      args.pcd, args.real, args.gazebo, args.mixed)

    if args.gazebo:
      hist_subpath = 'csv_gz_hists'
      img_mode_suffix = '_gz'

      # Meta file should specify full path, some may have timestamps
      tri_suffix = ''

    elif args.real:
      hist_subpath = 'csv_bx_hists'
      if args.iros2016:
        hist_subpath = os.path.join (hist_subpath, 'iros2016')

      img_mode_suffix = '_bx'
      tri_suffix = ''  # '_robo'

    # Directory in which triParams_* folders are found
    hist_parent_path = tactile_config.config_paths ('custom',
      os.path.join ('triangle_sampling', hist_subpath))
    # Directory in which hist_conf.csv can be found
    hist_path = os.path.join (hist_parent_path, tri_nbins_subpath)

  elif args.mixed:

    sampling_subpath, tri_nbins_subpath, tri_paramStr = parse_subpath_params (
      [args.histSubdirParam1] + args.histSubdirParam2,
      args.pcd, args.real, args.gazebo, args.mixed)

    # Use hist_conf.csv from PCL data
    hist_subpath = 'csv_hists'
    # Directory in which triParams_* folders are found
    hist_parent_path = tactile_config.config_paths ('custom',
      os.path.join ('triangle_sampling', hist_subpath, sampling_subpath))
    # Directory in which hist_conf.csv can be found
    hist_path = os.path.join (hist_parent_path, tri_nbins_subpath)

    img_mode_suffix = '_mixed'

    tri_suffix = ''


  # Move the underscore '_' from beginning to end
  if not mode_append:
    img_mode_suffix = img_mode_suffix [1:] + '_'

  return sampling_subpath, tri_nbins_subpath, hist_parent_path, hist_path,\
    img_mode_suffix, tri_suffix, tri_paramStr

