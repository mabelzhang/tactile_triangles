#!/usr/bin/env python

# Mabel Zhang
# 19 Feb 2016
#
# Renamed the specified set of .pcd, csv_gz_tri triangle _hand.csv and
#   _robo.csv files, and pcd_gz_collected_params .csv files, outputted by
#   sample_gazebo.py.
#
# The set of files is specified by their timestamp name. Their original names
#   should look like this:
#   ~/graspingRepo/train/triangle_sampling/pcd_gz_collected/2016-02-19-19-14-00.pcd
#   ~/graspingRepo/train/triangle_sampling/csv_gz_tri/2016-02-19-19-14-00_hand.csv
#   ~/graspingRepo/train/triangle_sampling/csv_gz_tri/2016-02-19-19-14-00_robo.csv
#   ~/graspingRepo/train/triangle_sampling/pcd_gz_collected_params/2016-02-19-19-14-00.csv
#
#   For this example, user should specify 2016-02-19-19-14-00 as input.
#   This script automatically finds the directories, using config_paths.py file.
#
# The specified output prefix will be prepended to the timestamp names. e.g.
#   118644ba80aa5048ac59fb466121cd17_2016-02-19-23-19-33.csv
#
# Example usage:
#   Put the timestamped file into subdirectory apple, with the new permanent
#     object name:
#   $ python move_gazebo_files.py --gazebo 2016-02-19-23-19-33 apple 118644ba80aa5048ac59fb466121cd17
#
#   Put the file into subdirectory teapot, without changing file name (pass
#     in empty string for out_prefix):
#   $ python move_gazebo_files.py --gazebo teapot_faf62b14_2016-02-20-01-12-57 teapot ""
#

# Python
import os
import argparse
import shutil

# My packages
from util.ansi_colors import ansi_colors
from triangle_sampling.config_paths import get_pcd_path, \
  get_robot_tri_path, get_robot_obj_params_path, get_probs_root


def main ():

  arg_parser = argparse.ArgumentParser ()

  arg_parser.add_argument ('in_prefix', type=str,
    help='Timestamp prefix of file')
  arg_parser.add_argument ('out_category', type=str,
    help='Subfolder to put the output file in. This should be named by object category')
  arg_parser.add_argument ('out_prefix', type=str,
    help='Prefix of output file, to pre-pend to timestamp file name')

  # I don't like this naming scheme. Not using this. Just append suffix in
  #   your out_prefix arg. I like my _robo and _hand as the final suffixes of
  #   the triangle csv files, so don't put anything after that.
  #arg_parser.add_argument ('--out_suffix', type=str, default='',
  #  help='(Optional) suffix of output file, after the timestamp name. A preceding underscore will be added for you.')

  arg_parser.add_argument ('--gazebo', action='store_true', default=False,
    help='Boolean flag, no args. Specify this for Gazebo files. Move files in csv_gz_tri, pcd_gz_collected, pcd_gz_collected_params.')
  arg_parser.add_argument ('--real', action='store_true', default=False,
    help='Boolean flag, no args. Specify this for real-robot files. Move files in csv_tri, pcd_collected, pcd_collected_params.')
  arg_parser.add_argument ('--pcd', action='store_true', default=False,
    help='Boolean flag, no args. Run on synthetic data in csv_tri_lists/ from point cloud')

  arg_parser.add_argument ('--testtime', action='store_true', default=False,
    help='Test-time files do not store probs .pkl files, so will not look for them if this flag is specified.')

  args = arg_parser.parse_args ()

  # Default to Gazebo files
  if args.gazebo + args.real > 1:
    print ('More than one of --gazebo, --real are specified. Can only pick one. Check your args and retry. Terminating...')
    return
  if args.gazebo:
    csv_suffix = 'gz_'
    mode_suffix = '_gz'
  elif args.real:
    csv_suffix = 'bx_'
    mode_suffix = '_bx'
  elif args.pcd:
    mode_suffix = '_pcd'

  in_prefix = args.in_prefix
  out_subdir = args.out_category
  # Allow empty prefix, if user just wants to move input file into a folder
  #   without changing file name.
  out_prefix = args.out_prefix
  if out_prefix:
    out_prefix = out_prefix + '_'

  #out_suffix = args.out_suffix
  #if out_suffix:
  #  out_suffix = '_' + out_suffix


  #####
  # Set up paths
  #####

  if args.gazebo or args.real:
    pcd_path = get_pcd_path (csv_suffix)
    tri_path = get_robot_tri_path (csv_suffix)
    obj_params_path = get_robot_obj_params_path (csv_suffix)
  costs_root, probs_root = get_probs_root (mode_suffix)
 
  # Input file basenames
  if args.gazebo or args.real:
    pcd_base = in_prefix + '.pcd'
    trih_base = in_prefix + '_hand.csv'
    trir_base = in_prefix + '_robo.csv'
    obj_params_base = in_prefix + '.csv'
    per_move_base = in_prefix + '_per_move.csv'
    #costs_base = in_prefix + '.pkl'
  probs_base = in_prefix + '.pkl'

  # Input file full paths
  if args.gazebo or args.real:
    pcd_in = os.path.join (pcd_path, pcd_base)
    trih_in = os.path.join (tri_path, trih_base)
    trir_in = os.path.join (tri_path, trir_base)
    obj_params_in = os.path.join (obj_params_path, obj_params_base)
    per_move_in = os.path.join (obj_params_path, per_move_base)
    #costs_in = os.path.join (costs_root, costs_base)
  probs_in = os.path.join (probs_root, probs_base)

  # Output file full paths
  if args.gazebo or args.real:
    pcd_out = os.path.join (pcd_path, out_subdir, out_prefix + \
      pcd_base)
    trih_out = os.path.join (tri_path, out_subdir, out_prefix + \
      trih_base)
    trir_out = os.path.join (tri_path, out_subdir, out_prefix + \
      trir_base)
    obj_params_out = os.path.join (obj_params_path, out_subdir,
      out_prefix + obj_params_base)
    per_move_out = os.path.join (obj_params_path, out_subdir,
      out_prefix + per_move_base)
    #costs_out = os.path.join (costs_root, out_subdir, out_prefix + costs_base)
  probs_out = os.path.join (probs_root, out_subdir, out_prefix + probs_base)


  # Check if input files exist
  if args.gazebo or args.real:
    if not os.path.exists (pcd_in):
      print ('%sInput pcd file %s does not exist. Check your input. Terminating...%s' % ( \
        ansi_colors.FAIL, pcd_in, ansi_colors.ENDC))
      return
 
    if not os.path.exists (trih_in):
      print ('%sInput triangle hand frame file %s does not exist. Check your input. Terminating...%s' % ( \
        ansi_colors.FAIL, trih_in, ansi_colors.ENDC))
      return
 
    if not os.path.exists (trir_in):
      print ('%sInput triangle robot frame file %s does not exist. Check your input. Terminating...%s' % ( \
        ansi_colors.FAIL, trir_in, ansi_colors.ENDC))
      return
 
    if not os.path.exists (obj_params_in):
      print ('%sInput object parameter file %s does not exist. Check your input. Terminating...%s' % ( \
        ansi_colors.FAIL, obj_params_in, ansi_colors.ENDC))
      return
 
    if not os.path.exists (per_move_in):
      print ('%sInput per-move file %s does not exist. Check your input. Terminating...%s' % ( \
        ansi_colors.FAIL, per_move_in, ansi_colors.ENDC))
      return
 
    #if not os.path.exists (costs_in):
    #  print ('%sInput costs file %s does not exist. Check your input. Terminating...%s' % ( \
    #    ansi_colors.FAIL, costs_in, ansi_colors.ENDC))
    #  return

  if args.testtime:
    print ('ATTENTION: --testtime specified, will NOT look for probabilities files. Check this is what you want!!!')
  else:
    if not os.path.exists (probs_in):
      print ('%sInput probabilities file %s does not exist. Check your input. Terminating...%s' % ( \
        ansi_colors.FAIL, probs_in, ansi_colors.ENDC))
      return


  # Move the files

  print ('Will be renaming these files:')
  if args.gazebo or args.real:
    print ('  %s' % pcd_in)
    print ('    to %s' % pcd_out)
    print ('  %s' % trih_in)
    print ('    to %s' % trih_out)
    print ('  %s' % trir_in)
    print ('    to %s' % trir_out)
    print ('  %s' % obj_params_in)
    print ('    to %s' % obj_params_out)
    print ('  %s' % per_move_in)
    print ('    to %s' % per_move_out)
    #print ('  %s' % costs_in)
    #print ('    to %s' % costs_out)
  if not args.testtime:
    print ('  %s' % probs_in)
    print ('    to %s' % probs_out)

  uinput = raw_input ('Do you want to proceed with this renaming (Y/N)? ')
  if uinput.lower () == 'y':

    if args.gazebo or args.real:
      if not os.path.exists (os.path.dirname (pcd_out)):
        os.makedirs (os.path.dirname (pcd_out))
      if not os.path.exists (os.path.dirname (trih_out)):
        os.makedirs (os.path.dirname (trih_out))
      if not os.path.exists (os.path.dirname (trir_out)):
        os.makedirs (os.path.dirname (trir_out))
      if not os.path.exists (os.path.dirname (obj_params_out)):
        os.makedirs (os.path.dirname (obj_params_out))
      if not os.path.exists (os.path.dirname (per_move_out)):
        os.makedirs (os.path.dirname (per_move_out))
      #if not os.path.exists (os.path.dirname (costs_out)):
      #  os.makedirs (os.path.dirname (costs_out))

      shutil.move (pcd_in, pcd_out)
      shutil.move (trih_in, trih_out)
      shutil.move (trir_in, trir_out)
      shutil.move (obj_params_in, obj_params_out)
      shutil.move (per_move_in, per_move_out)
      #shutil.move (costs_in, costs_out)

    if not args.testtime:
      if not os.path.exists (os.path.dirname (probs_out)):
        os.makedirs (os.path.dirname (probs_out))
      shutil.move (probs_in, probs_out)

    print ('Files moved.')

  # Do nothing if user didn't enter y
  else:
    print ('No files will be changed.')
    return


if __name__ == '__main__':
  main ()

