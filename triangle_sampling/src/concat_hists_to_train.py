#!/usr/bin/env python

# Mabel Zhang
# 24 Aug 2015
#
# Combine all objects' individual histogram .csv files into one big .csv file,
#   for passing to convert_csv_to_libsvm.py in tactile_collect package,
#   which converts them to pass to libsvm.
#
# This script creates numeric object class labels to prepend to each row of
#   object histogram. `.` libsvm wants labels in first column.
#
# Usage:
#   For default, if your meta list is in models.txt:
#   $ rosrun triangle_sampling concat_hists_to_train.py
#
#   If your meta list file is in some other file in config/ dir:
#   $ rosrun triangle_sampling concat_hists_to_train.py [metafile_name]
#       [catID1 catID2 ...]
#
# Example:
#   For train 3dnet (167 objs) and test archive3d (25 objs), total 192 objs:
#   $ rosrun triangle_sampling concat_hists_to_train.py models_3dnet.txt
#   $ rosrun triangle_sampling concat_hists_to_train.py models_archive3d_long.txt 6 2 5
#


# ROS
import rospkg

# Python
import csv
import os
import sys

# Local
from tactile_collect import tactile_config
from triangle_sampling.parse_models_list import parse_meta_one_line


# Parameters:
#   catlist: Python list of strings. Maybe be altered. String name of cateogry,
#     e.g. 'cup'.
#   catcounts: Python list of ints. May be altered. Number of samples in each
#     category. e.g. 10 cup samples.
#   catids: Python list of ints. May be altered. Assignment of official class
#     ID to be used in recognition. e.g. cup is class ID number 1.
# Returns True on success, False if line is empty or commented - then caller
#   can just skip this line in file.
def parse_meta_one_line (line, catlist, catcounts, catids, custom_catids):

  # If this line is empty or commented, ignore it
  if not line:
    return False
  elif line.startswith ('#'):
    return False


  # Get base name of file, drop extension
  basename = os.path.basename (line)
  basename = os.path.splitext (basename) [0]

  # Append .csv extension
  basename = basename + '.csv'


  # Find the category name in the path string

  # Drop base name
  catname, _ = os.path.split (line)
  # Grab last dir name. This won't contain a slash, guaranteed by split()
  _, catname = os.path.split (catname)

  # This indexes catlist and catids
  cat_idx = -1

  # If category not in list yet, append it
  if catname not in catlist:
    catlist.append (catname)
    cat_idx = len (catlist) - 1

    catcounts.append (1)

    if not custom_catids:
      catids.append (cat_idx)

  else:
    cat_idx = catlist.index (catname)

    catcounts [cat_idx] += 1;

  return True


def concat (args):

  # If no args specified, use defaults
  if not args:
    metafile_name = 'models.txt'
    out_name = 'hists_all.csv'

  # If at least one arg specified, use first arg as custom meta file name
  else:
    # Take the first string in list ONLY
    metafile_name = args [0]

    # Use the model file name, swap extension to .csv, for big csv file
    out_name = os.path.splitext (metafile_name) [0] + '.csv'


  # If more than 1 args are specified, take the remaining as custom cateogry
  #   IDs.
  custom_catids = False
  catids = list ()
  if len (args) > 1:

    custom_catids = True

    # Take the remaining args as custom class IDs. Convert strings to ints
    catids = [int(i) for i in args [1:]]

  print ('User specified %d custom category IDs:' % len(catids))
  print (catids)
  print ('')



  # Input and output path
  # This is same as sample_pcl_calc_hist.py
  hist_path = tactile_config.config_paths ('custom',
    'triangle_sampling/csv_hists/')
    #'triangle_sampling/csv_hists_1d/')

  # These are same as sample_pcl.cpp
  rospack = rospkg.RosPack ()
  pkg_path = rospack.get_path ('triangle_sampling')
  model_metafile_path = os.path.join (pkg_path, 'config/', metafile_name)

  # To hold names of all categories.
  # Indices to this list are category IDs to be put in new big .csv file
  catlist = list ()
  catcounts = list ()


  # Open output big .csv file
  out_path = os.path.join (hist_path, out_name)
  out_file = open (out_path, 'wb')
  out_writer = csv.writer (out_file)

  print ('Inputting individual histogram object names and class strings ' + \
    'from %s' % model_metafile_path)
  print ('')
  print ('Outputting all histograms, with class labels prepended, to %s' %\
    out_path)
  print ('')


  # Read meta list file line by line
  with open (model_metafile_path, 'rb') as metafile:

    # http://stackoverflow.com/questions/15599639/whats-perfect-counterpart-in-python-for-while-not-eof
    #while True:
    #  line = metafile.readline ()
    #  if not line:
    #    break


    for line in metafile:

      # Parse line in file, for base name and category info
      parse_result = parse_meta_one_line (line, catlist, catcounts,
        catids, custom_catids)
      if not parse_result:
        continue

      basename = parse_result [0]
      cat_idx = parse_result [1]

      # Sanity check
      if len (catids) < len (catlist):
        print ('concat_hists_to_train.py ERROR: Not enough custom category IDs specified! Saw more category names (%d) than custom category IDs (%d). Specify more category IDs and rerun.' % (len(catlist), len(catids)))
        break
 
 
      # Open individual object's histogram file
      with open (os.path.join (hist_path, basename), 'rb') as hist_file:
 
        # Read csv file
        hist_reader = csv.reader (hist_file)
 
        # There's only 1 row per file, the whole histogram flattened
        row = hist_reader.next ()
 
        # Prepend object class ID to front of list
        row.insert (0, catids [cat_idx])
 
        # Write to big .csv file
        out_writer.writerow (row)

  out_file.close ()


  for i in range (0, len (catlist)):
    print ('Object class %d: %15s (%d)' % (catids[i], catlist[i], catcounts[i]))
  print ('Total %d objects' % sum(catcounts))


if __name__ == '__main__':

  # Using a colon would pass in [], even if no args are specified, whereas
  #   without the colon, argv[1] would crash if there isn't a argv[1].
  concat (sys.argv[1:])

