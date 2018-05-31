#!/usr/bin/env python

# Mabel Zhang
# 4 Sep 2015
#
# Refactored from concat_hists_to_train.py.
#
# Parses one line in config/models.txt (or files formatted the same way).
#   Each line in the file is some kind of relative path,with the object name
#   in the basename of file, category name in the directory name immediately
#   before basename.
# Returns the basename (object name string) and the category index of this
#   object.
# Add a count to the catcounts[] for this class;
#   appends a new index to catids and appends new string to catnames, if this
#   category name has not been seen before.
#

# Python
import os


# Read meta file, return list of lines in file, with empty lines and comment
#   lines removed, endline characters stripped.
#   Copied from triangles_reader.py
def read_meta_file (meta_list_name):

  meta_list_file = open (meta_list_name)

  lines = []

  # Read meta list file line by line
  for line in meta_list_file:

    #if rospy.is_shutdown ():
    #  break

    # Skip empty lines
    if not line:
      continue

    # Strip endline char
    # Ref: https://docs.python.org/2/library/string.html
    line = line.rstrip ()

    # Skip comment lines
    if line.startswith ('#') or line == '':
      continue


    #print ('\n%s' % line)

    lines.append (line)

  return lines


# Parameters:
#   catnames: Python list of strings. Maybe be altered. String name of cateogry,
#     e.g. 'cup'.
#   catcounts: Python list of ints. May be altered. Number of samples in each
#     category. e.g. 10 cup samples.
#   catids: Python list of ints. May be altered. Assignment of official class
#     ID to be used in recognition. e.g. cup is class ID number 1.
# Returns [basename, cat_idx] extracted from line on success.
#   cat_idx indexes catnames list.
#   Returns [] if line is empty or commented - then caller can just skip this
#     line in file.
def parse_meta_one_line (line, catnames, catcounts, catids, custom_catids=False):

  # If this line is empty or commented, ignore it
  if not line.strip ():
    return []
  elif line.startswith ('#'):
    return []


  # Get base name of file, drop extension
  basename = os.path.basename (line)
  basename = os.path.splitext (basename) [0]

  # Append .csv extension
  basename = basename + '.csv'


  # Find the category name in the path string

  # Drop base name
  #catname, _ = os.path.split (line)
  # Grab last dir name. This won't contain a slash, guaranteed by split()
  #_, catname = os.path.split (catname)
  catname = get_meta_cat_name (line)

  # This indexes catnames and catids
  cat_idx = -1

  # If category not in list yet, append it
  if catname not in catnames:
    catnames.append (catname)
    cat_idx = len (catnames) - 1

    catcounts.append (1)

    if not custom_catids:
      catids.append (cat_idx)

  else:
    cat_idx = catnames.index (catname)

    catcounts [cat_idx] += 1;


  return [basename, cat_idx]


def get_meta_cat_name (line_orig):

  line = line_orig.strip ()

  # Drop base name
  catname, _ = os.path.split (line)
  # Grab last dir name. This won't contain a slash, guaranteed by split()
  _, catname = os.path.split (catname)

  return catname

