#!/usr/bin/env python

# Mabel Zhang
# 17 Feb 2015
#
# Configure paths used by multiple files
#

import os, inspect

# Variable parameter size:
#   args[0]: option. 'train', 'test', 'custom', or 'full'.
#     If 'custom' or 'full', user must supply args[1].
#   args[1]:
#     if arg[0] == 'custom': subpath name inside train/ directory.
#     if arg[0] == 'full': full path.
# Example usage:
#   config_paths ('custom', 'task_grasping/deep_rl_visual_tactile/imgs')
# Non-existent paths will be automatically created.
#
# Ref variable arguments:
#   http://stackoverflow.com/questions/919680/can-a-variable-number-of-arguments-be-passed-to-a-function
def config_paths (*args):

  #print (args)

  if len (args) == 0:
    print ('ERROR in tactile_config.py config_paths(): Must supply at least one argument.')
    raise Exception ()
    return ''

  option = args [0]
  if len (args) > 1:
    subpath = args[1]

  # Sanity checks
  if option == 'custom' and len (args) < 2:
    print ('ERROR in tactile_config.py config_paths(): custom option specified, but a custom subpath is not specified.')
    raise Exception ()
    return ''

  if option == 'full' and len (args) < 2:
    print ('ERROR in tactile_config.py config_paths(): full option specified, but a full path is not specified.')
    raise Exception ()
    return ''


  # Ref: http://stackoverflow.com/questions/50499/in-python-how-do-i-get-the-path-and-name-of-the-file-that-is-currently-executin
  datapath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
  # Ref: https://docs.python.org/2/library/os.path.html
  datapath = os.path.join (datapath, '../../../../../../train/')

  # Configure this for specific object paths
  if option == 'train':
    datapath = os.path.join (datapath, 'noiseMeasure1/train/real')
  elif option == 'test':
    datapath = os.path.join (datapath, 'noiseMeasure1/test/')
  elif option == 'custom':
    datapath = os.path.join (datapath, subpath)
  elif option == 'full':
    datapath = subpath
  else:
    print ('ERROR in tactile_config.py config_paths(): option string invalid. Specify train or test.')
    # Ref: http://stackoverflow.com/questions/6720119/setting-exit-code-in-python-when-an-exception-is-raised
    raise Exception ()
    return ''

  # realpath() returns canonical path eliminating symbolic links
  #   Ref: https://docs.python.org/2/library/os.path.html
  datapath = os.path.realpath (datapath)

  # http://stackoverflow.com/questions/273192/in-python-check-if-a-directory-exists-and-create-it-if-necessary
  if not os.path.exists (datapath):
    os.makedirs (datapath)

  return datapath

