#!/usr/bin/env python

# Mabel Zhang
# 22 Jul 2015
#
# This file tells catkin to put src/*py python modules into PYTHONPATH.
#
# Refs:
# Confusing tutorial because both Python files are in same ROS package:
#   http://wiki.ros.org/rospy_tutorials/Tutorials/Makefile
# Clears confusion as this person has Python files in 2 different ROS pkgs:
#   http://answers.ros.org/question/147391/importing-python-module-from-another-ros-package-setuppy/
# Not useful and repeats tutorial:
#   http://docs.ros.org/api/catkin/html/user_guide/setup_dot_py.html
#

## ! DO NOT MANUALLY INVOKE THIS setup.py, USE catkin_python_setup in
##   CMakeLists.txt INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    # scripts argument lets you keep .py files in whatever directory you want!
    #   http://docs.ros.org/api/catkin/html/user_guide/setup_dot_py.html
    # To import these, simply import filename, where filename.py is a file
    #   in the directory, e.g. src/filename.py.
    # This doesn't work for all systems!!! Use way below, the sure fire way.
    #scripts=['src'],

    # This requires you to put .py files inside src/packages folder.
    # To import these, you need to import packages.filename, where packages is
    #   name of subdir inside src, and filename.py is the file you're importing
    packages=['tactile_collect'],
    package_dir={'': 'src'}
)

setup(**setup_args)

