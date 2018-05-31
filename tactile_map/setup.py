#!/usr/bin/env python

# Mabel Zhang
# 19 Mar 2015
#
# This file tells catkin to put src/tactile_map/*py python modules into
#   PYTHONPATH.
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
    packages=['tactile_map'],
    package_dir={'': 'src'}
)

setup(**setup_args)

