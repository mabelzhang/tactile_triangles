#!/usr/bin/env python

# Mabel Zhang
# 29 Jan 2017
#
# Define object names, dimensions, and centers, for real robot triangle
#   sampling.
# Used by plan_poses.py and execute_poses.py.
#
# The need to have dimensions and fixed center for each object is that robot
#   arm motion planning feasibility depends on the object size and center!
#   If you move the center, many poses behind object or to the right of object
#   (in the case of using left arm) would become unfeasible to move to!
# The center is tested by plan_poses.py to have the maximum number of
#   reachable poses (as defined in plan_poses.py), empirically.
#

import numpy as np

class ObjectSpecs:

  # It's okay to skip numbers, these are keys to a dictionary, not a list.
  CANISTER38 = 0
  MUG28 = 1
  SQR_BOWL_GRAY = 2
  BOWL39 = 3
  BOTTLE43 = 4
  GLASS31 = 5
  BOTTLE44 = 6
  MUG37 = 7
  JAR35 = 8
  GLASS46 = 9
  #VASE38 =   # This might get confused btw cup and mug?
  #CUP30 = 
  #CUP27 = 
  #MUG_BEER = 
  #GSK_FLASK_500 = 

  # Objects no longer in use
  #GSK_SQR_1000 =   # This object physically disappeared. Dammit thieves!!
  #BOUNTY =   # This object deforms, not a good object
  #BUCKET48 =   # This object's bottom is too deep, cannot put dual locks

  names = {}
  cats = {}
  centers = {}
  dims = {}

  names [CANISTER38] = 'canister38'
  names [MUG28] = 'mug28'
  names [SQR_BOWL_GRAY] = 'sqr_bowl_gray'
  names [BOWL39] = 'bowl39'
  names [BOTTLE43] = 'bottle43'
  names [GLASS31] = 'cup31'
  names [BOTTLE44] = 'bottle44'
  names [MUG37] = 'mug37'
  names [JAR35] = 'jar35'
  names [GLASS46] = 'glass46'
  #names [GSK_SQR_1000] = 'gsk_sqr_1000'
  #names [BOUNTY] = 'bounty'
  #names [BUCKET48] = 'bucket48'

  cats [CANISTER38] = 'canister'
  cats [MUG28] = 'mug'
  cats [SQR_BOWL_GRAY] = 'bowl'
  cats [BOWL39] = 'bowl'
  cats [BOTTLE43] = 'bottle'
  cats [GLASS31] = 'glass'
  cats [BOTTLE44] = 'bottle'
  cats [MUG37] = 'mug'
  cats [JAR35] = 'jar'
  cats [GLASS46] = 'glass'
  #cats [GSK_SQR_1000] = 'bottle'
  #cats [BOUNTY] = 'toilet_paper'
  #cats [BUCKET48] = 'bowl'


  # center x y is SAME FOR ALL OBJECTS, so that can use all objects' trialed
  #   *feasible* poses in training, to use at test time. They will be in robot
  #   frame!!! (That means inflexible toward object pose changes, hence object
  #   center must all be same!! But that is the only way I can ensure most
  #   poses are feasible! That's what you get for not using a mobile robot!)
  # center z is calculated by plan_poses.py, then copied over by hand.
  # plan_poses.py only needs object dimensions (dims) to run. x and y are
  #   specified by hand, and the best x and y are found by running
  #   plan_poses.py in batch over many x and y's in a grid.
  #   It will calculate the z according to table height, rosparam
  #   moveit_planner/tabletop_z, set in baxter_reflex moveit_planner.cpp by
  #   measuring by hand.

  CX = 0.547763
  CY = 0.489033

  # Horizontally, wide length along robot x
  centers [CANISTER38] = np.array ([CX, CY, -0.073])
  dims [CANISTER38] = np.array ([0.08, 0.16, 0.17])

  # Handle along and pointing to -y of robot frame.
  # Object is off-centered, `.` base is centered on pedestal, but handle sticks
  #   out from pedestal. Handle is 4 cm, half of it is 2 cm, off-centered.
  centers [MUG28] = np.array ([CX, CY, -0.073000])
  centers [MUG28] [1] -= 0.02  # TODO Ehhh shouldn't change center, EVER! Test time depends on center being the same! Retrain this!!!
  dims [MUG28] = np.array ([0.10, 0.14, 0.14])

  # Plastic gray and white square bowl from FroGro
  centers [SQR_BOWL_GRAY] = np.array ([CX, CY, -0.122000])
  dims [SQR_BOWL_GRAY] = np.array ([0.152, 0.152, 0.072])

  centers [BOWL39] = np.array ([CX, CY, -0.113000])
  dims [BOWL39] = np.array ([0.168, 0.168, 0.09])

  # Plastic cylinder bottle with segments
  centers [BOTTLE43] = np.array ([CX, CY, -0.060500])
  dims [BOTTLE43] = np.array ([0.085, 0.085, 0.195])

  # Coca Cola glass cup
  centers [GLASS31] = np.array ([CX, CY, -0.080500])
  dims [GLASS31] = np.array ([0.085, 0.085, 0.155])

  # Glass bottlest-looking bottle
  centers [BOTTLE44] = np.array ([CX, CY, -0.020500])
  dims [BOTTLE44] = np.array ([0.09, 0.09, 0.275])

  # mug37, copied from active_touch execute_actions.py
  #centers [MUG37] = np.array ([CX, CY, -0.058500])  # On tall tabletop
  #centers [MUG37] = np.array ([CX, CY, -0.003500])  # On 4" mount, short table
  centers [MUG37] = np.array ([CX, CY, -0.066500])
  dims [MUG37] = np.array ([0.13, 0.19, 0.183])

  centers [JAR35] = np.array ([CX, CY, -0.113000])
  dims [JAR35] = np.array ([0.10, 0.10, 0.09])

  centers [GLASS46] = np.array ([CX, CY, -0.050500])  # TODO update z
  dims [GLASS46] = np.array ([0.055, 0.055, 0.215])


  # Objects no longer in use

  # gsk_sqr_1000. GSK square media bottle 1000 ml
  # 69 out of 96 poses feasible, with deg_step=45
  #centers [GSK_SQR_1000] = np.array ([CX, CY, -0.042500])
  #dims [GSK_SQR_1000] = np.array ([0.09, 0.09, 0.215])

  # bounty, toilet_paper type. Tall object, requires lower table to reach
  #   upper top.
  #centers [BOUNTY] = np.array ([CX, CY, -0.058500])  # TODO update z
  #dims [BOUNTY] = np.array ([0.13, 0.13, 0.28])

  # bucket48, metal bucket, short fat object
  #centers [BUCKET48] = np.array ([CX, CY, -0.004000])  # On 4" mount, short table
  #dims [BUCKET48] = np.array ([0.135, 0.135, 0.122])



