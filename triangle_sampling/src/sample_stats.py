#!/usr/bin/env python

# Mabel Zhang
# 11 Sep 2015
#
# Simple script to see what is the relationship btw nSamples and nSamplesRatio
#


import numpy as np


def main ():

  # 1~10, 20~100 step 10, 200~1000 step 100, 1000~3000 step 1000
  nPts = np.concatenate ((np.arange (0, 11, 1), np.arange (20, 101, 10),
    np.arange (200, 1001, 100), np.arange (2000, 3001, 1000)))

  # 10~40 step 10, 50~300 step 50
  nSamplesPerPassPerVoxel = np.concatenate ((np.arange (10, 50, 10),
    np.arange (50, 301, 50)))
  #nSamplesPerPassPerVoxel = [300]

  #nSamplesRatio = np.concatenate ((np.arange (0.6, 0.95, 0.05),
  #  np.arange (0.95, 1, 0.01)))
  nSamplesRatio = [0.95]

  # Column titles are the number of points
  print ('nPts samples ratio   min')

  for s in nSamplesPerPassPerVoxel:
    for r in nSamplesRatio:

      for p in nPts:
        print ('%4d' % p),
        print ('%7d  %3.2f ' % (s, r)),

        print ('%4d' % np.min ([s, np.floor (r * p)])),

        print ('')

        #print (s)
        #print (np.floor (r * p))

    print ('#')



if __name__ == '__main__':
  main ()

