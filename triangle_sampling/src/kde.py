#!/usr/bin/env python

# Mabel Zhang
# 9 Sep 2015
#
#
#

import numpy as np

from sklearn.neighbors import KernelDensity  # For KDE

# For debugging KDE
import matplotlib.pyplot as plt


def main ():

    '''
    # edgesdd is Python list[] of 3 elements. Each element is a numpy array
    #   with number of elements equal to the number of bins in that dimension
    #   + 1. `.` for b bins, there are b+1 edges.
    # See ret vals of this fn using the 3-line ex on numpy histogramdd() API page.
    histdd, edgesdd = np.histogramdd (data, bins=bins, range=bin_range,
      normed=False)


    #####
    # Find bin centers of histogram, so can feed these bin centers as data
    #   to KDE.
    #####

    nDims = len (edgesdd)

    # You don't need histogram centers. You shouldn't use them. Use bin_range.
    # Wait no, edges ARE already incorporating bin_range, because you pass
    #   bin_range to histogramdd()! So deifnitely need this.
    centersdd = []
    centersdd.append (np.zeros ([bins[0], ]))
    centersdd.append (np.zeros ([bins[1], ]))
    centersdd.append (np.zeros ([bins[2], ]))

    # Get histogram centers from edges. n+1 edges means there are n centers.
    #   Copied from plot_hist_dd.py.
    for i in range (0, nDims):
      centersdd[i][:] = ( \
        edgesdd[i][0:len(edgesdd[i])-1] + edgesdd[i][1:len(edgesdd[i])]) * 0.5


    #####
    # Construct the histogram into data to feed to KDE
    #####

    # Make a (n x d) matrix out of the histogram
    hist_as_data = []

    # For each bin in UNNORMALIZED histogram, append its counts
    # bins[0]
    for x in range (0, len (centersdd [0])):
      # Bin center
      bincx = centersdd [0] [x]

      # bins[1]
      for y in range (0, len (centersdd [1])):
        bincy = centersdd [1] [y]

        # bins[2]
        for z in range (0, len (centersdd [2])):
          bincz = centersdd [2] [z]

          if histdd [x, y, z] > 0:

            # Be careful when use [[]] * n, the inner lists are instances, not
            #   copies. So if you change one, it will change all the others!
            # Make sure histdd is UNNORMALIZED. Here is why, we are copying the
            #   list a (bin count) number of times. So bin count cannot be
            #   floating points!
            hist_as_data.extend ([[bincx, bincy, bincz]] * histdd [x, y, z])
    '''

    histdd = np.array ([[[2, 0], [0, 0]], [[0, 0], [0, 1]]])

    bins = [2, 2, 2]
    hist_as_data = np.array ([[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]])

    #####
    # Fit KDE to histogram
    #####

    # TODO: try different bandwidth variable, see if makes big difference

    #print (np.shape (hist_as_data))
    #print (hist_as_data)

    # Calculate KDE smoothed histograms
    #   Ref: http://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html#example-neighbors-plot-kde-1d-py
    kde = KernelDensity (kernel='epanechnikov').fit (hist_as_data)


    #####
    # Generate some fine-grained x-values, to request KDE
    #####

    # Increase this increase factor to get more smooth
    factor_detail = 1.0

    nbins_smoothed = np.array (bins) * factor_detail
    nbins_total = int (np.product (nbins_smoothed))

    print (bins)
    print ('Number of bins in KDE smoothed histogram: %d' % nbins_total)

    # Must be 0 min, step 1, `.` I'm using unravel_index()! This makes it easier
    #   to loop through the 3 dimensions and produce a n x 3 array. If manually
    #   do it, will need 3 nested for-loops. Not as clean and prone to mistakes.
    request_X = np.unravel_index (np.arange (0, nbins_total, 1),
      nbins_smoothed)
    request_X = np.asarray (request_X).T

    # Then make the "smoothed bin centers" floating point
    request_X = request_X / float (factor_detail)


    #####
    # Evaluated fitted KDE on the requested indices, to give us a smoothed
    #   histogram.
    #####

    # 1D array. Number of elements is number of bins in 3D histogram. So like
    #   8 * 8 * 9. This is also the number of rows in your request_X.
    density = kde.score_samples (request_X)

    # TODO why? Did I not construct hist_as_data correctly?
    # Flip signs to positive. For some reason KDE always returns negative
    density = -density

    #print ('%d inf elements' % np.sum (np.isinf (density)))

    # If negative infinity, just set to 0
    # Ref np.where: http://stackoverflow.com/questions/4588628/find-indices-of-elements-equal-to-zero-from-numpy-array
    inf_idx = np.where (np.isinf (density))
    if inf_idx:
      density [inf_idx] = 0

    #print ('%d inf elements after setting to zero' % np.sum (np.isinf (density)))
    #print ('%d zero elements' % np.size (np.where (density == 0)))

    print ('%d non-zero elements' % np.size (np.where (density != 0)))

    # Linearize the matrix, write to file
    density_linear = np.reshape (density, [density.size, ]);
    #file_writer.writerow (density_linear.tolist ())


    #####
    # Just plot 3D histogram in 1D anyway, the whole thing, to see if smoothed
    #   resembles original, even though the plot won't look like the actual
    #   histogram, `.` plot is linearized.
    #####

    # Just use linear index
    hist_linear = np.reshape (histdd, [histdd.size, ]);
    plt.plot (range (0, np.product (bins)), hist_linear)
    plt.show ()

    plt.plot (range (0, nbins_total), density)
    plt.show ()
 


if __name__ == '__main__':
  main ()

