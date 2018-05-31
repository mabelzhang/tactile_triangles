# Mabel Zhang
# 1 Sep 2015
#
# Refactored from sample_pcl_calc_hist.py and
#   triangles_on_robot_to_hists.py,
#   so that they can call same function. Then don't need to worry about
#   changing one and forgetting to change the other.
#
#   `.` they are for training data and test data, respectively! Must be
#   written the same way.
#

import rospy  # Only for catching Ctrl+C

# Python
import os
import csv

import numpy as np

from sklearn.neighbors import KernelDensity  # For KDE

# For debugging KDE
import matplotlib.pyplot as plt

# My packages
from tactile_collect import tactile_config
from triangle_sampling.config_hist_params import TriangleHistogramParams as \
  HistP

# Local
from plot_hist_dd import flatten_hist


def write_kde_config (config_path, kernel, bandwidth, factor_granular, bins):

  config_name = os.path.join (config_path, 'kde_config.csv')

  with open (config_name, 'wb') as config_file:

    column_titles = ['kernel', 'bandwidth', 'factor_granular',
      'bins0', 'bins1', 'bins2']

    config_writer = csv.DictWriter (config_file, fieldnames=column_titles)
    config_writer.writeheader ()

    print (zip (column_titles, [kernel, bandwidth, 
      factor_granular, bins[0], bins[1], bins[2]]))

    row = dict ()
    row.update (zip (column_titles, [kernel, bandwidth, 
      factor_granular, bins[0], bins[1], bins[2]]))

    config_writer.writerow (row)


# Parameters:
#   shape: Tuple of 3 elements, specifying the shape of the goal 3D matrix
def flat_to_3d_hist (density, shape):
     
  # Init a 3D matrix
  density_3d = np.zeros (shape)
 
  # Convert 1D index to 3D
  den_3d_idx = np.unravel_index (range (0, len (density)), shape)
 
  for i in range (0, len (density)):
 
    # Assign element in 3D index
    density_3d [den_3d_idx[0][i], den_3d_idx[1][i], den_3d_idx[2][i]] = \
      density [i]

  return density_3d


# Parameters:
#   file_writer: Obtained from csv.writer()
#   data: n x d Numpy array. n = # points, d = # dimensions.
#     Data to pass as param 1 to np.histogramdd().
#   bins, bin_range: Args to numpy histogramdd(), which has default 10 bins,
#     default range None. So if you don't know, then don't supply these args.
#   normed: Only observed if kde=False. For KDE, must use unnormalized hist,
#     so that it's possible to construct the data to pass to KDE!
#   All default values are copied from np.histogramdd. If the official NumPy
#     API changes, also need to change here!
def write_hist_3d_csv (file_writer, data, bins=[10,10,10], bin_range=None,
  normed=False, kde=False, obj_name=None, debug=True, config_path=None,
  prs=HistP.PRS):

  DEBUG_PLOTS = debug


  # Write 3D histogram as linearized 1D histogram
  if not kde:

    histdd, edgesdd = np.histogramdd (data, bins=bins, range=bin_range,
      normed=normed)

    # For debugging histogram itself, and bins - whether decimetered twice
    #print ('write_hist_3d_csv(): %d nonzero values in orig 3D hist' % len (np.nonzero (histdd) [0]))
    #print ('write_hist_3d_csv(): %d nonzero values in orig 3D hist')
    #print (np.nonzero (histdd))
    #print ('write_hist_3d_csv(): edgesdd:')
    #print (edgesdd)

    # For debugging decimeters mode. If data is out of histogram bin range,
    #   will get division by 0 error, and all histogram values will be nan.
    #print (data)
    #print (bins)
    #print (bin_range)
    #print (edgesdd)
    # All nans for some objs
    #print (histdd)

    # TODO 5 Dec 2016: ... This isn't even used later! Is this supposed to be
    #   BEFORE the histogramdd() call above????? I won't change anything now,
    #   but later if have time, test moving this above!!! Maybe I always pass
    #   in bin_range anyway, so didn't find this bug?
    # If bin_range wasn't supplied, populate it with edges from histogramdd
    if not bin_range:
      bin_range = []
      bin_range.append ((edgesdd [0][0], edgesdd [0][len (edgesdd [0]) - 1]))
      bin_range.append ((edgesdd [1][0], edgesdd [1][len (edgesdd [1]) - 1]))
      bin_range.append ((edgesdd [2][0], edgesdd [2][len (edgesdd [2]) - 1]))
      print ('bin_range: [mins (%.2f, %.2f, %.2f), max (%.2f, %.2f, %.2f)]' % \
        (bin_range[0][0], bin_range[1][0], bin_range[2][0],
         bin_range[0][1], bin_range[1][1], bin_range[2][1]))

      # Generalize the code above to d-dimensional. Not tested. Replace code
      #   above with this code, when have time to test it
      '''
      bin_range = []
      for d in range (0, len (self.edgesdd)):
        bin_range.append ((self.edgesdd [d][0], self.edgesdd [d][len (self.edgesdd [d]) - 1]))

      print ('bin_range: [mins ('),
      for d in range (0, len (self.edgesdd)):
        print ('%.2f, ' % bin_range[d][0]),

      print ('), max ('),
      for d in range (0, len (self.edgesdd)):
        print ('%.2f, ' % bin_range[d][1]),
      print (')]')
      '''


    # Reshape into linear shape, so don't need nested for-loops!
    # Test in bare Python shell:
    '''
    import numpy as np
    a = np.array ([[[1,2,3], [2,3,4]], [[5,6,7], [6,7,8]]])
    a.size
    a.ndim
    a.shape
    np.reshape (a, [a.size, ])

    # Test shaping back to 3D
    a = np.array ([[[1,2,3,4], [5,6,7,8]], [[9,10,11,12],[13,14,15,16]], [[17,18,19,20],[21,22,23,24]]])
    a.shape  # This prints (3,2,4)
    # Numbers are in order in the linear version
    a_lin = np.reshape (a, (a.size,))
    np.reshape (a_lin, (3,2,4))
    np.reshape (a_lin, a.shape)
    '''
    # Doesn't matter which bin is where, as long as all objects are saved the
    #   same way. `.` axes can be swapped around, values will still distribute
    #   the same way in the space.
    hist_linear = np.reshape (histdd, [histdd.size, ]);
    #if np.any (np.isnan (hist_linear)):
    if np.any (np.isnan (histdd)):
      print ('Saw nan in histogram')

    # Convert to Python list, then write to csv
    file_writer.writerow (hist_linear.tolist ())

    return (histdd, edgesdd, hist_linear)


  # Do KDE, write densities
  else:

    # Only for comparison with KDE. KDE itself is evaluated using raw data, not
    #   the histogram. Compare to the original histogram to see if KDE
    #   approximates it well.
    #
    # edgesdd is Python list[] of 3 elements. Each element is a numpy array
    #   with number of elements equal to the number of bins in that dimension
    #   + 1. `.` for b bins, there are b+1 edges.
    # See ret vals of this fn using the 3-line ex on numpy histogramdd() API
    #   page.
    histdd, edgesdd = np.histogramdd (data, bins=bins, range=bin_range,
      normed=False)

    # If bin_range wasn't supplied, populate it with edges from histogramdd
    if not bin_range:
      bin_range = []
      bin_range.append ((edgesdd [0][0], edgesdd [0][len (edgesdd [0]) - 1]))
      bin_range.append ((edgesdd [1][0], edgesdd [1][len (edgesdd [1]) - 1]))
      bin_range.append ((edgesdd [2][0], edgesdd [2][len (edgesdd [2]) - 1]))
      print ('bin_range: [mins (%.2f, %.2f, %.2f), max (%.2f, %.2f, %.2f)]' % \
        (bin_range[0][0], bin_range[1][0], bin_range[2][0],
         bin_range[0][1], bin_range[1][1], bin_range[2][1]))


    #####
    # Find bin centers of histogram, so can feed these bin centers as data
    #   to KDE.
    #####

    nDims = len (edgesdd)

    # You don't need histogram centers. You shouldn't use them. Use bin_range.
    # Wait no, edges ARE already incorporating bin_range, because you pass
    #   bin_range to histogramdd()! So definitely need this.
    centersdd = []
    centersdd.append (np.zeros ([bins[0], ]))
    centersdd.append (np.zeros ([bins[1], ]))
    centersdd.append (np.zeros ([bins[2], ]))

    # Get histogram centers from edges. n+1 edges means there are n centers.
    #   Copied from plot_hist_dd.py.
    for i in range (0, nDims):
      centersdd[i][:] = ( \
        edgesdd[i][0:len(edgesdd[i])-1] + edgesdd[i][1:len(edgesdd[i])]) * 0.5


    '''
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

    #print (hist_as_data)
    '''


    #####
    # Fit KDE to raw data
    #####

    # TODO: try different bandwidth variable, see if makes big difference

    kernel = 'tophat'
    bandwidth = 0.08

    #print (np.shape (hist_as_data))
    #print (hist_as_data)

    # Calculate KDE smoothed histograms
    #   Ref: http://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html#example-neighbors-plot-kde-1d-py
    # bandwidths for bins [15 15 18]:
    #   (1D example on sklearn used 0.5 for all 3 kernels.)
    #
    #   'epanechnikov'
    #     0.02 too small for all
    #     0.08 looks good
    #   'gaussian'
    #     0.08, 0.05 too large for lengths, to small for angles
    #
    #   Rescaled lengths (l0, l1, etc) to decimeters:
    #     'gaussian'
    #       0.02 too large for lengths, to small for angles
    #     'tophat'
    #       0.02 too small for all
    #       0.5 slightly too small
    #       0.6 seems best
    #       0.7 too big
    #       0.8 slightly too big
    #       1 too big for all
    # bandwidths for bins [15 15 15], decimeters:
    #     'linear'
    #       0.6 compared to tophat 0.6, angles plots too many sharp ups downs
    # bandwidths for bins [30 30 30], decimeters:
    #   'tophat'
    #     0.6 still the best, even though it looks more rounded
    #     0.5 slightly small
    #     0.3 too small
    # tophat 0.6 looks better than others, for bins [15 15 15]
    #kde = KernelDensity (kernel='tophat', bandwidth=0.6).fit (hist_as_data)

    # Pass in raw data, not histograms. Input data is still n x 3!
    # bins [8 8 9]:
    #   'tophat':
    #     0.6 best. though too small, jagged artificial dips
    #     0.7 too large, still more rounded than 0.6
    #     0.8 too large, curves too rounded
    #   'epanechnikov'
    #     0.6 good for lengths, too small for angle
    kde = KernelDensity (kernel=kernel, bandwidth=bandwidth).fit (data)


    #####
    # Generate some fine-grained x-values, to request KDE
    #####

    # User adjust param
    # This should increase as bandwidth decreases. `.` bandwidth at finer
    #   widths require finer granularity to evaluate.
    # Increase this increase factor to get more smooth. Remember this will be
    #   CUBIC in effect, `.` histogram is 3D!
    # Increasing this exponentially increases number of values you request,
    #   too big a value makes it run super slow.
    factor_granular = 3.0

    nbins_smoothed = np.array (bins) * factor_granular
    nbins_total = int (np.product (nbins_smoothed))

    #print (bins)
    print ('Number of bins (%d %d %d) in original histogram: %d' % \
      (bins[0], bins[1], bins[2], int(np.product(bins))))
    print ('Number of bins (%d %d %d) in KDE smoothed histogram: %d' % \
      (bins[0] * factor_granular, bins[1] * factor_granular,
       bins[2] * factor_granular, nbins_total))

    if config_path:
      write_kde_config (config_path, kernel, bandwidth, factor_granular, bins)


    xmin = bin_range[0][0]
    xmax = bin_range[0][1]
    ymin = bin_range[1][0]
    ymax = bin_range[1][1]
    zmin = bin_range[2][0]
    zmax = bin_range[2][1]

    xbins = bins[0]
    ybins = bins[1]
    zbins = bins[2]

    request_X = []

    # Use bin_range min max to define the x-values to request KDE
    # request_X must be (n x d), required by score_samples().
    for x in np.linspace (xmin, xmax, xbins * factor_granular):
      for y in np.linspace (ymin, ymax, ybins * factor_granular):
        for z in np.linspace (zmin, zmax, zbins * factor_granular):
          request_X.append ([x, y, z])

    request_X = np.array (request_X)


    # Try mgrid(). I don't like this, it returns x * y * z dimensions matrices!
    #   Then you still have to linearize them before passing to score_samples()!
    #   Error prone. So much easier to just do the way above, it's guaranteed
    #   correct.
    # You'd have to calculate the step yourself, and the end of range is
    #   excluded, python style. So use linspace, it doesn't exclude the end
    #   range.
    #x, y, z = np.mgrid [np.linspace (xmin, xmax, xbins * factor_granular),
    #  np.linspace (ymin, ymax, ybins * factor_granular),
    #  np.linspace (zmin, zmax, zbins * factor_granular)]


    #print (request_X)
    #print (np.shape (request_X))


    #####
    # Evaluated fitted KDE on the requested indices, to give us a smoothed
    #   histogram.
    #####

    if not rospy.is_shutdown ():
      try:
        # 1D array. Number of elements is number of bins in 3D histogram. So
        #   like 8 * 8 * 9. This is also the number of rows in your request_X.
        density = kde.score_samples (request_X)
      except rospy.exceptions.ROSInterruptException, err:
        return (None, None, None)
    else:
      return (None, None, None)

    #print (np.shape(density))

    #print ('%d inf elements' % np.sum (np.isinf (density)))

    # If negative infinity, just set to 0
    # Ref np.where: http://stackoverflow.com/questions/4588628/find-indices-of-elements-equal-to-zero-from-numpy-array
    inf_idx = np.where (np.isinf (density))
    if inf_idx:
      density [inf_idx] = 0

    # Raise to exponent, since score_samples() returns in log scale
    density = np.exp (density)

    #print ('%d inf elements after setting to zero' % np.sum (np.isinf (density)))
    #print ('%d zero elements' % np.size (np.where (density == 0)))

    print ('%d non-zero elements' % np.size (np.where (density != 0)))

    # density is just a 1D vector, `.` that's what score_samples() returns
    # If debugging, don't write! `.` histogram bin_range would be wrong!
    if not DEBUG_PLOTS:
      file_writer.writerow (density.tolist ())



    if DEBUG_PLOTS:

      # Convert density vector to 3D first, `.` currently in a flattened 1D
      #   array, becaues that's what score_samples() returns.
      #   score_samples() only deals with (n x d) data, where d is dimension of
      #   histogram, n is the total number of counts in all bins. Data it takes
      #   are bin positions in histogram. It does not deal with 3D histogram
      #   directly.

      density_3d = flat_to_3d_hist (density, [xbins * factor_granular,
        ybins * factor_granular, zbins * factor_granular])


      # ['l0', 'l1', 'a0']
      param_names = prs

      out_path = tactile_config.config_paths ('custom',
        'triangle_sampling/imgs/kde/')


      #####
      # Plot 2D slice of 3D histogram, as heat map
      #####

      # Turn off interactive mode, so you don't have to close every single
      #   window. It's faster. Just look at the images once saved.
      plt.ioff ()

      fig = plt.figure ()

      # These correspond to indices 0 1 2, of selected dimension
      other_dim = [[1,2], [0,2], [0,1]]

      # Pick a dimension, let's say 3rd. This is the slice we'll plot the other
      #   2 dimensions in.
      for pick_dim in range (0, 3):

        #   other_dim[pick_dim][0] is the first unpicked dim, other_dim[*][1]
        #     is the second unpicked dim, when pick_dim is picked.
        other_dim1 = other_dim [pick_dim] [0]
        other_dim2 = other_dim [pick_dim] [1]


        #####
        # Plot comparable histogram
        #####

        # Look at the middle slice. This is the index of the slice.
        #   (You could also look at the values returned by score_samples(), take
        #   the max, see which slice it's in, and look at that slice).
        slices = range (0, np.shape (histdd) [pick_dim])
        mid_slice = np.floor (np.median (slices))
        print (slices)
        print ('median slice index, of the list of indices above: %d' % mid_slice)

        if pick_dim == 0:
          # http://matplotlib.org/examples/pylab_examples/matshow.html
          slice_mat = histdd [mid_slice, :, :]
        elif pick_dim == 1:
          slice_mat = histdd [:, mid_slice, :]
        elif pick_dim == 2:
          slice_mat = histdd [:, :, mid_slice]

        ax = plt.matshow (slice_mat, figure=fig)

        # Ref move title up: http://stackoverflow.com/questions/12750355/python-matplotlib-figure-title-overlaps-axes-label-when-using-twiny
        plt.title ('Slice %d out of %d, along %s in 3D Histogram' % \
          (mid_slice+1, len(slices), param_names[pick_dim]), y=1.1)
        plt.xlabel (param_names [other_dim1])
        plt.ylabel (param_names [other_dim2])

        # Sanity check. Print out max and mins of histogram and KDE to match
        print ('%s:' % param_names [other_dim1])
        print ('%s range: %f, %f' % (param_names [other_dim1],
          bin_range [other_dim1] [0], bin_range [other_dim1] [1]))
        print ('%s:' % param_names [other_dim2])
        print ('%s range: %f, %f' % (param_names [other_dim2],
          bin_range [other_dim2] [0], bin_range [other_dim2] [1]))

        # For histograms, ticks are the histogram bins.
        # x and y axes are the unpicked dimension, when this dim is picked
        #   bin_range[*][0] is min, bin_range[*][1]
        plt_ax = plt.gca ()
        plt_ax.set_xticklabels ([format ('%.1f' % i) for i in \
          np.linspace (bin_range [other_dim1] [0],
            bin_range [other_dim1] [1], bins [other_dim1])], rotation='vertical')
        plt_ax.set_yticklabels ([format ('%.1f' % i) for i in \
          np.linspace (bin_range [other_dim2] [0],
            bin_range [other_dim2] [1], bins [other_dim2])])

        plt.xticks (np.arange (0, np.shape (slice_mat) [0], 1.0))
        plt.yticks (np.arange (0, np.shape (slice_mat) [1], 1.0))

        plt.colorbar (ax)
       
        if obj_name:
          out_name = os.path.join (out_path,
            obj_name + '_matshow_' + param_names [other_dim1] + '_' + \
            param_names [other_dim2] + '.eps')
          plt.savefig (out_name, bbox_inches='tight')
          print ('Plot saved to %s' % out_name)

        #plt.show ()
        # Ref clf: http://stackoverflow.com/questions/8213522/matplotlib-clearing-a-plot-when-to-use-cla-clf-or-close
        plt.clf ()


        #####
        # Plot KDE
        #####

        # Look at the middle slice
        #   (You could also look at the values returned by score_samples(), take
        #   the max, see which slice it's in, and look at that slice).
        slices = range (0, np.shape (density_3d) [pick_dim])
        mid_slice = np.floor (np.median (slices))
        print (slices)
        print ('median slice index, of the list of indices above: %d' % mid_slice)

        if pick_dim == 0:
          slice_mat = density_3d [mid_slice, :, :]
        elif pick_dim == 1:
          slice_mat = density_3d [:, mid_slice, :]
        elif pick_dim == 2:
          slice_mat = density_3d [:, :, mid_slice]

        ax = plt.matshow (slice_mat, figure=fig)

        # Ref move title up: http://stackoverflow.com/questions/12750355/python-matplotlib-figure-title-overlaps-axes-label-when-using-twiny
        plt.title ('Slice %d out of %d, along %s in 3D KDE Distribution' % \
          (mid_slice+1, len(slices), param_names[pick_dim]), y=1.1)
        plt.xlabel (param_names [other_dim1])
        plt.ylabel (param_names [other_dim2])

        # Sanity check. Print out max and mins of histogram and KDE to match
        #print ('%s range: %f, %f' % (param_names [other_dim1],
        #  np.min (request_X [:, other_dim1]), np.max (request_X [:, other_dim1])))
        #print ('%s range: %f, %f' % (param_names [other_dim2],
        #  np.min (request_X [:, other_dim2]), np.max (request_X [:, other_dim2])))

        print ('%s:' % param_names [other_dim1])
        print (np.unique (request_X [:, other_dim1]))
        print ('%s:' % param_names [other_dim2])
        print (np.unique (request_X [:, other_dim2]))
        #print ([format ('%.1f' % i) for i in \
        #  np.unique (request_X [:, other_dim1])])
        #print ([format ('%.1f' % i) for i in \
        #  np.unique (request_X [:, other_dim2])])

        # For KDE distributions, the ticks are the values we requested KDE to
        #   evaluate at.
        # x and y axes are the unpicked dimension, when this dim is picked
        #   bin_range[*][0] is min, bin_range[*][1]
        # Ref ax.set_xticklabels()
        #   http://stackoverflow.com/questions/3529666/matplotlib-matshow-labels
        # Ref set textual tick labels: http://stackoverflow.com/questions/5439708/python-matplotlib-creating-date-ticks-from-string
        plt_ax = plt.gca ()
        plt_ax.set_xticklabels ([format ('%.1f' % i) for i in \
          np.unique (request_X [:, other_dim1])], rotation='vertical')
        plt_ax.set_yticklabels ([format ('%.1f' % i) for i in \
          np.unique (request_X [:, other_dim2])])

        # Show ALL ticks (by default, it doesn't show all of them)
        # Ref: http://stackoverflow.com/questions/12608788/changing-the-tick-frequency-on-x-or-y-axis-in-matplotlib
        plt.xticks (np.arange (0, np.shape (slice_mat) [0], 1.0))
        plt.yticks (np.arange (0, np.shape (slice_mat) [1], 1.0))
       
        plt.colorbar (ax)

        if obj_name:
          out_name = os.path.join (out_path,
            obj_name + '_matshow_' + param_names [other_dim1] + '_' + \
            param_names [other_dim2] + '_kde.eps')
          plt.savefig (out_name, bbox_inches='tight')
          print ('Plot saved to %s' % out_name)

        #plt.show ()
        plt.clf ()


      #####
      # Plot flattened dimensions of 3D histogram
      #####

      plt.ioff ()

      # Plot flattened 3D histograms, for original and smoothed versions

      # For each dimension of the histogram
      for d in range (0, 3):

        hist_flat = flatten_hist (histdd, d)
        density_flat = flatten_hist (density_3d, d)
       
        # bin_range[i][j] is ith dimension, j=0 for min, j=1 for max.
        hist_xticks = np.linspace (bin_range[d][0], bin_range[d][1], bins[d])
        dens_xticks = np.linspace (bin_range[d][0], bin_range[d][1],
          nbins_smoothed[d])

        # Just use linear index
        hist_linear = np.reshape (histdd, [histdd.size, ]);
        plt.plot (hist_xticks, hist_flat, figure=fig)

        if obj_name:
          out_name = os.path.join (out_path,
            obj_name + '_' + param_names[d] + '.eps')
          plt.savefig (out_name, bbox_inches='tight')
          print ('Plot saved to %s' % out_name)

        #plt.show ()
        plt.clf ()

       
        print ('nbins_total / (factor_granular cubed): %f' % (nbins_total / (factor_granular ** 3)))
       
        # Divide by the cubic of how much we shrank the scale, to get the correct
        #   floating point scales.
        plt.plot (dens_xticks, density_flat, figure=fig)

        if obj_name:
          out_name = os.path.join (out_path,
            obj_name + '_' + param_names[d] + '_kde.eps')
          plt.savefig (out_name, bbox_inches='tight')
          print ('Plot saved to %s' % out_name)

        #plt.show ()
        plt.clf ()

      plt.close (fig)

    return (histdd, edgesdd, density)

