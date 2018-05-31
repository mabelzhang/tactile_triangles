#!/usr/bin/env python

# Mabel Zhang
# 8 Sep 2015
#
# Metrics to use in SVM (as kernels) and NN
#

# Python
from copy import deepcopy

# NumPy
import numpy as np
# For chisquare dist for histograms
#from sklearn.metrics.pairwise import chi2_kernel

# Local
from calc_hists import linearize_hist


# Custom SVM kernel, use with sklearn svm.SVC(kernel=hist_inter_kernel).
#
# Parameters:
#   A: nSamples1 x nDims, 2D NumPy matrix
#   B: nSamples2 x nDims, 2D NumPy matrix
# Returns nSamples1 x nSamples2 kernel matrix, 2D NumPy matrix.
#
# Ref: p.18-19 in Malik slides. Hist inter can be used as kernel DIRECTLY, even
#   though it's a inverted distance - larger intersection means smaller
#   distance. It's comparable to negative chi square.
#   http://web.stanford.edu/group/mmds/slides2008/malik.pdf
# Ref custom kernel example
#   http://scikit-learn.org/stable/auto_examples/svm/plot_custom_kernel.html#example-svm-plot-custom-kernel-py
#   Function signature requirements, search "Using Python function as kernels"
#     on this page
#     http://scikit-learn.org/stable/modules/svm.html
def hist_inter_kernel (A, B):

  # Simplest test case, works for 2D data or 1D data
  #return np.dot(A, B.T)

  #print (np.shape(A))
  #print (np.shape(B))

  try:
    assert (np.shape (A) [1] == np.shape (B) [1])
  except AssertionError:
    print ('Number of features in A and B are not equal: %s vs. %s' % ( \
      A.shape [1], B.shape [1]))


  nSamples_a = np.shape (A) [0]
  nSamples_b = np.shape (B) [0]

  # Preallocate to save time
  # na x nb
  kernel = np.zeros ((nSamples_a, nSamples_b))

  # For each sample in a
  for a_i in range (0, nSamples_a):

    # Find hist intersection (minimum) btw this row in A, and all rows in B
    #   (1 x nDims) intersects (nSamples_b x nDims)
    # Sum each row (axis=1). Each row is the intersection result with one
    #   sample in B
    # Vector of size nSamples_b
    kernel [a_i, :] = np.sum (np.minimum (A[a_i, :], B), axis=1)

  return kernel


# Custom distance function. Use with sklearn
#   NearestNeighbors(metric=hist_inter_dist)
# Parameters:
#   a, b: NumPy 1D arrays
# Returns scalar indicating the distance btw a and b.
#
# Ref: metric arg on this page http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
def hist_inter_dist (a, b):

  # Return the negative, because bigger histogram intersection means closer,
  #   i.e. "smaller" distance in the usual sense, so need to flip it.
  return -np.sum (np.minimum (a, b))

  # A better way is to normalize to [0, 1], by multiplying the product of the
  #   bin widths. But you need the bin widths to do that! Since this is a
  #   kernel function, it doesn't take extra inputs. Currently the
  #   normalization to [0, 1] has to be done in the caller, e.g. see
  #   active_touch classifier.py



def hist_minus_hist_inter_dist (a, b):

  inter = np.minimum (a, b)

  # a - inter and b - inter should both be positive, `.` inter is the minimum
  #   of a and b.
  # Bigger means farther.
  return (np.sum (a - inter) + np.sum (b - inter))



# This doesn't seem to work for anything. Don't use this
#def neg_chi2_kernel (*arg):

#  return -chi2_kernel (*arg)


# Custom SVM kernel, use with sklearn svm.SVC(kernel=kl_divergence_kernel).
# Symmetric KL Divergence (symmetry is required for SVM!)
#
# Parameters:
#   A: nSamples1 x nDims, 2D NumPy matrix
#   B: nSamples2 x nDims, 2D NumPy matrix
# Returns nSamples1 x nSamples2 kernel matrix, 2D NumPy matrix.
#
# Ref:
#   KL Divergence symmetrized
#     http://en.wikipedia.org/wiki/Kullback-Leibler_divergence#Symmetrised_divergence
#   Function signature requirements, search "Using Python function as kernels"
#     on this page
#     http://scikit-learn.org/stable/modules/svm.html
def kl_divergence_kernel (A, B):

  # Must be floating points for division. Caller should convert if they aren't
  #   already floats.
  assert (A.dtype == np.dtype ('float32') or \
    A.dtype == np.dtype ('float16') or \
    A.dtype == np.dtype ('float64'))

  nSamples_a = np.shape (A) [0]
  nSamples_b = np.shape (B) [0]

  # Preallocate to save time
  # na x nb
  kernel = np.zeros ((nSamples_a, nSamples_b))

  small_nonzero = 1e-6


  # For each sample in a
  for a_i in range (0, nSamples_a):

    # This is what you would do, if there are no zeros in A and B, no nested
    #   for-loop required:
    # One row in A, all of B. Symmetric KL divergence means need
    #   KL(A||B) + KL(B||A).
    # * and / operators are element-wise. log() is element-wise too.
    #kernel [a_i, :] = np.sum (A [a_i, :] * np.log (A [a_i, :] / B), axis=1) +
    #  np.sum (B * np.log (B / A [a_i, :]), axis=1)

    # Check for zeros in denominator. KL Divergence requires the two
    #   distributions satisfy Q(i)=0 implies P(i)=0, in division Pi*log(Pi/Qi).
    #   Then when Qi = 0, instead of division by 0, you simply set
    #   Pi*log(Pi/Qi) = 0. (Source: Wikipedia)

    # ??? How would this work? Based on the B, the row a would have to be
    #   different for each row B!! Because rows in B may have 0s in different
    #   places!! Then you're using a different a (because different places of a
    #   are set to 0) for different B's!!! Then you alter your original samples,
    #   and the samples being computed with each other aren't always altered
    #   the same way!!
    # So you can't ever do this without a nested for-loop!?


    for b_i in range (0, nSamples_b):

      '''
      # Copied from kl_divergence_dist() below. More clear if just do one
      #   sample at a time! So confusing in matrix form, because you can't set
      #   all elements outside, it has to be set differently for each pairing,
      #   depending on the vector you're pairing up with.

      azi = np.where (A [a_i, :] == 0)
      bzi = np.where (B [b_i, :] == 0)
  
      # Set them to the same, so that numer / nonzero_denom = 1, and np.log(1)=0.
      #   For KL divergence, when Pi*log(Pi/Qi) has Qi=0, we want to set product
      #   to just 0 (source: Wikipedia).
      # Since a and b are only 1 sample each, just set one unique row for each
      a_nonzeros = deepcopy (A [a_i, :])
      a_nonzeros [azi] = small_nonzero
      a_nonzeros [bzi] = small_nonzero
      b_nonzeros = deepcopy (B [b_i, :])
      b_nonzeros [azi] = small_nonzero
      b_nonzeros [bzi] = small_nonzero
  
      # Symmetric kl divergence is the sum KL(a,b) + KL(b,a)
      kernel [a_i, b_i] = \
        (np.sum (a_nonzeros * np.log (a_nonzeros / b_nonzeros)) + \
         np.sum (b_nonzeros * np.log (b_nonzeros / a_nonzeros)))
      '''

      kernel [a_i, b_i] = kl_divergence_dist (A [a_i, :], B [b_i, :])

  return kernel


# Doesn't work well.
def kl_divergence_dist (a, b):

  small_nonzero = 1e-6

  azi = np.where (a == 0)
  bzi = np.where (b == 0)

  '''
  # Set them to the same, so that numer / nonzero_denom = 1, and np.log(1)=0.
  #   For KL divergence, when Pi*log(Pi/Qi) has Qi=0, we want to set product
  #   to just 0 (source: Wikipedia).
  # Since a and b are only 1 sample each, just set one unique row for each
  a_nonzeros = deepcopy (a)
  a_nonzeros [azi] = small_nonzero
  a_nonzeros [bzi] = small_nonzero
  b_nonzeros = deepcopy (b)
  b_nonzeros [azi] = small_nonzero
  b_nonzeros [bzi] = small_nonzero

  # The problem with KL divergence assumption of Qi=0 => Pi=0 => Pi*log(Pi/Qi)
  #   is that, when Pi is very different from Qi, say Pi=100, Qi=0, you assume
  #   Pi=0 too, and set the divergence of Pi and Qi to 0, then you threw away
  #   a ton of useful information! It may be that this bin is the most dicerning
  #   one to differentiate btw P and Q!!
  # And it shouldn't be symmetric if you don't sum. So when only one of Pi and
  #   Qi is zero, you shouldn't set both to small_nonzero! If Pi is 0 and Qi is
  #   huge, then let it be that way, don't set Qi to be same as Pi, just set Pi
  #   to a small nonzero number, so that log(0.0001/big_number)=something.
  #   Result: even worse. Everywhere on the per-sample NN plot is red...

  return (np.sum (a_nonzeros * np.log (a_nonzeros / b_nonzeros)) + \
          np.sum (b_nonzeros * np.log (b_nonzeros / a_nonzeros)))
  '''


  # This works better than above, at producing per-sample NN distance plot.
  #   But as kernel in SVM, this is terrible, most things are classified as
  #   apples and bananas.
  #'''
  # What if I divide by the intersection, instead of setting things to
  #   small_nonzero and throwing away good information? Dividing by
  #   intersection should give us the difference btw the curves' "shape" in
  #   the sense of ratio too, right?

  a_nonzeros = deepcopy (a)
  a_nonzeros [azi] = small_nonzero
  b_nonzeros = deepcopy (b)
  b_nonzeros [bzi] = small_nonzero

  # Histogram intersection
  inter = np.minimum (a, b)
  b_minus = b_nonzeros - inter
  a_minus = a_nonzeros - inter
  b_minus [b_minus == 0] = small_nonzero
  a_minus [a_minus == 0] = small_nonzero

  # This produces reasonable per-sample NN plots, looks like the default NN
  #   minkowski distance's plot. Definitely better than the orig KL divergence
  #   definition above!
  #return (np.sum (a_nonzeros * np.log (a_minus / b_minus)) + \
  #        np.sum (b_nonzeros * np.log (b_minus / a_minus)))
  # This produces something reasonable, looks like the minkowski distance's
  #   plot. Definitely better than the orig KL divergence definition above!
  return (np.sum (a_minus * np.log (a_minus / b_minus)) + \
          np.sum (b_minus * np.log (b_minus / a_minus)))

  # These are guesses that didn't work
  #return (np.sum (a_minus * np.log (a_minus / b_nonzeros)) + \
  #        np.sum (b_minus * np.log (b_minus / a_nonzeros)))
  #return (np.sum (a_nonzeros * np.log (a_minus / b_nonzeros)) + \
  #        np.sum (b_nonzeros * np.log (b_minus / a_nonzeros)))
  #return (np.sum (a_nonzeros * np.log (b_minus / a_nonzeros)) + \
  #        np.sum (b_nonzeros * np.log (a_minus / b_nonzeros)))
  #'''



# Parameters:
#   h1, h2: Two normalized histograms btw which to compute the chi-squared
#     distance
def chisqr_dist (h1, h2):

  # Ref formula: http://stats.stackexchange.com/questions/184101/comparing-two-histograms-using-chi-square-distance
  # Add a small offset, so don't get division by 0.
  return np.sum ((h2 - h1) ** 2 / (h2 + h1 + 1e-6))


# Parameters:
#   h1, h2: Two normalized histograms btw which to compute the L2 distance
def l2_dist (h1, h2):

  # sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  return np.sqrt (np.sum ((h2 - h1) ** 2))


# Returns a distance btw 0 and 1
# Parameters:
#   h1, h2: Two normalized 3D histograms btw which to compute the inner product
#     distance.
def inner_prod_dist (h1, h2):

  # Dot product is cos. If h1 == h2 or h1 == -h2, then cos is 1 or -1, resp.
  #   If h1 _|_ h2, then cos is 0. So bigger the cosine magnitude, the closer
  #   the two vectors. This is opposite to definition of "distance", which is
  #   smaller the closer. So do 1 minus.
  # dot product = |A||B| cos (theta)
  #   cos(theta) = dot product / (|A| |B|)
  return 1 - np.abs (np.dot (linearize_hist (h1), linearize_hist (h2)) /
    (np.linalg.norm (h1) * np.linalg.norm (h2)))


# Custom SVM kernel, use with sklearn svm.SVC(kernel=inner_prod_kernel).
# Parameters:
#   A: nSamples1 x nDims, 2D NumPy matrix
#   B: nSamples2 x nDims, 2D NumPy matrix
# Returns nSamples1 x nSamples2 kernel matrix, 2D NumPy matrix.
# Ref:
#   Function signature requirements, search "Using Python function as kernels"
#     on this page
#     http://scikit-learn.org/stable/modules/svm.html
def inner_prod_kernel (A, B):

  # (na x d) * (nb x d)^T
  # (na x d) * (d x nb), each row sample of A is multiplied by each column
  #   (row transposed) sample of B. This is na x nb operations, which gives you
  #   na * nb scalar products.
  # Result is a na x nb matrix. Each (ia, ib)th element is the dot product of
  #   (ia)th row sample in A and (ib)th sample in B.
  # Add a small number, in case of division by 0.
  D = np.divide (np.dot (A, B.T),
    np.linalg.norm (A, axis=1).reshape (A.shape[0], 1) + 1e-6)
  E = np.divide (D, np.linalg.norm (B, axis=1).reshape (1, B.shape[0]) + 1e-6)

  return np.abs (E)
  # I would think this is more correct... but it gives accuracy 0% out of 2
  #   shapes, a cube and a sphere! Without the 1 minus, it gets 100%... weird..
  #   does that mean my inner_prod_dist() is wrong??
  #return (1 - np.abs (E))

  # To test on Python shell:
  '''
na = 3
nb = 2

# 3 samples, 4 dimensions per sample. d = 4, na = 3
A = np.random.rand (na,4)
# 2 samples, 4 dimensions per sample. d = 4, nb = 2
B = np.random.rand (nb,4)

# This gives norm of rows of A. 1 x 3 vector
np.linalg.norm (A, axis=1)
# This gives norm of rows of B. 1 x 2 vector
np.linalg.norm (B, axis=1)

# na x nb. Each (ia, ib)th element is the dot product of (ia)th row sample in
#   A and (ib)th sample in B.
C = np.dot (A, B.T)

# Divide each (ia, ib)th element of A*B^T by the magnitude of (ia)th sample
#   in A, and the magnitude of (ib)th sample in B. In MATLAB, this would be
#   D = bsxfun (@rdiv, C, np.linalg.norm (A, axis=1))
#   E = bsxfun (@rdiv, D, np.linalg.norm (B, axis=1).T)
# In NumPy 1.10.0, can simply use keepdims=True instead of the reshape().
D = np.divide (C, np.linalg.norm (A, axis=1).reshape (na, 1))
E = np.divide (D, np.linalg.norm (B, axis=1).reshape (1, nb))

# This is the final inner product distance
dist = 1 - np.abs (E)


  '''

