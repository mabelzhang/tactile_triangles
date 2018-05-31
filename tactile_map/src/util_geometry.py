#!/usr/bin/env python

# Mabel Zhang
# 15 Apr 2015
#
# Utility functions using geometry
#

import numpy as np
from scipy import linalg as sp_linalg


# Estimate the axis of revolution as the line that intersects the most lines
#   (normals) passed in. Pottmann CAD 1999.
# Parameters:
#   lines: [[(x1 y1 z1), (x2 y2 z2)]_1, .... [(x1 y1 z1), (x2 y2 z2)]_n]
#     A list of k lists of size-2 Numpy arrays. Each size-2 Numpy array 
#     represents a line, described by 2 points on the line.
#     [] denotes Python list, () denotes Numpy.
#     lines[i][0] is line i point 1, lines[i][1] is line i point 2.
# Returns the estimated axis of revolution, the line that intersects the most
#   number of lines passed in param. In format of two points on the axis,
#   [(ax1 ay1 az1), (ax2 ay2 az2)], a list of two size-3 Numpy arrays.
# Ref: http://www.sciencedirect.com/science/article/pii/S0010448598000761
'''
To test on Python command line (this is a bit outdated, as I changed some lines
  in actual code. This needs to be updated here):
import numpy as np
lines = [[np.asarray([1,3,6]), np.asarray([3,4,6])],
  [np.asarray([9,3,5]), np.asarray([6,2,5])],
  [np.asarray([3,5,1]), np.asarray([5,2,8])]]

l = [li[1] - li[0] for li in lines]
l = [li / np.linalg.norm (li) for li in l]
l_bar = [np.cross (lines[i][0], l[i]) for i in range (0, nLines)]
lines_plucker = [np.concatenate ((l_bar[i], l[i])) for i in range (0, nLines)]

L = np.reshape (lines_plucker, (3, 6)).transpose ()
L = np.matrix (L)

M = L * L.transpose ()
eigvals, eigvecs = np.linalg.eig (M)
eva_nonneg = np.asarray ([(i, eigvals[i]) for i in range (0, len(eigvals)) if eigvals[i] >= 0])
eva_min = [eva_nonneg[i] for i in range (0, len (eva_nonneg))
  if eva_nonneg[i][1] == min (eva_nonneg [:, 1])]
lambda_val = eva_min [0][1]
lambda_idx = int (eva_min [0][0])

C = eigvecs [:, lambda_idx]
c = np.asarray (C [0:3])
c_bar = np.asarray (C [3:6])

p = np.dot (c.transpose(), c_bar) / np.dot (c.transpose(), c)
A = np.concatenate ((c, c_bar - p * c))

'''
def est_linear_complex (lines, lambda_idx=-1):

  nLines = len (lines)


  ## Convert each of the k lines into Plucker coordinates.
  # Plucker coordinates:
  #   A line is represented as unit vector and moment,
  #     (u, m) = (u1 u2 u3 m1 m2 m3),
  #     u = (p2 - p1) / ||p2 - p1||, m = cross (p, u), p is a point on line, 
  #     can be p1 or p2.
  # Ref:
  #   http://www.sciencedirect.com/science/article/pii/S0010448598000761
  #   http://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates
  #     Wikipedia definition is diff from Pottmann, but more intuitive to see.

  # l. Unit vector of line. (p2 - p1) / ||p2 - p1||
  # List of k size-3 Numpy arrays. Each size-3 array is unit vector for line i.
  l = [li[1] - li[0] for li in lines]
  # Normalize
  l = [li / np.linalg.norm (li) for li in l]

  # l^bar. Moment. l^bar = cross (p1, l)
  # List of k size-3 Numpy arrays. Each size-3 array is moment for line i.
  # Assumption: len(lines) == len(l)
  l_bar = [np.cross (lines[i][0], l[i]) for i in range (0, nLines)]

  # (l^bar, l), reversed on purpose for matrix L later.
  #   Need reversed `.` moment is defined by dot(c^bar, l) + dot(c, l^bar), 
  #   i.e. 1st 3 components dot the last 3 components. So if C is in correct
  #   order (c, c^bar), then L has to be in reverse, (l^bar, l). Then
  #   C * L * L^T * C will give the correct moment.
  # List of k size-6 Numpy arrays.
  lines_plucker = [np.concatenate ((l_bar[i], l[i])) for i in range (0, nLines)]


  # Construct L, 6 x k matrix
  # Reshape the (list of k size-6 Numpy arrays) into a matrix. 6 x k.
  # Easiest way is reshape into k x 6, then transpose it into 6 x k.
  L = np.reshape (lines_plucker, (nLines, 6)).T
  L = np.matrix (L)
  print ('L:\n' + str(L))


  # Construct moments (half of the moments really, m is C and L) matrix M.
  #   M = L * L^T. 6 x 6
  M = L * L.T


  ## Compute lambda, smallest non-negative eigenvalue of M.

  # Use generalized eigval instead of ordinary. Pass in D for right hand side
  #   (default I, for ordinary eigval problem).
  # API: http://docs.scipy.org/doc/scipy-0.9.0/reference/generated/scipy.linalg.eig.html#scipy.linalg.eig
  # Ref: http://stackoverflow.com/questions/24752393/numpy-generalized-eigenvalue-probiem
  D = np.diag ([1,1,1,0,0,0])
  (eigvals, eigvecs) = sp_linalg.eig (M, D)

  # Keep the real parts
  # Ref: http://stackoverflow.com/questions/24752393/numpy-generalized-eigenvalue-probiem
  eigvecs = eigvecs.real
  eigvals = eigvals.real

  # If user didn't pass in a test choice, find the correct lambda
  if lambda_idx < 0:
    # Find non-negative eigenvalues
    # (index, value) pairs of non-negative eigenvalues
    # Convert to Numpy array to enable multi-axis slice indicing
    #   Ref: http://ilan.schnell-web.net/prog/slicing/
    # This needs to be absolute 0, not a soft 0 like 1e-6. Because this gets the
    #   real min eigval. We need the smallest, in order for X^T * D * X = 1 and
    #   (M-lambda*D)-X = 0 to be as close to 1 and 0 as possible. Even if it's
    #   1e-16 scale, it's okay. Just check soft zero when convert Plucker A to
    #   Cartesian.
    #eva_nonneg = np.asarray (
    #  [(i, eigvals[i]) for i in range (0, len(eigvals)) if eigvals[i] >= 0])
    # Looks like correct eigval should not eliminate negative
    #   Do eliminate the inf's!
    eva_nonneg = np.asarray (
      [(i, eigvals[i]) for i in range (0, len(eigvals))
      if not np.isinf (eigvals[i])])
 
    # Find min, among the non-neg eigvals
    # [(index, value)] pair, hopefully list size is 1 (unique eigvals)
    eva_min = [eva_nonneg[i] for i in range (0, len (eva_nonneg))
      if eva_nonneg[i][1] == min (eva_nonneg [:, 1])]
 
    # Just take first one, if eigvals aren't unique
    lambda_val = eva_min [0][1]
    lambda_idx = int (eva_min [0][0])

  else:
    lambda_val = eigvals [lambda_idx]

  print ('Eigvals: \n' + str(eigvals))
  #print (eva_nonneg)
  print ('Smallest eigval: \n' + str(lambda_val))


  # Find C, eigenvector corresponding to smallest non-neg eigval of M. Eigvecs
  #   are columns.
  #   Ref: http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html
  # 6 x 1 Numpy matrix
  C = np.reshape (eigvecs [:, lambda_idx], (6, 1))
  # 3 x 1 Numpy array
  c = np.asarray (C [0:3])
  c_bar = np.asarray (C [3:6])


  ## Check correctness of solution

  #print (eigvecs)
  #print (lambda_idx)
  #print (C)

  #print (c)
  #print (c_bar)

  # Solution X = C should satisfy:
  #   1. (M - lambda * D) * X = 0,
  #      X^T * D * X = 1
  #      where D = diag (1, 1, 1, 0, 0, 0)
  #   2. ||C[0:3]|| = 1
  print ('Checking correctness of Pottmann solution...')
  print ('This should be 0:')
  print ((M - lambda_val * D) * C)
  print ('First three should be close to 1 (usually only one 1 though):')
  print (np.diag (C.T * D * C))

  #print ('C.T * D * C for all eigvecs:')
  #for i in range (0, 6):
  #  C_tmp = np.reshape (eigvecs [:, i], (6, 1))
  #  print (np.diag (C_tmp.T * D * C_tmp))
    

  # It doesn't have to be, for solution to be correct. I get correct sols w/o
  #   this being 1.
  #print ('This |c| should print unit length:')
  #print (np.linalg.norm (c))


  ## Compute axis A = (a, a^bar) = (c, c^bar - pc),
  #   where pitch p = dot(c, c^bar) / c^2

  # Scalar. 1 x 1 Numpy array. (1 x 3) dot (3 x 1)
  p = np.dot (c.T, c_bar) / np.dot (c.T, c)

  # 6 x 1 Numpy array. Plucker coordinates of axis of revolution
  # A is not nec unit length (this is good for us to retrieve Cartesian line).
  A = np.concatenate ((c, c_bar - p * c))

  #print (np.dot (c.T, c_bar))
  #print (c[0] * c_bar[0])
  #print (c[1] * c_bar[1])
  #print (c[2] * c_bar[2])
  #print (np.dot (c.T, c))
  #print (p)

  print ('A: \n%f\n%f\n%f\n%f\n%f\n%f' % (A[0], A[1], A[2], A[3], A[4], A[5]))
  print ('Unit A[0:3].T: ' + str(A[0:3].T / np.linalg.norm (A[0:3])))
  

  ## Convert Plucker coordinate to two points on line in R3 space.
  #   http://www.loria.fr/~lazard/ARC-Visi3D/Pant-project/files/Plucker-3D.html
  #   They use a differnet convention for plucker coords than Pottmann, L[2]
  #     L[4] L[5] are the difference, L[0] L[1] L[3] are the cross products.
  #   http://www.loria.fr/~lazard/ARC-Visi3D/Pant-project/files/plucker.html
  # See my derivation on paper for converting Pottmann's Plucker to Cartesian.

  axis_pt1 = np.asarray ([0.0, 0.0, 0.0])
  axis_pt2 = np.asarray ([0.0, 0.0, 0.0])

  # Use a soft 0. Else might end up dividing 1e-16 magnitude numbers, getting
  #   super huge results in Cartesian.
  SOFT_ZERO = 1e-6

  # Nonzero
  if abs (A[0]) > SOFT_ZERO:
    axis_pt1 [0] = 1
    axis_pt1 [1] = (A[1] - A[5]) / A[0]
    axis_pt1 [2] = (A[2] + A[4]) / A[0]

    axis_pt2 [0] = -1
    axis_pt2 [1] = - (A[1] + A[5]) / A[0]
    axis_pt2 [2] = - (A[2] - A[4]) / A[0]

    print ('A[0] nonzero')

  elif abs (A[1]) > SOFT_ZERO:
    axis_pt1 [0] = (A[0] + A[5]) / A[1]
    axis_pt1 [1] = 1
    axis_pt1 [2] = (A[2] - A[3]) / A[1]

    axis_pt2 [0] = - (A[0] - A[5]) / A[1]
    axis_pt2 [1] = -1
    axis_pt2 [2] = - (A[2] + A[3]) / A[1]

    print ('A[1] nonzero')

  elif abs (A[2]) > SOFT_ZERO:
    axis_pt1 [0] = (A[0] - A[4]) / A[2]
    axis_pt1 [1] = (A[1] + A[3]) / A[2]
    axis_pt1 [2] = 1

    axis_pt2 [0] = - (A[0] + A[4]) / A[2]
    axis_pt2 [1] = - (A[1] - A[3]) / A[2]
    axis_pt2 [2] = -1

    print ('A[2] nonzero')

  else:
    # This should never happen!
    print ('est_linear_complex(): None of the first three Plucker coordinates in resulting axis A is non-zero. This should never happen! Check code for math.')


  #print (axis_pt1)
  #print (axis_pt2)
  # Print in readable notation
  print ('%f %f %f' % (axis_pt1[0], axis_pt1[1], axis_pt1[2]))
  print ('%f %f %f' % (axis_pt2[0], axis_pt2[1], axis_pt2[2]))
  print ('')

  return [axis_pt1, axis_pt2]



# As presented in Bicchi IJRR 1993, Contact estimation from force measurements.
#   This function is Section 5 method, for soft finger area contact.
#   If measured torque q = 0, then calls Section 4 method to solve for point
#   contact.
# Parameters:
#   force f, torque m: Numpy arrays, 3 x 1. Measured force f and torque m at 
#     force-torque sensor, at some offset frame from unknown contact centroid
#     on fingertip.
#   A: 3 x 3 diagonal constant coefficient matrix, in equation for surface S.
#      Surface S defined as:
#      S(r) = r^T * A^T * A * r - R^2 = 0. See Bicchi IJRR 1993 eqn (9).
#      A can be written in diagonal form, diag(1/alpha, 1/beta, 1/gamma).
#   R: scalar scale factor, in eqn for surface S(*) = 0 above.
# Outputs:
#   contact centroid c of soft finger contact (area contact),
#   contact normal n at contact centroid
#   torque q at contact centroid
#   K used in calculations
#   (force p at contact centroid is same as measured force f at sensor)
def calc_contact_normal (f, m, A, R, e):

  fT_dot_m = np.dot (f.T, m)

  ## Check Method 4 assumption: dot (f, m) = 0
  #   If true, that means measured torque at sensor q = 0. This means contact
  #   is a single point, not a soft areas. In reality, this will never be true.
  thresh = 1e-6
  if abs (fT_dot_m) < thresh:
    print ('Detected f.T * m = %g. Calculating estimate for point contact instead...' % (fT_dot_m))
    return calc_contact_normal_point (f, m, A, R, e)


  # Diagonal elts of A. 3 x 1
  A_vec = np.diag (A)
  A_alpha = 1.0 / A_vec [0]
  A_beta = 1.0 / A_vec [1]
  A_gamma = 1.0 / A_vec [2]

  det_A = np.linalg.det (A)

  sigma = det_A*det_A * (np.linalg.norm (np.linalg.inv(A) * m) ** 2) - \
    R*R * (np.linalg.norm (A*f) ** 2)


  ## Solve for one unknown, K. Bicchi eqn (16)

  K_term1 = -np.sign(fT_dot_m) / (np.sqrt(2)*R*det_A)
  K_term1 = np.squeeze (np.asarray (K_term1))

  K_inner_sqrt = sigma*sigma + 4 * det_A*det_A * R*R * (fT_dot_m)**2
  # Convert 2D matrix to scalar
  K_inner_sqrt = np.squeeze (np.asarray (K_inner_sqrt))

  # Inner sqrt has 2 results.
  # sqrt(2) and outer sqrt(sigma + sqrt(...)) have total of 2 results, not 4,
  #   `.` their signs multiply. So just need to put + or - on outer result.
  # Total 4 possible solutions for K.
  Ks = np.asarray ([ \
      K_term1 * np.sqrt (sigma + np.sqrt (K_inner_sqrt)),
    - K_term1 * np.sqrt (sigma + np.sqrt (K_inner_sqrt)),
      K_term1 * np.sqrt (sigma - np.sqrt (K_inner_sqrt)),
    - K_term1 * np.sqrt (sigma - np.sqrt (K_inner_sqrt))])

  #print ('4 K\'s: %g %g %g %g' %(Ks[0], Ks[1], Ks[2], Ks[3]))


  # Find the valid K.
  #   Does not make assumption that correct K must > 0. Just check eqn (7),
  #     to be safe.
  for i in range (0, len (Ks)):

    # Current possible solution for K
    Ki = Ks [i]

    # Skip impossible solutions
    if np.isnan (Ki):
      continue

    # Gamma. 3 x 3, in terms of K and measured force components f1 f2 f3
    Gamma_i = np.matrix ([
      [Ki / (A_alpha * A_alpha), f[2,0],               -f[1,0]],
      [-f[2,0],                Ki / (A_beta * A_beta), f[0,0]],
      [f[1,0],                 -f[0,0],              Ki / (A_gamma * A_gamma)]])
 

    # Determinant of Gamma
    det_Gamma_i = Ki * (Ki*Ki * det_A*det_A + np.linalg.norm (A*f) ** 2)

    # Solve for the other unknown, c.
    # Contact centroid c. Bicchi eqn (14). Paper specifically says to substitute
    #   K into (14) to get solution for c.
    #   A^(-2) means 1 over the square of each diagonal term in A.
    #   Ref: http://math.stackexchange.com/questions/340321/raising-a-square-matrix-to-a-negative-half-power
    ci = (1 / det_Gamma_i) * (
          Ki*Ki * det_A*det_A * A**(-2) * m +
          Ki*np.cross ((A*A*f), m, axis=0) +
          np.multiply (fT_dot_m, f)) + e
 
    # Bicchi eqn (14) first part
    # Result didn't change from above
    #ci = np.linalg.inv (Gamma_i) * m + e
 
    # 5.1.1 simplified c solution for spheres
    # Result didn't change from above
    #c = (1 / (Ki * (Ki*Ki + np.linalg.norm(f)**2))) * (
    #  Ki*Ki*m + K*(np.cross(f, m, axis=0)) + np.multiply (fT_dot_m, f)) + e
 

    # Surface normal n at contact centroid. 3 x 1. Bicchi eqn (10)
    (AAci, ni) = calc_normal (A, ci, e)

    # Check satisfaction of eqn (7), the assumption of nonadhesive force,
    #   f.T * n(c) < 0.
    if np.dot (f.T, ni) < 0:
      #print ('Found correct K: %g' %(Ki))
      K = Ki
      c = ci
      AAc = AAci
      n = ni
      #break


  # Contact torque at contact centroid. 3 x 1. Bicchi eqn (10). Paper
  #   specifically says to substitute K into (10) to get solution for q.
  q = K * AAc
  # Answer seems to use eqn (11), instead of (10), `.` it satisfies (11) but not (10). But in paper, it specifically says to substitue K into (10) to get q!
  #q = m - np.cross (c, f, axis=0)

  return (c, n, q, K)


# Do not call this function from outside. Outside function should call
#   calc_contact_normal() instead. That function will direct to this one if 
#   measured torque q = 0.
# As presented in Bicchi IJRR 1993, Contact estimation from force measurements.
#   This function is Section 4 method, for point contact with low friction,
#   assumption is q = 0.
# Parameters:
#   f: 3 x 1 measured force at force-torque sensor
#   m: 3 x 1 measured torque (i.e. moment) at force-torque sensor
#   A: 3 x 3 constant coefficient matrix, in equation for surface S(*) = 0.
#      Surface S defined as:
#      S(r) = r^T * A^T * A * r - R^2 = 0. See Bicchi IJRR 1993 eqn (9).
#      A can be written in diagonal form, diag(1/alpha, 1/beta, 1/gamma).
#   R: scalar scale factor, in eqn for surface S(*) = 0 above.
# Returns:
#   (c, n, q, None). The None is a placeholder for K, so that this fn returns
#   same format as calc_contact_normal(). But this method doesn't have a K, so
#   just return None.
def calc_contact_normal_point (f, m, A, R, e):

  f_norm = np.linalg.norm (f)

  # Sanity check
  if f_norm == 0:
    print ('Error in util_geometry calc_contact_normal_point(): Force vector magnitude is 0. Skipping this calculation.')
    return [np.matrix ([float('nan')] * 3).T] * 4

  ## r0. A point on line r intersecting fingertip surface S. 3 x 1
  #   Bicchi eqn (6)
  r0 = np.asmatrix (np.cross (f, m, axis=0) / (f_norm ** 2))


  ## Find intersection of wrench axis (a line) r with fingertip surface S
  #    Bicchi Section 5.

  # 3 x 1
  f_prime = A * f
  # 3 x 1
  r0_prime = A * r0

  # 1 x 1 Numpy matrix. Dot product
  fp_dot_r0p = f_prime.T * r0_prime

  # Scalar float. Norm
  fp_norm = np.linalg.norm (f_prime)

  # For debugging
  '''
  print ('f: ' + str(f.T))
  print ('||f||^2: ' + str(np.linalg.norm(f)**2))
  print ('r0: ' + str(r0.T))
  print ('||r0||^2: ' + str(np.linalg.norm(r0)**2))
  print ('second term in sqrt: ' + str(fp_norm * fp_norm * (np.linalg.norm (r0_prime) ** 2 - R * R)))
  print ('f.T * r0: ' + str (np.dot (f.T, r0)))
  print ('(f.T * r0)^2: ' + str(fp_dot_r0p * fp_dot_r0p))
  print ('')
  '''

  # 1 x 1 Numpy matrix
  val_in_sqrt = fp_dot_r0p * fp_dot_r0p - \
    fp_norm * fp_norm * (np.linalg.norm (r0_prime) ** 2 - R * R)
  # 5.1.1 eqn for sphere
  #val_in_sqrt = R*R - np.linalg.norm (np.cross (f, m, axis=0))**2 / \
  #    np.linalg.norm(f)**4

  # If no solution, return no solution
  # Guessing "no solution" happens in two cases. 1. When value inside sqrt is
  #   negative, then sqrt is imaginary. 2. Denominator is 0.
  #   TODO verify this guess is true, that there are no other no-solution cases.
  # Ref: http://stackoverflow.com/questions/9814577/identifying-a-complex-number
  if fp_norm == 0:
    print ('Error in util_geometry calc_contact_normal_point(): No solution. Denominator is 0, when solving for lambda s.t. estimated contact force line r(lambda) intersects fingertip surface S.')
    return [np.matrix ([float('nan')] * 3).T] * 4
  if val_in_sqrt < 0:
    print ('Error in util_geometry calc_contact_normal_point(): No solution. Value (%g) inside sqrt is negative, when solving for lambda s.t. estimated contact force line r(lambda) intersects fingertip surface S.' % (val_in_sqrt))
    return [np.matrix ([float('nan')] * 3).T] * 4

  # TODO: If don't get any intersections, set some small threshold to allow
  #   to bypass if-stmt above. For generated case, this shouldn't be needed.
  #   For real world, there is noise, so this is needed. Accept points on r
  #   that are within some threshold to S, doesn't need to be exactly
  #   intersecting S. This is a tip in Bicchi section 7 paragraph 2.
  #   "A good approximation can be assumed to be the point on the sensor surface
  #   [S] closest to the calculated centroids [r1, r2]."
  #   Not sure how to find that though. Sample S(r), with r in a for-loop??

  # lambda. 1 x 1 Numpy matrix. Parametrizes r = r0 + lambda * f.
  #   We want the lambda at intersection of line r and surface S. At this point,
  #   r(lambda) is an approximataion of c' for contact centroid c, `.`
  #   contact must be on surface S.
  # Two solution, +/- of a sqrt.
  lambda1 = (- fp_dot_r0p - np.sqrt (val_in_sqrt)) / (fp_norm * fp_norm)
  lambda2 = (- fp_dot_r0p + np.sqrt (val_in_sqrt)) / (fp_norm * fp_norm)

  # 5.1.1 equation for lambda, for a sphere
  #   Note: If use this, remember to uncomment the val_in_sqrt for 5.1.1 above
  #lambda1 = - (1 / np.linalg.norm (f)) * np.sqrt (val_in_sqrt)
  #lambda1 = (1 / np.linalg.norm (f)) * np.sqrt (val_in_sqrt)


  ## c'_i = r(lambda_i). An approximation of contact centroid c.
  #  r is a line intersecting fingertip surface S. "wrench axis." The
  #    intersection(s) is an approximation c' for contact centroid c.
  #    Bicchi eqn (5)
  #  r(lambda) is the point of intersection.
  r1 = r0 + np.multiply (lambda1, f) + e
  r2 = r0 + np.multiply (lambda2, f) + e


  ## c'. Approximation for contact centroid c, but c' does not have the
  #   properties of contact centroid.
  # Pick the lambda at which r directs into surface S, opposite the direction
  #   of surface normal n(lambda). `.` we want compressive force, force that
  #   presses into fingertip surface S.

  # 3 x 1 numpy matrix
  (_, n1) = calc_normal (A, r1, e)
  (_, n2) = calc_normal (A, r2, e)

  # Pick the r at which the surface normal points inwards, i.e. oppo dir of r,
  #   which is an approximation of contact force. Dot product < 0 indicates
  #   the two vectors are more than 90 degrees apart. When measured against
  #   surface tangent, which is 90 degs offset from surface normal, force 
  #   vector being > 90 degs from normal (which points outwards) means the 
  #   force points inwards. Then it is a compressive force that we want.
  # Bicchi eqn (7)
  if f.T * n1 < 0:
    # 1 x 1 Numpy matrix
    lambda_intersect = lambda1
    # 3 x 1 Numpy matrix
    c_prime = r1
    # 3 x 1 Numpy matrix
    n = n1
  elif f.T * n2 < 0:
    lambda_intersect = lambda2
    c_prime = r2
    n = n2
  else:
    print ('Error in util_geometry calc_contact_normal_point(): No solution. No compressive force r found that points inwards of fingertip surface.')
    return [np.matrix ([float('nan')] * 3).T] * 4


  # 3 x 1 numpy matrix. Bicchi eqn (3).
  # This should be close to 0. Otherwise you shouldn't be calling this fn!
  #   This fn assumes q = 0.
  q = m - np.cross (c_prime, f, axis=0)

  return (c_prime, n, q, None)


# Helper fn for Bicchi methods, for checking solution's consistency with
#   assumption of nonadhesive contact, f.T * n(c) < 0, Bicchi eqn (7).
#   Where f is force, n is normal at point c on surface S.
#   This fn returns n(c).
# Surface normal of surface S at point c. Bicchi eqn (10).
#   Surface is defined by: S(c) = c^T * A^T * A * c - R^2 = 0, Bicchi eqn (9)
# Parameters:
#   A: Constant coefficient matrix in eqn of S above.
#   c: Parameterizes the surface. Surface S(c) (paper uses S(r) and S(c)
#      interchangeably).
#   e: Coordinates of the center of ellipsoid.
# Returns:
#   AAc: A * A * (c - e). Gradient of surface S at point c
#   n: gradient(S(c)) / ||gradient(S(c))||, normal at point c on surface S
def calc_normal (A, c, e=np.matrix([[0.0, 0.0, 0.0]]).T):
  # (3 x 1) = (3 x 3) * (3 x 3) * (3 x 1)
  #gradient_S_at_c = A * A * c
  gradient_S_at_c = A * A * (c - e)
  return (gradient_S_at_c, gradient_S_at_c / np.linalg.norm (gradient_S_at_c))




# Parameters:
#   l1, l2: Two lines, each represented by Numpy array (vx, vy, vz, x0, y0, z0),
#     where (vx, vy, vz) is a vector colinear to the line, and
#           (x0, y0, z0) is a point on the line.
#def intersect_two_lines (l1, l2):

  # Line equation in 3D
  #   (x, y, z) = (x0, y0, z0) + t (vx, vy, vz)
  #   where (x0, y0, z0) is a pointo n the line
  #     t is the parameter
  #     (vx, vy, vz) is direction vector
  # Ref: http://www.netcomuk.co.uk/~jenolive/vect17.html



