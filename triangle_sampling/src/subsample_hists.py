# Mabel Zhang
# 31 Aug 2015
#
# This file is only for testing validity of the MATLAB file with same name.
#   Goal is to see why the 1D histograms outputted from
#   sample_pcl_calc_hist.py is different from the ones normalized
#   in matlab by subsample_hists.m.
# Hopefully this file's version is same as one of the above two, then we know
#   which is more correct.
#
# This file only loads one hardcoded file, a 3D histogram, normalize it
#   manually to see whether the result is same as the .py version or the .m
#   version.
#
# If that is inconclusive, then might have to read from the triangles file
#   and call np.histogramdd() again in here, to see if output is same as
#   sample_pcl_calc_hist.py. I really don't see why it would be
#   different, but... we'll have to see. Ugh!!!
#


# 'csv_hists/cup_2b46c83c.csv'

