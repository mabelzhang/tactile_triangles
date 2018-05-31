#!/usr/bin/env python

# Mabel Zhang
# 10 Sep 2015
#
# Print stats of the number of triangles, from
#   [csv_tri | csv_tri_lists]/num_triangles.csv
# saved from triangles_reader.py.
#


# Python
import csv
import argparse
import os

# Numpy
import numpy as np
# For n choose k
from scipy.misc import comb

# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
#from matplotlib.lines import Line2D  # For marker styles
from matplotlib.markers import MarkerStyle

# My packages
from tactile_collect import tactile_config
from util.ansi_colors import ansi_colors
from triangle_sampling.config_paths import get_sampling_subpath


class TrianglesStats:

  def __init__ (self, tri_path, draw_title=True):

    self.tri_path = tri_path
    self.draw_title = draw_title


  def load_nTriangles (self, tri_suffix):

    tri_name = os.path.join (self.tri_path, 'num_triangles' + \
      tri_suffix + '.csv')

    print ('%sLoading triangle stats from %s%s' % \
      (ansi_colors.OKCYAN, tri_name, ansi_colors.ENDC))
 

    #####
    # Read triangles file
    #####
 
    nTriangles = []
 
    with open (tri_name, 'rb') as tri_file:
 
      tri_reader = csv.reader (tri_file)
 
      # There is only one row
      for row in tri_reader:
 
        # Convert strings to ints
        nTriangles = [int (i) for i in row]
 
    print ('Triangle counts for %d objects loaded' % len (nTriangles))

    # TEMPORARY for removing a single outlier in Gazebo IROS data to see true
    #   max. Outlier was 20000+, everything else was 15379 and fewer.
    #nTriangles.remove (24509)
    #print nTriangles

 
 
    #####
    # Print stats
    #####
 
    print ('===== Triangle Stats =====')
 
    print ('Min number of triangles in an object: %d' % \
      (min (nTriangles)))
    print ('Max number of triangles in an object: %d' % \
      (max (nTriangles)))
    print ('Average number of triangles per object: %f' % \
      (np.mean (nTriangles)))
    print ('Median number of triangles: %f' % \
      (np.median (nTriangles)))
    print ('')
 
    return nTriangles
 

  # Entry point
  def plot_multi (self, nSamples, nSamplesRatio, tri_suffixes,
    linestyles=None, mstyles=None):

    #####
    # Init plotting colors
    #####

    # Can set higher numbers when have more things to plot!
    # +1 to get rid of the faint gray at end of nipy_spctral
    n_colors = len (nSamples) + 1
 
    # colormap_name: string from
    #   http://matplotlib.org/examples/color/colormaps_reference.html
    colormap_name = 'nipy_spectral'
 
    # 11 for max # sensors on any segment of hand (palm has 11)
    colormap = get_cmap (colormap_name, n_colors)
 
    # Argument here needs to be in range [0, luc], where luc is the 2nd arg
    #   specified to get_cmap().
    #color = colormap (color_idx)


    #####
    # Plot number of moves vs number of contacts per move
    #####

    fig = None
    fig_log = None

    dots = []
    dots_log = []

    if not linestyles:
      linestyles = ['-'] * len (nSamples)

    if not mstyles:
      mstyles = [None] * len (nSamples)

    for s in range (0, len (nSamples)):

      for r in range (0, len (nSamplesRatio)):

        tri_suffix = tri_suffixes [s] [r]

        nTriangles = self.load_nTriangles (nSamples [s],
          nSamplesRatio [r])

        fig, fig_log, curr_dots, curr_dots_log = \
          self.plot_n_moves_vs_n_contacts (
            nTriangles, tri_suffix, fig=fig, fig_log=fig_log,
            multi=True, color=colormap(s), linestyle=linestyles[s],
            mstyle=mstyles[s])

        # Save the plotting object handles, for plotting legend later
        dots.append (curr_dots)
        dots_log.append (curr_dots_log)


    # Assumption: There's only one nSamplesRatio, many nSamples.
    #   So will only label using nSamples.
    plt.figure (fig.number)
    plt.legend (dots, nSamples)

    plt.figure (fig_log.number)
    plt.legend (dots_log, nSamples)


    # Save to file
    out_name = os.path.join (self.tri_path, 'nMoves_vs_nSensors_multi.eps')
    fig.savefig (out_name, bbox_inches='tight')
    print ('Plot saved to %s' % out_name)

    out_name = os.path.join (self.tri_path,
      'nMoves_vs_nSensors_logScale_multi.eps')
    fig_log.savefig (out_name, bbox_inches='tight')
    print ('Plot saved to %s' % out_name)

    plt.show (fig)
    plt.show (fig_log)



  # Entry point
  def plot_single (self, tri_suffix, plot_nMoves=True):

    nTriangles = self.load_nTriangles (tri_suffix)


    #####
    # Plot number of triangles per object, as histogram
    #####
 
    nbins = 30
 
    # Don't normalize, `.` we want y-axis to be the actual count of objects!
    hist, edges = np.histogram (nTriangles, bins=nbins)
 
    # Note this is NOT histogram width!!! It's only for plotting.
    bar_width = np.max (nTriangles) / nbins * 0.5
 
    plt.bar (edges [0 : len (edges) - 1], hist, width=bar_width)
 
    if self.draw_title:
      plt.title (format ('Histogram of Number of Triangles in Objects,\n Min %d, Max %d, Median %d' % \
        (min(nTriangles), max(nTriangles), np.median(nTriangles))))
    plt.xlabel ('Number of Triangles')
    plt.ylabel ('Objects Count')
 
    # MATLAB "grid on"
    plt.grid (True, color='gray')
 
    # Save to file
    out_name = os.path.join (self.tri_path, 'nTriangles' + tri_suffix + '.eps')
    plt.savefig (out_name, bbox_inches='tight')
    print ('Plot saved to %s' % out_name)
 
    plt.show ()
 

    #####
    # Plot number of moves vs number of contacts per move
    #####

    if plot_nMoves:
      fig, fig_log, _, _ = self.plot_n_moves_vs_n_contacts (nTriangles,
        tri_suffix)
     
      #plt.show (fig)
      #plt.show (fig_log)



  def plot_n_moves_vs_n_contacts (self, nTriangles, tri_suffix,
    fig=None, fig_log=None, multi=False, color=None, linestyle='-', mstyle='o'):


    save_fig=True
    annotate=True

    if multi:
      save_fig=False
      annotate=False

 
    #####
    # Plot number of moves required vs number of sensors that fire on hand,
    #   for mean number of triangles
    #####
 
    # Calculate the data to be plotted
 
    n_sensors_min = 3
    n_sensors_max = 27
    n_sensors = range (n_sensors_min, n_sensors_max + 1)
 
    mean_tris = np.mean (nTriangles)
 
    n_moves = []
 
    for i in n_sensors:
 
      n_moves.append (mean_tris / comb (i, 3))
 
      print ('If contact %d points per move, that is %d triangles exhaustively, average number of triangles needs %f moves' %\
        (i, comb(i, 3), n_moves [len (n_moves) - 1]))
 
 
    # Plot
 
    if not fig:
      fig = plt.figure ()
    else:
      # Ref: http://stackoverflow.com/questions/7986567/matplotlib-how-to-set-the-current-figure
      plt.figure (fig.number)

    # Remember to add comma after returned handle
    # Ref http://stackoverflow.com/questions/11983024/matplotlib-legends-not-working
    dots, = plt.plot (n_sensors, n_moves, mstyle, markersize=5,
      markeredgewidth=0, figure=fig, color=color)
    plt.plot (n_sensors, n_moves, linestyle, figure=fig, color=color)
 
    if self.draw_title:
      title = 'Min Number of Moves Required vs. Number of Sensed Contacts per Move,\n on Average Object'
      if not multi:
        title += format (' (%d triangles)' % (mean_tris))
     
      plt.title (title)
    plt.xlabel ('Number of Contacts per Move')
    plt.ylabel ('Minimum Number of Moves')
 
    # MATLAB "grid on"
    plt.grid (True, color='gray')
 
    if annotate:
      # Ref: http://stackoverflow.com/questions/22272081/label-python-data-points-on-plot
      for i in range (0, len (n_sensors)):
        # Ref: http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.annotate
        plt.annotate ('%.1f' % (n_moves[i]),
          xy=(n_sensors[i], n_moves[i]), rotation=50, verticalalignment='bottom',
          horizontalalignment='left')
 
    # Save to file
    if save_fig:
      out_name = os.path.join (self.tri_path, 'nMoves_vs_nSensors' + tri_suffix + \
        '.eps')
      plt.savefig (out_name, bbox_inches='tight')
      print ('Plot saved to %s' % out_name)
 
    #plt.show ()
 
 
    #####
    # Plot in log scale
    #####

    if not fig_log:
      fig_log = plt.figure ()
    else:
      # Ref: http://stackoverflow.com/questions/7986567/matplotlib-how-to-set-the-current-figure
      plt.figure (fig_log.number)
 
    log_n_moves = np.log (n_moves)
 
    # Remember to add comma after returned handle
    dots_log, = plt.plot (n_sensors, log_n_moves, mstyle, markersize=5,
      markeredgewidth=0, figure=fig_log, color=color, fillstyle='full')
    plt.plot (n_sensors, log_n_moves, linestyle, figure=fig_log, color=color)
   
    if self.draw_title:
      title = 'Log of Min Number of Moves Required vs. Number of Sensed Contacts per Move,\n on Average Object'
      if not multi:
        title += format (' (%d triangles)' % (mean_tris))
      plt.title (title)
    plt.xlabel ('Number of Contacts per Move')
    plt.ylabel ('Log of Minimum Number of Moves')
 
    # MATLAB "grid on"
    plt.grid (True, color='gray')
 
    if annotate:
      # Ref: http://stackoverflow.com/questions/22272081/label-python-data-points-on-plot
      for i in range (0, len (n_sensors)):
        plt.annotate ('%.1f' % (log_n_moves[i]),
          xy=(n_sensors[i], log_n_moves[i]), rotation=30,
          verticalalignment='bottom', horizontalalignment='left')
 
    # Save to file
    if save_fig:
      out_name = os.path.join (self.tri_path, 'nMoves_vs_nSensors_logScale' + \
        tri_suffix + '.eps')
      plt.savefig (out_name, bbox_inches='tight')
      print ('Plot saved to %s' % out_name)
 
    #plt.show ()

    return fig, fig_log, dots, dots_log



def main ():

  #####
  # Parse command line arguments
  #####

  arg_parser = argparse.ArgumentParser ()

  arg_parser.add_argument ('histSubdirParam1', type=str,
    help='Used to create directory name to read from.\n' + \
      'For point cloud, nSamples used when triangles were sampled.\n' + \
      'For real robot data, specify the sampling density you want to classify real objects with, e.g. 10, will be used to load histogram bin configs.\n' + \
      'For Gazebo, triangle params desired, with no spaces, e.g. l0,l1,a0\n' + \
      'For mixed, enter 2 point cloud params first, then 2 Gazebo params, 4 total')
  arg_parser.add_argument ('histSubdirParam2', type=str, nargs='+',
    help='Used to create directory name to read from.\n' + \
      'For point cloud, nSamplesRatio used when triangles were sampled.\n' + \
      'For real robot data, specify the sampling density you want to classify real objects with, e.g. 0.95, will be used to load histogram bin configs.\n' + \
      'For Gazebo, number of bins in 3D histogram, with no spaces, e.g. 10,10,10\n' + \
      'For mixed, enter 2 point cloud params first, then 2 Gazebo params, 4 total.')

  # Ref: Boolean (Ctrl+F "flag") https://docs.python.org/2/howto/argparse.html
  arg_parser.add_argument ('--pcd', action='store_true', default=False,
    help='Boolean flag, no args. Run on point cloud data in csv_tri_lists/')
  arg_parser.add_argument ('--real', action='store_true', default=False,
    help='Boolean flag, no args. Run on real robot data in csv_tri/')
  arg_parser.add_argument ('--gazebo', action='store_true', default=False,
    help='Boolean flag, no args. Run on Gazebo data in csv_gz_tri/')

  arg_parser.add_argument ('--multi', action='store_true', default=False,
    help='Plot overlapped number of moves vs number of contacts per move, for many sampling densities (adjust these in code). nSamples and nSamplesRatio are ignored if this flag is specified.')

  arg_parser.add_argument ('--truetype', action='store_true', default=False,
    help='Tell matplotlib to generate TrueType 42 font, instead of rasterized Type 3 font. Specify this flag for uploading to ICRA.')
  arg_parser.add_argument ('--notitle', action='store_true', default=False,
    help='Do not plot titles, for paper figures, description should all be in caption.')

  args = arg_parser.parse_args ()

  if args.pcd & args.real & args.gazebo == 1 or \
    args.pcd | args.real | args.gazebo == 0:
    print ('%sYou must choose ONE of --pcd, --real, and --gazebo. Check your args and try again. Terminating...%s' % ( \
      ansi_colors.FAIL, ansi_colors.ENDC))
    return

  print ('%sPlot multi set to %s %s' % \
    (ansi_colors.OKCYAN, args.multi, ansi_colors.ENDC))


  # Set to True to upload to ICRA. (You can't view the plot in OS X)
  # Set to False if want to see the plot for debugging.

  # For ICRA PDF font compliance. No Type 3 font (rasterized) allowed
  #   Ref: http://phyletica.org/matplotlib-fonts/
  # You can do this in code, or edit matplotlibrc. But problem with matplotlibrc
  #   is that it's permanent. When you export EPS using TrueType (42), Mac OS X
  #   cannot convert to PDF. So you won't be able to view the file you
  #   outputted! Better to do it in code therefore.
  #   >>> import matplotlib
  #   >>> print matplotlib.matplotlib_fname()
  #   Ref: http://matplotlib.1069221.n5.nabble.com/Location-matplotlibrc-file-on-my-Mac-td24960.html
  if args.truetype:
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

  draw_title = not args.notitle


  #####
  # Set input path
  #####

  if args.pcd:
    # Synthetic 3D model data, saved from running sample_pcl.cpp and
    #   sample_pcl_calc_hist.py
    tri_path = tactile_config.config_paths ('custom',
      'triangle_sampling/csv_tri_lists/_stats/')
  elif args.real:
    # Real-robot tactile data, collected by triangles_collect.py
    tri_path = tactile_config.config_paths ('custom',
      'triangle_sampling/csv_tri/_stats/')
  elif args.gazebo:
    tri_path = tactile_config.config_paths ('custom',
      'triangle_sampling/csv_gz_tri/_stats/')

  thisNode = TrianglesStats (tri_path, draw_title)


  if args.multi:
    # Gazebo doesn't have this mode, I never needed it.
    if args.pcd or args.real:

      nSamples = [10, 25, 50, 100, 150, 200, 250, 300]
      #nSamples = [10, 25]#, 50, 100, 150, 200, 250, 300]
      nSamplesRatio = [0.95]

      tri_suffixes = []
      for s in range (0, len (nSamples)):
        tri_suffixes.append ([])
        for r in range (0, len (nSamplesRatio)):
          tri_suffixes [s].append ('_' + get_sampling_subpath (nSamples,
            nSamplesRatio, endSlash=False))
     
      # Ref: matplotlib.org/examples/lines_bars_and_markers/line_styles_reference.html
      #   matplotlib.org/1.3.1/examples/pylab_examples/line_styles.html
      #linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
      linestyles = None
     
      # Ref: matplotlib.org/api/markers_api.html MarkerStyle()
      #   matplotlib.org/api/markers_api.html
      markerstyles = ['o', 'd', '^', 'D', 's', '*', '<', 'v']
      #markerstyles = MarkerStyle.filled_markers
     
      thisNode.plot_multi (nSamples, nSamplesRatio, tri_suffixes,
        linestyles, markerstyles)

  else:
    if args.pcd or args.real:
      tri_suffix = '_' + get_sampling_subpath (nSamples, nSamplesRatio,
        endSlash=False)
      thisNode.plot_single (tri_suffix, plot_nMoves=True)

    elif args.gazebo:
      # No sampling rate in Gazebo. Triangle params and nbins don't affect
      #   number of triangles either, they are more downstream parameters,
      #   after triangles have been computed. They're for the histograms.
      thisNode.plot_single ('', plot_nMoves=False)


if __name__ == '__main__':
  main ()

