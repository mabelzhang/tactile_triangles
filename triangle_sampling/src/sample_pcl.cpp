// Mabel Zhang
// 27 Jul 2015
//
// Modified and updated from my ./sample_pt_cloud.py
//
// Quick note:
//   There's a "Fast" mode and a "Slow" mode for running this file.
//   For "Fast" mode, set doRViz = false.
//   For "Slow" mode, set doRViz = true.
//   doRViz lets you visualize the triangles to make sure they are correct.
//     Slows down running considerably. So if you are outputting after sure
//     that everything is correct, disable it.
//   If you run the script and it's slower than 1 second per object, then check
//     if you forgot to set doRViz = false.
//
// Samples sets of 3 points from a point cloud (or actually, vertices of mesh
//   in an OBJ file).
//
//   Since size of triangles vary in different OBJ files, preferably you use
//   OBJ files with very small triangles, so that sampling them is really
//   like sampling from ANYWHERE on an object surface, i.e. a very dense
//   point cloud. This is the best way to simulate finger touches, and
//   therefore train data for use to model noisy robot finger touches.
//
//   If you have OBJ files with very big triangles, sampling from their
//   vertices may not be optimal to train as data to use on real robot hand,
//   since hand's sampling will be a lot more dense - not just limited to a
//   few vertices of very big triangles, but anywhere inside a triangle as 
//   well! So your training data won't represent the real data a hand would
//   get.
//
//   To get smaller triangles, you can use some software (Blender maybe) to
//   break up the triangles, i.e. increase polygon count, to get denser
//   triangles.
//
// The 3 sampled points are recorded as the triangle formed among them,
//   described by 3 parameters of the triangles - a combination of side
//   lengths and angles of the triangle.
//
// Any given sampled triangle is constrained to fit in ReFlex Hand's finger
//   lengths.
//
// This file has to be in C++ (not Python) because sampling is done via
//   PCL Octree.
//
// This file produces no output files. It only publishes triangles by rosmsg.
//   Run a Python node to plot the histograms and save triangles and/or
//   histograms to csv files.
//
//
// To compile:
//   $ catkin_make --pkg triangle_sampling
// To run:
//   $ rosrun tactile_map keyboard_interface.py
//   $ rosrun triangle_sampling sample_pcl
// To see visualization:
//   $ rosrun rviz rviz
//   Select point cloud size (m) = 0.001, alpha = 0.5
//   Enable Marker.
//     For per-iteration, only enable namespaces without *_cumu suffix.
//     For cumulative, enable *_cumu suffix.
//     For clearer triangles, disable sample_voxel* and voxel_hand_sphere.
//
// To plot 3D histograms (and save plots to .eps, using matplotlib):
//   $ rosrun triangle_sampling sample_pcl_plotter.py
//
// To plot histogram intersections (and save them using matplotlib), output
//   3D histograms to csv for training, and/or raw triangles to csv:
//   $ rosrun triangle_sampling sample_pcl_calc_hist.py
//
//
// Ref:
//   http://docs.pointclouds.org/trunk/group__octree.html
//   http://docs.pointclouds.org/trunk/classpcl_1_1octree_1_1_octree_point_cloud_search.html
//   http://pointclouds.org/documentation/tutorials/octree.php
//   My linemod_msgs/src/octree_raytrace.cpp
//


// C++
#include <fstream>  // for ifstream
#include <boost/filesystem.hpp>

// ROS
#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>
#include <std_msgs/String.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Float32.h>

// PCL
#include <pcl/point_cloud.h>
//#include <pcl_ros/point_cloud.h>  // 18 Dec 2016: if pcl/point_cloud.h doesn't work, you might need this
#include <pcl/PCLPointCloud2.h>
#include <pcl/io/obj_io.h>  // loadOBJFile ()
#include <pcl/io/pcd_io.h>  // loadPCDFile ()
#include <pcl/octree/octree.h>
#include <pcl/filters/voxel_grid.h>  // For downsampling point cloud
// Converts btw pcl::PointCloud<> and pcl::PCLPointCloud2
#include <pcl/conversions.h>
#include <pcl/common/transforms.h>

//#include <pcl/io/pcd_io.h>  // PCD files
#include <pcl/point_types.h>

// ROS PCL interface
//#include <pcl_conversions/pcl_conversions.h> // Don't use this. This breaks compile. Use the <triangle_sampling/pcl_conversions_partial.h> instead. If anything is not in there yet, pull in from original file, and REMOVE THE "inline"s!!
// To publish pcl::PointCloud as sensor_msgs::PointCloud2
//   Ref: http://wiki.ros.org/pcl_ros
//   Copied from pcl_ros, because whenver I try to build that package, it
//     messes things up. It thinks pcl_conversions.h types are wrong and thinks
//     every pcl:: type is ROS type (e.g. sensor_msgs::) for some reason!
#include "triangle_sampling/point_cloud.h"

//#include <pcl_ros/transforms.h>

// Eigen
#include <Eigen/Dense>  // Eigen mats http://eigen.tuxfamily.org/dox/group__TutorialMatrixArithmetic.html

// My packages
#include <triangle_sampling_msgs/TriangleParams.h>

// Local
// This is a symlink to the file in grasp_point_calc package, `.` that package
//   is rosmake and doesn't work with catkin.
#include "triangle_sampling/skeleton.h"  // create_marker()
#include "triangle_sampling/read_scaling_config.h"



// Copied and updated from my ./sample_pt_cloud.py
class SampleObjPCL
{
public:

  bool doPause_;
  bool doSkip_;
  bool doTerminate_;

  bool goToNextObj_;

  SampleObjPCL (ros::NodeHandle nh)
  {
    //ros::Subscriber key_sub = nh.subscribe ("/keyboard_interface/key", 1,
    //  &SampleObjPCL::keyCB, this);
    //ros::Subscriber next_obj_sub = nh.subscribe ("/sample_pcl/next_obj", 1,
    //  &SampleObjPCL::nextObjCB, this);
    ros::Subscriber next_obj_sub = nh.subscribe ("/sample_pcl/next_obj", 1,
      &SampleObjPCL::nextObjCB, this);

    doPause_ = false;
    doSkip_ = false;
    doTerminate_ = false;
    goToNextObj_ = false;
  }

  void keyCB (const std_msgs::StringConstPtr& msg)
  {
    // If you need to convert more than 1 char to lower case, use this
    //   http://blog.fourthwoods.com/2013/12/10/convert-c-string-to-lower-case-or-upper-case/
    //   else you can also use std::lower() in a for-loop. It only does 1 char.
    if (! msg->data.compare (" "))
    {
      fprintf (stderr, "Got user signal to toggle pause / resume...\n");
      doPause_ = ! doPause_;
    }
    else if ((! msg->data.compare ("s")) || (! msg->data.compare ("S")))
    {
      fprintf (stderr, "Got user signal to skip rest of this object model, "
        "skipping...\n");
      doSkip_ = true;
    }
    else if ((! msg->data.compare ("q")) || (! msg->data.compare ("Q")))
    {
      fprintf (stderr, "Got user signal to terminate program, "
        "terminating...\n");
      doTerminate_ = true;
    }

  }

  void nextObjCB (const std_msgs::BoolConstPtr& msg)
  {
    fprintf (stderr, "Got signal to advance to next object\n");
    goToNextObj_ = msg -> data;
  }

};

/* Don't actually calculate factorial. It overflows very quickly.
long factorial (long n)
{
  long prod = 1;
  for (long i = 2; i <= n; i ++)
  {
    prod *= i;
  }

  // If overflown, set to max
  if (prod < 0)
    prod = LONG_MAX;

  return prod;
}
*/

long choose (long n, long k)
{
  // Formula:
  //   n choose k = n! / (k! (n-k)!)
  // But don't calculate the factorial directly, it will overflow very fast.
  //   Instead, reduce the fraction first, by dividing (n-k)!, as follows:
  // Assume k is small (k = 3 for this file):
  //     n! / (k! (n-k)!)
  //   = (n * (n-1) * (n-2) * ... * (n-k+1)) / k!

  // n * (n-1) * (n-2) * ... * (n-k+1)
  long nf = 1;
  for (long i = n; i >= n-k+1; i --)
  {
    nf *= i;
  }

  // k!
  long kf = 1;
  for (long i = 2; i <= k; i ++)
  {
    kf *= i;
  }

  long ret_val = nf / kf;

  // If overflown, set to max
  //   Check for < 0, don't check for <= 0. Because if have 2 points, 2C3=0,
  //   and that's not an overflow!
  if (ret_val < 0)
  {
    fprintf (stderr, "WARNING: nCk (%ld choose %ld) overflown. "
      "Setting to LONG_MAX.\n", n, k);
    ret_val = LONG_MAX;
  }

  return ret_val;
}


// Sort a vector's indices in place, return the indices.
//   The vector is not changed.
// Access the sorted vector by:
//   for (auto i: sort_indices(v))
//     cout << v[i] << endl;
// From http://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
// Requires c++11 compiler
template <typename T>
std::vector<size_t> sort_indices (const std::vector<T> &v)
{
  // initialize original index locations
  std::vector<size_t> idx(v.size());
  for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

  // sort indexes based on comparing values in v
  sort (idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}


void get_object_center_radii (pcl::PointCloud <pcl::PointXYZ> & cloud,
  Eigen::Vector3f & model_center, Eigen::Vector3f & model_radii)
{
  std::vector <float> xs;
  std::vector <float> ys;
  std::vector <float> zs;
  for (int i = 0; i < cloud.size (); i ++)
  {
    // pcl::PointCloud API http://docs.pointclouds.org/trunk/classpcl_1_1_point_cloud.html
    model_center [0] += cloud [i].x;
    model_center [1] += cloud [i].y;
    model_center [2] += cloud [i].z;

    xs.push_back (cloud [i].x);
    ys.push_back (cloud [i].y);
    zs.push_back (cloud [i].z);
  }
  model_center /= float (cloud.size ());

  // Find max and min of each dimension, to get model radius
  float xmin = *(std::min_element (xs.begin (), xs.end ()));
  float xmax = *(std::max_element (xs.begin (), xs.end ()));
  float ymin = *(std::min_element (ys.begin (), ys.end ()));
  float ymax = *(std::max_element (ys.begin (), ys.end ()));
  float zmin = *(std::min_element (zs.begin (), zs.end ()));
  float zmax = *(std::max_element (zs.begin (), zs.end ()));

  model_radii [0] = (xmax - xmin) * 0.5;
  model_radii [1] = (ymax - ymin) * 0.5;
  model_radii [2] = (zmax - zmin) * 0.5;

  fprintf (stderr, "Model center: %.2f %.2f %.2f, model radii: %.2f %.2f %.2f\n",
    model_center[0], model_center[1], model_center[2],
    model_radii[0], model_radii[1], model_radii[2]);
}


int main (int argc, char ** argv)
{
  //========
  // User adjust params
  //========

  bool doRViz = false;
  // Number of iterations to publish essential msgs, even if not visualizing.
  //   This is to ensure subscribers receive msgs.
  int noVizIters = 50;

  bool DEBUG_VOXEL = false;
  bool DEBUG_SAMPLING = false;
  bool DEBUG_ANGLES = false;
  bool DEBUG_CORNER_CASES = false;

  // Defaults. Can be changed by cmd line args.
  // Number of triangles per sampling sphere (i.e. per pass) per voxel is 
  //   determined by min() of these two numbers.
  // Each voxel has 8 passes, to cover 50% overlap. So total # triangle samples
  //   per voxel is 8 times nSamplesPerPassPerVoxel.
  int nSamplesPerPassPerVoxel = 100;
  float nSamplesRatio = 0.95;
  std::string meta_base = std::string ("models.txt");

  const float cumu_marker_dur = 60.0f;
  const float text_height = 0.008f;


  //========
  // Parse cmd line args to set some params
  //========

  // Make some colors to print info
  // From http://stackoverflow.com/questions/287871/print-in-terminal-with-colors-using-python
  //   ANSI colors https://gist.github.com/chrisopedia/8754917
  //   C++ http://stackoverflow.com/questions/7414983/how-to-use-the-ansi-escape-code-for-outputting-colored-text-on-console
  static const char * OKCYAN = "\033[96m";
  static const char * ENDC = "\033[0m";

  for (int i = 0; i < argc; i ++)
  {
    if (! strcmp (argv [i], "--nSamples"))
    {
      nSamplesPerPassPerVoxel = std::strtol (argv [++i], NULL, 10);
      fprintf (stderr, "User specified nSamples %d\n", nSamplesPerPassPerVoxel);
    }
    else if (! strcmp (argv [i], "--nSamplesRatio"))
    {
      nSamplesRatio = std::strtof (argv [++i], NULL);
      fprintf (stderr, "User specified nSamplesRatio %f\n", nSamplesRatio);
    }
    else if (! strcmp (argv [i], "--meta"))
    {
      meta_base = std::string (argv [++i]);
      fprintf (stderr, "User specified meta file name %s\n", meta_base.c_str ());
    }
  }
  fprintf (stderr, "%s""nSamplesPerPassPerVoxel is set to %d, "
    "nSamplesRatio to %f, meta file %s""%s\n",
    OKCYAN, nSamplesPerPassPerVoxel, nSamplesRatio, meta_base.c_str (), ENDC);


  //========
  // Init ROS
  //========

  ros::init (argc, argv, "sample_pcl");
  ros::NodeHandle nh;

  SampleObjPCL thisNode = SampleObjPCL (nh);

  // Prompt to display at keyboard_interface.py prompt
  ros::Publisher prompt_pub = nh.advertise <std_msgs::String> (
    "keyboard_interface/prompt", 1);
  std_msgs::String prompt_msg;
  prompt_msg.data = "Press space to pause, ";

  ros::Publisher tri_pub = nh.advertise <triangle_sampling_msgs::TriangleParams> (
    "/sample_pcl/triangle_params", 1);

  ros::Publisher vis_pub = nh.advertise <visualization_msgs::Marker> (
    "visualization_marker", 0);
  ros::Publisher vis_arr_pub = nh.advertise <visualization_msgs::MarkerArray> (
    "visualization_marker_array", 0);

  // Keep topic name similar to sample_pt_cloud.py, for ease of switching btw scripts
  // To publish pcl::PointCloud in ROS, which appear to ROS as
  //   sensor_msgs/PointCloud2!
  //   Ref: http://www.ros.org/wiki/pcl_ros
  // Use this if you're using pcl::PointCloud<> human-readable type
  ros::Publisher cloud_pub = nh.advertise <pcl::PointCloud
    <pcl::PointXYZ> > ("/sample_obj/cloud2", 2);
  // Use this if you're using the more complicated pcl::PCLPointCloud2
  //ros::Publisher cloud2_pub = nh.advertise <
  //  sensor_msgs::PointCloud2> ("/sample_obj/cloud", 2);

  // It doesn't work if I place these in SampleObjPCL constructor. I don't know
  //   why. It works for Python.
  ros::Subscriber key_sub = nh.subscribe ("/keyboard_interface/key", 1,
    &SampleObjPCL::keyCB, &thisNode);
  ros::Subscriber next_obj_sub = nh.subscribe ("/sample_pcl/next_obj", 1,
    &SampleObjPCL::nextObjCB, &thisNode);

  // To determine what folder the data will be saved in, by an external node
  //   (sample_pcl_calc_hist.py) that subscribes to this node.
  ros::Publisher nSamples_pub = nh.advertise <std_msgs::Int32> (
    "/sample_pcl/nSamples", 1);
  ros::Publisher nSamplesRatio_pub = nh.advertise <std_msgs::Float32> (
    "/sample_pcl/nSamplesRatio", 1);
  std_msgs::Int32 nSamples_msg;
  nSamples_msg.data = nSamplesPerPassPerVoxel;
  std_msgs::Float32 nSamplesRatio_msg;
  nSamplesRatio_msg.data = nSamplesRatio;

  std::string frame_id = "/base";

  // Hertz
  ros::Rate wait_rate = ros::Rate (50);

  fprintf (stderr, "sample_pcl node initialized...\n");

  // Used to delete all markers in btw objects
  int prev_seqs = 0;

  // Used to profile running time
  //std::clock_t start_time;

  // clock() above doesn't work. It times ~190 s as 17 s, ridiculous.
  //   http://www.cplusplus.com/reference/ctime/time/
  struct tm y2k = {0};
  y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
  y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;
  time_t start_time;


  //========
  // Define model file paths
  //========

  // Define the dataset we're using
  const int MENGLONG = 0;
  const int ARCHIVE3D = 1;
  const int _3DNET = 2;
  const int COMBO = 3;
  const int SIMPLE = 4;
  const int DATASET = COMBO;  // _3DNET

  // A tell-tale signature of what dataset a model file is from, just based
  //   on this SUBSTRING being found in file path.
  // Used for DATASET == COMBO case, to detect which dataset a file is from.
  // NOTE: Make sure this is indeed a SUBSTRING of the path!! Else won't be
  //   detected correctly! In COMBO case, the path should be inside
  //   models.txt, not in model_path. So it will be read in as model_name.
  std::string MENGLONG_SIG ("Menglong");
  std::string ARCHIVE3D_SIG ("archive3d");
  std::string _3DNET_SIG ("3DNet");
  std::string SIMPLE_SIG ("simple");


  // model_path is prefix concat to model_name[i]. model_name[] is read from
  //   models.txt in config/ directory.

  // Path of this ROS package
  //   Add roslib to CMakeLists.txt and package.xml to compile.
  //   http://docs.ros.org/api/roslib/html/c++/namespaceros_1_1package.html
  //   http://wiki.ros.org/Packages
  std::string pkg_path = ros::package::getPath ("triangle_sampling");

  // Check paths only on a specific computer, by computer name
  // Ref: http://cboard.cprogramming.com/cplusplus-programming/68619-how-find-computer-name.html
  char * computer = getenv ("HOSTNAME");

  // Model file root paths
  std::string model_path;
  // Menglong's 16 objects
  if (DATASET == MENGLONG)
  {
    // This dataset is only tested on MacBook Air
    // (It's present on MacBook Pro Ubuntu too, but I don't know the path and
    //    haven't tested this code on that computer.)
    if (! strcmp (computer, "masters-MacBook-Air.local"))
    {
      model_path = std::string (
        "/Users/master/courses/15summer/MenglongModels/");
    }
    else
    {
      fprintf (stderr, "DATASET specified to be MENGLONG objects, "
        "but not on a computer that has the dataset! "
        "Run a different dataset, or run on the correct computer. "
        "Terminating...\n");
      return -1;
    }
  } 

  // Archive3D objects
  else if (DATASET == ARCHIVE3D)
  {
    model_path = pkg_path + 
      "../../../../train/triangle_sampling/models/archive3d/pcd/";
  }

  // 3DNet Cat10 training objects
  else if (DATASET == _3DNET)
  {
    model_path = pkg_path + 
      "/../../../../train/3DNet/cat10/train/pcd/";
  }

  // Combined multiple datasets
  else if (DATASET == COMBO)
  {
    // To make the models.txt file, run echo_model_paths.sh from here, with
    //   input path being the root directory of your dataset.
    // $ cd /Users/master/graspingRepo/train
    // $ echo_model_paths triangle_sampling/models/archive3d/pcd/ models.txt
    // $ echo_model_paths 3DNet/cat10/train/pcd/ models.txt
    model_path = pkg_path + "/../../../../train/";
  }

  // Invalid dataset value
  else
  {
    fprintf (stderr, "sample_pcl: ERROR: DATASET invalid. "
      "Specify a valid one in code. Terminating...\n");
    return -1;
  }

  // File containing a list of relative paths of model files, from the model
  //   root path above.
  // Archive3D objects
  //std::string meta_path (pkg_path + "/config/models_partial.txt");
  // 3DNet Cat10 training objects
  std::string meta_path = pkg_path + "/config/";
  meta_path = meta_path + meta_base;


  /*
  if (! strcmp (computer, "203-32-97-4"))
  {
    fprintf (stderr, "Detected computer is MacBook Pro Ubuntu.\n");
  }
  else if (! strcmp (computer, "masters-MacBook-Air.local"))
  {
    fprintf (stderr, "Detected computer is MacBook Air OS X.\n");
  }
  else if (! strcmp (computer, "calvisii"))
  {
    fprintf (stderr, "Detected computer is Baxter desktop calvisii.\n");
  }
  */


  // Traditional way of declaring a static vector of strings
  /*
  const int nFiles = 2;
  //std::string model_name_arr [nFiles] = {
  //  "axe1_c.obj", "bigbox_c.obj", "bottle_c.obj", "broom_c.obj",
  //  "brush_c.obj", "flowerspray_c.obj", "gastank_c.obj", "handlebottle_c.obj",
  //  "heavyranch1_c.obj", "pan_c.obj", "pipe_c.obj", "shovel_c.obj",
  //  "spadefork_c.obj", "spraybottle_c.obj", "watercan_c.obj",
  //  "wreckbar1_c.obj"};

  // Shove into a std::vector
  std::vector <std::string> model_name;
  for (int i = 0; i < nFiles; i ++)
    model_name.push_back (model_name_arr [i]);
  fprintf (stderr, "%u files.\n", (unsigned int) model_name.size ());
  */

  // Way from Peter-Michael. Must build with c++11.
  // Menglong's objects
  /*std::vector <std::string> model_name {
    "axe1_c.obj", "bigbox_c.obj", "bottle_c.obj", "broom_c.obj",
    "brush_c.obj", "flowerspray_c.obj", "gastank_c.obj", "handlebottle_c.obj",
    "heavyranch1_c.obj", "pan_c.obj", "pipe_c.obj", "shovel_c.obj",
    "spadefork_c.obj", "spraybottle_c.obj", "watercan_c.obj",
    "wreckbar1_c.obj"};
  */
  // Archive3D objects
  /*
  std::vector <std::string> model_name {
    "cup_2b46c83c.pcd", "cup_4b05736e.pcd",
    "cup_baecbac6.pcd", "cup_e9a70951.pcd",
    "pliers_6a5e5943.pcd",
    "scissors_d7c76ec5.pcd",
    "screwdriver_827757e4.pcd",
    "screwdriver_e054371d.pcd"
  };
  */


  // Read list of models from config file
  // Ref: http://stackoverflow.com/questions/7868936/read-file-line-by-line
  std::vector <std::string> model_name;

  std::ifstream infile (meta_path);

  // Test 2 lines above, then delete this hardcoded line
  //std::ifstream infile ("/Users/master/graspingRepo/reFlexHand/catkin_ws/src/triangle_sampling/config/models.txt");

  std::string line;
  // Read a full line, including spaces (getline(... '\n') lets you read
  //   including spaces. ifsteam >> std::string reads a token at a time, so
  //   it does not effectively ignore commented lines.)
  while (std::getline (infile, line, '\n'))
  {
    // If this line starts with #, it's a comment. Ignore it.
    if (! line.compare (0, 1, "#"))
      continue;
    // Ignore empty line
    else if (! line.compare (0, 1, ""))
      continue;
    else
      model_name.push_back (line);
  }


  int nFiles = model_name.size ();

  // Only for concatenating paths
  boost::filesystem::path model_path_bst (model_path);



  // Read other configs for specific datasets
  std::map <std::string, float> rescale_map;
  if ((DATASET == _3DNET) || (DATASET == COMBO))
  {
    std::string rescale_path (pkg_path +
      "/config/resize_3dnet_cat10_train.csv");
    read_scaling_config (rescale_path, rescale_map);
  }



  //======
  // Descriptors
  //======

  const int nTriParams = 3;

  // Seed random number generator with current time
  // Don't seed it with time(), if want consistent output. Find a constant that
  //   works well.
  //srand (time (NULL));
  // For now, I'll just seed with 0. When have time, find one that works well
  //   for final data.
  srand (0);


  //========
  // Octree parameters
  //========

  //pcl::VoxelGrid <pcl::PCLPointCloud2> voxel;

  // Approx length of each finger on ReFlex Hand
  const float fin_len = 0.16f;
  // Radius of sphere formed by ReFlex Hand. We'll assume each finger's length
  //   is 1/4 of the sphere's circumference.
  //   C = 4 * fin_len = 2 * pi * r
  //       2 * fin_len = pi * r
  //                 r = 2 * fin_len / pi
  // But Hand probably won't always be fully extended for a touch, so make it
  //   smaller. When fully extend, 2 fore-fingers can cover approx the diameter
  //   of the sphere.
  //   This buffer factor is visually gauged by looking at RViz and seeing
  //   whether size of sphere on obj looks possible in real life for Hand to
  //   span.
  const float buffer_factor = 1.0f;
  const float radius = 2 * fin_len / M_PI * buffer_factor;
  // Side length of voxels. Unit: meters.
  //   Each voxel is defined to be the biggest box that can fit inside the
  //     sphere formed by ReFlex Hand. This is chosen so that we can sample
  //     from overlapping spheres, not entirely exclusive. Putting voxels
  //     inside the spheres, rather than hull of spheres, lets the spheres
  //     overlap.
  //   Cube diagonal of voxel is the diagonal of sphere, 2 * radius.
  //     Formula: diagonal = side * sqrt(3)
  //   So side length of cube is diagonal / sqrt(3) = 2 * radius / sqrt(3).
  const float voxel_side = 2 * radius * sqrt(3) / 3;

  fprintf (stdout, "radius %f m, voxel_side %f m \n", radius, voxel_side);

  // Percent overlap btw voxels that I want to sample from
  const float pc_overlap = 0.5;
  const int nOverlapPasses = floor (1 / pc_overlap) - 1;

  // Define sphere center for the 8 passes
  //   +x, +y, +z, +xz, +yz, +xy, +xyz.
  const float overlapInc = pc_overlap * voxel_side;
  std::vector <Eigen::Vector3f> overlapIncrements;
  overlapIncrements.push_back (Eigen::Vector3f (0, 0, 0));
  overlapIncrements.push_back (Eigen::Vector3f (overlapInc, 0, 0));
  overlapIncrements.push_back (Eigen::Vector3f (0, overlapInc, 0));
  overlapIncrements.push_back (Eigen::Vector3f (0, 0, overlapInc));
  overlapIncrements.push_back (Eigen::Vector3f (overlapInc, 0, overlapInc));
  overlapIncrements.push_back (Eigen::Vector3f (0, overlapInc, overlapInc));
  overlapIncrements.push_back (Eigen::Vector3f (overlapInc, overlapInc, 0));
  overlapIncrements.push_back (Eigen::Vector3f (overlapInc, overlapInc, overlapInc));


  //fprintf (stderr, "Sampling %d triangles per pass per voxel\n",
  //  nSamplesPerPassPerVoxel);

  //========
  // Loop through each obj file
  //========

  // Start clocking running time
  // Ref: http://stackoverflow.com/questions/3220477/how-to-use-clock-in-c
  //start_time = std::clock ();

  start_time = time (NULL);


  for (int obj_idx = 0; obj_idx < model_name.size (); obj_idx ++)
  {
    fprintf (stderr, "\n");

    // API:
    // ros::ok() vs. ros::isShuttingDown(). Use latter if in a callback fn.
    //   http://wiki.ros.org/roscpp/Overview/Initialization%20and%20Shutdown
    if (! ros::ok ())
      break;


    //========
    // Storage to hold model and triangle descriptors
    //========

    pcl::PointCloud <pcl::PointXYZ> cloud;
    // If you need a pointer, it must be allocated on the heap. Otherwise boost
    //   frees it incorrectly, if you just call pcl::...<...>::Ptr(&cloud), to
    //   cast a static var to a ::Ptr type!
    // For pcl::PointCloud<> type, luckily there is a makeShared() function, it
    //   works without memory leaks. For pcl::PCLPointCloud2, it doesn't have it.
    //pcl::PointCloud <pcl::PointXYZ>::Ptr cloud_ptr (
    //  new pcl::PointCloud <pcl::PointXYZ> ());
 
    // Create a Ptr type when cloud is empty 
    // Cannot use static allocation. Boost shared ptr throws freeing ptr without
    //   allocating. I remember discovering this 2 years ago too, PCL examples
    //   all allocate on heap too. WHY???
    //pcl::PCLPointCloud2::Ptr cloud2_ptr (new pcl::PCLPointCloud2 ());


    // Triangle descriptors. One per object. Clear at end of object loop
    std::vector <float> l0;
    std::vector <float> l1;
    std::vector <float> l2;

    std::vector <float> a0;
    std::vector <float> a1;
    std::vector <float> a2;


    //========
    // Load OBJ file
    //   API: http://docs.pointclouds.org/trunk/group__io.html
    //========

    // Ref: http://stackoverflow.com/questions/6297738/how-to-build-a-full-path-string-safely-from-separate-strings
    //   http://stackoverflow.com/questions/4179322/how-to-convert-boost-path-type-to-string
    boost::filesystem::path model_name_bst (model_name [obj_idx]);
    boost::filesystem::path model_full_path_bst =
      model_path_bst / model_name_bst;
    std::string file_path = model_full_path_bst.string ();

    // If models are from multiple datasets, figure out which dataset THIS 
    //   model is in, so can modify it accordingly below.
    int dataset = DATASET;
    if (dataset == COMBO)
    {
      // Figure out which dataset this file is in

      if (model_name [obj_idx].find (MENGLONG_SIG) != std::string::npos)
      {
        dataset = MENGLONG;
        //fprintf (stderr, "This file is part of Menglong models. Will rescale.\n");
      }
      else if (model_name [obj_idx].find (ARCHIVE3D_SIG) != std::string::npos)
      {
        dataset = ARCHIVE3D;
        //fprintf (stderr, "This file is part of Archive3D. Will rotate 90 wrt X.\n");
      }
      else if (model_name [obj_idx].find (SIMPLE_SIG) != std::string::npos)
      {
        dataset = SIMPLE;
      }
      else if (model_name [obj_idx].find (_3DNET_SIG) != std::string::npos)
      {
        dataset = _3DNET;
        //fprintf (stderr, "This file is part of 3DNet. Will rescale.\n");
      }
      else
      {
        fprintf (stderr, "WARNING: sample_pcl.cpp: "
          "Did not find what dataset this file belongs to! "
          "Check path in models.txt. Will treat as PCD, "
          "and will not do any rescaling or rotation to this model.\n");
      }
    }
    // Debugging seg fault in pcd::io::loadPCDfile() on Ubuntu
    //fprintf (stderr, "dataset: %d, file_path: %s", dataset, file_path.c_str ());
    //return 0;


    // Parse object file path, to get category name

    // Find last slash before model file name
    // Ref: www.cplusplus.com/reference/string/string/find_last_of/
    std::size_t tmp_idx = file_path.find_last_of ("/");
    // Get substring from beginning to the char before last "/"
    std::string obj_cat = file_path.substr (0, tmp_idx);

    // Find second-to-last slash
    tmp_idx = obj_cat.find_last_of ("/");
    // Get substring from the char after second-to-last "/"
    //   If "/" not found, find_last_of() returns std::string::npos, which
    //   is -1. So this would take index [0] onwards. So it still works.
    obj_cat = obj_cat.substr (tmp_idx+1);


    int nPts = 0;

    // Menglong objects: scale down
    // Ref string ends with:
    //   http://stackoverflow.com/questions/874134/find-if-string-endswith-another-string-in-c
    //if (! file_path.compare (file_path.length () - 4, 4, ".obj"))
    if (dataset == MENGLONG)
    {
      // loadObjFile() only exists in PCL 1.8.0.
      // Ref: http://stackoverflow.com/questions/32539837/how-to-determine-pcl-point-cloud-library-version-in-c-code
      #if PCL_VERSION_COMPARE(>=, 1, 8, 0)

        pcl::io::loadOBJFile (file_path, cloud);
        //pcl::io::loadOBJFile (model_full_path_bst.string (), *cloud2_ptr);
       
        nPts = cloud.size ();
        // Shrink model, because orig points are in hundreds scale, too big for
        //   meters. I'm guessing they are in milimeters?
        // This is just for Menglong's OBJ files. For my PCD files, I resize in
        //   Blender manually to real-world size.
        for (int i = 0; i < nPts; i ++)
        {
          // pcl::PointCloud API http://docs.pointclouds.org/trunk/classpcl_1_1_point_cloud.html
          cloud [i].x *= 0.001;
          cloud [i].y *= 0.001;
          cloud [i].z *= 0.001;
        }

      #else
        printf ("Not on MacBook Air. loadOBJFile() requires PCL 1.8, will not load any models!\n");

      #endif
    }

    // Archive3D objects: Rotate
    else if (dataset == ARCHIVE3D)
    {
      pcl::io::loadPCDFile (file_path, cloud);

      // Rotate 90 degs wrt X. This is an artifact from Blender exports
      //   Ref: http://pointclouds.org/documentation/tutorials/matrix_transform.php
      Eigen::Affine3f transform = Eigen::Affine3f::Identity ();
      // 20 Dec 2015: I corrected all OBJ files, but PCD files weren't
      //   resampled. When have time, recreate the PCD files from the new OBJ
      //   files. Then don't need this anymore.
      transform.rotate (Eigen::AngleAxisf (90.0 * M_PI / 180.0, 
      //transform.rotate (Eigen::AngleAxisf (0.0 * M_PI / 180.0, 
        Eigen::Vector3f::UnitX ()));
      pcl::transformPointCloud (cloud, cloud, transform);
    }

    else if (dataset == SIMPLE)
    {
      pcl::io::loadPCDFile (file_path, cloud);
    }

    // 3DNet objects: scale down by factor loaded in config file
    else if (dataset == _3DNET)
    {
      pcl::io::loadPCDFile (file_path, cloud);

      //fprintf (stderr, "Rescaling %s by %f\n", cat, rescale_map[cat]);

      // Rescale the point cloud, using the scaling factor specified in config
      //   file for this object category.
      // Assumption: cat is in the keys of map
      // TODO 29 Dec 2016: Faster way would be to do Eigen::MatrixXf mat =
      //   pcl::PointCloud<>::getMatrixXfMap(), then just multiply the whole
      //   Eigen::MatrixXf by the scaling constant! See active_visual_tactile
      //   package.
      for (int i = 0; i < cloud.size (); i ++)
      {
        cloud [i].x *= (rescale_map [obj_cat]);
        cloud [i].y *= (rescale_map [obj_cat]);
        cloud [i].z *= (rescale_map [obj_cat]);
      }
    } 
    // Default, just load PCD
    else
    {
      pcl::io::loadPCDFile (file_path, cloud);
    }

    nPts = cloud.size ();
    
    //int nPts = cloud2_ptr -> height * cloud2_ptr -> width;
    fprintf (stderr, "Loaded file %d: %s.\n%d vertices\n", 
      obj_idx + 1, model_full_path_bst.string ().c_str (), nPts);

    // Debugging seg fault in pcd::io::loadPCDfile() on Ubuntu
    //return 0;


    // Get model center
    Eigen::Vector3f model_center (0.0, 0.0, 0.0);
    // Get model radii in 3 dimensions
    Eigen::Vector3f model_radii (0.0, 0.0, 0.0);
 
    get_object_center_radii (cloud, model_center, model_radii);


    // I don't need to downsample anymore, since you have to use Octree anyway.
    //   I'll just use Octree directly below.
    /*
    //========
    // Downsample cloud to 1 point every 8 mm (0.008 m)
    //   8 mm is to match MEMS barometer sensor size on ReFlex Hand.
    // Ref: They recommend using OctreePointCloudVoxelCentroid, instead of 
    //   VoxelGrid. Former is more memory efficient!
    //   http://www.pcl-users.org/VoxelGridFilter-Leaf-size-is-too-small-td4025570.html
    //   This comes from a search of VoxelGrid not letting me downsample to
    //   0.008 m, saying leaf size too small. Radu Rusu says VoxelGrid is 
    //   unstable after some changes, and use Octree instead.
    //   Code snippet provided.
    //
    //   Since that is the case, I decided that I don't really need to
    //   downsample, since I'm already using Octree anyway, and it's supposed
    //   to be memory efficient. When I sample triangles, It's better to sample
    //   from original cloud anyway.
    //   
    // Ref VoxelGrid:
    //   http://pointclouds.org/documentation/tutorials/voxel_grid.php#voxelgrid
    //========

    // Convert pcl type to the correct type to use with VoxelGrid
    //   ::Ptr http://stackoverflow.com/questions/10644429/create-a-pclpointcloudptr-from-a-pclpointcloud
    pcl::toPCLPointCloud2 (cloud, *cloud2_ptr);

    voxel.setInputCloud (cloud2_ptr);
    // Unit: meters
    // Not sure why it doesn't let me do 0.008, 8 mm.
    voxel.setLeafSize (0.9f, 0.9f, 0.9f);
    voxel.filter (*cloud2_ptr);

    // Convert back to my model type
    pcl::fromPCLPointCloud2 (*cloud2_ptr, cloud);

    fprintf (stderr, "%d vertices after pcl::VoxelGrid downsampling. %d vertices in copy of data.\n",
      (unsigned int) cloud.size (),
      (unsigned int) (cloud2_ptr->height * cloud2_ptr->width));

    // Clear out the copy of data right away. Don't waste memory space.
    // This is their official way of clearing out pcl::PCLPointCloud2
    //   http://docs.pointclouds.org/trunk/reconstruction_8hpp_source.html
    cloud2_ptr->width = cloud2_ptr->height = 0;
    cloud2_ptr->data.clear ();
    */

 
    //========
    // Make Octree. Add vertices in OBJ file into Octree
    //========

    pcl::octree::OctreePointCloudSearch <pcl::PointXYZ> octree (voxel_side);
    octree.setInputCloud (cloud.makeShared ());
    octree.addPointsFromInputCloud ();
 
 
    //========
    // Sample a set of 3 points, many sets.
    //========

    pcl::octree::OctreePointCloudSearch <pcl::PointXYZ>::AlignedPointTVector
      voxel_center_list;
    octree.getOccupiedVoxelCenters (voxel_center_list);
    fprintf (stderr, "%u occupied voxels\n",
      (unsigned int) voxel_center_list.size ());

    // For visualization only
    // "triangles" is a linear array. Stores 3 points at a time. Total 3 *
    //   nTriangles elements.
    std::vector <pcl::PointXYZ> triangles;
    std::vector <float> vis_text;
    std::vector <pcl::PointXYZ> voxel_sphere_centers;
    std::vector <pcl::PointXYZ> voxel_center_pts;
    std::vector <int> triangle_to_voxel_idx;
    std::vector <int> triangle_to_sphere_idx;

    // This records the indices in the CLOUD, NOT in the neighbors nbrs_idx!
    //   `.` indices in the cloud are unique references to each point, while
    //   indices in nbrs_idx do not always mean the same point - the same idx
    //   in nbrs_idx may mean different points in cloud, different idx in
    //   nbrs_idx may mean same pts in cloud!
    std::vector <std::set <int>> sampled_tris_idx;

    // Loop through each voxel. Do 8 passes for each voxel.
    for (int i = 0; i < voxel_center_list.size (); i ++)
    {
      if (DEBUG_SAMPLING)
        printf ("Voxel %d\n", i);

      // For visualization
      voxel_center_pts.push_back (voxel_center_list [i]);

      // Don't do this, `.` one vertex can be in many different triangles!
      //   Don't want to limit each vertex to only a single triangle.
      // To store unique numbers for sampling
      //std::vector <int> range;
      //for (int j = 0; j < occupied_voxels.size (); j ++)
      //  range.push_back (j);

      // 8 passes
      for (int pass = 0; pass < overlapIncrements.size (); pass ++)
      {
        // Voxel center (or augmented one)
        pcl::PointXYZ curr_center = pcl::PointXYZ (
          voxel_center_list [i].x + overlapIncrements [pass].x (),
          voxel_center_list [i].y + overlapIncrements [pass].y (),
          voxel_center_list [i].z + overlapIncrements [pass].z ());
        // For visualization
        voxel_sphere_centers.push_back (curr_center);

        // Get a set of points to sample from
        std::vector <int> nbrs_idx;
        std::vector <float> ignored;
        octree.radiusSearch (curr_center, radius, nbrs_idx, ignored);
        if (DEBUG_SAMPLING)
          printf ("Pass %d. # points in sphere in this pass: %lu\n", pass,
            nbrs_idx.size ());

        // This makes even more undistinctive histograms! Fixed number seems
        //   better.
        //int nSamplesPerPassPerVoxel = floor (0.1 * nbrs_idx.size ());

        // In case there are too few points in this sphere, see how many can
        //   be sampled.
        // Technically, you can use n choose 3 >= nSamplesPerPassPerVoxel,
        //   and n is the minimum number of points required:
        //   For nSamplesPerPassPerVoxel == 5,
        //     5 choose 3 = 10
        //     4 choose 3 = 4
        //   So n = 5. You can sample 5 triangles as long as you have 5 points.
        //   But that is a tight bound, and it assumes that your random
        //   sampling actually hits every single point. The quality of the
        //   triangles are also not that good, since many of them will share
        //   points. So we relax this bound to simply
        //   3 * nSamplesPerPassPerVoxel. This way, it is still possible to get 
        //   nSamplesPerPassPerVoxel triangles with all unique points, which
        //   is a better set of samples than repetitive points.
        int nSamplesCurrPass = std::min ((double) nSamplesPerPassPerVoxel, 
          floor (nbrs_idx.size () * nSamplesRatio));

        if (nSamplesCurrPass < nSamplesPerPassPerVoxel)
        {
          if (DEBUG_CORNER_CASES)
            fprintf (stderr, "Too few neighbors (%lu). Will only sample %d "
              "triangles from this pass\n", nbrs_idx.size (), nSamplesCurrPass);
        }

        long nChoose3 = 0;
        if (nSamplesCurrPass > 0)
          nChoose3 = choose ((long) nbrs_idx.size (), 3);

        std::vector <std::set <int> > pass_attempted_sets;


        // Sample nSamplesCurrPass triangles
        // Elegant way to sample without replacement
        //   for (int i = 0; i < m; i ++)
        //     swap (X [i], X [i + rand (n - i )]);
        //   where rand(a) returns uniform samples from 0, ..., a-1.
        //   X[0 ... m-1] is now a random sample.
        //   Ref: Tomazos answer here
        //      http://stackoverflow.com/questions/13772475/how-to-get-random-and-unique-values-from-a-vector
        for (int tri = 0; tri < nSamplesCurrPass; tri ++)
        {
          // Sanity check
          // If all neighbors have been exhaustively sampled, skip this pass
          // Test for >=, not >, because pass_attempted_sets size is 
          //   incremented in previous iteration. So placing this check at
          //   beginning of loop would check whether previous samples have
          //   reached nChoose3. You actually never go >, so checking for >
          //   would do nothing! > is just a safety measure in case you 
          //   skipped the == nChoose3 for some reason. So you're really
          //   checking for ==.
          if (pass_attempted_sets.size () >= nChoose3)
          {
            fprintf (stderr, "INFO: All sets of three in this pass have been "
              "exhaustively sampled. Skipping remainder of this pass even "
              "though did not reach %d samples.\n", nSamplesCurrPass);
            break;
          }


          std::vector <int> range;
          for (int j = 0; j < nbrs_idx.size (); j ++)
            range.push_back (j);

          //========
          // Sample 3 UNIQUE random points (`.` a triangle can't have any 2
          //   vertices at same point, then it's a line)
          //========
          int tmp_idx = 0;
          for (int j = 0; j < 3; j ++)
          {
            tmp_idx = j + rand () % (nbrs_idx.size () - j);

            //fprintf (stderr, "tmp_idx: %d. range.size(): %u\n", tmp_idx,
            //  (unsigned int) range.size ());

            // Check you're not swapping btw the same idx, 
            //   swap(range[0],range[0])
            //   which causes floating point exception 8.
            if (tmp_idx != j)
              std::swap (range [j], range [tmp_idx]);
          }

          if (DEBUG_SAMPLING)
          {
            fprintf (stderr, "range: %d %d %d, nbrs_idx.size() %u, cloud.size(): %u\n",
              range[0], range[1], range[2],
              (unsigned int) nbrs_idx.size (),
              (unsigned int) cloud.size ());
            fprintf (stderr, "These 3 points' indices were chosen: %d %d %d\n",
              nbrs_idx [range [0]],
              nbrs_idx [range [1]],
              nbrs_idx [range [2]]
              );
          }


          //========
          // Make sure this triangle has not been sampled before, i.e. duplicate
          //========

          // Make an unordered set of the sampled vertices
          std::set <int> sample_idx;
          sample_idx.insert (nbrs_idx [range [0]]);
          sample_idx.insert (nbrs_idx [range [1]]);
          sample_idx.insert (nbrs_idx [range [2]]);

          // If this triangle has not been attempted in this pass, record it.
          // Even if this is a repeat, add it, so that we know how many combos
          //   have been sampled in THIS pass. This lets us terminate, in
          //   the case e.g. there are only 3 points in this pass, but the
          //   set has been sampled in some other previous pass.
          //   If we do not check >= n choose k, this pass will be infinite
          //   loop!
          //   We cannot just check a valid sample, because this triangle is
          //   a repeat, so it will not be added. Then we always think we
          //   sampled 0 points. Then this will still be infinite loop.
          //   So we need to add even the invalid sample histories.
          if (std::find (pass_attempted_sets.begin (),
            pass_attempted_sets.end (), sample_idx) ==
            pass_attempted_sets.end ())
          {
            pass_attempted_sets.push_back (sample_idx);;
          }


          // If this triangle has been sampled, resample for another triangle
          // Ref: www.cplusplus.com/reference/algorithm/find/
          if (std::find (sampled_tris_idx.begin (), sampled_tris_idx.end (),
            sample_idx) != sampled_tris_idx.end ())
          {
            //if (DEBUG_SAMPLING)
              fprintf (stderr, "INFO: This triangle (%d %d %d) already sampled. "
                "Resampling (out of %lu neighbors)...\n",
                range[0], range[1], range[2], nbrs_idx.size ());
            tri --;
            continue;
          }


          //========
          // Temporarily store the 3 sampled points
          //========

          pcl::PointXYZ pt0_pcl = cloud [nbrs_idx [range [0]]];
          pcl::PointXYZ pt1_pcl = cloud [nbrs_idx [range [1]]];
          pcl::PointXYZ pt2_pcl = cloud [nbrs_idx [range [2]]];

          Eigen::Vector3f pt0 (pt0_pcl.x, pt0_pcl.y, pt0_pcl.z);
          Eigen::Vector3f pt1 (pt1_pcl.x, pt1_pcl.y, pt1_pcl.z);
          Eigen::Vector3f pt2 (pt2_pcl.x, pt2_pcl.y, pt2_pcl.z);


          //========
          // Compute triangle connected by the 3 sampled pts. Compute all 6
          //   params, not just the 3 params that uniquely define a triangle,
          //   `.` let python module decide which 3 to pick to describe
          //   triangles, then can easily compare which 3 are better.
          //========

          // Pick the first two sides
          //   Calc angle btw them to make sure they aren't in a straight line.
          // They must subtract by same point, in order for dot product to
          //   compute correctly! Else your sign may be wrong. Subtracting by
          //   same point puts the two vectors at same origin, w hich is what
          //   dot product requires, in order to give you correct theta btw
          //   the two vectors!
          Eigen::Vector3f s10 = pt0 - pt1;
          Eigen::Vector3f s12 = pt2 - pt1;

          Eigen::Vector3f s01 = pt1 - pt0; 
          Eigen::Vector3f s02 = pt2 - pt0;

          Eigen::Vector3f s20 = pt0 - pt2;
          Eigen::Vector3f s21 = pt1 - pt2;

          // Two side lengths
          float len_s10 = s10.norm ();
          float len_s12 = s12.norm ();

          float len_s01 = s01.norm ();
          float len_s02 = s02.norm ();

          float len_s20 = s20.norm ();
          float len_s21 = s21.norm ();

          float angle1 = 0;
          // Angle btw two vectors.
          //   Dot product is dot(a,b) = |a||b| cos (theta).
          //       dot(a,b) / (|a||b|) = cos(theta)
          //                     theta = acos (dot(a,b) / (|a||b|))
          if (len_s10 * len_s12 != 0)
          {
            angle1 = acos ((s10.dot (s12)) / (len_s10 * len_s12));
            if (std::isnan (angle1))
            {
              //angle1 = 0;
              fprintf (stderr, "angle1 is nan. Resampling...\n");
              tri --;
              continue;
            }
            // If angle is within 1 deg (~0.017 rads) from 180, treat it as
            //   a straight line
            else if (fabs (angle1 - M_PI) < 0.017)
            {
              fprintf (stderr, "angle1 ~ M_PI is a straight line, "
                "not triangle. Resampling from %lu neighbors...\n",
                nbrs_idx.size ());
              tri --;
              continue;
            }
          }
          // Exception case: length is 0. Re-select the triangle.
          else
          {
            fprintf (stderr, "Triangle had 0-length side 1. Resampling...\n");
            //fprintf (stderr, "Pass %d, triangle %d. "
            //  "%lu pts in sphere, taking %d samples. "
            //  "Chose %d %d %d.\n",
            //  pass, tri,
            //  nbrs_idx.size(), nSamplesCurrPass, range[0], range[1], range[2]);
            //std::cerr << s10 << std::endl << s12 << std::endl;
            tri --;
            continue;
          }

          float angle0 = 0;
          if (len_s01 * len_s02 != 0)
          {
            angle0 = acos ((s01.dot (s02)) / (len_s01 * len_s02));
            if (std::isnan (angle0))
            {
              //angle0 = 0;
              fprintf (stderr, "angle0 is nan. Resampling...\n");
              tri --;
              continue;
            }
            // If angle is within 1 deg (~0.017 rads) from 180, treat it as
            //   a straight line
            else if (fabs (angle0 - M_PI) < 0.017)
            {
              fprintf (stderr, "angle0 ~ M_PI is a straight line, "
                "not triangle. Resampling from %lu neighbors...\n",
                nbrs_idx.size ());
              tri --;
              continue;
            }
          }
          else
          {
            fprintf (stderr, "Triangle had 0-length side 2. Resampling...\n");
            tri --;
            continue;
          }

          // For some reason, when 0.00167885 / 0.00167885 = 1, acos(1) gives
          //   nan. It should give 0. So I'll assume nan => 0.
          if (DEBUG_ANGLES)
            std::cerr << s20.dot (s21) << " / " << len_s20 * len_s21 << " = "
              << (s20.dot (s21)) / (len_s20 * len_s21) << std::endl;

          float angle2 = 0;
          if (len_s20 * len_s21 != 0)
          {
            angle2 = acos ((s20.dot (s21)) / (len_s20 * len_s21));
            if (std::isnan (angle2))
            {
              //angle2 = 0;
              fprintf (stderr, "angle2 is nan. Resampling...\n");
              tri --;
              continue;
            }
            // If angle is within 1 deg (~0.017 rads) from 180, treat it as
            //   a straight line
            else if (fabs (angle2 - M_PI) < 0.017)
            {
              fprintf (stderr, "angle2 ~ M_PI is a straight line, "
                "not triangle. Resampling from %lu neighbors...\n",
                nbrs_idx.size ());
              tri --;
              continue;
            }
          }
          else
          {
            fprintf (stderr, "Triangle had 0-length side 3. Resampling...\n");
            tri --;
            continue;
          }

          if (DEBUG_ANGLES)
            printf ("angle0: %f, angle1: %f, angle2: %f, error: %f\n",
              angle0 * 180 / M_PI,
              angle1 * 180 / M_PI,
              angle2 * 180 / M_PI,
              fabs (angle0 + angle1 + angle2 - M_PI)
            );
          assert (fabs (angle0 + angle1 + angle2 - M_PI) < 1e-3);


          //========
          // Store in sorted order, so that parameters are consistent across
          //   different triangles, i.e. l0 is the longest side, l1 is medium
          //   side, l2 is shortest side; a0 is largest angle, a1 is medium
          //   angle, a2 is smallest angle.
          //========

          // This is in order of which point was picked first. C++11 style init
          std::vector <float> tmp_lens {len_s02, len_s10, len_s12};

          // Sort in increasing order, with access to sorted indices
          std::vector <size_t> sorted_lens_idx = sort_indices (tmp_lens);
          l2.push_back (tmp_lens [sorted_lens_idx [0]]);
          l1.push_back (tmp_lens [sorted_lens_idx [1]]);
          l0.push_back (tmp_lens [sorted_lens_idx [2]]);

          // This is in order of which point was picked first. C++11 style init
          std::vector <float> tmp_angs {angle0, angle1, angle2};

          // Sort in increasing order, with access to sorted indices
          std::vector <size_t> sorted_angs_idx = sort_indices (tmp_angs);
          a2.push_back (tmp_angs [sorted_angs_idx [0]]);
          a1.push_back (tmp_angs [sorted_angs_idx [1]]);
          a0.push_back (tmp_angs [sorted_angs_idx [2]]);


          // Old way of manual sorting. Might be faster than function overhead
          //   of sort_indices, which is an overkill for sorting 3 elts!
          //   Not as clean, but has some nice 
          //   tricks like remove() and erase() pair-operation, max_element,
          //   min_element.
          
          /*
          // This is in order of which point was picked first. C++11 style init
          std::vector <float> tmp_lens {len_s02, len_s10, len_s12};

          // l0 stores max side length
          // Ref max_element:
          //   http://en.cppreference.com/w/cpp/algorithm/max_element
          float tmp_l = *(std::max_element (tmp_lens.begin (), tmp_lens.end ()));
          l0.push_back (tmp_l);
          // Ref remove element by value:
          //   http://stackoverflow.com/questions/3385229/c-erase-vector-element-by-value-rather-than-by-position
          tmp_lens.erase (std::remove (tmp_lens.begin (), tmp_lens.end (),
            tmp_l), tmp_lens.end ());

          // l2 stores min side length
          tmp_l = *(std::min_element (tmp_lens.begin (), tmp_lens.end ()));
          l2.push_back (tmp_l);
          tmp_lens.erase (std::remove (tmp_lens.begin (), tmp_lens.end (),
            tmp_l), tmp_lens.end ());

          // l1 stores the remaining value (medium length)
          l1.push_back (tmp_lens [0]);


          // Do same thing for angles

          // This is in order of which point was picked first. C++11 style init
          std::vector <float> tmp_angs {angle0, angle1, angle2};

          // a0 stores max side length
          float tmp_a = *(std::max_element (tmp_angs.begin (), tmp_angs.end ()));
          a0.push_back (tmp_a);
          tmp_angs.erase (std::remove (tmp_angs.begin (), tmp_angs.end (),
            tmp_a), tmp_angs.end ());

          // a2 stores min side length
          tmp_a = *(std::min_element (tmp_angs.begin (), tmp_angs.end ()));
          a2.push_back (tmp_a);
          tmp_angs.erase (std::remove (tmp_angs.begin (), tmp_angs.end (),
            tmp_a), tmp_angs.end ());

          // a1 stores the remaining value (medium length)
          a1.push_back (tmp_angs [0]);
          */


          //========
          // At this point, there should be no more errors causing a need to
          //   resample. This sample is final. Store to vectors.
          //========

          sampled_tris_idx.push_back (sample_idx);

          // For visualization, and for active touch probabilities data
          triangles.push_back (pt0_pcl);
          triangles.push_back (pt1_pcl);
          triangles.push_back (pt2_pcl);

          // For visualization only. Side-Angle-Side, in order of sampling.
          //   I thought about visualizing the 3 chosen params, but that is too
          //   much unnecessary work to correspond, and too error-prone. So just
          //   visualize the first angle, by sampling order.
          vis_text.push_back (len_s10);
          vis_text.push_back (len_s12);
          vis_text.push_back (angle1);

          // For visualization only. This tells which voxel center and which
          //   sphere center this triangle corresponds to. Size of these two
          //   indexing arrays are same as size of "triangles" array.
          triangle_to_voxel_idx.push_back (voxel_center_pts.size () - 1);
          triangle_to_sphere_idx.push_back (voxel_sphere_centers.size () - 1);
        }
      }
    }
    // Now we have (nSamplesPerPassPerVoxel * 8 * (# voxels)) triangles.
    //   (except for when a pass has not enough points to do
    //    nSamplesPerPassPerVoxel, then that pass may only have
    //    nSamplesCurrPass, the number is specific to that pass.

    int nTriangles = l1.size ();
    printf ("%d triangles sampled total\n", nTriangles);
    if (DEBUG_SAMPLING)
      printf ("Sampled %u points (should be 3 times as # triangles)\n",
        (unsigned int) triangles.size ());

    // Print l1, l2, a1
    //for (int i = 0; i < l1.size (); i ++)
    //  fprintf (stdout, "[%5.3f   %5.3f   %5.3f]\n", l1[i], l2[i],
    //    a1[i] * 180 / M_PI);

    if (DEBUG_SAMPLING)
    {
      printf ("%u voxels, %u voxel spheres (should be 8 times of 1st number)\n",
        (unsigned int) voxel_center_pts.size (),
        (unsigned int) voxel_sphere_centers.size ());
      printf ("%u triangle-to-voxel indices, %u triangle-to-sphere indices (these should be same as # triangles)\n",
        (unsigned int) triangle_to_voxel_idx.size (),
        (unsigned int) triangle_to_sphere_idx.size ());
    }

    //if (DEBUG_SAMPLING)
      //for (int i = 0; i < nTriangles; i ++)
      //  printf ("%d, ", triangle_to_voxel_idx [i]);
      //printf ("\n");
  
 
    //========
    // Create triangle rosmsg to publish, so a Python node can take care of
    //   the rest.
    //========

    triangle_sampling_msgs::TriangleParams tri_msg;

    tri_msg.obj_seq = obj_idx;
    tri_msg.obj_name = model_name [obj_idx];
    tri_msg.obj_cat = obj_cat;

    tri_msg.l0.resize (nTriangles);
    tri_msg.l1.resize (nTriangles);
    tri_msg.l2.resize (nTriangles);
    tri_msg.a0.resize (nTriangles);
    tri_msg.a1.resize (nTriangles);
    tri_msg.a2.resize (nTriangles);

    tri_msg.pt0.resize (nTriangles);
    tri_msg.pt1.resize (nTriangles);
    tri_msg.pt2.resize (nTriangles);

    for (int i = 0; i < nTriangles; i ++)
    {
      tri_msg.l0 [i] = l0 [i];
      tri_msg.l1 [i] = l1 [i];
      tri_msg.l2 [i] = l2 [i];
      tri_msg.a0 [i] = a0 [i];
      tri_msg.a1 [i] = a1 [i];
      tri_msg.a2 [i] = a2 [i];

      geometry_msgs::Point p0;
      // Index the linear array
      p0.x = triangles [i * nTriParams + 0].x;
      p0.y = triangles [i * nTriParams + 0].y;
      p0.z = triangles [i * nTriParams + 0].z;
      tri_msg.pt0 [i] = p0;

      geometry_msgs::Point p1;
      // Index the linear array
      p1.x = triangles [i * nTriParams + 1].x;
      p1.y = triangles [i * nTriParams + 1].y;
      p1.z = triangles [i * nTriParams + 1].z;
      tri_msg.pt1 [i] = p1;

      geometry_msgs::Point p2;
      // Index the linear array
      p2.x = triangles [i * nTriParams + 2].x;
      p2.y = triangles [i * nTriParams + 2].y;
      p2.z = triangles [i * nTriParams + 2].z;
      tri_msg.pt2 [i] = p2;
    }
 
    tri_msg.obj_center.x = model_center [0];
    tri_msg.obj_center.y = model_center [1];
    tri_msg.obj_center.z = model_center [2];

    tri_msg.obj_radii.x = model_radii [0];
    tri_msg.obj_radii.y = model_radii [1];
    tri_msg.obj_radii.z = model_radii [2];

 
    //========
    // Create histogram from the 3 params, for this entire obj, i.e. for all 
    //   sets of 3 points sampled.
    // This is the final descriptor for this object.
    //========
 
    // Not doing this in C++. Pass l1, l2, a1 over to Python to do it!
    //   Python rules for prototyping. Gosh C++ is a waste of my time.
    //   Python : C++ :: Mac : PC

    //   See sample_pcl_calc_hist.py
 
 
    //========
    // Save histograms to file, to use as descriptors
    //========

    // Not doing this in C++. Pass over to Python
    //   See sample_pcl_calc_hist.py


    //========
    // Erase markers from previous object
    //========

    // Erase all cumulative markers from prev object, to see more clearly when
    //   new obj appears.
    visualization_msgs::MarkerArray erase_arr;
    for (int i = 0; i <= prev_seqs; i ++)
    {
      visualization_msgs::Marker erase_m;
      erase_m.id = i;
      erase_m.action = visualization_msgs::Marker::DELETE;

      // Shouldn't be needed. Just to appease the red transform errors in RViz
      erase_m.header.frame_id = frame_id;

      // Erase this id in all cumulative namespaces
      // Except these namespaces, which only have 1 marker (with id 0) at
      //   any given time, so will be automatically erase when we publish a
      //   new one below:
      //   "sample_hand_sphere", "sample_voxel", "sample_tri" "sample_pts"
      //   "text"
      erase_m.ns = std::string ("sample_voxel_cumu");
      erase_arr.markers.push_back (erase_m);
      erase_m.ns = std::string ("sample_tri_cumu");
      erase_arr.markers.push_back (erase_m);
      erase_m.ns = std::string ("sample_pts_cumu");
      erase_arr.markers.push_back (erase_m);
    }
    if (ros::ok ())
    {
      //fprintf (stderr, "erase_arr has %lu markers\n", erase_arr.markers.size());
      vis_arr_pub.publish (erase_arr);
      ros::spinOnce ();
      wait_rate.sleep ();
    }


    //========
    // Visualize sampled points and the triangle connecting them, in sequence
    //   in RViz
    //
    // Publish triangle data, for a Python node to take care of the rest.
    //========

    int seq = 0;
    while (ros::ok ())
    {
      //========
      // Spin and publish essential ONCE, even if doRViz is false
      //========

      prompt_pub.publish (prompt_msg);

      // To plot in matplotlib
      //fprintf (stderr, "Publishing obj_seq %d\n", obj_idx);
      tri_pub.publish (tri_msg);

      nSamples_pub.publish (nSamples_msg);
      nSamplesRatio_pub.publish (nSamplesRatio_msg);


      // Use this if you use pcl::PointCloud<> type
      // This doesn't work
      // Convert from ns to us. Copied from pcl_conversions.h
      cloud.header.stamp = ros::Time::now ().toNSec() / 1000ull;
      cloud.header.seq = seq;
      cloud.header.frame_id = frame_id;
      cloud_pub.publish (cloud);
      // This should work too, but don't need it
      //pcl::PCLPointCloud2 cloud2_pcl;
      //pcl::toPCLPointCloud2 (cloud, cloud2_pcl);
      //sensor_msgs::PointCloud2 cloud2;
      //pcl_conversions::fromPCL (cloud2_pcl, cloud2);
      //cloud_pub.publish (cloud2);
     
      // Use this if you use pcl::PCLPointCloud2 type
      //sensor_msgs::PointCloud2 cloud2_ros;
      //pcl_conversions::fromPCL (cloud2, cloud2_ros);
       
       
      if ((doRViz) && (! thisNode.doPause_))
      {
        //========
        // Visualize triangles in RViz
        // These correspond to sample_pt_cloud.py
        //========

        visualization_msgs::Marker marker_sample;
        create_marker (visualization_msgs::Marker::POINTS, frame_id, 0,
          0, 0, 0, 1, 0, 0, 0.8, 0.005, 0.005, 0.005,
          marker_sample, "sample_pts", 1, 0, 0, 0, 0);

        for (int i = 0; i < nTriParams; i ++)
        {
          //printf ("Point %u\n", (unsigned int) seq * nTriParams + i);

          geometry_msgs::Point tmp_pt;
          // Index the linear array
          tmp_pt.x = triangles [seq * nTriParams + i].x;
          tmp_pt.y = triangles [seq * nTriParams + i].y;
          tmp_pt.z = triangles [seq * nTriParams + i].z;

          marker_sample.points.push_back (tmp_pt);
        }

        // Make a copy for cumulative namespace
        visualization_msgs::Marker marker_sample_cumu;
        create_marker (visualization_msgs::Marker::POINTS, frame_id, seq,
          0, 0, 0, 1, 1, 0, 0.8, 0.002, 0.002, 0.002,
          marker_sample_cumu, "sample_pts_cumu", 1, 0, 0, 0, cumu_marker_dur);
        marker_sample_cumu.points = marker_sample.points;


        // Create a LINE_LIST Marker for the triangle
        // Simply connect the 3 points to visualize the triangle
        visualization_msgs::Marker marker_tri;
        create_marker (visualization_msgs::Marker::LINE_LIST, frame_id, 0,
          0, 0, 0, 1, 0, 0, 0.8, 0.001, 0, 0,
          marker_tri, "sample_tri", 1, 0, 0, 0, 0);
        marker_tri.points.push_back (marker_sample.points [0]);
        marker_tri.points.push_back (marker_sample.points [1]);
        marker_tri.points.push_back (marker_sample.points [1]);
        marker_tri.points.push_back (marker_sample.points [2]);
        marker_tri.points.push_back (marker_sample.points [2]);
        marker_tri.points.push_back (marker_sample.points [0]);

        // Make a copy for cumulative namespace
        visualization_msgs::Marker marker_tri_cumu;
        create_marker (visualization_msgs::Marker::LINE_LIST, frame_id, seq,
          0, 0, 0, 1, 0.5, 0, 0.8, 0.001, 0, 0,
          marker_tri_cumu, "sample_tri_cumu", 1, 0, 0, 0, cumu_marker_dur);
        marker_tri_cumu.points = marker_tri.points;


        // Create text labels for sides and angle, to visually see if I
        //   calculated correctly.
        // Ref boost: http://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
        //   str() http://www.boost.org/doc/libs/1_55_0/libs/format/example/sample_formats.cpp

        // Draw text at midpoint of side
        // NOTE: vis_text[0] is currently side btw pts [0] and [1]. Change if that
        //   changes.
        visualization_msgs::Marker marker_s10;
        create_marker (visualization_msgs::Marker::TEXT_VIEW_FACING, frame_id, 0,
          (marker_sample.points [0].x + marker_sample.points [1].x) * 0.5,
          (marker_sample.points [0].y + marker_sample.points [1].y) * 0.5,
          (marker_sample.points [0].z + marker_sample.points [1].z) * 0.5,
          1, 0, 0, 0.8, 0, 0, text_height,
          marker_s10, "text", 1, 0, 0, 0, 0);
        marker_s10.text = boost::str (boost::format ("%.2f") %
          vis_text [seq * nTriParams + 0]);

        // Draw text at midpoint of side
        // NOTE: vis_text[1] is currently side btw pts [1] and [2]. Change if that
        //   changes.
        visualization_msgs::Marker marker_s12;
        create_marker (visualization_msgs::Marker::TEXT_VIEW_FACING, frame_id, 1,
          (marker_sample.points [1].x + marker_sample.points [2].x) * 0.5,
          (marker_sample.points [1].y + marker_sample.points [2].y) * 0.5,
          (marker_sample.points [1].z + marker_sample.points [2].z) * 0.5,
          1, 0, 0, 0.8, 0, 0, text_height,
          marker_s12, "text", 1, 0, 0, 0, 0);
        marker_s12.text = boost::str (boost::format ("%.2f") %
          vis_text [seq * nTriParams + 1]);

        // NOTE: Angle currently is btw the sides [0][1] and [1][2], so plot
        //  angle at point [1]. If change definition of angle, need to change
        //  this too!
        visualization_msgs::Marker marker_angle1;
        create_marker (visualization_msgs::Marker::TEXT_VIEW_FACING, frame_id, 2,
          marker_sample.points [1].x, marker_sample.points [1].y,
          marker_sample.points [1].z,
          1, 0, 0, 0.8, 0, 0, text_height,
          marker_angle1, "text", 1, 0, 0, 0, 0);
        marker_angle1.text = boost::str ( boost::format ("%.2f") %
          (vis_text [seq * nTriParams + 2] * 180 / M_PI));


        // Voxel that corresponds to the triangle drawn above
        // Green
        visualization_msgs::Marker marker_vox;
        create_marker (visualization_msgs::Marker::CUBE, frame_id, 0,
          voxel_center_pts [triangle_to_voxel_idx [seq]].x,
          voxel_center_pts [triangle_to_voxel_idx [seq]].y,
          voxel_center_pts [triangle_to_voxel_idx [seq]].z,
          0, 0.8, 0, 0.3, voxel_side, voxel_side, voxel_side,
          marker_vox, "sample_voxel", 1, 0, 0, 0, 0);

        // Make a copy for cumulative namespace
        // Green
        visualization_msgs::Marker marker_vox_cumu;
        create_marker (visualization_msgs::Marker::CUBE, frame_id, 
          triangle_to_voxel_idx [seq],
          voxel_center_pts [triangle_to_voxel_idx [seq]].x,
          voxel_center_pts [triangle_to_voxel_idx [seq]].y,
          voxel_center_pts [triangle_to_voxel_idx [seq]].z,
          0, 0.5, 0, 0.2, voxel_side, voxel_side, voxel_side,
          marker_vox_cumu, "sample_voxel_cumu", 1, 0, 0, 0, cumu_marker_dur);


        // Spheres that are size of hand, at the pass in the voxel that 
        //   corresponds to the triangle drawn above.
        visualization_msgs::Marker marker_vox_spheres;
        create_marker (visualization_msgs::Marker::SPHERE, frame_id, 0,
          voxel_sphere_centers [triangle_to_sphere_idx [seq]].x,
          voxel_sphere_centers [triangle_to_sphere_idx [seq]].y,
          voxel_sphere_centers [triangle_to_sphere_idx [seq]].z,
          0.8, 0, 0, 0.3, radius*2, radius*2, radius*2,
          marker_vox_spheres, "sample_hand_sphere", 1, 0, 0, 0, 0);


        vis_pub.publish (marker_sample);
        vis_pub.publish (marker_sample_cumu);
        vis_pub.publish (marker_tri);
        vis_pub.publish (marker_tri_cumu);

        vis_pub.publish (marker_vox);
        vis_pub.publish (marker_vox_cumu);
        vis_pub.publish (marker_vox_spheres);

        vis_pub.publish (marker_s10);
        vis_pub.publish (marker_s12);
        vis_pub.publish (marker_angle1);

        seq ++;
        if (seq >= nTriangles)
        {
          prev_seqs = seq - 1;
          break;
        }
      }

      ros::spinOnce ();


      if (thisNode.goToNextObj_ || thisNode.doSkip_ || thisNode.doTerminate_)
        break;

      // If not visualizing, stop after noVizIters iterations
      if (! doRViz)
      {
        seq += 1;
        if (seq >= noVizIters)
        {
          //fprintf (stderr, "Proceeding to next model\n");
          break;
        }
      }

      wait_rate.sleep ();
    }


    // Clear for next iter
    cloud.clear ();
    octree.deleteTree ();
    triangles.clear ();
    vis_text.clear ();
    voxel_sphere_centers.clear ();
    voxel_center_pts.clear ();
    triangle_to_voxel_idx.clear ();
    triangle_to_sphere_idx.clear ();
    l0.clear ();
    l1.clear ();
    l2.clear ();
    a0.clear ();
    a1.clear ();
    a2.clear ();


    // Debug output
    //if (DEBUG_VOXEL)
    //  fprintf (stderr, "After clearing: %d vertices after pcl::VoxelGrid downsampling. %d vertices in copy of data.\n",
    //    (unsigned int) cloud.size (),
    //    (unsigned int) (cloud2_ptr->height * cloud2_ptr->width));


    if (thisNode.doSkip_)
    {
      thisNode.doSkip_ = false;
      continue;
    }

    if (thisNode.doTerminate_)
      break;

  }


  fprintf (stderr, "\n");

  // Print out running time
  //double duration = (std::clock () - start_time) / (double) CLOCKS_PER_SEC;

  time_t end_time = time (NULL);
  double duration = difftime (end_time, mktime (&y2k)) -
    difftime (start_time, mktime (&y2k));

  fprintf (stderr, "Total sample time for %lu objects: %f seconds. \n"
    "Average %f seconds per object.\n", model_name.size (), duration,
    duration / model_name.size ());


  // Publish a last msg to tell Python node we're done, so that they can show
  //   matplotlib plot images and save them
  // I think this only works if program shuts down normally. If ros::ok() is
  //   false, then these msgs don't get published.
  fprintf (stderr, "Publishing last %d messages to tell subscriber we are "
    "terminating...\n", noVizIters);
  triangle_sampling_msgs::TriangleParams tri_msg;
  tri_msg.obj_seq = -1;
  for (int i = 0; i < noVizIters; i ++)
  {
    tri_pub.publish (tri_msg);
    //fprintf (stderr, "Publishing obj_seq %d\n", tri_msg.obj_seq);
    wait_rate.sleep ();
  }



  //========
  // Calculate histogram intersection btw different objs
  //========

  // Not doing this in C++. Pass over to Python
  //   See sample_pcl_calc_hist.py


  //========
  // Publish histogram intersections to Python node, which will call matplotlib
  //   to plot them in Python.
  // I want to plot like confusion matrix. 10 objs in the row, same 10 objs
  //   in the column. 1st row and 1st col are pictures of these objs (make 
  //   this table in TeX, not in matplotlib. Just put the .eps plot in the
  //   multi-column multi-row 9x9 sub-table in the lower right corner.
  //   ____________________
  //   |\|_|_|_|_|_|_|_|_|_|
  //   |_|                 |
  //   |_|                 |
  //   |_|      .eps       |
  //   |_|                 |
  //   |_|_________________|
  //========

  // See sample_pcl_calc_hist.py


  // http://www.cplusplus.com/reference/memory/shared_ptr/get/
  //if (DEBUG_VOXEL)
  //  fprintf (stderr, "Pointer is: %p\n", cloud2_ptr.get());

}

