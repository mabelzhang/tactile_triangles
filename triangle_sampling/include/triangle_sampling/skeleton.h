// Mabel Zhang
// 2 Apr 2014
//
// Added this file so other packages can use the create_marker() function
//   defined in skeleton.cpp
//

#ifndef _SKELETON_H_
#define _SKELETON_H_

// To publisher RViz markers
#include <visualization_msgs/Marker.h>

// Fill in a Marker message
// Copied from /opt/ros/fuerte/stacks/object_manipulation/object_manipulator/src/object_manipulator/draw_functions.py
//   It is used in pick_and_place_manager.py. Adapted to C++.
// Parameters:
//   duration: in seconds
//   marker: return value

/* Header if put in .cpp file
void create_marker (int type, const std::string frame, int id,
  double tx, double ty, double tz,
  float r, float g, float b, float alpha,
  double sx, double sy, double sz,
  // Return value
  visualization_msgs::Marker& marker, 
  // Default values
  const std::string ns,
  double qw, double qx, double qy, double qz,
  double duration, bool frame_locked)
  //const std::string model_path_dae
*/

void create_marker (int type, const std::string frame, int id,
  double tx, double ty, double tz,
  float r, float g, float b, float alpha,
  double sx, double sy, double sz,
  // Return value
  visualization_msgs::Marker& marker,
  // Default values
  const std::string ns = std::string ("pose_marker"),
  // Identity is qw1 qx0 qy0 qz0
  double qw = 0., double qx = 0., double qy = 0., double qz = 0.,
  double duration = 60., bool frame_locked = false)
{
  marker.header.frame_id = frame;
  // If want marker to display forever
  //marker.header.stamp = ros::Time ();
  // If want marker to display only for a duration lifetime
  marker.header.stamp = ros::Time::now ();

  marker.ns = ns;
  // Must be unique. Same ID replaces the previously published one
  marker.id = id;

  marker.type = type;
  marker.action = visualization_msgs::Marker::ADD;

  marker.pose.position.x = tx;
  marker.pose.position.y = ty;
  marker.pose.position.z = tz;

  marker.pose.orientation.x = qx;
  marker.pose.orientation.y = qy;
  marker.pose.orientation.z = qz;
  marker.pose.orientation.w = qw;

  marker.scale.x = sx;
  marker.scale.y = sy;
  marker.scale.z = sz;

  marker.color.a = alpha;
  marker.color.r = r;
  marker.color.g = g;
  marker.color.b = b;

  marker.lifetime = ros::Duration(duration);

  // If this marker should be frame-locked, i.e. retransformed into its frame every timestep
  marker.frame_locked = frame_locked;

  // Only if using a MESH_RESOURCE marker type:
  //  Note: Path MUST be on the computer running RViz, not the PR2 computer!
  //  Pass this in param
  //const std::string model_path_dae ("package://MenglongModels/spraybottle_c_rotated_filledHole_1000timesSmaller.dae");
  // Equivalent to strcpy()
  //marker.mesh_resource = model_path_dae;
}

#endif
