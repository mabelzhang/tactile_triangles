# Mabel Zhang
# 11 Jul 2015
#
# Availability of poses depend on whether the targeted frame is broadcasted to
#   tf. Publisher should publish as None, if a Pose doesn't exist. Subscribers
#   should check for None type when using the poses.
#

Header header

float32[] pressures
# 0 for palm, 1 2 3 for fingers CCW, 3 is thumb.
int8[] finger_idx
int8[] sensor_idx

# http://docs.ros.org/jade/api/geometry_msgs/html/msg/Pose.html
bool hand_valid
geometry_msgs/Pose[] pose_wrt_hand
geometry_msgs/Point[] norm_endpt_wrt_hand
string hand_frame_id

bool robot_valid
geometry_msgs/Pose[] pose_wrt_robot
geometry_msgs/Point[] norm_endpt_wrt_robot
string robot_frame_id

bool cam_valid
geometry_msgs/Pose[] pose_wrt_cam
geometry_msgs/Point[] norm_endpt_wrt_cam
string cam_frame_id

bool obj_valid
geometry_msgs/Pose[] pose_wrt_obj
geometry_msgs/Point[] norm_endpt_wrt_obj
string obj_frame_id

