# Mabel Zhang
# 3 Aug 2015

int32 obj_seq
string obj_name
string obj_cat

# Triangle parameters
float32[] l0
float32[] l1
float32[] l2
float32[] a0
float32[] a1
float32[] a2

# Absolute position of 3 points sampled on each triangle.
# Size of each array is same as the parameter arrays l0, l1, etc. Each item is
#   one triangle.
geometry_msgs/Point[] pt0
geometry_msgs/Point[] pt1
geometry_msgs/Point[] pt2

# Absolute position of object center in the world. This should be the geometry
#   mesh center, not the file center. This is used to simulate wrist pose, so
#   that it points to inside of object. Therefore this point must actually be
#   in the "inside" center of the object point cloud!
geometry_msgs/Point obj_center

# Radius in x y z
geometry_msgs/Vector3 obj_radii

