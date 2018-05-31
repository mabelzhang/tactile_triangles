// Mabel Zhang
// 8 Mar 2015
//
// Subscribes to sensor_msgs/PointCloud, publish sensor_msgs/PointCloud2.
//
// PointCloud2 format allows you to save from topic to PCD file easily:
//   $ rosrun pcl_ros pointcloud_to_pcd input:=/topic
//   Ref: http://wiki.ros.org/pcl/Overview
//
//   View it using
//   $ pcd_viewer file.pcd
//
// Used in conjunction with manual_explore.py.
//

#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
// http://docs.ros.org/api/sensor_msgs/html/namespacesensor__msgs.html
#include <sensor_msgs/point_cloud_conversion.h>

class PointCloud2_publisher
{
  public:

    PointCloud2_publisher (ros::NodeHandle n)
    {
      cloud_sub_ = n.subscribe ("/tactile_map/contact/cloud", 5,
        &PointCloud2_publisher::PointCloud_CB, this);
      cloud2_pub_ = n.advertise <sensor_msgs::PointCloud2>
        ("/tactile_map/contact/cloud2", 5);

      got_first_ = false;
      got_another_ = false;
    }

    void PointCloud_CB (const sensor_msgs::PointCloud::ConstPtr& msg)
    {
      // Make a copy
      cloud_ = sensor_msgs::PointCloud (*msg);

      got_first_ = got_another_ = true;
    }

    void publish ()
    {
      // Safe guard
      if (! got_first_)
        return;

      // If got another message since previous publish, convert new msg.
      //   This if-stmt saves process time. If no new msg, just publish previous
      if (got_another_)
      {
        // API http://docs.ros.org/api/sensor_msgs/html/namespacesensor__msgs.html
        convertPointCloudToPointCloud2 (cloud_, cloud2_);
      }

      cloud2_pub_.publish (cloud2_);
      got_another_ = false;
    }

  private:

    ros::Subscriber cloud_sub_;
    ros::Publisher cloud2_pub_;

    bool got_first_;
    // Got another msg since previous published
    bool got_another_;

    sensor_msgs::PointCloud cloud_;
    sensor_msgs::PointCloud2 cloud2_;
};

int main (int argc, char * argv [])
{
  ros::init (argc, argv, "PointCloud2_publisher");
  ros::NodeHandle n;

  PointCloud2_publisher cloud2_pub (n);

  ros::Rate wait_rate (5);
  while (ros::ok ())
  {
    cloud2_pub.publish ();

    ros::spinOnce ();

    wait_rate.sleep ();
  }

  return 0;
}

