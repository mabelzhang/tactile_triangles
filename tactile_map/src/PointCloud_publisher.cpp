// Mabel Zhang
// 10 Mar 2015
//
// Modified from ./PointCloud2_publisher.cpp.
//
// Subscribes to sensor_msgs/PointCloud2, publish sensor_msgs/PointCloud.
// Ref: http://wiki.ros.org/pcl/Overview
//
// Used in conjunction with spin_seam.py.
//

#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
// http://docs.ros.org/api/sensor_msgs/html/namespacesensor__msgs.html
#include <sensor_msgs/point_cloud_conversion.h>

class PointCloud_publisher
{
  public:

    PointCloud_publisher (ros::NodeHandle n)
    {
      cloud2_sub_ = n.subscribe ("/cloud_pcd", 5,
        &PointCloud_publisher::PointCloud2_CB, this);
      cloud_pub_ = n.advertise <sensor_msgs::PointCloud>
        ("/cloud1_pcd", 5);

      got_first_ = false;
      got_another_ = false;
    }

    void PointCloud2_CB (const sensor_msgs::PointCloud2::ConstPtr& msg)
    {
      // Make a copy
      cloud2_ = sensor_msgs::PointCloud2 (*msg);

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
        convertPointCloud2ToPointCloud (cloud2_, cloud_);
      }

      cloud_pub_.publish (cloud_);
      got_another_ = false;
    }

  private:

    ros::Subscriber cloud2_sub_;
    ros::Publisher cloud_pub_;

    bool got_first_;
    // Got another msg since previous published
    bool got_another_;

    sensor_msgs::PointCloud cloud_;
    sensor_msgs::PointCloud2 cloud2_;
};

int main (int argc, char * argv [])
{
  ros::init (argc, argv, "PointCloud_publisher");
  ros::NodeHandle n;

  PointCloud_publisher cloud_pub (n);

  ros::Rate wait_rate (5);
  while (ros::ok ())
  {
    cloud_pub.publish ();

    ros::spinOnce ();

    wait_rate.sleep ();
  }

  return 0;
}

