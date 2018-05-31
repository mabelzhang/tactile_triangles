// Mabel Zhang
// 19 Oct 2015
// Modified from http://gazebosim.org/tutorials?cat=sensors&tut=contact_sensor&ver=1.9

#ifndef _GAZEBO_CONTACT_PLUGIN_HH_
#define _GAZEBO_CONTACT_PLUGIN_HH_

#include <string>

// ROS Gazebo plugin
#include <ros/ros.h>
#include <gazebo/common/Plugin.hh>

#include <gazebo/sensors/sensors.hh>

#include <std_msgs/Int32.h>

// My packages
#include <reflex_gazebo_msgs/Contact.h>

namespace gazebo
{
  /// \brief An example plugin for a contact sensor.
  class ContactTutorialPlugin : public SensorPlugin
  {
    /// \brief Constructor.
    public: ContactTutorialPlugin();

    /// \brief Destructor.
    public: virtual ~ContactTutorialPlugin();


    // Member functions

    // I thnk Gazebo calls this, upon reading your SDF <plugin> tag. It passes
    //   in the _sensor and _sdf. The function then uses the _sensor passed in
    //   to populates the parentSensor member field. Then you can use
    //   that var to do whatever you want.
    //   It also tells Gazebo to call the OnUpdate() function, when sensor is
    //   updated. That is where you do whatever you want.
    /// \brief Load the sensor plugin.
    /// \param[in] _sensor Pointer to the sensor that loaded this plugin.
    /// \param[in] _sdf SDF element that describes the plugin.
    public: virtual void Load(sensors::SensorPtr _sensor, sdf::ElementPtr _sdf);

    /// \brief Callback that receives the contact sensor's update signal.
    private: virtual void OnUpdate();


    // Member fields

  private:

    /// \brief Pointer to the contact sensor
    sensors::ContactSensorPtr parentSensor;

    /// \brief Connection that maintains a link between the contact sensor's
    /// updated signal and the OnUpdate callback.
    event::ConnectionPtr updateConnection;


    // ROS

    // Publish contact info on rostopic
    ros::NodeHandle nh_;
    ros::Publisher contact_pub_;

    reflex_gazebo_msgs::Contact contact_msg_;
  };

  // Make some colors to print info
  // From http://stackoverflow.com/questions/287871/print-in-terminal-with-colors-using-python
  //   ANSI colors https://gist.github.com/chrisopedia/8754917
  //   C++ http://stackoverflow.com/questions/7414983/how-to-use-the-ansi-escape-code-for-outputting-colored-text-on-console
  static const char * OKCYAN = "\033[96m";
  static const char * OKRED = "\033[31m";
  static const char * OKGREEN = "\033[32m";
  static const char * OKBLUE = "\033[34m";
  static const char * ENDC = "\033[0m";
}
#endif

