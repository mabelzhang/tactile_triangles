// Mabel Zhang
// 19 Oct 2015
//
// ROS Gazebo Contact Sensor plugin to simulate pressure sensors on ReFlex Hand
//   fingers, but with Boolean value only,
// Each sensor gets one instance of this plugin. 27 instances total.
//   Upon contact, publishes its finger number (1-3) and sensor number (1-9)
//   to reflex_gazebo/Contact type message.
//
// Modified from http://gazebosim.org/tutorials?cat=sensors&tut=contact_sensor&ver=1.9
//
// gazebo::ContactPlugin API: https://osrf-distributions.s3.amazonaws.com/gazebo/api/dev/classgazebo_1_1ContactPlugin.html
//

#include "contact_sensor_plugin.h"

using namespace gazebo;
GZ_REGISTER_SENSOR_PLUGIN(ContactTutorialPlugin)

/////////////////////////////////////////////////
ContactTutorialPlugin::ContactTutorialPlugin() : SensorPlugin()
{
}

/////////////////////////////////////////////////
ContactTutorialPlugin::~ContactTutorialPlugin()
{
  // Disconnect sensor, see if this deletes sensor from world properly when
  //   hand is removed from world using remove_model
  // API https://osrf-distributions.s3.amazonaws.com/gazebo/api/dev/classgazebo_1_1sensors_1_1ContactSensor.html
  if (this -> parentSensor)
  {
    ROS_INFO ("Disconnecting contact sensor %s", this -> GetHandle ().c_str ());
    this -> parentSensor -> DisconnectUpdated (this -> updateConnection);
  }
}

/////////////////////////////////////////////////
// Save _sensor in member var. Tells Gazebo to call OnUpdate() when sensor gets
//   new values.
void ContactTutorialPlugin::Load(sensors::SensorPtr _sensor, sdf::ElementPtr /*_sdf*/)
{
  // Make sure the ROS node for Gazebo has already been initialized
  if (!ros::isInitialized())
  {
    ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load plugin. "
      << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
    return;
  }

  // Get the parent sensor.
  this->parentSensor =
    // This line errors in ROS Kinetic:
    //   no matching function for call to ‘dynamic_pointer_cast(gazebo::sensors::SensorPtr&)’
    // Changing boost to std fixed it.
    //boost::dynamic_pointer_cast<sensors::ContactSensor>(_sensor);
    std::dynamic_pointer_cast<sensors::ContactSensor>(_sensor);

  // Make sure the parent sensor is valid.
  if (!this->parentSensor)
  {
    gzerr << "ContactTutorialPlugin requires a ContactSensor.\n";
    return;
  }

  // Connect to the sensor update event.
  this->updateConnection = this->parentSensor->ConnectUpdated(
      boost::bind(&ContactTutorialPlugin::OnUpdate, this));

  // Make sure the parent sensor is active.
  this->parentSensor->SetActive(true);


  /////
  // ROS publisher
  /////

  contact_pub_ = nh_.advertise <reflex_gazebo_msgs::Contact> (
    "/reflex_gazebo/contact", 5);


  // Find out which sensor on hand this instance of the plugin is hooked up to,
  //   assign a unique sensor number to this instance.

  // Get parent link name, parse the string to get the finger # and sensor #
  // SensorPlugin API: https://osrf-distributions.s3.amazonaws.com/gazebo/api/dev/classgazebo_1_1SensorPlugin.html
  // this->parentSensor ContactSensor API: https://osrf-distributions.s3.amazonaws.com/gazebo/api/dev/classgazebo_1_1sensors_1_1ContactSensor.html

  // finger_1_sensor_6
  //std::string sensor_name = this -> parentSensor -> GetName ();
  //printf ("Sensor name: %s\n", sensor_name.c_str ());

  // libcontact_tutorial_plugin.dylib
  //std::string file_name = this -> GetFilename ();
  //printf ("File name: %s\n", file_name.c_str ());

  // f1s6_plugin
  std::string hdl_name = this -> GetHandle ();
  printf ("Handle name: %s\n", hdl_name.c_str ());

  // Grab this from the URDF file, urdf/full_reflex_model.gazebo
  // ATTENTION if change name there, need to change here accordingly, otherwise
  //   sensors won't get assigned the correct number, and you won't get the
  //   correct /reflex_hand message that tells you which sensor is activated!
  std::string name_format = "f%ds%d_plugin";

  // 1 to 3
  int fin_num = 0;
  // 1 to 9
  int sen_num = 0;
  sscanf (hdl_name.c_str (), name_format.c_str (), &fin_num, &sen_num);

  printf ("Parsed: %d %d\n", fin_num, sen_num);

  contact_msg_.fin_num = fin_num;
  contact_msg_.sen_num = sen_num;


  ROS_INFO ("Gazebo contact sensor plugin initialized for finger %d sensor %d, sensor name %s",
    fin_num, sen_num, this -> parentSensor -> GetName ().c_str ());
}

/////////////////////////////////////////////////
void ContactTutorialPlugin::OnUpdate()
{
  // Get all the contacts.
  msgs::Contacts contacts;
  contacts = this->parentSensor->GetContacts();
  for (unsigned int i = 0; i < contacts.contact_size(); ++i)
  {
    // Valuable info. Skipping printing because of clutter. Turn on if debugging
    // Print the two bodies in collision
    // contacts.contact(i) is physics::Contact type
    //   API https://osrf-distributions.s3.amazonaws.com/gazebo/api/dev/classgazebo_1_1physics_1_1Contact.html
    //std::cout << "Collision between[" << OKBLUE << contacts.contact(i).collision1() << ENDC
    //          << "] and [" << OKCYAN << contacts.contact(i).collision2() << ENDC << "]\n";

    // Too much clutter
    // Print low level physics information
    /*
    for (unsigned int j = 0; j < contacts.contact(i).position_size(); ++j)
    {
      std::cout << j << "  Position:"
                << contacts.contact(i).position(j).x() << " "
                << contacts.contact(i).position(j).y() << " "
                << contacts.contact(i).position(j).z() << "\n";
      std::cout << "   Normal:"
                << contacts.contact(i).normal(j).x() << " "
                << contacts.contact(i).normal(j).y() << " "
                << contacts.contact(i).normal(j).z() << "\n";
      std::cout << "   Depth:" << contacts.contact(i).depth(j) << "\n";
    }
    */
  }

  // If this sensor got any contacts at all, publish this sensor's identity to
  //   the topic, so subscribers know this sensor was activated
  if (contacts.contact_size () > 0)
  {
    contact_msg_.header.stamp = ros::Time::now ();

    contact_pub_.publish (contact_msg_);
  }
}

