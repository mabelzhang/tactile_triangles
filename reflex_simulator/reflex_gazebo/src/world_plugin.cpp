// Mabel Zhang
// 3 Nov 2015
//
// Note: Currently only using this to set gravity to 0, because the rosservice
//   call way makes hand's twist params all nans, and afterwards nothing works.
//   Not using the rostopic in here, as this way of moving hand makes it
//   fly into space. Instead, use rosservice call set_model_state.
//
//
// Subscribes to rostopic and moves hand accordingly.
// Note if you print from the rostopic callback function, must print to stderr.
//   stdout doesn't get flushed!! OnUpdate() can print to stdout fine, but
//   rostopic callback function can't.
//
// Tutorial: http://www.gazebosim.org/tutorials?tut=ros_plugins&cat=connect_ros
// Registering OnUpdate() function using event::Events:
//   http://gazebosim.org/tutorials/?tut=plugins_model
//
// To test in bare shell:
//   Launch hand in Gazebo, which loads this plugin through reflex.world file
//   $ roslaunch reflex_gazebo reflex_world.launch
//

// ROS Gazebo plugin
#include <ros/ros.h>
#include <gazebo/common/Plugin.hh>

// Gazebo
#include <gazebo/sensors/sensors.hh>
#include <gazebo/math/gzmath.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/transport.hh>

// ROS
//#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <geometry_msgs/Vector3.h>
//#include <std_srvs/Empty.h>

// My packages
#include <reflex_gazebo_msgs/GetModelSize.h>
#include <reflex_gazebo_msgs/SetModelPose.h>
#include <reflex_gazebo_msgs/SetFixedHand.h>  // Should replace with SetBool
#include <reflex_gazebo_msgs/RemoveModel.h>
#include <reflex_gazebo_msgs/SetBool.h>


namespace gazebo
{
  class HandWorldPlugin : public WorldPlugin
  {
  private:

    int seq_seen_;

    physics::WorldPtr world_;
    physics::PhysicsEnginePtr physics_;

    std::string reflex_model_name_;

    bool found_wrist_;
    physics::LinkPtr world_link_;
    physics::LinkPtr base_link_;
    physics::JointPtr world_to_base_joint_;

    physics::JointPtr world_to_base_joint_fresh_;

    // Get Gazebo to call this plugin every iteration of simulation
    event::ConnectionPtr updateConnection_;

    // Seems this must be a member var. If local var in Load(), might not get
    //   any subscribed msgs?
    ros::NodeHandle nh_;

    //ros::Subscriber move_sub_;
    //ros::Subscriber static_sub_;
    //ros::Subscriber fixed_sub_;
    ros::ServiceServer model_size_srv_;
    ros::ServiceServer model_pose_srv_;
    ros::ServiceServer fixed_hand_srv_;
    ros::ServiceServer rm_hand_srv_;
    ros::ServiceServer rm_model_srv_;
    ros::ServiceServer model_visible_srv_;

    int times_attached_, times_detached_;

    // Convenience flags for debugging. Only ONE should be set to true!!!
    bool TestAttachDetach_;  // Detach() causes seg fault in Gazebo
    bool TestAttachStatic_;
    bool TestInitJoint_;

    bool hand_exists_;
    math::Pose hand_pose_;


  public:

    HandWorldPlugin () : WorldPlugin ()
    {
    }

    void find_link (std::string link_name, physics::LinkPtr& link)
    {
      //physics::LinkPtr link;

      if (! world_)
        return;

      physics::BasePtr link_genericType = world_ -> GetByName (
        link_name);
      if (link_genericType)
      {
        link =
          boost::dynamic_pointer_cast <physics::Link> (link_genericType);

        fprintf (stderr, "Found link %s\n",
          link -> GetScopedName ().c_str ());
 
        for (int i = 0; i < link -> GetParentJoints ().size (); i ++)
          fprintf (stderr, "  Parent joint %d: %s\n", i,
            link -> GetParentJoints ().at (i) -> GetName ().c_str ());
 
        for (int i = 0; i < link -> GetChildJoints ().size (); i ++)
          fprintf (stderr, "  Child joint %d: %s\n", i,
            link -> GetChildJoints ().at (i) -> GetName ().c_str ());
      }
      else
      {
        fprintf (stderr, "Did not find link %s.\n", link_name.c_str ());
      }
    }

    void find_model (std::string model_name, physics::ModelPtr& model)
    {
      if (! world_)
        return;

      // Is this local var making GetBoundingBox() hang? > Nope. It just hangs
      //physics::ModelPtr model;

      physics::BasePtr model_genericType = world_ -> GetByName (
        model_name);
      if (model_genericType)
      {
        model = boost::dynamic_pointer_cast <physics::Model> (model_genericType);
      }
      else
      {
        fprintf (stderr, "Did not find model %s.\n", model_name.c_str ());
      }
    }

    void find_wrist ()
    {
      if (! world_)
        return;

      found_wrist_ = true;


      // Save to member fields

      // Using find_link() instead. Delete this block when that works
      /*
      physics::BasePtr world_link_genericType = world_ -> GetByName (
        //"world");
        "ground_plane::link");
      if (world_link_genericType)
      {
        world_link_ =
          boost::dynamic_pointer_cast <physics::Link> (world_link_genericType);
        fprintf (stderr, "Found world link\n");
      }
      else
      {
        fprintf (stderr, "Did not find world link\n");
      }
      */

      //find_link ("world", world_link_);
      find_link ("ground_plane::link", world_link_);

      find_link ("base_link", base_link_);

      // Joint connecting hand root /base_link to parent, i.e. the wrist joint
      physics::BasePtr wrist_jnt_genericType = world_ -> GetByName (
        "world_to_base_link");
      if (wrist_jnt_genericType)
      {
        world_to_base_joint_ =
          boost::dynamic_pointer_cast <physics::Joint> (wrist_jnt_genericType);

        // physics::Joint API: https://osrf-distributions.s3.amazonaws.com/gazebo/api/dev/classgazebo_1_1physics_1_1Joint.html
        fprintf (stderr, "Found joint %s. Type %x\n",
          world_to_base_joint_ -> GetName ().c_str (),
          world_to_base_joint_ -> GetType ());

        // Doesn't compile
        //if (world_to_base_joint_ -> GetSDF () -> HasAttribute (std::string ("type")))
        //  fprintf (stderr, "Type?: %s\n", world_to_base_joint_ -> GetSDF () -> Get ("type"));

        if (world_to_base_joint_ -> GetParent ())
          fprintf (stderr, "  Parent: %s\n",
            world_to_base_joint_ -> GetParent () -> GetName ().c_str ());
        else
        {
          fprintf (stderr, "  No parent\n");


          /*
          fprintf (stderr, "  Adding world as parent.\n");

          // Add world_to_base_link joint as a child of world link
          world_link_ -> AddChildJoint (world_to_base_joint_);
          // This cast is needed to compile in OS X Yosemite ROS Indigo Gazebo
          //   2.2.6. Cast not needed in Ubuntu ROS Jade Gazebo 5.1.0.
          boost::dynamic_pointer_cast <physics::Base> (world_link_) -> Update ();

          fprintf (stderr, "World link %s\n", world_link_ -> GetScopedName ().c_str ());
          for (int i = 0; i < world_link_ -> GetChildJoints ().size (); i ++)
            fprintf (stderr, "  Child: %s\n",
              world_link_ -> GetChildJoints ().at (i) -> GetName ().c_str ());

         
          // Add world link as a parent of world_to_base_link joint
          world_to_base_joint_ -> SetParent (world_link_);
          world_to_base_joint_ -> Update ();

          fprintf (stderr, "Joint %s\n", world_to_base_joint_ -> GetName ().c_str ());
          if (world_to_base_joint_ -> GetParent ())
            fprintf (stderr, "  Parent: %s\n",
              world_to_base_joint_ -> GetParent () -> GetName ().c_str ());
          else
            fprintf (stderr, "  Still no parent\n");


          // Try 2nd way to add world link as parent of world_to_base_link joint
          fprintf (stderr, "  Attaching world link to base link\n");

          world_to_base_joint_ -> Attach (world_link_, base_link_);
          world_to_base_joint_ -> Update ();

          if (world_to_base_joint_ -> GetParent ())
            fprintf (stderr, "  Parent: %s\n",
              world_to_base_joint_ -> GetParent () -> GetScopedName ().c_str ());
          else
            fprintf (stderr, "  Still no parent\n");
          */
        }

        if (world_to_base_joint_ -> GetChild ())
        {
          fprintf (stderr, "  Child: %s\n",
            world_to_base_joint_ -> GetChild () -> GetName ().c_str ());
        }


        if (TestAttachStatic_)
        {
          // To test AttachStaticModel(), have to remove hand from world joint
          //   first, to allow hand to move freely.
          //   Will only connect hand with world by using Link
          //   AttachStaticModel().
          // TODO: I should eliminate this, `.` this approach doesn't use 
          //   Detach() calls at all, `.` it gives seg faults! Just remove 
          //   the world_to_base_link joint from URDF should be sufficient.
          //   > Removed it. Now don't use this var anymore here, because it'll
          //     be null.
          //if (world_to_base_joint_)
          //  world_to_base_joint_ -> Detach ();
         
          // Attach hand as a StaticModel to world link, so hand sticks to
          //   ground as initial state
          physics::ModelPtr reflex_model;
          find_model (reflex_model_name_, reflex_model);
          if (reflex_model)
          {
            reflex_model -> SetStatic (true);
            math::Pose pose = math::Pose (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            world_link_ -> AttachStaticModel (reflex_model, pose);
            ROS_ERROR ("AttachStaticModel base_link to world");
          }
        }

        else if (TestInitJoint_)
        {
          // Maybe it shouldn't be the joint itself. PhysicsEngine::CreateJoint
          //   API says the 2nd arg is supposed to be a parent! So it should be
          //   ground plane!
          physics::ModelPtr ground_model;
          find_model ("ground_plane", ground_model);
          if (! ground_model)
            ROS_ERROR ("ERROR: ground_plane model NOT found!");

          // Create the joint from fresh. It serves same purpose as
          //   world_to_base_joint_, but is created in code, instead of by URDF.
          world_to_base_joint_fresh_ = physics_ -> CreateJoint ("revolute",
            ground_model);

          // Detach the original joint loaded from URDF file
          world_to_base_joint_ -> Detach ();

          // Load the new one
          world_to_base_joint_fresh_ -> Load (world_link_, base_link_,
            math::Pose ());
          world_to_base_joint_fresh_ -> Init ();

          ROS_ERROR ("Created new fixed joint. Parent: %s, Child: %s",
            world_to_base_joint_fresh_ -> GetParent () -> GetScopedName ().c_str (),
            world_to_base_joint_fresh_ -> GetChild () -> GetScopedName ().c_str ());
        } 
      }
      else
      {
        fprintf (stderr, "Did not find world_to_base_link joint.\n");
      }
    }

    void OnUpdate (const common::UpdateInfo & /*_info*/)
    {
      //printf ("OnUpdate() is called");

      // Does this help hand from being bumped by cup?
      //   > Yes!!! Now hand moves a lot less, almost as good as fixing to
      //     fixed joint!
      physics::ModelPtr reflex_model;
      find_model (reflex_model_name_, reflex_model);
      if (reflex_model && hand_exists_)
      {
        reflex_model -> SetWorldTwist (math::Vector3 (), math::Vector3 ());
      }

      // Does this help hand from being bumped by cup?
      //   > No doesn't help much
      //if (base_link_)
      //  base_link_ -> SetLinkStatic (true);

      // Does this help hand from being bumped by cup?
      //   > Not sure... the pcd point clouds look very similar. Maybe this
      //     doesn't help much?
      /*
      if (base_link_)
      {
        base_link_ -> SetAngularAccel (math::Vector3 ());
        base_link_ -> SetAngularVel (math::Vector3 ());
        base_link_ -> SetLinearAccel (math::Vector3 ());
        base_link_ -> SetLinearVel (math::Vector3 ());
      }
      */

      // Does this help hand from being bumped by cup?
      //   > Not sure... the pcd point clouds look very similar. Maybe this
      //     doesn't help much?
      /*
      if (reflex_model)
      {
        reflex_model -> SetAngularAccel (math::Vector3 ());
        reflex_model -> SetAngularVel (math::Vector3 ());
        reflex_model -> SetLinearAccel (math::Vector3 ());
        reflex_model -> SetLinearVel (math::Vector3 ());
      }
      */

      if (reflex_model && hand_exists_)
      {
        // This prints "reflex", wtf?
        //fprintf (stderr, "Parent model of reflex_model: %s\n",
        //  reflex_model -> GetParentModel () -> GetName ().c_str ());

        // Does this help hand be fixed (immovable when bumped by cup)??
        reflex_model -> SetWorldPose (hand_pose_);
      }

      // Check if a rosmsg comes in to the subscribed topic
      ros::spinOnce ();
    }

    void print_world_child ()
    {
      if (world_link_)
      {
        ROS_ERROR ("INFO: world link has %lu children",
          world_link_ -> GetChildJoints ().size ());
        for (int i = 0; i < world_link_ -> GetChildJoints ().size (); i ++)
          fprintf (stderr, "  Child: %s\n",
            world_link_ -> GetChildJoints ().at (i) -> GetName ().c_str ());
      }
      else
        ROS_ERROR ("ERROR: world link is NULL!");
    }

    void print_joint_parent_child (physics::JointPtr joint)
    {
      if (joint)
      {
        ROS_ERROR ("INFO: %s has parent:", joint -> GetName ().c_str ());
        if (joint -> GetParent ())
          fprintf (stderr, "  Parent: %s\n",
            joint -> GetParent () -> GetScopedName ().c_str ());
        else
          fprintf (stderr, "  No parent\n");

        ROS_ERROR ("INFO: %s has children:", joint -> GetName ().c_str ());
        if (joint -> GetChild ())
          fprintf (stderr, "  Child: %s\n",
            joint -> GetChild () -> GetName ().c_str ());
        else
          fprintf (stderr, "  No children\n");
      }
      else
        ROS_ERROR ("ERROR: Joint passsed in is NULL!");
    }

    void print_base_parent ()
    {
      // Check how many parent joints base_link link now has, after
      //   attach/detach
      if (base_link_)
      {
        ROS_ERROR ("INFO: base_link now has %lu parent joints",
          base_link_ -> GetParentJoints ().size ());
        for (int i = 0; i < base_link_ -> GetParentJoints ().size (); i ++)
        {
          if (base_link_ -> GetParentJoints().at (i))
          {
            fprintf (stderr, "  Parent: %s\n",
              base_link_ -> GetParentJoints ().at (i) -> GetName ().c_str ());
          }
        }
      }
      else
        ROS_ERROR ("ERROR: base link is NULL!");
    }


    // Set hand to be static
    // This doesn't work. Hand can still be moved
    /*
    void static_hand_cb (const std_msgs::Bool::ConstPtr& msg)
    {
      fprintf (stderr, "Setting hand static to %s\n", msg -> data ? "true" :
        "false");

      physics::BasePtr wrist_link_genericType = world_ -> GetByName (
        "base_link");
      physics::LinkPtr wrist_link =
        boost::dynamic_pointer_cast <physics::Link> (wrist_link_genericType);

      fprintf (stderr, "Found link %s\n", wrist_link -> GetName ().c_str ());

      for (int i = 0; i < wrist_link -> GetParentJoints ().size (); i ++)
        fprintf (stderr, "  Parent joint %d: %s\n", i,
          wrist_link -> GetParentJoints ().at (i) -> GetName ().c_str ());

      for (int i = 0; i < wrist_link -> GetChildJoints ().size (); i ++)
        fprintf (stderr, "  Child joint %d: %s\n", i,
          wrist_link -> GetChildJoints ().at (i) -> GetName ().c_str ());

      wrist_link -> SetStatic (msg -> data);
    }
    */

    // Set hand joint type to fixed, so it cannot be moved during collision
    //   with other objects
    //void fixed_hand_cb (const std_msgs::Bool::ConstPtr& msg)
    bool fixed_hand_cb (reflex_gazebo_msgs::SetFixedHand::Request & req,
      reflex_gazebo_msgs::SetFixedHand::Response & res)
    {
      //fprintf (stderr, "Setting hand fixed to %s\n", msg -> data ? "true" :
      //  "false");
      ROS_ERROR ("Setting hand fixed to %s", req.fixed ? "true" :
        "false");

      // Link
      /*
      physics::BasePtr wrist_link_genericType = world_ -> GetByName (
        "base_link");
      physics::LinkPtr wrist_link =
        boost::dynamic_pointer_cast <physics::Link> (wrist_link_genericType);

      fprintf (stderr, "Found link %s\n", wrist_link -> GetName ().c_str ());

      for (int i = 0; i < wrist_link -> GetParentJoints ().size (); i ++)
        fprintf (stderr, "  Parent joint %d: %s\n", i,
          wrist_link -> GetParentJoints ().at (i) -> GetName ().c_str ());

      for (int i = 0; i < wrist_link -> GetChildJoints ().size (); i ++)
        fprintf (stderr, "  Child joint %d: %s\n", i,
          wrist_link -> GetChildJoints ().at (i) -> GetName ().c_str ());


      // Haven't tried this yet, but think this would have to be set recursively
      //   on every single link, including fingers, excluding sensors.
      //wrist_link -> SetCollideMode ("fixed");
      */


      // Joint
      // This doesn't work. No such link as world_to_base_link, SDF removed it
      /*
      // Joint connecting hand root /base_link to parent, i.e. the wrist joint
      physics::BasePtr wrist_jnt_genericType = world_ -> GetByName (
        "world_to_base_link");
      physics::JointPtr wrist_jnt =
        boost::dynamic_pointer_cast <physics::Joint> (wrist_jnt_genericType);

      // physics::Joint API: https://osrf-distributions.s3.amazonaws.com/gazebo/api/dev/classgazebo_1_1physics_1_1Joint.html
      fprintf (stderr, "Found joint %s\n", wrist_jnt -> GetName ().c_str ());
      fprintf (stderr, "  Parent: %s\n", wrist_jnt -> GetParent () -> GetName ().c_str ());
      fprintf (stderr, "  Child: %s\n", wrist_jnt -> GetChild () -> GetName ().c_str ());

      // Set joint type to fixed

      // sdf::Element http://osrf-distributions.s3.amazonaws.com/gazebo/api/1.2.5/classsdf_1_1Element.html
      sdf::ElementPtr sdf_vals = wrist_jnt -> GetSDF ();

      sdf::ParamPtr jnt_type = sdf_vals -> GetAttribute ("type");
      std::string jnt_type_value;
      jnt_type -> Get (jnt_type_value);
      fprintf (stderr, "This joint type: %s\n", jnt_type_value.c_str ());

      jnt_type -> Set ("fixed");
      wrist_jnt -> UpdateParameters (sdf_vals);
      */


      // Add a joint
      // This doesn't work. Doesn't compile. I don't know how to make a Joint
      //   object from code correctly.
      /*
      physics::BasePtr world_link_genericType = world_ -> GetByName (
        "world");
      physics::LinkPtr world_link =
        boost::dynamic_pointer_cast <physics::Link> (world_link_genericType);

      physics::BasePtr base_link_genericType = world_ -> GetByName (
        "base_link");
      physics::LinkPtr base_link =
        boost::dynamic_pointer_cast <physics::Link> (base_link_genericType);

      // Make a new link to attach the two
      //physics::ODEJointPtr wrist_jnt (new physics::ODEJoint (
      //  world_link_genericType));
      physics::HingeJoint <physics::BulletJoint> wrist_jnt =
        new physics::HingeJoint <physics::BulletJoint> (
        world_link_genericType);
      wrist_jnt -> SetName ("world_to_base_link");
      wrist_jnt -> Attach (world_link, base_link);

      // Check if actually attached
      // physics::Joint API: https://osrf-distributions.s3.amazonaws.com/gazebo/api/dev/classgazebo_1_1physics_1_1Joint.html
      fprintf (stderr, "Found joint %s\n", wrist_jnt -> GetName ().c_str ());
      fprintf (stderr, "  Parent: %s\n", wrist_jnt -> GetParent () -> GetName ().c_str ());
      fprintf (stderr, "  Child: %s\n", wrist_jnt -> GetChild () -> GetName ().c_str ());

      */


      // This doesn't work, not because of the code, but because world_to_base_link joint is 0-limits revolute joint in URDF, so nothing can move it. It's a fixed joint.
      /*
      if (! found_wrist_)
        find_wrist ();

      base_link_ -> SetLinkStatic (msg -> data);
      */


      // This is the only way that works! But it gets intermittent seg fault
      //   after Detach() call. But gdb shows seg fault is somewhere else, not
      //   in this file.
      if (TestAttachDetach_)
      {
        // Joint detach (if fixed==true) and re-attach (else)
        if (! found_wrist_)
          find_wrist ();
       
        // Fixed link
        if (req.fixed)
        {
          // Check null ptr (Boost shared ptr is null if you don't set it. It 
          //   doesn't allow assigning to NULL explicitly.)
          //   http://stackoverflow.com/questions/621220/null-pointer-with-boostshared-ptr
          if (world_to_base_joint_)
          {
            //ROS_ERROR ("world_link_ is valid pointer? %s", world_link_?"true":"false");
       
            // This line is needed even if world_link_ is null. Otherwise the
            //   hand doesn't stay fixed.
            world_to_base_joint_ -> Attach (world_link_, base_link_);
            world_to_base_joint_ -> Update ();
       
            // What if I don't do this? Doesn't seem to make a difference. Seg
            //   fault happens with or without these. Maybe I don't need this.
            /*
            // Have to do this ourselves, because Gazebo physics::Joint::Attach()
            //   unbelievably doesn't add joint to the parent and child links as
            //   child and parent joints, respectively.
            if (world_link_)
            {
              bool found = false;
              physics::Joint_V children = world_link_ -> GetChildJoints ();
              for (int i = 0; i < children.size (); i ++)
              {
                if ((children [i]) && (! children [i] -> GetName ().compare (
                  world_to_base_joint_ -> GetName ())))
                {
                  found = true;
                  break;
                }
              }
              if (! found)
                world_link_ -> AddChildJoint (world_to_base_joint_);
            }
       
            if (base_link_)
            {
              bool found = false;
              physics::Joint_V parents = base_link_ -> GetParentJoints ();
              for (int i = 0; i < parents.size (); i ++)
              {
                if ((parents [i]) && (! parents [i] -> GetName ().compare (
                  world_to_base_joint_ -> GetName ())))
                {
                  found = true;
                  break; 
                }
              }
              if (! found)
                base_link_ -> AddParentJoint (world_to_base_joint_);
            }
            */
       
            times_attached_ ++;
       
            ROS_ERROR ("Attached base_link to world. Cumulative %d times",
              times_attached_);
          }
        }
       
        // Not fixed, so detach joint
        else
        {
          if (world_to_base_joint_)
          {
            world_to_base_joint_ -> Detach ();
            world_to_base_joint_ -> Update ();
       
            // Added this to see if fixes seg fault. Nope it doesn't.
            // Detach() only removes this joint from its parent and child links,
            //   but doesn't remove the links from this joint's parentLink and
            //   childLink fields!
            //world_to_base_joint_ -> RemoveChild ("/base_link");
       
            times_detached_ ++;
       
            // Printing to stderr so it gets flushed. Not an actual error
            ROS_ERROR ("INFO: Detached base_link from world. Cumulative %d times",
              times_detached_);
          }
        }
       
        // Sanity checks
        print_world_child ();
        print_joint_parent_child (world_to_base_joint_);
        print_base_parent ();
      }


      // This is the only way that works, and no seg fault!!!!
      //
      // Hand remains fixed to ground
      //   Fix for that: world_to_base_joint -> Detach() in find_wrist(), and
      //     world_link_ -> AttachStaticModel(), to init the hand to ground.
      // Now hand moves, but the fingers bang on the hand itself, causing a
      //   non-zero torque. So the next time hand is moved, it still has the
      //   torque and it's moving by itself after moving to new pose, probably
      //   after calling Detach.
      //   Can we set the twist to zero, before detaching?
      //   Yes this worked, for after moving hand anyway. It doesn't keep
      //     moving anymore.
      // Now cup's force (since it's fixed) on hand makes hand fly away and
      //   fingers close in onto the hand itself. How to make fingers stop
      //   closing in? I thought they were before, when I had hand attached to
      //   fixed joint world_to_base_joint. Now that hand is not fixed, it 
      //   doesn't have enough force to counter the cup's infinite mass.
      //   But setting the hand "static" should be the same as attaching hand
      //   to the fixed world_to_base_link joint. Why isn't hand static? Does
      //   "static" mean something else, not actually fixed?
      //
      //   When the hand spins away from the cup, looks like the controllers
      //   are dismantled, the fingers move through joint limits to where they
      //   aren't supposed to be able to move! Is the infinite force from the
      //   cup that big? Could be. When hand is fixed to the ground and I try
      //   to teleport it, this does happen too.
      //
      //   Maybe I just need to pause longer after guraded_move? Or decrease
      //   how much I close each step in the guarded_move cycles?
      //
      //   The main problem is not the cup's force, it's the hand's controller's
      //   force! Cup isn't forcing hand to go one way, the hand's own
      //   controllers are! As the hand forces to close the fingers, and they
      //   can't close because of the obstacle (object), the controllers exert
      //   too much force, and end up slipping past the obstacle, banging onto
      //   the palm, sending the hand rotating and flying away.
      // 
      //   So the root problem is still that the hand isn't fixed! Why isn't it
      //   "fixed" when I set it to static?? Model object API says SetStatic()
      //   means immovable! But it moves!
      //
      //   What if I comment the call SetStatic(false) after detaching? Maybe
      //   this is setting the hand loose?
      //   Nope. That didn't help. That sends the hand always spinning whenever
      //   I teleport it! So must need to set it to false after detach.
      //
      //   In btw attaching and detaching hand, when it's doing guarded_move,
      //   do I do something else to the hand? I move it to the cylinder grid
      //   pose in model_pose_cb(). Could that have removed the static flag?
      //   What if I call SetStatic(true) again in that function?
      //   Actually it is done right after that function, when caller calls
      //   set fixed hand. So this is unnecessary and already done.
      //
      //   Change guarded_move increment to be smaller.
      //   Didn't help, hand still spins away. So this is not the problem.
      //
      //   3rd time coming back to the conclusion that the root problem is that
      //   the hand model isn't really fixed! How to make it fixed??? That's
      //   probably why I resorted to the world_to_base_joint joint...
      //   attaching to it made the hand fixed, and that was the only way I
      //   tried that worked...
      //
      //   Maybe still call world_to_base_joint_ -> Detach() and Attach() here?
      //   Could it be that this was the gap causing the segmentation fault,
      //   that I only connect the two links with a joint, didn't connect the
      //   two links directly with each other? AttachStaticModel() does the
      //   latter. Maybe seg fault wouldn't happen if I call Attach() and
      //   Detach() here now??
      //   Nope. Seg faulted immediately, at the 3rd grid point! Wow, quick to
      //   see effect of seg fault... Hand did stay fixed, didn't spin away.
      //
      // Solved by putting SetWorldPose(hand_pose_) in OnUpdate(). Now works!!!
      if (TestAttachStatic_)
      {
        if (! found_wrist_)
          find_wrist ();
       
        physics::ModelPtr reflex_model;
        find_model (reflex_model_name_, reflex_model);
        if (! reflex_model)
        {
          res.success = false;
          return true;
        }
       
        if (req.fixed)
        {
          if (world_link_)
          {
            reflex_model -> SetStatic (true);
            world_link_ -> AttachStaticModel (reflex_model, math::Pose ());
       
            // This makes hand's spinning and flying away after hitting cup
            //   milder. Fingers don't slam into palm anymore, though hand
            //   still moves after touching cup.
            reflex_model -> SetStatic (false);

            ROS_ERROR ("AttachStaticModel base_link to world");
          }
        }
        else
        {
          if (world_link_)
          {
            // Application-specific: Set model torque to 0, so hand doesn't go
            //   flying after detach, if it had non-zero torque due to fingers
            //   closing and banging onto palm, or object bumping hand.
            // This reduces hand's flying away by a lot.
            reflex_model -> SetWorldTwist (math::Vector3 (), math::Vector3 ());

            world_link_ -> DetachStaticModel ("reflex");
       
            // This makes hand fly away less
            reflex_model -> SetStatic (false);

            ROS_ERROR ("DetachStaticModel base_link from world");
          }
        }
      }


      /*
      if (! found_wrist_)
        find_wrist ();

TODO
      if (req.fixed)
      {
        base_link_ -> AddParentJoint (world_to_base_joint_);
        world_to_base_joint_ -> Update ();
        //base_link_ -> Update ();
      }
      else
      {
        world_link_ -> RemoveChildJoint ("world_to_base_link");
        base_link_ -> RemoveParentJoint ("world_to_base_link");
        //world_link_ -> Update ();
        //base_link_ -> Update ();
        world_to_base_joint_ -> Update ();
      }
      */


      // AndreiHaidu's suggestion to look at Gripper.cc.
      //   It CreateJoint() in Gripper::Load(), and at each update call to
      //   HandleAttach(), it Load()s a bunch of palm links, and Init() the
      //   joint from fresh.
      //
      // This works the same as my original approach that worked!! However,
      //   This still gets seg faults at Detach()!!! I doubt that their
      //   Gripper.cc doesn't get seg faults. They must get it too. Maybe not
      //   enough people use it to find this bug. They need to fix how they
      //   do Detach(), or whatever update it results in that's causing the seg
      //   fault.

      if (TestInitJoint_)
      {
        if (! found_wrist_)
          find_wrist ();
       
        physics::ModelPtr reflex_model;
        find_model (reflex_model_name_, reflex_model);
        if (! reflex_model)
        {
          res.success = false;
          return true;
        }
       
       
        if (req.fixed)
        {
          reflex_model -> SetStatic (true);
        
          // Load and init the joint

          // Specify joint's parent and child links.
          // Gripper.cc never specifies children for the joint using SetChild(),
          //   because they call fixedJoint->Load(palm_model), and that
          //   automatically adds the palm as parent of the fixed joint, and
          //   a bunch of contact links as children of the fixed joint.
          //   Then they don't need to call Attach(). Maybe that's how they
          //   got around calling Attach()!!
          //   Joint::Load(_parent, _child_, _pose) is the signature
          //   of function called in Gripper.cc. Load() loads the joint, not 
          //   the link children! Gripper.cc calls Load() in a loop, which means
          //   they probably create multiple identical joints, each connecting
          //   one contact link to the palm model.
          world_to_base_joint_fresh_ -> Load (world_link_, base_link_,
            math::Pose ());

          world_to_base_joint_fresh_ -> Init ();
          world_to_base_joint_fresh_ -> SetHighStop (0, 0);
          world_to_base_joint_fresh_ -> SetLowStop (0, 0);
        }
        else
        {
          // Need this, else hand quickly spins and flies away upwards, right
          //   after teleporting!
          math::Vector3 lin = math::Vector3 (0, 0, 0);
          math::Vector3 ang = math::Vector3 (0, 0, 0);
          reflex_model -> SetWorldTwist (lin, ang);
       
          // Need this, else hand slips away horizontally right after
          //   teleporting
          reflex_model -> SetStatic (false);
       
          // Does this help with seg fault? > Nope. Still get it.
          //world_to_base_joint_fresh_ -> RemoveChildren ();

          // This line is the real problem, now that I've eliminated the
          //   Attach() call. Also, Detach() is always the line after which seg
          //   fault happens! So there is no more doubt!!! It HAS to be
          //   Detach() bug!!!
          // I need to find a way to call some other function to destroy the
          //   joint, instead of Detach(). Since I always Load() the joint
          //   fresh above, I should be able to just destroy it entirely!
          world_to_base_joint_fresh_ -> Detach ();

          // Tried Fini(). Doesn't detach joint. Out of ideas... now going
          //   back to add a Load() function to the AttachStaticModel() way,
          //   see if that fixes the hand not being fixed and fingers slamming
          //   into hand, sending hand spinning away.
        }

        // Sanity checks
        print_world_child ();
        print_joint_parent_child (world_to_base_joint_fresh_);
        print_base_parent ();
      }


      res.success = true;
      return true;
    }

    bool rm_hand_cb (reflex_gazebo_msgs::RemoveModel::Request & req,
      reflex_gazebo_msgs::RemoveModel::Response & res)
    {
      ROS_ERROR ("Removing hand and contact sensors from world");

      // Just implement removing ContactSensors here, if works, then add
      //   srv call to /gazebo/delete_model to remove hand. Currently the
      //   /gazebo/delete_model srv call is done in sample_gazebo.py already,
      //   easier in Python.


      // This unloads contact sensors correctly!!!! But Gazebo freezes
      //   afterwards, I don't even see where the newly spawned hand in Gazebo,
      //   though it says it's spawned. Then Gazebo just freezes.
      //
      // Function doesn't exist in ROS Indigo, does in Jade (Gazebo 5)
      // Tips from poster's code http://answers.gazebosim.org/question/6369/how-to-delete-a-model-entirely-including-contact-filter/
      // ContactManager API https://osrf-distributions.s3.amazonaws.com/gazebo/api/dev/classgazebo_1_1physics_1_1ContactManager.html
      // This function doesn't exist in ROS Indigo either!!!
      //   Merged here to default branch 2014-02-03, no idea what version of
      //   Gazebo that is.
      //   https://bitbucket.org/osrf/gazebo/pull-requests/934/add-function-to-remove-filter-in/diff
      // I think I just have to install Jade. No other way. That is probably
      //   the quickest anyway. Okay reinstalled Jade, was quick, half an hour
      //   or less.

      // See the version in /opt/ros/jade/include/ros/common.h
      #if ROS_VERSION_MINIMUM(1, 11, 16)  // Jade and newer
        physics::ContactManager * cm = physics_ -> GetContactManager ();
        if (cm)
        {
          ROS_ERROR ("Removing finger 1 contact filters from world");
          cm -> RemoveFilter ("finger_1_sensor_1");
          cm -> RemoveFilter ("finger_1_sensor_2");
          cm -> RemoveFilter ("finger_1_sensor_3");
          cm -> RemoveFilter ("finger_1_sensor_4");
          cm -> RemoveFilter ("finger_1_sensor_5");
          cm -> RemoveFilter ("finger_1_sensor_6");
          cm -> RemoveFilter ("finger_1_sensor_7");
          cm -> RemoveFilter ("finger_1_sensor_8");
          cm -> RemoveFilter ("finger_1_sensor_9");
 
          ROS_ERROR ("Removing finger 2 contact filters from world");
          cm -> RemoveFilter ("finger_2_sensor_1");
          cm -> RemoveFilter ("finger_2_sensor_2");
          cm -> RemoveFilter ("finger_2_sensor_3");
          cm -> RemoveFilter ("finger_2_sensor_4");
          cm -> RemoveFilter ("finger_2_sensor_5");
          cm -> RemoveFilter ("finger_2_sensor_6");
          cm -> RemoveFilter ("finger_2_sensor_7");
          cm -> RemoveFilter ("finger_2_sensor_8");
          cm -> RemoveFilter ("finger_2_sensor_9");
 
          ROS_ERROR ("Removing finger 3 contact filters from world");
          cm -> RemoveFilter ("finger_3_sensor_1");
          cm -> RemoveFilter ("finger_3_sensor_2");
          cm -> RemoveFilter ("finger_3_sensor_3");
          cm -> RemoveFilter ("finger_3_sensor_4");
          cm -> RemoveFilter ("finger_3_sensor_5");
          cm -> RemoveFilter ("finger_3_sensor_6");
          cm -> RemoveFilter ("finger_3_sensor_7");
          cm -> RemoveFilter ("finger_3_sensor_8");
          cm -> RemoveFilter ("finger_3_sensor_9");
        }

      #else  // Older than Jade

        ROS_ERROR ("This rosservice is not supported for ROS versions lower than Jade (Gazebo 5). Indigo and lower cannot remove contact sensors properly when hand model is removed. All fixes are for newer versions than Gazebo in Indigo - RemoveModel(), RemoveFilter(), etc.");

      #endif

      // This gives this error:
      //   ***** Internal Program Error - assertion (this->inertial != __null) failed in virtual void gazebo::physics::ODELink::OnPoseChange():
      //   /tmp/buildd/gazebo-5.0.1+dfsg/gazebo/physics/ode/ODELink.cc(271): Inertial pointer is NULL
      //
      // Function doesn't exist in ROS Indigo, does in Jade (Gazebo 5)
      // Ref: http://answers.ros.org/question/9562/how-do-i-test-the-ros-version-in-c-code/
      /*
      // See the version in /opt/ros/jade/include/ros/common.h
      #if ROS_VERSION_MINIMUM(1, 11, 16)  // Jade and newer

        // Function only exists in Gazebo 3.1+
        // This post says there's problem deleting contact sensors, when deleting
        //   models!!! That's the problem I'm having, and why I tried to delete
        //   contact sensors manually below!
        //   http://answers.gazebosim.org/question/6369/how-to-delete-a-model-entirely-including-contact-filter/
        //   https://bitbucket.org/osrf/gazebo/pull-requests/1106/added-world-removemodel-to-fix-issue-1177/diff
        // This RemoveModel() doesn't seem to be in Gazebo 2 with ROS Indigo...
        //   probably need Jade with Gazebo 5... Oh, the pull request says it's
        //   merged into gazebo_3.1 branch. So yeah I won't have it... - -
        //   Why is ROS's gazebo so OLD!!!
        physics::ModelPtr reflex_model;
        find_model (req.model_name, reflex_model);
        if (reflex_model)
        {
          world_ -> RemoveModel (reflex_model);
        }
        //world_ -> RemoveModel (req.model_name);

      #else  // Older than Jade
        ROS_ERROR ("This rosservice is not supported for ROS versions lower than Jade (Gazebo 5). Indigo and lower cannot remove contact sensors properly when hand model is removed. All fixes are for newer versions than Gazebo in Indigo - RemoveModel(), RemoveFilter(), etc.");
      #endif
      */

      // Entity names are from ../urdf/full_reflex_model.gazebo <plugin name>
      //   tags. 9 for each finger, 3 fingers.
      /*
      // Maybe not remove plugin. Maybe need to remove the sensor, since the
      //   Gazebo error is issued by "ContactManager.cc".
      // These get this error:
      //   Error [SensorManager.cc:354] Unable to remove sensor[default::reflex::Distal_1/sensor_1::finger_1_sensor_6] because it does not exist.
      ROS_ERROR ("Removing finger 1 sensors from world");
      sensors::remove_sensor ("finger_1_sensor_1");
      sensors::remove_sensor ("finger_1_sensor_2");
      sensors::remove_sensor ("finger_1_sensor_3");
      sensors::remove_sensor ("finger_1_sensor_4");
      sensors::remove_sensor ("finger_1_sensor_5");
      sensors::remove_sensor ("finger_1_sensor_6");
      sensors::remove_sensor ("finger_1_sensor_7");
      sensors::remove_sensor ("finger_1_sensor_8");
      sensors::remove_sensor ("finger_1_sensor_9");

      ROS_ERROR ("Removing finger 2 sensors from world");
      sensors::remove_sensor ("finger_2_sensor_1");
      sensors::remove_sensor ("finger_2_sensor_2");
      sensors::remove_sensor ("finger_2_sensor_3");
      sensors::remove_sensor ("finger_2_sensor_4");
      sensors::remove_sensor ("finger_2_sensor_5");
      sensors::remove_sensor ("finger_2_sensor_6");
      sensors::remove_sensor ("finger_2_sensor_7");
      sensors::remove_sensor ("finger_2_sensor_8");
      sensors::remove_sensor ("finger_2_sensor_9");

      ROS_ERROR ("Removing finger 3 sensors from world");
      sensors::remove_sensor ("finger_3_sensor_1");
      sensors::remove_sensor ("finger_3_sensor_2");
      sensors::remove_sensor ("finger_3_sensor_3");
      sensors::remove_sensor ("finger_3_sensor_4");
      sensors::remove_sensor ("finger_3_sensor_5");
      sensors::remove_sensor ("finger_3_sensor_6");
      sensors::remove_sensor ("finger_3_sensor_7");
      sensors::remove_sensor ("finger_3_sensor_8");
      sensors::remove_sensor ("finger_3_sensor_9");
      */

      /*
      ROS_ERROR ("Removing finger 1 plugins from world");
      world_ -> RemovePlugin ("f1s1_plugin");
      world_ -> RemovePlugin ("f1s2_plugin");
      world_ -> RemovePlugin ("f1s3_plugin");
      world_ -> RemovePlugin ("f1s4_plugin");
      world_ -> RemovePlugin ("f1s5_plugin");
      world_ -> RemovePlugin ("f1s6_plugin");
      world_ -> RemovePlugin ("f1s7_plugin");
      world_ -> RemovePlugin ("f1s8_plugin");
      world_ -> RemovePlugin ("f1s9_plugin");

      ROS_ERROR ("Removing finger 2 plugins from world");
      world_ -> RemovePlugin ("f2s1_plugin");
      world_ -> RemovePlugin ("f2s2_plugin");
      world_ -> RemovePlugin ("f2s3_plugin");
      world_ -> RemovePlugin ("f2s4_plugin");
      world_ -> RemovePlugin ("f2s5_plugin");
      world_ -> RemovePlugin ("f2s6_plugin");
      world_ -> RemovePlugin ("f2s7_plugin");
      world_ -> RemovePlugin ("f2s8_plugin");
      world_ -> RemovePlugin ("f2s9_plugin");

      ROS_ERROR ("Removing finger 3 plugins from world");
      world_ -> RemovePlugin ("f3s1_plugin");
      world_ -> RemovePlugin ("f3s2_plugin");
      world_ -> RemovePlugin ("f3s3_plugin");
      world_ -> RemovePlugin ("f3s4_plugin");
      world_ -> RemovePlugin ("f3s5_plugin");
      world_ -> RemovePlugin ("f3s6_plugin");
      world_ -> RemovePlugin ("f3s7_plugin");
      world_ -> RemovePlugin ("f3s8_plugin");
      world_ -> RemovePlugin ("f3s9_plugin");
      */

      /*
      physics::BasePtr snsr_genericType = world_ -> GetByName ("");
      if (snsr_genericType)
      {
        snsr = boost::dynamic_pointer_cast <sensors::ContactSensor> (
          snsr_genericType);

      }
      else
      {
        ROS_ERROR ("Sensor not found");
      }
      */

      // Safe guard for when hand is deleted, but Gazebo did not remove model
      //   cleanly, then find_model() will still find it, but setting pose and
      //   twist on it in OnUpdate() would give inertial pointer is NULL error.
      // Once hand is removed, set to false, so OnUpdate() stop trying to set
      //   hand's pose.
      hand_exists_ = false;

      return true;
    }

    bool rm_model_cb (reflex_gazebo_msgs::RemoveModel::Request & req,
      reflex_gazebo_msgs::RemoveModel::Response & res)
    {
      ROS_ERROR ("Removing model %s from world", req.model_name.c_str ());

      /*
      physics::ModelPtr model;
      find_model (req.model_name, model);
      if (model)
        world_ -> RemoveModel (model);
      */

      // This may need newer Gazebo than that in ROS Indigo. Gazebo 5 works
      // TEMPORARY commented out on Baxter computer.
      //world_ -> RemoveModel (req.model_name);
      ROS_ERROR ("RemoveModle() temporarilly commented out to run on MacBook Air. Uncomment and recompile reflex_gazebo when get back on MacBook Pro with gazebo 5!!!");

      res.success = true;
      return true;
    }

    /*
    // Currently not in use. This way of moving hand doesn't work.
    //
    // To test, publish a whatever pose:
    //   $ rostopic pub /reflex_gazebo/move_hand geometry_msgs/PoseStamped -1 "{header: {frame_id: '', stamp: 0, seq: 0}, pose: {position: {x: 0, y: 0, z: 0}, orientation: {x: 0, y: 0, z: 0, w: 0}}}"
    void move_hand_cb (const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
      // header.seq is uint32, so must convert to signed for normal operations
      // TODO: Why is this always 1???? Doesn't make any sense!! How to specify
      //   header.seq on command line line, maybe I'm doing it wrong (seq: 0)?
      int seq = (int) (msg -> header.seq);

      // Must use stderr, otherwise never gets printed!!
      fprintf (stderr, "move_hand_cb() got message seq %d\n", seq);

      // TODO seq is always 1, can't use this until I fix it
      // If we've seen this request, ignore it
      //if (seq <= seq_seen_)
      //{
      //  fprintf (stderr, "move_hand_cb() ignoring repeated message seq %d, seen seq id up to %d\n",
      //    seq, seq_seen_);
      //  return;
      //}

      physics::BasePtr wrist_link_genericType = world_ -> GetByName (
        "base_link");
      physics::LinkPtr wrist_link =
        boost::dynamic_pointer_cast <physics::Link> (wrist_link_genericType);

      fprintf (stderr, "Found link %s\n", wrist_link -> GetName ().c_str ());

      for (int i = 0; i < wrist_link -> GetParentJoints ().size (); i ++)
        fprintf (stderr, "  Parent joint %d: %s\n", i,
          wrist_link -> GetParentJoints ().at (i) -> GetName ().c_str ());

      for (int i = 0; i < wrist_link -> GetChildJoints ().size (); i ++)
        fprintf (stderr, "  Child joint %d: %s\n", i,
          wrist_link -> GetChildJoints ().at (i) -> GetName ().c_str ());


      wrist_link -> SetStatic (true);
      //wrist_link -> SetGravityMode (false);
      // Disable in physics engine. Wait... this means collisions and contact
      //   sensors won't work too, right?
      //wrist_link -> SetEnabled (false);

      // This doesn't work. Makes the hand fly into space! Proper way is to
      //   use rosservice call set_model_state, after setting gravity to 0
      //   using world_->GetPhysicsEngine(). Then all parts of hand move
      //   together, and hand stays in place without gravity.
      // TODO: must convert quaternion to rpy. tf, bullet, or Eigen should have
      //   such conversions. Just use 0s for now.
      math::Pose pose = math::Pose (
        msg -> pose.position.x, msg -> pose.position.y, msg -> pose.position.z,
        0.0, 0.0, 0.0);
      wrist_link -> SetWorldPose (pose);


      // Assumes header.frame_id indicates the link that needs to be moved.
      //   e.g. if frame_id = "/base_link", we move the joint connecting
      //   frame_id to its parent. We let caller specify the link to move,
      //   instead of the parent link wrt which to move - this way, we can move
      //   /base_link even after hand is attached to a robot. Then instead of
      //   having to change parent link from "/" to "/left_gripper", we don't
      //   need to change anything. Just use GetParent() to look up parent all
      //   the same.
      ////msg -> pose.orientation.x
      //msg -> pose.orientation.y
      //msg -> pose.orientation.z
      //msg -> pose.orientation.w

      seq_seen_ = msg -> header.seq;
    }
    */

    // To test:
    //   $ rosservice call /reflex_gazebo/get_model_size '{model_name: object}'
    bool model_size_cb (reflex_gazebo_msgs::GetModelSize::Request & req,
      reflex_gazebo_msgs::GetModelSize::Response & res)
    {
      fprintf (stderr, "Got get_model_size rosservice call\n");

      // physics::Model https://osrf-distributions.s3.amazonaws.com/gazebo/api/dev/classgazebo_1_1physics_1_1Model.html
      physics::ModelPtr model;
      find_model (req.model_name, model);

      if (! model)
      {
        fprintf (stderr, "Model %s not found.\n", req.model_name.c_str ());
        res.success = false;
        return false;
      }

      fprintf (stderr, "Found model %s. Has %u children\n",
        model -> GetName().c_str (), model -> GetChildCount ());

      // Sanity check in case of null pointer error
      if (model -> GetChildCount () > 0)
      {
        // This function hangs at the first call when Gazebo starts! So caller
        //   of rosservice will need to make the call with a timeout. For
        //   Python, use the python signal package like here:
        //   http://stackoverflow.com/questions/492519/timeout-on-a-python-function-call
        // math::Box https://osrf-distributions.s3.amazonaws.com/gazebo/api/dev/classgazebo_1_1math_1_1Box.html
        math::Box bbox = model -> GetBoundingBox ();
       
        math::Vector3 size = bbox.GetSize ();
        res.model_size.x = size.x;// [0];
        res.model_size.y = size.y;// [1];
        res.model_size.z = size.z;// [2];
       
        // Center of box wrt world frame of Gazebo, probably
        math::Vector3 center = bbox.GetCenter ();
        res.model_center.x = center.x;// [0];
        res.model_center.y = center.y;// [1];
        res.model_center.z = center.z;// [2];

        res.success = true;
      }
      // Object is empty. This can happen when an object has been deleted.
      else
      {
        fprintf (stderr, "Model has 0 children. This is abnormal. Returning without doing anything.");
        res.success = false;
      }

      fprintf (stderr, "get_model_size returning\n");
      return true;
    }

    bool model_pose_cb (reflex_gazebo_msgs::SetModelPose::Request & req,
      reflex_gazebo_msgs::SetModelPose::Response & res)
    {
      fprintf (stderr, "Got set_model_pose rosservice call\n");

      // Safe guard for when hand is deleted, but Gazebo did not remove model
      //   cleanly, then find_model() will still find it, but setting pose and
      //   twist on it in OnUpdate() would give inertial pointer is NULL error.
      hand_exists_ = true;

      // physics::Model https://osrf-distributions.s3.amazonaws.com/gazebo/api/dev/classgazebo_1_1physics_1_1Model.html
      physics::ModelPtr model;
      find_model (req.model_name, model);

      if (! model)
      {
        fprintf (stderr, "Model %s not found.\n", req.model_name.c_str ());
        res.success = false;
        return false;
      }

      fprintf (stderr, "Found model %s. Has %u children\n",
        model -> GetName().c_str (), model -> GetChildCount ());


      // Sanity check in case of null pointer error
      if (model -> GetChildCount () > 0)
      {
        // Convert ROS geometry_msgs/Pose to Gazebo gazebo::math::Pose
       
        math::Vector3 pos = math::Vector3 (req.pose.position.x,
          req.pose.position.y, req.pose.position.z);
       
        // w first!!!
        // API https://osrf-distributions.s3.amazonaws.com/gazebo/api/dev/classgazebo_1_1math_1_1Quaternion.html
        math::Quaternion quat = math::Quaternion (req.pose.orientation.w,
          req.pose.orientation.x, req.pose.orientation.y,
          req.pose.orientation.z);
       
        //math::Pose pose = math::Pose (pos, quat);
        hand_pose_ = math::Pose (pos, quat);
       
        // Set pose
        if (req.relative)
        {
          model -> SetRelativePose (hand_pose_);
        }
        else
        {
          model -> SetWorldPose (hand_pose_);
        }
       
        res.success = true;
      }
      // Object is empty. This can happen when an object has been deleted.
      else
      {
        fprintf (stderr, "Model has 0 children. This is abnormal. Returning without doing anything.");
        res.success = false;
      }

      fprintf (stderr, "set_model_pose returning\n");
      return true;
    }

    // Not being used. Originally thought objects disappearing after removing
    //   is visibility problem, but really it's another bug in Gazebo. They
    //   don't remove models cleanly. If change name of model, then reload and
    //   it'd be visible. So now just using random number generator for random
    //   name of model.
    // This service call does works on objects that have not been removed with
    //   the same name beforehand though!
    bool model_visible_cb (reflex_gazebo_msgs::SetBool::Request & req,
      reflex_gazebo_msgs::SetBool::Response & res)
    {
      fprintf (stderr, "Got set_model_visible rosservice call\n");

      /*
      // Debug info
      // physics::Model https://osrf-distributions.s3.amazonaws.com/gazebo/api/dev/classgazebo_1_1physics_1_1Model.html
      physics::ModelPtr model;
      find_model ("object", model);
      if (! model)
      {
        fprintf (stderr, "Model object not found.\n");
      }
      else if (model -> GetParent ())
      {
        fprintf (stderr, "Found model %s. Parent: %s\n",
          model -> GetName().c_str (), model -> GetParent () -> GetName ().c_str ());
      }

      physics::LinkPtr link;
      find_link ("object::object_link", link);
      if (! link)
        fprintf (stderr, "Link object::object_link not found.\n");
      else
      {
        if (link -> GetParent ())
        {
          fprintf (stderr, "Found link %s. Parent: %s\n",
            link -> GetName().c_str (), link -> GetParent () -> GetName ().c_str ());
        }

        msgs::Visual orig_vis = link -> GetVisualMessage (
          "object::object_link::object_visual");
        fprintf (stderr, "Original object_visual is %s\n", (bool) (orig_vis.visible ()) ? "visible" : "invisible");
      }
      */


      // Ref: http://answers.gazebosim.org/question/7637/toggle-visibility-of-a-link/
      //   This doesn't seem to do anything. Object still transparency = 0.0 in
      //   Gazebo GUI.
      //   Doesn't look like this guy ever got it to work though...
      // This is a better one, with an actual answer:
      //   http://answers.gazebosim.org/question/1377/visuals-material-update-from-modelplugin-in-ros-gazebo/
      //   Code in ModelPlugin:
      //   https://bitbucket.org/osrf/gazebo/src/ff38b1f7b7a8d8decbdadc33a84a83eb7cc44b47/examples/plugins/model_visuals/model_visuals.cc?at=default&fileviewer=file-view-default
      //   Difference from above is, this node->Init() uses world name, and
      //     set_name() uses a <visual>'s name, not a <link>'s! Then it works.
      
      transport::NodePtr node(new transport::Node());
      node->Init(world_ -> GetName ());
  
      // gztopic echo /gazebo/default/visual shows this 
      transport::PublisherPtr visPub =
        node->Advertise<msgs::Visual>("~/visual");
      visPub->WaitForConnection();
  
      // Entity names are defined in the sdf generated by triangle_sampling
      //   sample_gazebo.py generate_sdf().
      msgs::Visual msg; 
      // Set the visual's name. This should be unique
      msg.set_name ("object::object_link::object_visual");
      // Set the visual's parent. This visual will be attached to the parent
      msg.set_parent_name ("object::object_link");

      msg.set_visible (req.data);
      // Visible
      if (req.data)
        msg.set_transparency (0.0);
      // Invisible
      else
        msg.set_transparency (1.0);

      visPub -> Publish (msg);

      res.success = true;
      return true;
    }

    // sdf:: API (hard to find from API main page):
    //   http://osrf-distributions.s3.amazonaws.com/gazebo/api/1.9.1/namespacesdf.html
    // sdf::Element API: http://osrf-distributions.s3.amazonaws.com/gazebo/api/1.9.1/classsdf_1_1Element.html
    void Load (physics::WorldPtr _world, sdf::ElementPtr _sdf)
    {
      // Specify this in <plugin><model_name> tag in .world file.
      // To find what value it's supposed to be, look in Gazebo pane, expand
      //   the "Models" list. The one that's your robot is the name to pass in.
      if (! _sdf -> HasElement ("model_name"))
      {
        reflex_model_name_ = "reflex";
        // if parameter tag does NOT exist
        fprintf (stderr, "Missing <model_name> tag in <plugin> in .world file, defaulting to %s\n",
          reflex_model_name_.c_str ());
      }
      // if parameter tag exists, get its value
      else
      {
        // Ref: http://answers.gazebosim.org/question/2764/passing-parameters-to-included-sdf/
        // Deprecated.
        //reflex_model_name_ = _sdf -> GetElement ("model_name") -> GetValueString ();

        // All deprecated GetX() have been replaced with Get<template type>
        // Ref: http://answers.gazebosim.org/question/5070/gazebo-ros-plugin-getvaluedouble-deprecated/
        reflex_model_name_ = _sdf -> Get <std::string> ("model_name");
      }


      // Make sure the ROS node for Gazebo has already been initialized
      if (!ros::isInitialized())
      {
        ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load plugin. "
          << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
        return;
      }

      world_ = _world;


      // Turn off gravity
      // This way doesn't make the hand model's twist params nan. If you set
      //   gravity to 0 using command line rosservice call set_physics_properties, then it makes hand model's twist params nan, and you will get very weird bugs - like fingers move but palm doesn't follow at first subsequent set_model_state, and relative movements or no movements at all at second and on! Undefined behavior basically!
      // So must turn off here.
      math::Vector3 gravity = math::Vector3 (0.0, 0.0, 0.0);
      physics_ = world_ -> GetPhysicsEngine ();
      if (physics_)
        physics_ -> SetGravity (gravity);


      // Advertise rostopic, for nodes to call this plugin to do stuff
      // Tutorial: http://wiki.ros.org/roscpp_tutorials/Tutorials/UsingClassMethodsAsCallbacks
      //move_sub_ = nh_.subscribe ("/reflex_gazebo/move_hand", 5,
      //  &gazebo::HandWorldPlugin::move_hand_cb, this);
      //seq_seen_ = -1;

      //static_sub_ = nh_.subscribe ("/reflex_gazebo/set_static_hand", 5,
      //  &gazebo::HandWorldPlugin::static_hand_cb, this);

      //fixed_sub_ = nh_.subscribe ("/reflex_gazebo/set_fixed_hand", 5,
      fixed_hand_srv_ = nh_.advertiseService ("/reflex_gazebo/set_fixed_hand",
        &gazebo::HandWorldPlugin::fixed_hand_cb, this);

      rm_hand_srv_ = nh_.advertiseService ("/reflex_gazebo/remove_hand",
        &gazebo::HandWorldPlugin::rm_hand_cb, this);

      model_size_srv_ = nh_.advertiseService ("/reflex_gazebo/get_model_size",
        &gazebo::HandWorldPlugin::model_size_cb, this);

      model_pose_srv_ = nh_.advertiseService ("/reflex_gazebo/set_model_pose",
        &gazebo::HandWorldPlugin::model_pose_cb, this);

      model_visible_srv_ = nh_.advertiseService ("/reflex_gazebo/set_model_visible",
        &gazebo::HandWorldPlugin::model_visible_cb, this);

      rm_model_srv_ = nh_.advertiseService ("/reflex_gazebo/remove_model",
        &gazebo::HandWorldPlugin::rm_model_cb, this);


      // Listen to the update event. This event is broadcast every
      //   simulation iteration.
      // From ModelPlugin tutorial 
      //   http://gazebosim.org/tutorials/?tut=plugins_model
      this -> updateConnection_ = event::Events::ConnectWorldUpdateBegin(
          boost::bind (&HandWorldPlugin::OnUpdate, this, _1));


      found_wrist_ = false;

      times_attached_ = 0;
      times_detached_ = 0;

      // Convenience flags for debugging. Only ONE should be set to true!!!
      TestAttachDetach_ = false;
      TestAttachStatic_ = true;
      TestInitJoint_ = false;

      hand_exists_ = false;
      hand_pose_ = math::Pose ();

      printf ("World plugin initialized\n");
    }


  };

GZ_REGISTER_WORLD_PLUGIN (HandWorldPlugin)
}

