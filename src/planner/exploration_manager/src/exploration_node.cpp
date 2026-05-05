#include <ros/ros.h>
#include <exploration_manager/exploration_fsm.h>

#ifdef APEXNAV_ENABLE_TRAJ_FSM
#include <exploration_manager/exploration_fsm_traj.h>
#endif

#include <exploration_manager/backward.hpp>
namespace backward {
backward::SignalHandling sh;
}

using namespace apexnav_planner;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "apexnav_node");
  ros::NodeHandle nh("~");

  // Check if real-world mode
  bool is_real_world = false;
  nh.param("is_real_world", is_real_world, false);

  if (is_real_world) {
#ifdef APEXNAV_ENABLE_TRAJ_FSM
    ROS_INFO("========================================");
    ROS_INFO("  Starting in REAL WORLD mode");
    ROS_INFO("========================================");
    ExplorationFSMReal expl_fsm;
    expl_fsm.init(nh);
    ros::Duration(1.0).sleep();
    ros::spin();
#else
    ROS_ERROR("Real-world trajectory mode was not built. Rebuild with "
              "-DAPEXNAV_ENABLE_TRAJ_FSM=ON to enable it.");
    return 1;
#endif
  }
  else {
    ROS_INFO("========================================");
    ROS_INFO("  Starting in SIMULATION mode");
    ROS_INFO("========================================");
    ExplorationFSM expl_fsm;
    expl_fsm.init(nh);
    ros::Duration(1.0).sleep();
    ros::spin();
  }

  return 0;
}
