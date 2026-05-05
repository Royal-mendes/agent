cmake_minimum_required(VERSION 2.8.3)
project(exploration_manager)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O0 -g0 -Wall")
option(APEXNAV_ENABLE_TRAJ_FSM "Build real-world continuous trajectory FSM" OFF)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  plan_env
  path_searching
  vis_utils
  cv_bridge
  lkh_mtsp_solver
  trajectory_manager
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES exploration_manager
  CATKIN_DEPENDS plan_env path_searching vis_utils lkh_mtsp_solver
  #  DEPENDS system_lib
)

include_directories(
  include
  SYSTEM
  ${catkin_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

set(EXPLORATION_NODE_SOURCES
  src/exploration_node.cpp
  src/exploration_fsm.cpp
  src/exploration_manager.cpp
  src/reflective_agent_bridge.cpp
)

if(APEXNAV_ENABLE_TRAJ_FSM)
  add_definitions(-DAPEXNAV_ENABLE_TRAJ_FSM)
  list(APPEND EXPLORATION_NODE_SOURCES src/exploration_fsm_traj.cpp)
endif()

add_executable(exploration_node
  ${EXPLORATION_NODE_SOURCES}
)
target_link_libraries(exploration_node
  ${catkin_LIBRARIES}
)
