cmake_minimum_required(VERSION 3.0.2)
project(arena_evaluation)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

catkin_python_setup()

catkin_package()
include_directories(${catkin_INCLUDE_DIRS})

catkin_install_python(PROGRAMS
  scripts/data_recorder_node
  scripts/get_metrics
  scripts/create_plots
  scripts/process_data

  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
