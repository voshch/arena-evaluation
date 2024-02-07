cmake_minimum_required(VERSION 3.0.2)
project(arena_evaluation_msgs)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
)

add_service_files(FILES
  ChangeDirectory.srv
)
generate_messages()

catkin_package(
  CATKIN_DEPENDS message_runtime
)
include_directories(${catkin_INCLUDE_DIRS})
