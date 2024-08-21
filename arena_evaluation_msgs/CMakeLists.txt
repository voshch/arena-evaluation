cmake_minimum_required(VERSION 3.5)
project(arena_evaluation_msgs)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Define the service files (message & service generation)
# Automatically takes care of generating the necessary include directories for the messages and services
# These directories are added to the include path as needed.
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/ChangeDirectory.srv"
)

ament_package()

