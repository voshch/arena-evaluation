cmake_minimum_required(VERSION 3.5)
project(arena_evaluation)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Check Compiler
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Python setup
ament_python_install_package(${PROJECT_NAME})

# Include directories
include_directories(include)

# Set dependencies 
set(dependencies
  ament_cmake
  rclcpp
  rclpy
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
)

# Install Python scripts
install(PROGRAMS
  scripts/data_recorder_node
  scripts/get_metrics
  scripts/create_plots
  scripts/process_data
  DESTINATION lib/${PROJECT_NAME}
)

# Export custom include directory (headers) for downstream packages 
ament_export_include_directories(include)

# Export dependencies of the package for downstream packages (inherit these dependencies)
# Handle both include directories and the libraries required for linking.
ament_export_dependencies(${dependencies})

# Package configuration
ament_package()



