cmake_minimum_required(VERSION 3.16)
project(arena_evaluation LANGUAGES CXX)

# --- ctypes solver libraries -------------------------------------------------
# `arena_evaluation/processing/acoustics/solver.cpp` and
# `arena_evaluation/processing/path/solver.cpp` are CPython-free C++17 sources
# that Python loads through ctypes (impedance_grid.py, theta_star.py) from a
# `solver.so` next to the importing module. This file is the only place their
# compile flags live:
#
#   * `arena build` runs setup.py's build_ext, which configures this file with
#     -DARENA_SOLVERS_ONLY=ON and builds both targets into the package tree;
#   * a direct `cmake -S arena_evaluation -B <dir> -DARENA_SOLVERS_ONLY=ON`
#     followed by `cmake --build <dir>` builds the same two libraries.
#
# ARENA_SOLVER_OUTPUT_DIR is the Python package root the libraries are written
# below, `processing/acoustics/solver.so` and `processing/path/solver.so`.
set(ARENA_SOLVER_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/arena_evaluation"
    CACHE PATH "Python package root the ctypes solver libraries are written below")

option(ARENA_SOLVERS_ONLY "Configure only the ctypes solver libraries, without ROS dependencies" OFF)

# The solvers are plain C++17: no Python headers, no ament, no rclcpp.
function(arena_add_solver target source relative_output_dir)
  add_library(${target} SHARED "${source}")
  set_target_properties(${target} PROPERTIES
    PREFIX ""
    OUTPUT_NAME solver
    LIBRARY_OUTPUT_DIRECTORY "${ARENA_SOLVER_OUTPUT_DIR}/${relative_output_dir}"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON)
  target_compile_options(${target} PRIVATE -O3 -fPIC -Wall -Wextra -Wpedantic)
endfunction()

arena_add_solver(arena_acoustics_solver
  arena_evaluation/processing/acoustics/solver.cpp
  processing/acoustics)
arena_add_solver(arena_theta_star_solver
  arena_evaluation/processing/path/solver.cpp
  processing/path)

# Solver-only configure: everything below needs a ROS 2 ament workspace.
if(ARENA_SOLVERS_ONLY)
  return()
endif()

# --- ROS package -------------------------------------------------------------
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
find_package(arena_evaluation_msgs REQUIRED)

# Python setup. Sets PYTHON_INSTALL_DIR, which the solver install rules need.
ament_python_install_package(${PROJECT_NAME})

# Both solver libraries sit next to the module that loads them, in the source
# tree and in the installed package.
install(TARGETS arena_acoustics_solver
  LIBRARY DESTINATION "${PYTHON_INSTALL_DIR}/arena_evaluation/processing/acoustics")
install(TARGETS arena_theta_star_solver
  LIBRARY DESTINATION "${PYTHON_INSTALL_DIR}/arena_evaluation/processing/path")

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
  arena_evaluation_msgs
  hunav_msgs
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
