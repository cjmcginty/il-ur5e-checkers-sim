cmake_minimum_required(VERSION 3.8)
project(checkers_gz_plugins)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ROS dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# Gazebo dependencies
find_package(gz-cmake3 REQUIRED)
find_package(gz-plugin2 REQUIRED)
find_package(gz-sim8 REQUIRED)

add_library(checkers_piece_state_plugin SHARED
  src/checkers_piece_state_plugin.cpp
)

target_compile_features(checkers_piece_state_plugin PUBLIC cxx_std_17)

target_include_directories(checkers_piece_state_plugin PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(checkers_piece_state_plugin
  rclcpp
  std_msgs
)

target_link_libraries(checkers_piece_state_plugin
  gz-plugin2::gz-plugin2
  gz-sim8::gz-sim8
)

install(
  TARGETS checkers_piece_state_plugin
  LIBRARY DESTINATION lib
)

install(
  DIRECTORY include/
  DESTINATION include
)

ament_package()