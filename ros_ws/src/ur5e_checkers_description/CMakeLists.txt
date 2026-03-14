cmake_minimum_required(VERSION 3.8)
project(ur5e_checkers_description)

find_package(ament_cmake REQUIRED)

# Install non-code resources so FindPackageShare() works
install(DIRECTORY
  launch
  config
  urdf
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/robot_description_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
