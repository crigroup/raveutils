cmake_minimum_required(VERSION 2.8.3)
project(raveutils)

find_package(catkin REQUIRED)
find_package(OpenRAVE REQUIRED)

catkin_python_setup()

catkin_package(
  DEPENDS OpenRAVE
)
