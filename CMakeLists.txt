cmake_minimum_required(VERSION 2.8.3)
project(raveutils)

find_package(catkin REQUIRED COMPONENTS
  openrave_catkin
  rospy
)

find_package(OpenRAVE REQUIRED)

catkin_python_setup()

catkin_package(
  DEPENDS OpenRAVE
)

# Tests
if (CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(tests)
endif()

install(DIRECTORY data DESTINATION ${OpenRAVE_DEVEL_DIR})
install(DIRECTORY data DESTINATION ${OpenRAVE_INSTALL_DIR})
