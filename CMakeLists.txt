cmake_minimum_required(VERSION 2.8.3)
project(raveutils)

find_package(catkin REQUIRED)
find_package(OpenRAVE REQUIRED)

catkin_python_setup()

catkin_package(
  DEPENDS OpenRAVE
)

# Tests
catkin_add_nosetests(tests/test_conversions.py)
catkin_add_nosetests(tests/test_kinematics.py)
catkin_add_nosetests(tests/test_planning.py)
catkin_add_nosetests(tests/test_transforms.py)
# test_visual.py is skipped because it will make travis fail
