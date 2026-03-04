cmake_minimum_required(VERSION 3.8)
project(my_demo_gui)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)

find_package(rviz_common REQUIRED)
find_package(rviz_rendering REQUIRED)
find_package(rviz_default_plugins REQUIRED)

find_package(Qt5 REQUIRED COMPONENTS Widgets)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

add_executable(my_demo_gui_node
  include/my_demo_gui/main_window.hpp
  include/my_demo_gui/camera_view.hpp
  include/my_demo_gui/rviz_sim_view.hpp
  src/main.cpp
  src/main_window.cpp
  src/camera_view.cpp
  src/rviz_sim_view.cpp
  src/rs2_concept.ui
)
target_include_directories(my_demo_gui_node PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(my_demo_gui_node
  ament_index_cpp
  rclcpp
  rviz_common
  rviz_rendering
  rviz_default_plugins
)

target_link_libraries(my_demo_gui_node
  Qt5::Widgets
  ${OpenCV_LIBRARIES}
)

install(TARGETS my_demo_gui_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
