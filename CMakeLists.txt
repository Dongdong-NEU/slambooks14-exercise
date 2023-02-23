cmake_minimum_required(VERSION 2.8.3)

project(imageProcess)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(undistortImage undistort_image.cpp ) 

target_link_libraries(undistortImage ${OpenCV_LIBS})