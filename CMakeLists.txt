cmake_minimum_required(VERSION 2.8.3)

project(eigenMatrix)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# 寻找Eigen库
find_package(Eigen3 REQUIRED)
# 将Eigen库include进来
include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(eigenMatrix main.cpp ) #src/image_process.cpp