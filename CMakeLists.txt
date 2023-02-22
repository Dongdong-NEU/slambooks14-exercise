cmake_minimum_required(VERSION 2.8.3)

project(drawTrajectory)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# 寻找Eigen库
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)

# 将Eigen库include进来
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${Sophus_INCLUDE_DIRS})
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(useSophus useSophus.cpp)
add_executable(drawTrajectory draw_trajectory.cpp) 
add_executable(trajctoryError trajectoryError.cpp) 

target_link_libraries(drawTrajectory ${Sophus_LIBRARIES} ${Pangolin_LIBRARIES})
target_link_libraries(trajctoryError ${Sophus_LIBRARIES} ${Pangolin_LIBRARIES})
target_link_libraries(useSophus Sophus::Sophus)