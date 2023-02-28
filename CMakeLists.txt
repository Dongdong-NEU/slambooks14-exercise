cmake_minimum_required(VERSION 2.8.3)

project(imageProcess)

set(CMAKE_CXX_FLAGS "-std=c++11 -O2")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Ceres REQUIRED)
find_package(G2O REQUIRED)

include_directories("/usr/include/eigen3")
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${Ceres_INCLUDE_DIRS})
include_directories(${G20_INCLUDE_DIRS})

add_executable(undistortImage undistort_image.cpp ) 
add_executable(disParity disparity.cpp)
add_executable(guassNewton gaussnewton.cpp)
add_executable(ceresCurveFitting ceresCurveFitting.cpp)
add_executable(g2oCurveFitting g2oCurveFitting.cpp)

target_link_libraries(undistortImage ${OpenCV_LIBS})
target_link_libraries(disParity ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
target_link_libraries(guassNewton ${OpenCV_LIBS})
target_link_libraries(ceresCurveFitting  ${OpenCV_LIBS} ${CERES_LIBRARIES})
target_link_libraries(g2oCurveFitting ${OpenCV_LIBS} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY})

add_subdirectory(rgbd)