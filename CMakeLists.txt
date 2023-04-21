cmake_minimum_required(VERSION 2.8.3)

project(ORBPranctice)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories("/usr/include/eigen3/")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})
find_package(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_executable(computeORB ./../orb_feature/computeORB.cpp ) 
target_link_libraries(computeORB ${OpenCV_LIBS})

add_executable(E2Rt ./../E2Rt/E2Rt.cpp ) 
target_link_libraries(E2Rt Sophus::Sophus)

add_executable(triangulation ./../E2Rt/triangulation.cpp)
target_link_libraries(triangulation  ${OpenCV_LIBS})

add_executable(PNP ./../GN-BA/pose_estimation_3d2d.cpp)
target_link_libraries(PNP 
${OpenCV_LIBS} 
${CSPARSE_LIBRARY}
g2o_core g2o_stuff  g2o_types_sba g2o_types_slam3d 
g2o_csparse_extension g2o_solver_csparse
cholmod g2o_solver_cholmod 
${Sophus_LIBRARIES})

add_executable(PNP_My ./../PNP_ICP_Practice/pnp_g2o.cpp)
target_link_libraries(PNP_My 
${OpenCV_LIBS} 
${CSPARSE_LIBRARY}
g2o_core g2o_stuff  g2o_types_sba g2o_types_slam3d 
g2o_csparse_extension g2o_solver_csparse
cholmod g2o_solver_cholmod 
${Sophus_LIBRARIES})

add_executable(PNP_g2o ./../PNP_ICP_Practice/pnp_g2o_stone.cpp)
target_link_libraries(PNP_g2o 
${OpenCV_LIBS} 
${CSPARSE_LIBRARY}
g2o_core g2o_stuff  g2o_types_sba g2o_types_slam3d 
g2o_csparse_extension g2o_solver_csparse
cholmod g2o_solver_cholmod 
${Sophus_LIBRARIES})

add_executable(pnp_ceres ./../PNP_ICP_Practice/pnp_ceres.cpp)
target_link_libraries(pnp_ceres ${OpenCV_LIBS} ${CERES_LIBRARIES})

add_executable(PNP_Err ./../GN-BA/PNP_Err_test.cpp)
target_link_libraries(PNP_Err ${OpenCV_LIBS} g2o_core g2o_stuff  ${Sophus_LIBRARIES} )

add_executable(GN-BA ./../GN-BA/GN-BA.cpp)
target_link_libraries(GN-BA ${OpenCV_LIBS} g2o_core g2o_stuff)

add_executable(myICP ./../ICP/myICP.cpp)
target_link_libraries(myICP ${OpenCV_LIBS} Sophus::Sophus  ${Pangolin_LIBRARIES})

add_executable(pose_estimation_3d3d ./../ICP/pose_estimation_3d3d.cpp)
target_link_libraries(pose_estimation_3d3d ${OpenCV_LIBS} Sophus::Sophus  ${Pangolin_LIBRARIES} g2o_core g2o_stuff)

add_executable(icp_g2o_tmp ./../PNP_ICP_Practice/icp_g2o_tmp.cpp)
target_link_libraries(icp_g2o_tmp ${OpenCV_LIBS} 
${CSPARSE_LIBRARY}
g2o_core g2o_stuff  g2o_types_sba g2o_types_slam3d 
g2o_csparse_extension g2o_solver_csparse
cholmod g2o_solver_cholmod 
${Sophus_LIBRARIES})



