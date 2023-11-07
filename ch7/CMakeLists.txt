project(7_1)
cmake_minimum_required(VERSION 3.16)


#设置cmkae模式
set(CMAKE_BUILD_TYPE,Release)

#set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")
set(CMAKE_CXX_FLAGS "-std=c++14 -O2 ${SSE_FLAGS} -msse4") 

#添加Opencv库
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#Eigen
include_directories("/usr/local/include/Eigen3")

#SOphus
find_package(Sophus REQUIRED)
find_package(fmt REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# g2o
LIST( APPEND CMAKE_MODULE_PATH /usr/local/include/g2o/cmake_modules )
set(G2O_ROOT /usr/local/include/g2o)
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

#可执行文件以及连接库
add_executable(orb_cv orb_cv.cpp)
target_link_libraries(orb_cv ${OpenCV_LIBS})

add_executable(orb_self orb_self.cpp)
target_link_libraries(orb_self ${OpenCV_LIBS})

add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})

add_executable(triangulation triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS})

add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
target_link_libraries(pose_estimation_3d2d ${OpenCV_LIBS} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} fmt::fmt)
 
add_executable(pose_estimation_3d3d pose_estimation_3d3d.cpp)
target_link_libraries(pose_estimation_3d3d   ${OpenCV_LIBS} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} fmt)
