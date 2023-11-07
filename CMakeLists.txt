cmake_minimum_required(VERSION 3.16)

project(helloSlam)

add_executable(helloSlam helloSlam.cpp)
add_library(hello libHelloSlam.cpp)
add_library(hello_shared SHARED libHelloSlam.cpp)
add_executable(useHello useHello.cpp)

target_link_libraries(useHello hello_shared)