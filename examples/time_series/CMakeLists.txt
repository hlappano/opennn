cmake_minimum_required(VERSION 2.8.10)

# Project's name

project(time_series)

link_directories(${CMAKE_SOURCE_DIR}/opennn)

include_directories(${CMAKE_SOURCE_DIR}/opennn)

add_executable(time_series main.cpp)

target_link_libraries(time_series opennn)


