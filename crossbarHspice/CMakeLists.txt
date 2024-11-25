cmake_minimum_required(VERSION 3.9)
project(crossbarsim)

#set(CMAKE_C_COMPILER "/opt/rh/devtoolset-8/root/usr/bin/gcc")
#set(CMAKE_CXX_COMPILER "/opt/rh/devtoolset-8/root/usr/bin/g++")

#set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS, "-std=c++14")

# find_package(Armadillo REQUIRED)
# include_directories(${ARMADILLO_INCLUDE_DIRS})

add_library(d2array STATIC 2dArray.cc)
add_library(d3varray STATIC 3dVArray.cc)
add_library(genarray STATIC GenArray.cc)
add_library(d3harray STATIC 3dHArray.cc)
add_library(lgu STATIC LogicUnit.cc)

add_executable(sim main.cc)
# target_link_libraries(sim PRIVATE d2array d3varray d3harray genarray lgu  ${ARMADILLO_LIBRARIES})
target_link_libraries(sim PRIVATE d2array d3varray d3harray genarray lgu) 
set_property(TARGET sim PROPERTY CXX_STANDARD 14)

add_executable(check fastmode_answer_check.cc)
set_property(TARGET check PROPERTY CXX_STANDARD 14)

add_executable(gen getIRAS-table.cc)
set_property(TARGET gen PROPERTY CXX_STANDARD 14)


