cmake_minimum_required(VERSION 3.9)
project(crossbarsim)


#set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS, "-std=c++17")

# find_package(Armadillo REQUIRED)
# include_directories(${ARMADILLO_INCLUDE_DIRS})

add_library(genarray STATIC GenArray.cc)
add_library(lgu STATIC LogicUnit.cc)

add_executable(sim main.cpp)
target_link_libraries(sim PRIVATE genarray lgu)
set_property(TARGET sim PROPERTY CXX_STANDARD 17)




