cmake_minimum_required(VERSION 3.4...3.18)
project(henon_map_cpp LANGUAGES CUDA CXX)
find_package(CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ version selection")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(PYBIND)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(pybind11)
pybind11_add_module(henon_map_engine src/bind.cpp src/henon.cu src/modulation.cpp src/dynamic_indicator.cpp)

target_link_libraries(henon_map_engine PRIVATE Threads::Threads)
target_link_libraries(henon_map_engine PUBLIC ${CMAKE_CUDA_LIBRARIES})
target_link_libraries(henon_map_engine PUBLIC fftw3)
