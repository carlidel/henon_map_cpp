cmake_minimum_required(VERSION 3.4...3.18)
project(henon_map_cpp LANGUAGES CUDA CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

add_subdirectory(pybind11)
pybind11_add_module(henon_map_engine src/bind.cpp src/henon.cu src/modulation.cpp)

target_link_libraries(henon_map_engine PRIVATE Threads::Threads)
set_property(TARGET henon_map_engine PROPERTY CXX_STANDARD_REQUIRED 14)