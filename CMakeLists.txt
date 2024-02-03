cmake_minimum_required(VERSION 3.4...3.18)
# Set the path to the desired g++ version
set(CMAKE_CXX_COMPILER "/usr/bin/g++-9")
set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++14")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
# set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ version selection")
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# specify path of cuda compiler
# N.B. REMEMBER TO USE devtoolset-7!!!
# set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-10")

option(WITH_CUDA "Enable CUDA support" ON)

project(henon_map_cpp LANGUAGES CUDA CXX)
find_package(CUDA)

# flag to enable/disable CUDA

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CUDA_FOUND AND WITH_CUDA)
  add_compile_definitions(USE_CUDA)
endif()

add_compile_definitions(PYBIND)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
if(CUDA_FOUND AND WITH_CUDA)
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

add_subdirectory(pybind11)

if(CUDA_FOUND AND WITH_CUDA)
  message(STATUS "CUDA found. Compiling with CUDA support.")
  pybind11_add_module(henon_map_engine src/bind.cpp src/henon.cu src/henon_gpu.cu src/modulation.cpp src/dynamic_indicator.cpp src/utils.cu)
#   target_compile_options(henon_map_engine PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_70,code=sm_70>)
  target_link_libraries(henon_map_engine PRIVATE Threads::Threads)
  target_link_libraries(henon_map_engine PUBLIC ${CMAKE_CUDA_LIBRARIES})
  target_link_libraries(henon_map_engine PUBLIC fftw3)
else()
  if(WITH_CUDA)
    message(WARNING "CUDA not found. Compiling without CUDA support.")
  else()
    message(STATUS "CUDA not requested. Compiling without CUDA support.")
  endif()
  add_library(henon_map_engine MODULE src/bind.cpp src/henon.cu src/modulation.cpp src/dynamic_indicator.cpp src/utils.cu)
  target_link_libraries(henon_map_engine PRIVATE Threads::Threads fftw3)
endif()
