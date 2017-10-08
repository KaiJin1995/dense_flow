cmake_minimum_required(VERSION 2.8)
project( denseFlow )

find_package( OpenCV REQUIRED )
find_package( CUDA )
include_directories("/usr/local/cuda-8.0/include/")
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( denseImage test.cpp )
target_link_libraries( denseImage ${OpenCV_LIBS} )
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

