cmake_minimum_required (VERSION 2.8.12)
project(start_simple_lr)

include("${PROJECT_SOURCE_DIR}/help.cmake")
define_colors()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wno-shadow -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_STANDARD 14)

option(BUILD_DEBUG "build project for debug or development propurse..." ON)

# 编译结果中的库和可执行文件的路径
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

set(PADDLE_PREFIX
    "/Users/taozj/Dropbox/repos/machine_learning/PADDLE/paddle-r1.7.z" )
set(INCLUDE_DIR
    ${PADDLE_PREFIX}/inference_root/paddle/include
    ${PADDLE_PREFIX}/inference_root/third_party/
    ${PADDLE_PREFIX}/inference_root/third_party/install/glog/include/
    ${PADDLE_PREFIX}/inference_root/third_party/install/gflags/include/
)
include_directories( ${INCLUDE_DIR} )

link_directories(
    ${PADDLE_PREFIX}/inference_root/paddle/lib
    ${PADDLE_PREFIX}/inference_root/third_party/install/glog/lib/
)

set (EXTRA_LIBS ${EXTRA_LIBS} paddle_fluid)
set (EXTRA_LIBS ${EXTRA_LIBS} glog)
set (EXTRA_LIBS ${EXTRA_LIBS} pthread)


add_executable(start_simple_lr native_main.cc)
target_link_libraries(
    start_simple_lr
    ${EXTRA_LIBS}
)
