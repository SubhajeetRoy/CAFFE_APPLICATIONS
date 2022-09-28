cmake_minimum_required(VERSION 2.8)

project( DEMO )

set(SOURCES 
demo.cpp 
classify.cpp

)
include_directories(.)
## CHANGE THIS PATH TO YOUR CAFFE DIR PATH
include_directories(/home/roy/caffe_1_0/caffe/include)

## CHANGE THIS PATH TO YOUR CAFFE DIR PATH
include_directories(/home/roy/caffe_1_0/caffe/distribute/include)


set(WILLP_FLAGS_COMMON " -MMD -DUSE_OPENCV -fno-exceptions -ffunction-sections -fdata-sections -funsigned-char -Wall -Wformat -Wparentheses -Wno-variadic-macros -Wno-sign-compare -DHAVE_PTHREADS -DHAVE_POSIX_CLOCKS -D__i386__ -DCPU_ONLY")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} ${WILLP_FLAGS_COMMON}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WILLP_FLAGS_COMMON} -Wno-reorder -fpermissive -fexceptions")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections,--stats,--Map,map.info")
set(CMAKE_C_FLAGS_DEBUG   "-DDEBUG -g -ggdb -O0")
set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG -g -ggdb -O0")

set(CMAKE_C_FLAGS_RELEASE   "-DRELEASE -Os")
set(CMAKE_CXX_FLAGS_RELEASE " -DRELEASE -Os")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-Wl,--strip-all")
add_executable(${PROJECT_NAME} ${SOURCES})

# CHANGE THIS PATH TO YOUR CAFFE DIR PATH for libcaffe.so
TARGET_LINK_LIBRARIES(${PROJECT_NAME} glog boost_system opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_videoio /home/roy/caffe_1_0/caffe/build/lib/libcaffe.so)

install(TARGETS ${PROJECT_NAME}
        DESTINATION sbin)


