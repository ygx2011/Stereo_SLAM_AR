SET(execName zed_slam_ar)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)

SET(EXECUTABLE_OUTPUT_PATH ".")

IF(WIN32) # Windows
    message(FATAL_ERROR "Multi ZED not available on Windows")
    EXIT()
ELSE() # Linux
    find_package(ZED 1.0 REQUIRED)
    SET(Eigen_INCLUDE_DIRS "/usr/include/eigen3") ##depends if installed by packet or by own build install
    SET(Eigen_INCLUDE_DIRS_2 "/usr/local/include/eigen3") ##depends if installed by packet or by own build install
    include_directories(${Eigen_INCLUDE_DIRS_2}) 

##For Jetson, OpenCV4Tegra is based on OpenCV2.4
     exec_program(uname ARGS -p OUTPUT_VARIABLE CMAKE_SYSTEM_NAME2)
     if ( CMAKE_SYSTEM_NAME2 MATCHES "aarch64" OR CMAKE_SYSTEM_NAME2 MATCHES "armv7l" )
        find_package(OpenCV 2.4)
     else()
      find_package(OpenCV 3.1)
     endif()

ENDIF(WIN32)

find_package(CUDA REQUIRED)
 
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${Eigen_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)#ygx

link_directories(${ZED_LIBRARY_DIR})
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

SET(SRC_FOLDER src)
FILE(GLOB_RECURSE SRC_FILES "${SRC_FOLDER}/*.cpp")

ADD_EXECUTABLE(${execName} ${SRC_FILES})
set_property(TARGET ${execName} PROPERTY OUTPUT_NAME "zed_slam_ar")

#ygx
find_package( OpenGL REQUIRED )
include_directories(${OPENGL_INCLUDE_DIR})
find_package( GLUT REQUIRED )
include_directories(${GLUT_INCLUDE_DIR})
find_package( GLEW REQUIRED )
include_directories(${GLEW_INCLUDE_DIR})
find_package( PNG REQUIRED )
include_directories(${PNG_INCLUDE_DIR})
LIST(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})
find_package(GLFW REQUIRED)
include_directories(${GLFW_INCLUDE_DIRS})

#ygx
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
include_directories(
        ${PROJECT_SOURCE_DIR}
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS})

add_definitions(-std=c++0x)

TARGET_LINK_LIBRARIES(${execName}
                        ${ZED_LIBRARIES}
                        ${OpenCV_LIBRARIES}
                        ${CUDA_LIBRARIES} ${CUDA_npps_LIBRARY} ${CUDA_nppi_LIBRARY}
                        #ygx
                        ${OPENGL_LIBRARIES}
                        ${GLUT_LIBRARIES}
                        ${GLEW_LIBRARIES}
                        /usr/lib/libglfw.so.3
                        ${PNG_LIBRARY}
                        #ygx
                        ${EIGEN3_LIBS}
                        ${Pangolin_LIBRARIES}
                        ${PROJECT_SOURCE_DIR}/lib/libORB_SLAM2.so)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3" ) # Release Perf mode
