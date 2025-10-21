cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM3)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# ---------------------------------------
# Compiler flags
# ---------------------------------------
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations -Wno-attributes -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-attributes -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# Use With Caution
# -Wno-deprecated-copy
# -Wno-conversion-null
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
    -Wno-deprecated-declarations \
    -Wno-attributes \
    -Wno-deprecated-copy \
    -Wno-conversion-null \
    -Wno-cpp \
    -O3")

# ---------------------------------------
# CMake modules path
# ---------------------------------------
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# ---------------------------------------
# OpenCV
# ---------------------------------------
find_package(OpenCV 4.2)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 4.4 not found.")
   endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(realsense2)

# Python3 - Mission systems
find_package(Python3 3.8 EXACT REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python3_EXECUTABLE = ${Python3_EXECUTABLE}")
message(STATUS "Python3_INCLUDE_DIRS = ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_LIBRARIES = ${Python3_LIBRARIES}")

# Pytorch - Mission systems
find_package(Torch REQUIRED)
message(STATUS "Pytorch includes: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Pytorch libs: ${TORCH_LIBRARIES}")

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/include/CameraModels
${PROJECT_SOURCE_DIR}/Thirdparty/pybind11/include
${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${Python3_INCLUDE_DIRS}
${TORCH_INCLUDE_DIRS}
)

# ---------------------------------------
# Debug messages
# ---------------------------------------
foreach(var
    Eigen3_DIR
    EIGEN3_INCLUDE_DIR
    Pangolin_DIR
    Pangolin_INCLUDE_DIRS
    Pangolin_LIBRARIES
    Python3_LIBRARIES
    Python3_INCLUDE_DIRS
    Python3_EXECUTABLE
    CONDA_LIBS
    OpenCV_DIR
    catkin_DIR
)
  message(STATUS "${var} = ${${var}}")
endforeach()

# ---------------------------------------
# Output path
# ---------------------------------------
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# ---------------------------------------
# Source files
# ---------------------------------------
set(SOURCE_FILES
src/System.cc
#src/System_util.cc
src/Tracking.cc
src/Tracking_util.cc
src/LocalMapping.cc
src/LocalMapping_util.cc
src/LoopClosing.cc
#src/LoopClosing_util.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/MapPoint_util.cc
src/MapObject.cc
src/KeyFrame.cc
src/KeyFrame_util.cc
src/Atlas.cc
src/Atlas_util.cc
src/Map.cc
src/Map_util.cc
src/MapDrawer.cc
src/ObjectDrawer.cc
src/ObjectDetection.cc
src/ObjectRenderer.cc
src/Optimizer.cc
src/Optimizer_util.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Viewer.cc
src/ImuTypes.cc
src/G2oTypes.cc
src/CameraModels/Pinhole.cpp
src/CameraModels/KannalaBrandt8.cpp
src/OptimizableTypes.cpp
src/MLPnPsolver.cpp
src/GeometricTools.cc
src/TwoViewReconstruction.cc
src/Config.cc
src/Settings.cc)

# ---------------------------------------
# Headers
# ---------------------------------------
set(HEADER_FILES
include/System.h
include/Tracking.h
include/LocalMapping.h
include/LoopClosing.h
include/ORBextractor.h
include/ORBmatcher.h
include/FrameDrawer.h
include/Converter.h
include/MapPoint.h
include/KeyFrame.h
include/Atlas.h
include/Map.h
include/MapDrawer.h
include/MapObject.h
include/ObjectDrawer.h
include/ObjectDetection.h
include/ObjectRenderer.h
include/ObjectPoseGraph.h
include/Renderer.hpp
include/Optimizer.h
include/Frame.h
include/KeyFrameDatabase.h
include/Sim3Solver.h
include/Viewer.h
include/ImuTypes.h
include/G2oTypes.h
include/CameraModels/GeometricCamera.h
include/CameraModels/Pinhole.h
include/CameraModels/KannalaBrandt8.h
include/OptimizableTypes.h
include/MLPnPsolver.h
include/GeometricTools.h
include/TwoViewReconstruction.h
include/SerializationUtils.h
include/Config.h
include/Settings.h
include/tcp_interface.hpp)

# ---------------------------------------
# Library target
# ---------------------------------------
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES})

# ---------------------------------------
# Thirdparty subdirs
# ---------------------------------------
add_subdirectory(Thirdparty/g2o)
add_subdirectory(Thirdparty/pybind11) # Mission systems

target_link_libraries(${PROJECT_NAME}
torch_python # Mission systems
${Python3_LIBRARIES} # Mission systems
pybind11::embed # Mission systems
pybind11::module # Mission systems
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
-lboost_serialization
-lcrypto
-lboost_system   # Mission systems
-lpthread   # Mission systems
-lncurses   # for Jetson
-lncursesw   # for Jetson
-lffi   # for Jetson
snark_imaging   # Mission systems
snark_ouster  # Mission systems
comma_csv  # Mission systems
comma_string  # Mission systems
comma_xpath  # Mission systems
)

# If RealSense SDK is found the library is added and its examples compiled
if(realsense2_FOUND)
    include_directories(${PROJECT_NAME}
    ${realsense_INCLUDE_DIR}
    )
    target_link_libraries(${PROJECT_NAME}
    ${realsense2_LIBRARY}
    )
endif()

# ---------------------------------------
# Optional: print target summary
# ---------------------------------------
message(STATUS "Object SLAM build complete: ${PROJECT_NAME}")


# Build examples

#Monocular examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

add_executable(mono_euroc
        Examples/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME})

add_executable(mono_kitti  # Mission-Systems
        Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})

add_executable(mono_vulcan  # Mission-Systems
        Examples/Monocular/mono_vulcan.cc)
target_link_libraries(mono_vulcan ${PROJECT_NAME})

add_executable(mono_geo_vulcan  # Mission-Systems
        Examples/Monocular/mono_geo_vulcan.cc)
target_link_libraries(mono_geo_vulcan ${PROJECT_NAME})


#Monocular inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular-Inertial)

add_executable(mono_inertial_euroc
        Examples/Monocular-Inertial/mono_inertial_euroc.cc)
target_link_libraries(mono_inertial_euroc ${PROJECT_NAME})

add_executable(mono_inertial_vulcan
        Examples/Monocular-Inertial/mono_inertial_vulcan.cc)
target_link_libraries(mono_inertial_vulcan ${PROJECT_NAME})

add_executable(mono_inertial_kitti
        Examples/Monocular-Inertial/mono_inertial_kitti.cc)
target_link_libraries(mono_inertial_kitti ${PROJECT_NAME})

#Stereo examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

add_executable(stereo_kitti  # Mission-Systems
        Examples/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti ${PROJECT_NAME})


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo-Inertial)
add_executable(stereo_inertial_kitti  # Mission-Systems
        Examples/Stereo-Inertial/stereo_inertial_kitti.cc)
target_link_libraries(stereo_inertial_kitti ${PROJECT_NAME})

