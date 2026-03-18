cmake_minimum_required(VERSION 3.8)
project(cuda_cone_rush LANGUAGES CXX C CUDA)

add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20012,20015>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

option(ENABLE_VERBOSE "Print essential timings information for each callback" OFF)
option(LATENCY_TESTING "Print latency information for each BARQ callback" OFF)
option(USE_CUDA_POINTCLOUD_CONVERTER "Build pointcloud_converter from CUDA implementation" ON)
option(NSIGHT_SDK "Set an execution limit for NVIDIA Nsight Systems profiling" OFF)

# Pinned memory: auto-detect architecture and set the default accordingly.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
  option(USE_PINNED_MEMORY "Use CUDA pinned (page-locked) host memory" ON)
else()
  option(USE_PINNED_MEMORY "Use CUDA pinned (page-locked) host memory" OFF)
endif()

if(USE_PINNED_MEMORY)
  message(STATUS "Pinned host memory: ENABLED")
  add_definitions(-DUSE_PINNED_MEMORY)
else()
  message(STATUS "Pinned host memory: DISABLED")
endif()

if(ENABLE_VERBOSE)
  message(STATUS "Verbose timing information: ENABLED")
  add_definitions(-DENABLE_VERBOSE)
else()
  message(STATUS "Verbose timing information: DISABLED")
endif()

if(LATENCY_TESTING)
  message(STATUS "Latency testing: ENABLED")
  add_definitions(-DLATENCY_TESTING)
else()
  message(STATUS "Latency testing: DISABLED")
endif()

if(USE_CUDA_POINTCLOUD_CONVERTER)
  message(STATUS "PointCloud converter backend: CUDA")
  set(POINTCLOUD_CONVERTER_SOURCE src/utils/cuda_pointcloud_converter.cu)
  add_definitions(-DUSE_CUDA_POINTCLOUD_CONVERTER)
else()
  message(STATUS "PointCloud converter backend: CPU")
  set(POINTCLOUD_CONVERTER_SOURCE src/utils/pointcloud_converter.cpp)
endif()

if(NSIGHT_SDK)
  message(STATUS "NVIDIA Nsight Systems profiling: ENABLED")
  add_definitions(-DNSIGHT_SDK)
else()
  message(STATUS "NVIDIA Nsight Systems profiling: DISABLED")
endif()

#set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.8")
# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(CUDA  REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)

find_package(PCL REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

# Find include directories
include_directories(${PCL_INCLUDE_DIRS})
include_directories(/opt/ros/humble/include/pcl_ros)
include_directories(/opt/ros/humble/include/pcl_conversions)
include_directories(/opt/ros/humble/include/message_filters)
include_directories(/opt/ros/humble/include/pcl_msgs)   
include_directories(/opt/ros/humble/include/tf2)   
include_directories(/opt/ros/humble/include/tf2_ros)   
include_directories(/opt/ros/humble/include/tf2_msgs) 

include_directories(include ${catkin_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${Eigen_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_definitions(-std=c++17 -g -lboost_thread-mt)

add_library(${PROJECT_NAME} src/controller_node.cu)

add_executable(cuda_cone_rush_node
  src/main.cpp
)
target_link_libraries(cuda_cone_rush_node
  ${PCL_LIBRARIES}
  ${CUDA_LIBRARIES}
  cuda_clustering
  cuda_filtering
  cuda_segmentation
  pointcloud_converter
)
target_include_directories(cuda_cone_rush_node PUBLIC include)

add_library(cuda_clustering src/clustering/cuda_clustering.cu)
add_library(cluster_filtering src/clustering/cluster_filtering/dimension_filter.cpp)
add_library(cuda_filtering src/filtering/cuda_filtering.cu)
add_library(cuda_segmentation src/segmentation/cuda_segmentation.cu)
add_library(barq_lib src/barq/barq.cpp)

target_include_directories(barq_lib PUBLIC include)

# Optimized pointcloud converter library
add_library(pointcloud_converter ${POINTCLOUD_CONVERTER_SOURCE})
target_compile_options(pointcloud_converter PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -mtune=native -ffast-math>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)
target_include_directories(pointcloud_converter PUBLIC include)
ament_target_dependencies(pointcloud_converter
  sensor_msgs
  pcl_conversions
  PCL
)

ament_target_dependencies(cluster_filtering
  geometry_msgs
)

ament_target_dependencies(cuda_clustering
  rclcpp
  visualization_msgs
)
ament_target_dependencies(cuda_filtering
  rclcpp
)
ament_target_dependencies(cuda_segmentation
  rclcpp
)

target_link_libraries(${PROJECT_NAME}
  ${PCL_LIBRARIES}
  ${CUDA_LIBRARIES}
  cuda_clustering
  cuda_filtering
  cuda_segmentation
  pointcloud_converter
  barq_lib
)
ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  sensor_msgs
  visualization_msgs
  pcl_conversions
  PCL
)

target_link_libraries(${PROJECT_NAME}_node 
  ${PROJECT_NAME}
)

target_include_directories(cuda_cone_rush PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(cuda_cone_rush PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

install(TARGETS
  cuda_cone_rush_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch directory
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install config directory
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
