#pragma once

#include <cstdint>
#include <cstring>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_types.h>

namespace pointcloud_utils {

void convertPointCloud2ToFloatArray(
    const sensor_msgs::msg::PointCloud2::SharedPtr& sub_cloud,
    float* output_data
);

} 
