#pragma once

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <thrust/device_vector.h>

class IConverter
{
public:
    virtual void convert(
        const sensor_msgs::msg::PointCloud2::SharedPtr& sub_cloud,
        thrust::device_vector<float>& d_out) = 0;
    virtual ~IConverter() = default;
};
