#pragma once

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <thrust/device_vector.h>

#include <cstdint>
#include <cstddef>

class IConverter
{
public:
    virtual unsigned int convert(
        const sensor_msgs::msg::PointCloud2::SharedPtr& sub_cloud,
        thrust::device_vector<float>& d_out) = 0;

    // Raw-buffer overload: packs an interleaved point buffer (x,y,z,intensity at
    // byte offsets 0,4,8,12 of every point_step bytes) into float4. Shared by the
    // ROS2 and BARQ paths so they run the identical GPU conversion kernel.
    virtual unsigned int convert(
        const std::uint8_t* data, std::size_t in_bytes,
        std::uint32_t width, std::uint32_t height,
        std::uint32_t row_step, std::uint32_t point_step,
        thrust::device_vector<float>& d_out) = 0;

    virtual ~IConverter() = default;
};
