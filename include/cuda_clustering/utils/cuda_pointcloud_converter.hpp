#pragma once

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace pointcloud_utils
{

struct ConverterResources
{
    std::uint8_t* d_input      = nullptr;
    std::size_t   input_bytes  = 0;
    cudaStream_t  stream       = nullptr;

    ConverterResources();
    ~ConverterResources();

    ConverterResources(const ConverterResources&)            = delete;
    ConverterResources& operator=(const ConverterResources&) = delete;
    ConverterResources(ConverterResources&&)                 = delete;
    ConverterResources& operator=(ConverterResources&&)      = delete;

    void reserve(std::size_t in_bytes);
};

void convertPointCloud2ToFloatArray(
    const sensor_msgs::msg::PointCloud2::SharedPtr& sub_cloud,
    thrust::device_vector<float>& d_out,
    ConverterResources& res);

} // namespace pointcloud_utils