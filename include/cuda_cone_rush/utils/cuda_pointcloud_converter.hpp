#pragma once

#include "cuda_cone_rush/utils/iconverter.hpp"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <cstdint>
#include <cstddef>

class CudaConverter : public IConverter
{
public:
    CudaConverter();
    ~CudaConverter() override;

    CudaConverter(const CudaConverter&)            = delete;
    CudaConverter& operator=(const CudaConverter&) = delete;

    unsigned int convert(const sensor_msgs::msg::PointCloud2::SharedPtr& sub_cloud,
                         thrust::device_vector<float>& d_out) override;

    unsigned int convert(const std::uint8_t* data, std::size_t in_bytes,
                         std::uint32_t width, std::uint32_t height,
                         std::uint32_t row_step, std::uint32_t point_step,
                         thrust::device_vector<float>& d_out) override;

private:
    std::uint8_t*  d_input_      = nullptr;
    std::size_t    input_bytes_  = 0;
    cudaStream_t   stream_       = nullptr;

    void reserve(std::size_t in_bytes);
};
