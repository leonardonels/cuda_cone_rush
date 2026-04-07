#pragma once

#include "cuda_cone_rush/utils/iconverter.hpp"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <cstdint>
#include <cstddef>

class CudaRingConverter : public IConverter
{
public:
    CudaRingConverter();
    ~CudaRingConverter() override;

    CudaRingConverter(const CudaRingConverter&)            = delete;
    CudaRingConverter& operator=(const CudaRingConverter&) = delete;

    unsigned int convert(const sensor_msgs::msg::PointCloud2::SharedPtr& sub_cloud,
                         thrust::device_vector<float>& d_out) override;

    /// Raw device pointer to ring data (valid after convert())
    float* getRingPtr() { return thrust::raw_pointer_cast(d_ring_.data()); }

private:
    std::uint8_t*  d_input_      = nullptr;
    std::uint32_t* d_count_      = nullptr;
    std::size_t    input_bytes_  = 0;
    cudaStream_t   stream_       = nullptr;

    thrust::device_vector<float> d_ring_;

    void reserve(std::size_t in_bytes);
};
