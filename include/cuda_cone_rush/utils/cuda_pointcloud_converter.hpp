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

    explicit CudaConverter(bool filter_zeros_ = false);

    CudaConverter(const CudaConverter&)            = delete;
    CudaConverter& operator=(const CudaConverter&) = delete;

    unsigned int convert(const sensor_msgs::msg::PointCloud2::SharedPtr& sub_cloud,
                         thrust::device_vector<float>& d_out) override;

private:
    std::uint8_t*  d_input_      = nullptr;
    std::uint32_t* d_count_      = nullptr;
    std::size_t    input_bytes_  = 0;
    cudaStream_t   stream_       = nullptr;
    
    bool filter_zeros_;

    void reserve(std::size_t in_bytes);
};
