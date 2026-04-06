#pragma once

#include "cuda_cone_rush/filtering/ifiltering.hpp"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <rclcpp/rclcpp.hpp>
#include <iostream>
#include <chrono>

class CudaRingFilter : public IFilter
{
private:
    float upLimitX, downLimitX;
    float upLimitY, downLimitY;
    float upLimitZ, downLimitZ;
    bool filterX, filterY, filterZ;

    int block_size_ = 768;

    thrust::device_vector<float>        d_temp_;
    thrust::device_vector<unsigned int> d_count_;

    // Ring support
    float* d_ring_in_ = nullptr;
    thrust::device_vector<float> d_ring_temp_;

public:
    CudaRingFilter(float upFilterLimitsX, float downFilterLimitsX,
                   float upFilterLimitsY, float downFilterLimitsY,
                   float upFilterLimitsZ, float downFilterLimitsZ);
    ~CudaRingFilter() = default;

    void filterPoints(float* input, unsigned int inputSize,
                     float** output, unsigned int* outputSize,
                     cudaStream_t stream) override;

    /// Set the ring input pointer before calling filterPoints
    void setRingInput(float* ring_ptr, unsigned int size);

    /// Get compacted ring output (valid after filterPoints)
    float* getRingOutput() { return thrust::raw_pointer_cast(d_ring_temp_.data()); }
};
