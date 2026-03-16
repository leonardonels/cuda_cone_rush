#pragma once

#include <rclcpp/rclcpp.hpp>
#include <cuda_runtime.h>
#include <cstddef>
#include <vector>

class Isegmentation
{
public:
    /**
     * → input point buffer (as floats xyzxyz...)
     * num_points: number of points
     * out_points: caller-allocated buffer, at least as big as input
     * out_num_points: actual number of points in segmentation result
     */
    virtual void segment(float *inputData,
                         unsigned int nCount,
                         float *out_points,
                         unsigned int* out_num_points,
                         cudaStream_t stream) = 0;
    virtual ~Isegmentation() = default;
};
