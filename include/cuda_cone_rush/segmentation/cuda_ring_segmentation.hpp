#pragma once

#include "cuda_cone_rush/segmentation/isegmentation.hpp"
#include "cuda_cone_rush/segmentation/cuda_segmentation.hpp" // for segParam_t

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

class CudaRingSegmentation : public Isegmentation
{
public:
    CudaRingSegmentation(segParam_t& params);
    ~CudaRingSegmentation() = default;

    void segment(float* inputData,
                 unsigned int nCount,
                 float* out_points,
                 unsigned int* out_num_points,
                 cudaStream_t stream) override;

    /// Set ring input before calling segment
    void setRingInput(float* ring_ptr, unsigned int size);

    /// Get compacted ring output (valid after segment)
    float* getRingOutput() { return thrust::raw_pointer_cast(d_ring_out_.data()); }

private:
    segParam_t segP_;
    float* d_ring_in_ = nullptr;
    thrust::device_vector<float> d_ring_out_;
};
