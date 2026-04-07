#pragma once

#include "cuda_cone_rush/segmentation/isegmentation.hpp"
#include "cuda_cone_rush/segmentation/cuda_segmentation.hpp" // for segParam_t
#include "cuda_cone_rush/utils/cached_allocator.hpp"

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
    int num_ring_groups_;
    int num_patches_;       // = num_ring_groups_ * 2 (left + right)

    float* d_ring_in_ = nullptr;
    thrust::device_vector<float> d_ring_out_;

    CachedAllocator alloc_;

    // Partitioning buffers
    thrust::device_vector<int> d_patch_of_;      // [N] patch id per point
    thrust::device_vector<int> d_indices_;        // [N] point indices sorted by patch
    thrust::device_vector<int> d_patch_counts_;   // [num_patches]
    thrust::device_vector<int> d_patch_offsets_;  // [num_patches] exclusive prefix sum
    thrust::device_vector<int> d_patch_write_;    // [num_patches] atomic write cursors

    // RANSAC buffers (num_patches * iters_per_patch)
    thrust::device_vector<int>    d_ransac_counts_;
    thrust::device_vector<float4> d_ransac_planes_;
    thrust::device_vector<float4> d_best_planes_;  // [num_patches] winning plane per patch

    // Compaction
    thrust::device_vector<int>          d_index_;      // [N] inlier marks
    thrust::device_vector<unsigned int> d_out_count_;   // [1]
};
