#pragma once

#include "isegmentation.hpp"
#include "cuda_cone_rush/utils/cached_allocator.hpp"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

typedef struct
{
    double distanceThreshold;
    int maxIterations;
    double probability;
    double maxSegmentationDistance; // new parameter to limit how far points can be from the sensor for plane fitting
    bool optimizeCoefficients;
} segParam_t;

class CudaSegmentation : public Isegmentation
{
private:
    segParam_t segP;
    CachedAllocator alloc;  // reusable memory pool for thrust temp buffers

    double totalTime = 0.0;
    unsigned int iterations = 0;

    int compactInliersKernel_block_size = 768; // default block size for compactInliers kernel
    int ransacPlaneKernel_block_size = 768; // default block size for ransacPlane kernel
    int markInliersKernel_block_size = 768; // default block size for markInliers kernel
    int markInliersFromDeviceKernel_block_size = 768; // default block size for markInliersFromDevice kernel

    // -------------------------------------------------------------------------
    // Device vectors for GPU data
    // -------------------------------------------------------------------------
    thrust::device_vector<int> d_index;
    thrust::device_vector<float> d_input;

    // RANSAC temporary buffers
    thrust::device_vector<int> d_counts;
    thrust::device_vector<float4> d_planes;
    thrust::device_vector<float4> d_bestPlane;  // single-element: best plane model

    // -------------------------------------------------------------------------
    // a single device integer to count how many points survived filtering
    // -------------------------------------------------------------------------
    thrust::device_vector<unsigned int> d_out_count;

public:
    CudaSegmentation(segParam_t& params);
    ~CudaSegmentation() = default;
    
    void segment(float *inputData,
                 unsigned int nCount,
                 float *out_points,
                 unsigned int *out_num_points,
                 cudaStream_t stream) override;
};
