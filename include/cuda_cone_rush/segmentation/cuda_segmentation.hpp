#pragma once

#include "isegmentation.hpp"
#include "cuda_cone_rush/utils/cached_allocator.hpp"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

typedef enum
{
    SACMODEL_PLANE = 0,
    SACMODEL_LINE,
    SACMODEL_CIRCLE2D,
    SACMODEL_CIRCLE3D,
    SACMODEL_SPHERE,
    SACMODEL_CYLINDER,
    SACMODEL_CONE,
    SACMODEL_TORUS,
    SACMODEL_PARALLEL_LINE,
    SACMODEL_PERPENDICULAR_PLANE,
    SACMODEL_PARALLEL_LINES,
    SACMODEL_NORMAL_PLANE,
    SACMODEL_NORMAL_SPHERE,
    SACMODEL_REGISTRATION,
    SACMODEL_REGISTRATION_2D,
    SACMODEL_PARALLEL_PLANE,
    SACMODEL_NORMAL_PARALLEL_PLANE,
    SACMODEL_STICK,
} SacModel;

typedef enum
{
    SAC_RANSAC = 0,
    SAC_LMEDS = 1,
    SAC_MSAC = 2,
    SAC_RRANSAC = 3,
    SAC_RMSAC = 4,
    SAC_MLESAC = 5,
    SAC_PROSAC = 6,
} SacMethod;

typedef struct
{
    double distanceThreshold;
    int maxIterations;
    double probability;
    double maxSegmentationDistance; // new parameter to limit how far points can be from the sensor for plane fitting
    bool optimizeCoefficients;
} segParam_t;

class cudaSegmentation
{
public:
    // Now Just support: SAC_RANSAC + SACMODEL_PLANE
    cudaSegmentation(int ModelType, int MethodType, cudaStream_t stream = 0);

    /*
    Input:
        cloud_in: data pointer for points cloud
        nCount: count of points in cloud_in
    Output:
        Index: data pointer which has the index of points in a plane from input
        modelCoefficients: data pointer which has the group of coefficients of the plane
    */
    int set(segParam_t param);
    void segment(float *cloud_in, int nCount,
                 int *index, float *modelCoefficients);

private:
};

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