#pragma once 
#include "cuda_clustering/filtering/ifiltering.hpp"
#include <ostream>
#include <iostream>
#include <chrono>
#include <rclcpp/rclcpp.hpp>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

class CudaFilter : public IFilter
{
    private:
        cudaStream_t stream = NULL;

        float upLimitX, downLimitX;
        float upLimitY, downLimitY;
        float upLimitZ, downLimitZ;
        bool filterX, filterY, filterZ;

        int passthroughFilterKernel_block_size = 768; // default block size

        // thrust device vectors replace cudaMallocManaged temp buffers
        thrust::device_vector<float> d_temp;
        thrust::device_vector<unsigned int> d_count;

        double totalTime = 0.0;
        unsigned int iterations = 0;
        
    public:
        CudaFilter(float upFilterLimitsX, float downFilterLimitsX,
                    float upFilterLimitsY, float downFilterLimitsY,
                    float upFilterLimitsZ, float downFilterLimitsZ);
        ~CudaFilter() = default;
        void filterPoints(float* input, unsigned int inputSize, float** output, unsigned int* outputSize, cudaStream_t stream) override;
};