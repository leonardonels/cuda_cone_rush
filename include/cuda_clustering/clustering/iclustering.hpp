#include <cuda_runtime.h>
#pragma once 
#include "cuda_clustering/clustering/cluster_filtering/icluster_filtering.hpp"

class IClustering 
{
    public:
        virtual ~IClustering() = default;
        virtual void extractClusters(float* input, unsigned int inputSize, float* outputEC, std::shared_ptr<visualization_msgs::msg::Marker> cones, cudaStream_t stream = 0) = 0;
};