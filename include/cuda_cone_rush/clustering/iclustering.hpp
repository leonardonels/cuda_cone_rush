#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <visualization_msgs/msg/marker.hpp>

#include "cuda_cone_rush/clustering/cluster_filtering/icluster_filtering.hpp"

class IClustering 
{
    public:
        virtual ~IClustering() = default;
        virtual void extractClusters(float* input, unsigned int inputSize, float* outputEC, std::shared_ptr<visualization_msgs::msg::Marker> cones, cudaStream_t stream = 0) = 0;
};