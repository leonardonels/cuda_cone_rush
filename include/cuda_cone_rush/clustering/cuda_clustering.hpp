#pragma once 

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <chrono>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "cuda_cone_rush/clustering/iclustering.hpp"
#include "cuda_cone_rush/utils/cached_allocator.hpp"

typedef struct {
  unsigned int minClusterSize;
  unsigned int maxClusterSize;
  float voxelX;
  float voxelY;
  float voxelZ;
  int countThreshold;
} extractClusterParam_t;

struct clustering_parameters
{
    struct clustering
    {
        float voxelX, voxelY, voxelZ;
        unsigned int countThreshold, minClusterSize, maxClusterSize;
    } clustering;

    cluster_filter filtering;
};


class CudaClustering : public IClustering
{
  private:
    extractClusterParam_t ecp;
    cluster_filter filterParams;
    cudaStream_t stream = NULL;
    CachedAllocator alloc;  // reusable memory pool for thrust temp buffers

    // -------------------------------------------------------------------------
    // Device vectors — all intermediate data stays on GPU
    // -------------------------------------------------------------------------
    // bounding box + grid dims (written by kernels, read by kernels)
    thrust::device_vector<float> d_bbox;        // [minX, minY, minZ, maxX, maxY, maxZ]
    thrust::device_vector<int>   d_grid;        // [gridX, gridY]

    // pipeline counts stored on device to avoid mid-pipeline D→H syncs
    // layout: [numVoxels, numFiltered, numClusters, numCones]
    int*         d_counts = nullptr;         // pinned host memory (4 ints)
    int*         d_countsDevice = nullptr;   // device-side counts

    // voxelization
    thrust::device_vector<int>          d_voxelKeys;     // voxel hash per point (sorted in-place)
    thrust::device_vector<int>          d_origKeys;      // original per-point voxel keys (before sort)
    thrust::device_vector<int>          d_uniqueKeys;    // unique voxel hashes
    thrust::device_vector<unsigned int> d_voxelCounts;   // points per voxel
    thrust::device_vector<int>          d_filteredKeys;  // voxels surviving countThreshold

    // union-find clustering
    thrust::device_vector<int>          d_parent;        // union-find parent array

    // per-point cluster labels
    thrust::device_vector<int>          d_pointLabels;   // cluster root label per point

    // per-cluster data
    thrust::device_vector<int>          d_uniqueLabels;  // unique root labels
    thrust::device_vector<int>          d_labelMap;      // root → compact cluster id
    thrust::device_vector<float>        d_clusterBBox;   // [6 * numClusters]
    thrust::device_vector<unsigned int> d_clusterSizes;  // points per cluster

    // output cones
    thrust::device_vector<float>        d_conePoints;    // [3 * maxCones]

    double totalTime = 0.0;
    unsigned int iterations = 0;

    int boundingBoxKernel_block_size = 768; // default block size for bounding box kernel
    int computeVoxelKeysKernel_block_size = 768; // default block size for computeVoxelKeys kernel
    int unionFindKernel_block_size = 768; // default block size for unionFind kernel
    int flattenParentKernel_block_size = 768; // default block size for flattenParent kernel
    int assignClusterLabelsKernel_block_size = 768; // default block size for assignClusterLabels kernel
    int clusterBBoxKernel_block_size = 768; // default block size for clusterBBox kernel
    int dimensionFilterKernel_block_size = 768; // default block size for dimensionFilter kernel

  public:
    CudaClustering(clustering_parameters& param);
    ~CudaClustering();

    void extractClusters(float* input, unsigned int inputSize, float* outputEC,
                         std::shared_ptr<visualization_msgs::msg::Marker> cones,
                         cudaStream_t stream) override;
};