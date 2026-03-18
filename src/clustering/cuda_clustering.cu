#include "cuda_cone_rush/clustering/cuda_clustering.hpp"
#include "cuda_cone_rush/utils/cached_allocator.hpp"

#include <iostream>
#include <algorithm>
#include <vector>

#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/pair.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

// ==========================================================================
//  KERNEL 1:  Bounding box reduction (GPU — no D→H copy needed)
// ==========================================================================
//  each block reduces its chunk to local min/max via shared memory, then
//  commits the block result to global output using atomicMinFloat/MaxFloat.
//
//  atomicMinFloat/MaxFloat perform the min/max comparison in float space
//  (via fminf/fmaxf) before storing the result as integer bits via CAS.
// ==========================================================================
__device__ inline void atomicMinFloat(float* addr, float val)
{
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int val_as_int   = __float_as_int(val);
    int old          = *addr_as_int, assumed;

    do {
        assumed = old;
        float current = __int_as_float(assumed);
        float newval  = fminf(val, current);
        old = atomicCAS(addr_as_int, assumed, __float_as_int(newval));
    } while (assumed != old);
}

__device__ inline void atomicMaxFloat(float* addr, float val)
{
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int old          = *addr_as_int, assumed;

    do {
        assumed = old;
        float current = __int_as_float(assumed);
        float newval  = fmaxf(val, current);
        old = atomicCAS(addr_as_int, assumed, __float_as_int(newval));
    } while (assumed != old);
}

// d_bbox layout: [minX, minY, minZ, maxX, maxY, maxZ]
__global__ void boundingBoxKernel(
    const float* __restrict__ points,
    unsigned int nPoints,
    float* __restrict__ d_bbox)
{
    extern __shared__ float s_data[];
    
    float* s_minX = &s_data[0];
    float* s_minY = &s_data[blockDim.x];
    float* s_minZ = &s_data[2 * blockDim.x];
    float* s_maxX = &s_data[3 * blockDim.x];
    float* s_maxY = &s_data[4 * blockDim.x];
    float* s_maxZ = &s_data[5 * blockDim.x];

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float lminX = 1e30f, lminY = 1e30f, lminZ = 1e30f;
    float lmaxX = -1e30f, lmaxY = -1e30f, lmaxZ = -1e30f;

    // grid-stride loop
    for (unsigned int i = gid; i < nPoints; i += blockDim.x * gridDim.x) {
        float x = points[i * 4 + 0];
        float y = points[i * 4 + 1];
        float z = points[i * 4 + 2];
        lminX = fminf(lminX, x); lmaxX = fmaxf(lmaxX, x);
        lminY = fminf(lminY, y); lmaxY = fmaxf(lmaxY, y);
        lminZ = fminf(lminZ, z); lmaxZ = fmaxf(lmaxZ, z);
    }

    s_minX[tid] = lminX; s_minY[tid] = lminY; s_minZ[tid] = lminZ;
    s_maxX[tid] = lmaxX; s_maxY[tid] = lmaxY; s_maxZ[tid] = lmaxZ;
    __syncthreads();

    // shared memory reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_minX[tid] = fminf(s_minX[tid], s_minX[tid + stride]);
            s_minY[tid] = fminf(s_minY[tid], s_minY[tid + stride]);
            s_minZ[tid] = fminf(s_minZ[tid], s_minZ[tid + stride]);
            s_maxX[tid] = fmaxf(s_maxX[tid], s_maxX[tid + stride]);
            s_maxY[tid] = fmaxf(s_maxY[tid], s_maxY[tid + stride]);
            s_maxZ[tid] = fmaxf(s_maxZ[tid], s_maxZ[tid + stride]);
        }
        __syncthreads();
    }

    // block winner atomically updates global bbox
    if (tid == 0) {
        atomicMinFloat(&d_bbox[0], s_minX[0]);
        atomicMinFloat(&d_bbox[1], s_minY[0]);
        atomicMinFloat(&d_bbox[2], s_minZ[0]);
        atomicMaxFloat(&d_bbox[3], s_maxX[0]);
        atomicMaxFloat(&d_bbox[4], s_maxY[0]);
        atomicMaxFloat(&d_bbox[5], s_maxZ[0]);
    }
}

// ==========================================================================
//  KERNEL 2:  Compute a voxel hash for every point
// ==========================================================================
//  reads bounding box from device pointer (output of boundingBoxKernel)
//  Hash = ix + iy * GRID + iz * GRID * GRID
// ==========================================================================
__global__ void computeVoxelKeysKernel(
    const float* __restrict__ points,
    int* __restrict__ keys,
    unsigned int nPoints,
    float voxelX, float voxelY, float voxelZ,
    const float* __restrict__ d_bbox,  // [minX, minY, minZ, maxX, maxY, maxZ]
    int* __restrict__ d_grid)          // output: [gridX, gridY] computed by thread 0
{
    // thread 0 computes grid dims from device-side bbox and writes to d_grid
    __shared__ float s_bbox[6];
    __shared__ int s_grid[2];

    if (threadIdx.x < 6) s_bbox[threadIdx.x] = d_bbox[threadIdx.x];
    __syncthreads();

    float minX = s_bbox[0], minY = s_bbox[1], minZ = s_bbox[2];

    if (threadIdx.x == 0) {
        float maxX = s_bbox[3], maxY = s_bbox[4];
        s_grid[0] = __float2int_ru((maxX - minX) / voxelX) + 1;
        s_grid[1] = __float2int_ru((maxY - minY) / voxelY) + 1;
        // also store to global so subsequent kernels can read gridX/gridY
        d_grid[0] = s_grid[0];
        d_grid[1] = s_grid[1];
    }
    __syncthreads();

    int gridX = s_grid[0];
    int gridY = s_grid[1];

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nPoints) return;

    float x = points[tid * 4 + 0];
    float y = points[tid * 4 + 1];
    float z = points[tid * 4 + 2];

    int ix = __float2int_rd((x - minX) / voxelX);
    int iy = __float2int_rd((y - minY) / voxelY);
    int iz = __float2int_rd((z - minZ) / voxelZ);

    keys[tid] = ix + iy * gridX + iz * gridX * gridY;
}

// ==========================================================================
//  Helper:  union-find with path compression (device)
// ==========================================================================
__device__ int uf_find(int* parent, int i)
{
    while (parent[i] != i) {
        parent[i] = parent[parent[i]];   // path splitting
        i = parent[i];
    }
    return i;
}

__device__ void uf_union(int* parent, int a, int b)
{
    while (true) {
        a = uf_find(parent, a);
        b = uf_find(parent, b);
        if (a == b) return;
        // Smaller root becomes child — deterministic to avoid races
        if (a > b) { int tmp = a; a = b; b = tmp; }
        int old = atomicCAS(&parent[b], b, a);
        if (old == b) return;   // success
        // Retry with updated roots
    }
}

// ==========================================================================
//  KERNEL 3:  Union-find on voxel grid — 26-connectivity
// ==========================================================================
//  reads gridX/gridY from device pointer (computed by computeVoxelKeysKernel)
// ==========================================================================
__global__ void unionFindKernel(
    const int* __restrict__ uniqueKeys,   // sorted unique voxel hashes
    int  numVoxels,
    int* __restrict__ parent,
    const int* __restrict__ d_grid)       // [gridX, gridY]
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (unsigned)numVoxels) return;

    int gridX = d_grid[0];
    int gridY = d_grid[1];

    int myKey = uniqueKeys[tid];
    int iz = myKey / (gridX * gridY);
    int rem = myKey % (gridX * gridY);
    int iy = rem / gridX;
    int ix = rem % gridX;

    // 26-connected neighbours
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                int nx = ix + dx;
                int ny = iy + dy;
                int nz = iz + dz;
                if (nx < 0 || ny < 0 || nz < 0) continue;

                int nKey = nx + ny * gridX + nz * gridX * gridY;

                // Binary-search for nKey in uniqueKeys
                int lo = 0, hi = numVoxels - 1;
                int found = -1;
                while (lo <= hi) {
                    int mid = (lo + hi) / 2;
                    int mk  = uniqueKeys[mid];
                    if (mk == nKey) { found = mid; break; }
                    else if (mk < nKey) lo = mid + 1;
                    else                hi = mid - 1;
                }
                if (found >= 0) {
                    uf_union(parent, (int)tid, found);
                }
            }
        }
    }
}

// ==========================================================================
//  KERNEL 4:  Flatten parent array (path compress to root)
// ==========================================================================
__global__ void flattenParentKernel(int* parent, int n)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (unsigned)n) return;
    parent[tid] = uf_find(parent, tid);
}

// ==========================================================================
//  KERNEL 5:  Assign cluster label to every point via sorted-key lookup
// ==========================================================================
//  each point already has a voxel key in d_voxelKeys.
//  we binary-search in the *filtered* unique keys to find its voxel index,
//  then read the flattened parent (= cluster root label).
//  Points whose voxel was filtered out get label -1.
// ==========================================================================
__global__ void assignClusterLabelsKernel(
    const int* __restrict__ pointKeys,      // voxel key per original point
    unsigned int nPoints,
    const int* __restrict__ filteredKeys,    // sorted filtered unique voxel hashes
    int numFiltered,
    const int* __restrict__ parent,          // flattened parent array (indexed by filtered-voxel-idx)
    int* __restrict__ pointLabels)           // output: cluster root label per point (-1 if filtered)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nPoints) return;

    int key = pointKeys[tid];

    // binary search in filteredKeys
    int lo = 0, hi = numFiltered - 1;
    int found = -1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int mk = filteredKeys[mid];
        if (mk == key) { found = mid; break; }
        else if (mk < key) lo = mid + 1;
        else               hi = mid - 1;
    }

    pointLabels[tid] = (found >= 0) ? parent[found] : -1;
}

// ==========================================================================
//  KERNEL 6:  Per-cluster bounding box via atomics
// ==========================================================================
//  each point contributes to its cluster's bounding box.
//  d_clusterBBox layout per cluster: [minX, minY, minZ, maxX, maxY, maxZ]
//  d_clusterSizes: atomically counted per cluster
// ==========================================================================
__global__ void clusterBBoxKernel(
    const float* __restrict__ points,
    const int* __restrict__ pointLabels,    // cluster root label per point
    const int* __restrict__ labelMap,        // root_label → compact cluster id
    int numLabels,                           // number of unique labels (for binary search in labelMap)
    const int* __restrict__ labelKeys,       // sorted unique root labels
    unsigned int nPoints,
    float* __restrict__ d_clusterBBox,       // [6 * numClusters]
    unsigned int* __restrict__ d_clusterSizes)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nPoints) return;

    int label = pointLabels[tid];
    if (label < 0) return;

    // binary search for label in labelKeys to get compact cluster id
    int lo = 0, hi = numLabels - 1;
    int cid = -1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int mk = labelKeys[mid];
        if (mk == label) { cid = labelMap[mid]; break; }
        else if (mk < label) lo = mid + 1;
        else                  hi = mid - 1;
    }
    if (cid < 0) return;

    float x = points[tid * 4 + 0];
    float y = points[tid * 4 + 1];
    float z = points[tid * 4 + 2];

    atomicMinFloat(&d_clusterBBox[cid * 6 + 0], x);
    atomicMinFloat(&d_clusterBBox[cid * 6 + 1], y);
    atomicMinFloat(&d_clusterBBox[cid * 6 + 2], z);
    atomicMaxFloat(&d_clusterBBox[cid * 6 + 3], x);
    atomicMaxFloat(&d_clusterBBox[cid * 6 + 4], y);
    atomicMaxFloat(&d_clusterBBox[cid * 6 + 5], z);

    atomicAdd(&d_clusterSizes[cid], 1u);
}

// ==========================================================================
//  KERNEL 7:  Dimension filter — check each cluster's bbox on GPU
// ==========================================================================
//  produces a compacted list of cone center points (x,y,z) for valid clusters.
//  d_numCones is part of d_countsDevice[3]
// ==========================================================================
__global__ void dimensionFilterKernel(
    const float* __restrict__ d_clusterBBox,   // [6 * numClusters]
    const unsigned int* __restrict__ d_clusterSizes,
    int numClusters,
    unsigned int minClusterSize,
    unsigned int maxClusterSize,
    float filterMinX, float filterMinY, float filterMinZ,
    float filterMaxX, float filterMaxY, float filterMaxZ,
    float maxHeight,
    float* __restrict__ d_conePoints,          // output: [3 * numClusters] max
    int* __restrict__ d_numCones)              // output: device counter (int)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (unsigned)numClusters) return;

    unsigned int sz = d_clusterSizes[tid];
    if (sz < minClusterSize || sz > maxClusterSize) return;

    float mnX = d_clusterBBox[tid * 6 + 0];
    float mnY = d_clusterBBox[tid * 6 + 1];
    float mnZ = d_clusterBBox[tid * 6 + 2];
    float mxX = d_clusterBBox[tid * 6 + 3];
    float mxY = d_clusterBBox[tid * 6 + 4];
    float mxZ = d_clusterBBox[tid * 6 + 5];

    float dx = mxX - mnX;
    float dy = mxY - mnY;
    float dz = mxZ - mnZ;

    // isCone check
    if (mnZ < maxHeight &&
        dx < filterMaxX && dy < filterMaxY && dz < filterMaxZ &&
        dx > filterMinX && dy > filterMinY && dz > filterMinZ)
    {
        int idx = atomicAdd(d_numCones, 1);
        d_conePoints[idx * 3 + 0] = (mxX + mnX) * 0.5f;
        d_conePoints[idx * 3 + 1] = (mxY + mnY) * 0.5f;
        d_conePoints[idx * 3 + 2] = (mxZ + mnZ) * 0.5f;
    }
}

// ==========================================================================
//  CudaClustering implementation
// ==========================================================================
CudaClustering::CudaClustering(clustering_parameters& param)
{
    ecp.minClusterSize  = param.clustering.minClusterSize;
    ecp.maxClusterSize  = param.clustering.maxClusterSize;
    ecp.voxelX          = param.clustering.voxelX;
    ecp.voxelY          = param.clustering.voxelY;
    ecp.voxelZ          = param.clustering.voxelZ;
    ecp.countThreshold  = param.clustering.countThreshold;

    filterParams = param.filtering;

    // pre-allocate small fixed-size device buffers
    d_bbox.resize(6);       // [minX, minY, minZ, maxX, maxY, maxZ]
    d_grid.resize(2);       // [gridX, gridY]

    // pinned host memory for async D→H count transfer (avoids sync stalls)
    // layout: [numVoxels, numFiltered, numClusters, numCones]
    cudaMallocHost(&d_counts, 4 * sizeof(int));

    // device-side counters written by thrust/kernels
    cudaMalloc(&d_countsDevice, 4 * sizeof(int));

    #ifdef ENABLE_VERBOSE
    // ---------------------------------------------------------------------------------------------------------------
    // TODO: optmize block usage of each kernel based on occupancy results, instead of hardcoding 768 threads for all
    // ---------------------------------------------------------------------------------------------------------------
        RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "---cluster kernels info---");
        int minGridSize = 0;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &this->boundingBoxKernel_block_size, boundingBoxKernel, 0, 0);
        RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "  CUDA Occupancy of boundingBoxKernel Max Potential Block Size: %d", this->boundingBoxKernel_block_size);
        
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &this->computeVoxelKeysKernel_block_size, computeVoxelKeysKernel, 0, 0);
        RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "  CUDA Occupancy of computeVoxelKeysKernel Max Potential Block Size: %d", this->computeVoxelKeysKernel_block_size);
        
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &this->unionFindKernel_block_size, unionFindKernel, 0, 0);
        RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "  CUDA Occupancy of unionFindKernel Max Potential Block Size: %d", this->unionFindKernel_block_size);
        
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &this->flattenParentKernel_block_size, flattenParentKernel, 0, 0);
        RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "  CUDA Occupancy of flattenParentKernel Max Potential Block Size: %d", this->flattenParentKernel_block_size);
        
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &this->assignClusterLabelsKernel_block_size, assignClusterLabelsKernel, 0, 0);
        RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "  CUDA Occupancy of assignClusterLabelsKernel Max Potential Block Size: %d", this->assignClusterLabelsKernel_block_size);
        
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &this->clusterBBoxKernel_block_size, clusterBBoxKernel, 0, 0);
        RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "  CUDA Occupancy of clusterBBoxKernel Max Potential Block Size: %d", this->clusterBBoxKernel_block_size);

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &this->dimensionFilterKernel_block_size, dimensionFilterKernel, 0, 0);
        RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "  CUDA Occupancy of dimensionFilterKernel Max Potential Block Size: %d", this->dimensionFilterKernel_block_size);
        #endif
}

CudaClustering::~CudaClustering()
{
    if (d_counts) cudaFreeHost(d_counts);
    if (d_countsDevice) cudaFree(d_countsDevice);
}

// --------------------------------------------------------------------------
//  extractClusters — full GPU pipeline (single D→H sync at the end)
// --------------------------------------------------------------------------
//  1. Bounding box                              (GPU kernel)
//  2. Compute voxel hash per point              (GPU kernel, reads bbox from device)
//  3. Sort points by voxel hash                 (GPU — thrust)
//  4. Reduce to unique voxels + per-voxel counts(GPU — thrust)
//  5. Filter voxels by countThreshold           (GPU — thrust::copy_if)
//  6. Union-find 26-connectivity                (GPU kernel)
//  7. Flatten parent labels                     (GPU kernel)
//  8. Assign cluster label per point            (GPU kernel)
//  9. Per-cluster bounding box via atomics       (GPU kernel)
//  10. Dimension filter → cone points           (GPU kernel)
//  11. Copy cone points D→H (tiny)             (single sync)
// --------------------------------------------------------------------------
void CudaClustering::extractClusters(
    float* input,             // device pointer  (x,y,z,i)*N
    unsigned int inputSize,
    float* /*outputEC*/,      // device pointer  (unused in this pipeline)
    std::shared_ptr<visualization_msgs::msg::Marker> cones,
    cudaStream_t stream)
{
    #ifdef ENABLE_VERBOSE
        auto t1 = std::chrono::steady_clock::now();
    #endif
    
    if (inputSize < ecp.minClusterSize) {
        RCLCPP_WARN(rclcpp::get_logger("cuda_cone_rush_node"),
                     "Not enough points for clustering (%u < %u)", inputSize, ecp.minClusterSize);
        return;
    }

    int blocks;

    // ------------------------------------------------------------------
    // Pre-size all vectors to inputSize once (no resizing mid-pipeline)
    // ------------------------------------------------------------------
    if (d_voxelKeys.capacity() < inputSize) {
        d_voxelKeys.reserve(inputSize);
        d_origKeys.reserve(inputSize);
        d_uniqueKeys.reserve(inputSize);
        d_voxelCounts.reserve(inputSize);
        d_filteredKeys.reserve(inputSize);
        d_parent.reserve(inputSize);
        d_pointLabels.reserve(inputSize);
        d_uniqueLabels.reserve(inputSize);
        d_labelMap.reserve(inputSize);
        d_clusterBBox.reserve(inputSize * 6);
        d_clusterSizes.reserve(inputSize);
        d_conePoints.reserve(inputSize * 3);
    }
    d_voxelKeys.resize(inputSize);
    d_origKeys.resize(inputSize);
    d_uniqueKeys.resize(inputSize);
    d_voxelCounts.resize(inputSize);
    d_pointLabels.resize(inputSize);

    // zero the device counter buffer: [numVoxels, numFiltered, numClusters, numCones]
    cudaMemsetAsync(d_countsDevice, 0, 4 * sizeof(int), stream);

    // ------------------------------------------------------------------
    // 1. Bounding box on GPU (no D→H copy)
    // ------------------------------------------------------------------
    float bbox_init[6] = {1e30f, 1e30f, 1e30f, -1e30f, -1e30f, -1e30f};
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_bbox.data()), bbox_init,
                    6 * sizeof(float), cudaMemcpyHostToDevice, stream);

    int maxThreads = this->boundingBoxKernel_block_size > 0 ? this->boundingBoxKernel_block_size : 256;
    size_t sharedMemBytes = 6 * maxThreads * sizeof(float);
    blocks = (inputSize + maxThreads - 1) / maxThreads;

    boundingBoxKernel<<<blocks, maxThreads, sharedMemBytes, stream>>>(
        input, inputSize, thrust::raw_pointer_cast(d_bbox.data()));

    // ------------------------------------------------------------------
    // 2. Compute voxel hash per point
    // ------------------------------------------------------------------
    blocks = (inputSize + computeVoxelKeysKernel_block_size - 1) / computeVoxelKeysKernel_block_size;
    computeVoxelKeysKernel<<<blocks, computeVoxelKeysKernel_block_size, 0, stream>>>(
        input, thrust::raw_pointer_cast(d_voxelKeys.data()), inputSize,
        ecp.voxelX, ecp.voxelY, ecp.voxelZ,
        thrust::raw_pointer_cast(d_bbox.data()),
        thrust::raw_pointer_cast(d_grid.data()));

    // ------------------------------------------------------------------
    // 3. Save original per-point keys, then sort
    // ------------------------------------------------------------------
    thrust::copy(thrust::cuda::par(alloc).on(stream),
                 d_voxelKeys.begin(), d_voxelKeys.end(),
                 d_origKeys.begin());

    thrust::sort(thrust::cuda::par(alloc).on(stream),
                 d_voxelKeys.begin(), d_voxelKeys.end());

    // ------------------------------------------------------------------
    // 4. Reduce to unique voxels + counts — sync to get numVoxels
    // ------------------------------------------------------------------
    auto new_end = thrust::reduce_by_key(
        thrust::cuda::par(alloc).on(stream),
        d_voxelKeys.begin(), d_voxelKeys.end(),
        thrust::make_constant_iterator(1u),
        d_uniqueKeys.begin(),
        d_voxelCounts.begin());

    int numVoxels = (int)(new_end.first - d_uniqueKeys.begin());

    // ------------------------------------------------------------------
    // 5. Filter voxels by countThreshold — sync to get numFiltered
    // ------------------------------------------------------------------
    d_filteredKeys.resize(numVoxels);

    int countThresh = ecp.countThreshold;
    auto filt_end = thrust::copy_if(
        thrust::cuda::par(alloc).on(stream),
        d_uniqueKeys.begin(), d_uniqueKeys.begin() + numVoxels,
        d_voxelCounts.begin(),  // stencil
        d_filteredKeys.begin(),
        [countThresh] __device__ (unsigned int c) { return (int)c >= countThresh; });

    int numFiltered = (int)(filt_end - d_filteredKeys.begin());

    // ------------------------------------------------------------------
    // 6. Union-find on 26-connected voxel grid
    // ------------------------------------------------------------------
    d_parent.resize(numFiltered);
    thrust::sequence(thrust::cuda::par(alloc).on(stream),
                     d_parent.begin(), d_parent.begin() + numFiltered);

    blocks = (numFiltered + unionFindKernel_block_size - 1) / unionFindKernel_block_size;
    unionFindKernel<<<blocks, unionFindKernel_block_size, 0, stream>>>(
        thrust::raw_pointer_cast(d_filteredKeys.data()),
        numFiltered,
        thrust::raw_pointer_cast(d_parent.data()),
        thrust::raw_pointer_cast(d_grid.data()));

    // ------------------------------------------------------------------
    // 7. Flatten parent array
    // ------------------------------------------------------------------
    flattenParentKernel<<<blocks, flattenParentKernel_block_size, 0, stream>>>(
        thrust::raw_pointer_cast(d_parent.data()), numFiltered);

    // ------------------------------------------------------------------
    // 8. Assign cluster label per point
    // ------------------------------------------------------------------
    blocks = (inputSize + assignClusterLabelsKernel_block_size - 1) / assignClusterLabelsKernel_block_size;
    assignClusterLabelsKernel<<<blocks, assignClusterLabelsKernel_block_size, 0, stream>>>(
        thrust::raw_pointer_cast(d_origKeys.data()),
        inputSize,
        thrust::raw_pointer_cast(d_filteredKeys.data()),
        numFiltered,
        thrust::raw_pointer_cast(d_parent.data()),
        thrust::raw_pointer_cast(d_pointLabels.data()));

    // ------------------------------------------------------------------
    // 9. Build compact cluster IDs + per-cluster bbox — sync to get numClusters
    // ------------------------------------------------------------------
    d_uniqueLabels.resize(numFiltered);
    thrust::copy(thrust::cuda::par(alloc).on(stream),
                 d_parent.begin(), d_parent.begin() + numFiltered,
                 d_uniqueLabels.begin());
    thrust::sort(thrust::cuda::par(alloc).on(stream),
                 d_uniqueLabels.begin(), d_uniqueLabels.begin() + numFiltered);
    auto ul_end = thrust::unique(thrust::cuda::par(alloc).on(stream),
                                  d_uniqueLabels.begin(), d_uniqueLabels.begin() + numFiltered);
    int numClusters = (int)(ul_end - d_uniqueLabels.begin());

    // build label → compact ID map (sequential 0..numClusters-1)
    d_labelMap.resize(numClusters);
    thrust::sequence(thrust::cuda::par(alloc).on(stream),
                     d_labelMap.begin(), d_labelMap.begin() + numClusters);

    // init per-cluster bbox + sizes
    d_clusterBBox.resize(numClusters * 6);
    d_clusterSizes.resize(numClusters);

    thrust::fill(thrust::cuda::par(alloc).on(stream),
                 d_clusterSizes.begin(), d_clusterSizes.begin() + numClusters, 0u);
    {
        float* raw_bbox = thrust::raw_pointer_cast(d_clusterBBox.data());
        thrust::for_each(thrust::cuda::par(alloc).on(stream),
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(numClusters * 6),
            [raw_bbox] __device__ (int i) {
                int channel = i % 6;
                raw_bbox[i] = (channel < 3) ? 1e30f : -1e30f;
            });
    }

    // per-cluster bbox accumulation
    blocks = (inputSize + clusterBBoxKernel_block_size - 1) / clusterBBoxKernel_block_size;
    clusterBBoxKernel<<<blocks, clusterBBoxKernel_block_size, 0, stream>>>(
        input,
        thrust::raw_pointer_cast(d_pointLabels.data()),
        thrust::raw_pointer_cast(d_labelMap.data()),
        numClusters,
        thrust::raw_pointer_cast(d_uniqueLabels.data()),
        inputSize,
        thrust::raw_pointer_cast(d_clusterBBox.data()),
        thrust::raw_pointer_cast(d_clusterSizes.data()));

    // ------------------------------------------------------------------
    // 10. Dimension filter on GPU → cone points
    // ------------------------------------------------------------------
    d_conePoints.resize(numClusters * 3);
    // zero the numCones counter (d_countsDevice[3])
    // cudaMemsetAsync(&d_countsDevice[3], 0, sizeof(int), stream);    // already done previously

    blocks = (numClusters + dimensionFilterKernel_block_size - 1) / dimensionFilterKernel_block_size;
    if (blocks == 0) blocks = 1;
    dimensionFilterKernel<<<blocks, dimensionFilterKernel_block_size, 0, stream>>>(
        thrust::raw_pointer_cast(d_clusterBBox.data()),
        thrust::raw_pointer_cast(d_clusterSizes.data()),
        numClusters,
        ecp.minClusterSize, ecp.maxClusterSize,
        filterParams.clusterMinX, filterParams.clusterMinY, filterParams.clusterMinZ,
        filterParams.clusterMaxX, filterParams.clusterMaxY, filterParams.clusterMaxZ,
        filterParams.maxHeight,
        thrust::raw_pointer_cast(d_conePoints.data()),
        &d_countsDevice[3]);

    // ------------------------------------------------------------------
    // 11. SINGLE D→H sync: copy numCones to pinned host memory
    // ------------------------------------------------------------------
    cudaMemcpyAsync(&d_counts[3], &d_countsDevice[3],
                    sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int numCones = d_counts[3];

    if (numCones > 0) {
        std::vector<float> h_cones(numCones * 3);
        cudaMemcpyAsync(h_cones.data(), thrust::raw_pointer_cast(d_conePoints.data()),
                        numCones * 3 * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        for (int i = 0; i < numCones; ++i) {
            geometry_msgs::msg::Point pnt;
            pnt.x = h_cones[i * 3 + 0];
            pnt.y = h_cones[i * 3 + 1];
            pnt.z = h_cones[i * 3 + 2];
            cones->points.push_back(pnt);
        }
    }

    #ifdef ENABLE_VERBOSE
        auto t2 = std::chrono::steady_clock::now();
        auto total_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t2 - t1).count();
        std::cout << "Clustering time: " << total_ms << " ms" << std::endl;
    #endif
}
