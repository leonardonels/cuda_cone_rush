#include "cuda_cone_rush/segmentation/cuda_ring_segmentation.hpp"

#include <chrono>
#include <algorithm>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>

// ==========================================================================
//  Note: This implementation assumes Y axis is forward, X axis is right, and Z axis is up (ROS standard).
//        But it is not: X is forward, Y is left, Z is up (Velodyne standard).
// ==========================================================================  

// ==========================================================================
//  Constant memory: ring group boundaries (uploaded once in constructor)
// ==========================================================================
static __constant__ int c_ring_bounds[65];   // max 64 groups → 65 boundaries
static __constant__ int c_num_ring_groups;

// ==========================================================================
//  Device helpers
// ==========================================================================
__device__ inline int ringToGroup(int ring_id) {
    for (int g = 0; g < c_num_ring_groups; g++) {
        if (ring_id < c_ring_bounds[g + 1]) return g;
    }
    return c_num_ring_groups - 1;  // fallback for out-of-range
}

__device__ inline unsigned int wang_hash_ring(unsigned int seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

// ==========================================================================
//  Phase 1a: Classify each point into a patch and count per-patch totals
// ==========================================================================
//  patch_id = ring_group * 2 + side
//    even patch_id → right (x >= 0)
//    odd  patch_id → left  (x <  0)
// ==========================================================================
__global__ void countPatchKernel(
    const float* __restrict__ points,
    const float* __restrict__ rings,
    int N,
    int* __restrict__ patch_of,
    int* __restrict__ patch_counts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int ring_id = __float2int_rn(rings[i]);
    int rg   = ringToGroup(ring_id);
    int side = (points[i * 4] >= 0.0f) ? 0 : 1;
    int patch = rg * 2 + side;

    patch_of[i] = patch;
    atomicAdd(&patch_counts[patch], 1);
}

// ==========================================================================
//  Phase 1b: Scatter point indices into contiguous per-patch segments
// ==========================================================================
__global__ void scatterIndicesKernel(
    const int* __restrict__ patch_of,
    const int* __restrict__ patch_offsets,
    int* __restrict__ patch_write,
    int* __restrict__ indices,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int patch = patch_of[i];
    int slot  = atomicAdd(&patch_write[patch], 1);
    indices[patch_offsets[patch] + slot] = i;
}

// ==========================================================================
//  Phase 2: Patched RANSAC — one block per (patch, iteration)
//    Grid:  num_patches * iters_per_patch  blocks
//    Block: 256 threads (for inlier counting + reduction)
// ==========================================================================
__global__ void ransacPatchedKernel(
    const float* __restrict__ points,
    const int*   __restrict__ indices,
    const int*   __restrict__ patch_offsets,
    const int*   __restrict__ patch_counts,
    int num_patches,
    int iters_per_patch,
    float threshold,
    float max_seg_dist,
    int*    __restrict__ ransac_counts,
    float4* __restrict__ ransac_planes,
    unsigned int seed)
{
    int global_iter = blockIdx.x;
    int patch_id    = global_iter / iters_per_patch;
    int local_iter  = global_iter % iters_per_patch;

    if (patch_id >= num_patches) return;

    int offset = patch_offsets[patch_id];
    int count  = patch_counts[patch_id];
    int result_idx = patch_id * iters_per_patch + local_iter;

    if (count < 3) {
        if (threadIdx.x == 0) ransac_counts[result_idx] = -1;
        return;
    }

    // --- sample 3 random points from this patch (O(1) via index lookup) ---
    unsigned int s = seed + global_iter * 199999u;
    int idx1 = indices[offset + wang_hash_ring(s)     % count];
    int idx2 = indices[offset + wang_hash_ring(s + 1) % count];
    int idx3 = indices[offset + wang_hash_ring(s + 2) % count];

    float p1x = points[idx1 * 4], p1y = points[idx1 * 4 + 1], p1z = points[idx1 * 4 + 2];
    float p2x = points[idx2 * 4], p2y = points[idx2 * 4 + 1], p2z = points[idx2 * 4 + 2];
    float p3x = points[idx3 * 4], p3y = points[idx3 * 4 + 1], p3z = points[idx3 * 4 + 2];

    // optional: reject seed points too far from sensor
    if (p1x * p1x + p1y * p1y + p1z * p1z > max_seg_dist) {
        if (threadIdx.x == 0) ransac_counts[result_idx] = -1;
        return;
    }

    // --- fit plane (ax + by + cz + d = 0) ---
    float v1x = p2x - p1x, v1y = p2y - p1y, v1z = p2z - p1z;
    float v2x = p3x - p1x, v2y = p3y - p1y, v2z = p3z - p1z;

    float a = v1y * v2z - v1z * v2y;
    float b = v1z * v2x - v1x * v2z;
    float c = v1x * v2y - v1y * v2x;

    float norm = sqrtf(a * a + b * b + c * c);
    if (norm < 1e-6f) {
        if (threadIdx.x == 0) ransac_counts[result_idx] = -1;
        return;
    }

    float inv_norm = 1.0f / norm;
    a *= inv_norm;  b *= inv_norm;  c *= inv_norm;
    float d = -(a * p1x + b * p1y + c * p1z);

    // --- count inliers within this patch only ---
    int local_count = 0;
    for (int j = threadIdx.x; j < count; j += blockDim.x) {
        int pi = indices[offset + j];
        float x = points[pi * 4];
        float y = points[pi * 4 + 1];
        float z = points[pi * 4 + 2];
        float dist = fabsf(a * x + b * y + c * z + d);
        if (dist <= threshold) local_count++;
    }

    // --- block reduction ---
    __shared__ int s_counts[256];
    s_counts[threadIdx.x] = local_count;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s_counts[threadIdx.x] += s_counts[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        ransac_counts[result_idx] = s_counts[0];
        ransac_planes[result_idx] = make_float4(a, b, c, d);
    }
}

// ==========================================================================
//  Phase 3: Find best plane per patch (segmented argmax)
//    Grid:  num_patches blocks
//    Block: next-power-of-2 >= iters_per_patch (≤ 1024)
// ==========================================================================
__global__ void findBestPlaneKernel(
    const int*    __restrict__ ransac_counts,
    const float4* __restrict__ ransac_planes,
    int iters_per_patch,
    float4* __restrict__ best_planes)
{
    int patch = blockIdx.x;
    int base  = patch * iters_per_patch;
    int tid   = threadIdx.x;

    __shared__ int s_counts[1024];
    __shared__ int s_indices[1024];

    int my_count = -1;
    if (tid < iters_per_patch)
        my_count = ransac_counts[base + tid];

    s_counts[tid]  = my_count;
    s_indices[tid] = tid;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_counts[tid + stride] > s_counts[tid]) {
                s_counts[tid]  = s_counts[tid + stride];
                s_indices[tid] = s_indices[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (s_counts[0] > 0) {
            best_planes[patch] = ransac_planes[base + s_indices[0]];
        } else {
            // no valid plane → impossible-to-match plane (all points pass through)
            best_planes[patch] = make_float4(0.0f, 0.0f, 0.0f, 1e10f);
        }
    }
}

// ==========================================================================
//  Phase 4a: Mark inliers using each point's patch-local best plane
// ==========================================================================
__global__ void markInliersPatchedKernel(
    const float*  __restrict__ points,
    int N,
    const int*    __restrict__ patch_of,
    const float4* __restrict__ best_planes,
    float threshold,
    int* __restrict__ out_index)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 plane = best_planes[patch_of[i]];

    float x = points[i * 4];
    float y = points[i * 4 + 1];
    float z = points[i * 4 + 2];
    float dist = fabsf(plane.x * x + plane.y * y + plane.z * z + plane.w);

    out_index[i] = (dist <= threshold) ? 1 : 0;
}

// ==========================================================================
//  Phase 4b: Compact non-inlier points + ring values in a single pass
// ==========================================================================
__global__ void compactNonInliersWithRingKernel(
    const float* __restrict__ in_points,
    const float* __restrict__ in_rings,
    const int*   __restrict__ index,
    float*        __restrict__ out_points,
    float*        __restrict__ out_rings,
    unsigned int* __restrict__ d_count,
    int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    if (index[tid] != 1) {   // NOT ground
        unsigned int w = atomicAdd(d_count, 1);
        out_points[w * 4 + 0] = in_points[tid * 4 + 0];
        out_points[w * 4 + 1] = in_points[tid * 4 + 1];
        out_points[w * 4 + 2] = in_points[tid * 4 + 2];
        out_points[w * 4 + 3] = in_points[tid * 4 + 3];
        out_rings[w] = in_rings[tid];
    }
}

// ==========================================================================
//  Constructor
// ==========================================================================
CudaRingSegmentation::CudaRingSegmentation(segParam_t& params)
{
    segP_.distanceThreshold      = params.distanceThreshold;
    segP_.maxIterations          = params.maxIterations;
    segP_.probability            = params.probability;
    segP_.maxSegmentationDistance = params.maxSegmentationDistance
                                   * params.maxSegmentationDistance;
    segP_.ringGroupBoundaries    = params.ringGroupBoundaries;

    num_ring_groups_ = static_cast<int>(segP_.ringGroupBoundaries.size()) - 1;
    num_patches_     = num_ring_groups_ * 2;  // left + right per ring group

    // Upload ring group boundaries to constant memory
    cudaMemcpyToSymbol(c_ring_bounds, segP_.ringGroupBoundaries.data(),
                       segP_.ringGroupBoundaries.size() * sizeof(int));
    cudaMemcpyToSymbol(c_num_ring_groups, &num_ring_groups_, sizeof(int));

    // Pre-allocate fixed-size buffers
    d_patch_counts_.resize(num_patches_);
    d_patch_offsets_.resize(num_patches_);
    d_patch_write_.resize(num_patches_);
    d_best_planes_.resize(num_patches_);
    d_out_count_.resize(1);
}

// ==========================================================================
//  setRingInput
// ==========================================================================
void CudaRingSegmentation::setRingInput(float* ring_ptr, unsigned int /*size*/)
{
    d_ring_in_ = ring_ptr;
}

// ==========================================================================
//  segment  —  ring-aware patched RANSAC ground removal
// ==========================================================================
void CudaRingSegmentation::segment(
    float* inputData,
    unsigned int nCount,
    float* out_points,
    unsigned int* out_num_points,
    cudaStream_t stream)
{
    // Fallback: no ring data or too few points → pass everything through
    if (nCount < 10 || !d_ring_in_) {
        if (nCount > 0) {
            cudaMemcpyAsync(out_points, inputData, nCount * 4 * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
            if (d_ring_in_) {
                if (d_ring_out_.size() < nCount) d_ring_out_.resize(nCount);
                cudaMemcpyAsync(thrust::raw_pointer_cast(d_ring_out_.data()),
                                d_ring_in_, nCount * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream);
            }
            cudaStreamSynchronize(stream);
        }
        *out_num_points = nCount;
        return;
    }

    // ------------------------------------------------------------------
    //  Resize per-point buffers
    // ------------------------------------------------------------------
    if (d_patch_of_.capacity() < nCount) {
        d_patch_of_.reserve(nCount);
        d_indices_.reserve(nCount);
        d_index_.reserve(nCount);
        d_ring_out_.reserve(nCount);
    }
    d_patch_of_.resize(nCount);
    d_indices_.resize(nCount);
    d_index_.resize(nCount);
    d_ring_out_.resize(nCount);

    // ------------------------------------------------------------------
    //  Reset per-patch counters
    // ------------------------------------------------------------------
    thrust::fill(thrust::cuda::par(alloc_).on(stream),
                 d_patch_counts_.begin(), d_patch_counts_.end(), 0);
    thrust::fill(thrust::cuda::par(alloc_).on(stream),
                 d_patch_write_.begin(),  d_patch_write_.end(),  0);

    const int threads = 256;
    const int blocks  = (nCount + threads - 1) / threads;

    // ==================================================================
    //  Phase 1: Classify & Partition
    // ==================================================================
    countPatchKernel<<<blocks, threads, 0, stream>>>(
        inputData, d_ring_in_, nCount,
        thrust::raw_pointer_cast(d_patch_of_.data()),
        thrust::raw_pointer_cast(d_patch_counts_.data()));

    // exclusive prefix sum → patch_offsets  (d_patch_counts_ is preserved)
    thrust::exclusive_scan(thrust::cuda::par(alloc_).on(stream),
                           d_patch_counts_.begin(), d_patch_counts_.end(),
                           d_patch_offsets_.begin());

    scatterIndicesKernel<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(d_patch_of_.data()),
        thrust::raw_pointer_cast(d_patch_offsets_.data()),
        thrust::raw_pointer_cast(d_patch_write_.data()),
        thrust::raw_pointer_cast(d_indices_.data()),
        nCount);

    // ==================================================================
    //  Phase 2: Patched RANSAC
    // ==================================================================
    int max_iter = segP_.maxIterations;
    if (max_iter <= 0) max_iter = 100;

    int iters_per_patch    = std::max(1, max_iter / num_patches_);
    int total_ransac_blocks = num_patches_ * iters_per_patch;

    if (static_cast<int>(d_ransac_counts_.size()) < total_ransac_blocks) {
        d_ransac_counts_.resize(total_ransac_blocks);
        d_ransac_planes_.resize(total_ransac_blocks);
    }

    thrust::fill(thrust::cuda::par(alloc_).on(stream),
                 d_ransac_counts_.begin(),
                 d_ransac_counts_.begin() + total_ransac_blocks, -1);

    auto now = std::chrono::high_resolution_clock::now();
    unsigned int seed = static_cast<unsigned int>(now.time_since_epoch().count());

    ransacPatchedKernel<<<total_ransac_blocks, 256, 0, stream>>>(
        inputData,
        thrust::raw_pointer_cast(d_indices_.data()),
        thrust::raw_pointer_cast(d_patch_offsets_.data()),
        thrust::raw_pointer_cast(d_patch_counts_.data()),
        num_patches_, iters_per_patch,
        static_cast<float>(segP_.distanceThreshold),
        static_cast<float>(segP_.maxSegmentationDistance),
        thrust::raw_pointer_cast(d_ransac_counts_.data()),
        thrust::raw_pointer_cast(d_ransac_planes_.data()),
        seed);

    // ==================================================================
    //  Phase 3: Best plane per patch
    // ==================================================================
    int best_threads = 1;
    while (best_threads < iters_per_patch) best_threads <<= 1;
    if (best_threads > 1024) best_threads = 1024;

    findBestPlaneKernel<<<num_patches_, best_threads, 0, stream>>>(
        thrust::raw_pointer_cast(d_ransac_counts_.data()),
        thrust::raw_pointer_cast(d_ransac_planes_.data()),
        iters_per_patch,
        thrust::raw_pointer_cast(d_best_planes_.data()));

    // ==================================================================
    //  Phase 4: Mark inliers & compact survivors
    // ==================================================================
    markInliersPatchedKernel<<<blocks, threads, 0, stream>>>(
        inputData, nCount,
        thrust::raw_pointer_cast(d_patch_of_.data()),
        thrust::raw_pointer_cast(d_best_planes_.data()),
        static_cast<float>(segP_.distanceThreshold),
        thrust::raw_pointer_cast(d_index_.data()));

    unsigned int* raw_count = thrust::raw_pointer_cast(d_out_count_.data());
    cudaMemsetAsync(raw_count, 0, sizeof(unsigned int), stream);

    compactNonInliersWithRingKernel<<<blocks, threads, 0, stream>>>(
        inputData, d_ring_in_,
        thrust::raw_pointer_cast(d_index_.data()),
        out_points,
        thrust::raw_pointer_cast(d_ring_out_.data()),
        raw_count, nCount);

    // Single sync: only out_num_points is needed on host
    cudaMemcpyAsync(out_num_points, raw_count, sizeof(unsigned int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}
