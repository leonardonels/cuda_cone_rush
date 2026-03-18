#include "cuda_cone_rush/segmentation/cuda_segmentation.hpp"

#include <iostream>
#include <chrono>
#include <vector>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/extrema.h>

// --------------------
// COMPACTION KERNEL
// --------------------
__global__ void compactInliersKernel(
    const float* in_points, 
    const int* index, 
    float* out_points, 
    unsigned int* d_count, 
    int max_points) 
{
    // calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // we don't want to read out of bounds
    if (tid < max_points) {
        // index != 1 means that the point is NOT part of the segmented ground plane
        if (index[tid] != 1) { 
            
            // atomically reserve a spot in the output array.
            // atomicAdd returns the old value, giving this specific thread a unique write index.
            unsigned int write_idx = atomicAdd(d_count, 1);
            
            // copy the 4 floats (X, Y, Z, Intensity) directly in VRAM
            out_points[write_idx * 4 + 0] = in_points[tid * 4 + 0];
            out_points[write_idx * 4 + 1] = in_points[tid * 4 + 1];
            out_points[write_idx * 4 + 2] = in_points[tid * 4 + 2];
            out_points[write_idx * 4 + 3] = in_points[tid * 4 + 3];
        }
    }
}


// ---------------------------------------------------------------------------------------
// RANSAC Kernels (Replacement for libcudasegmentation)
// ---------------------------------------------------------------------------------------

// Pseudo random generator
__device__ inline unsigned int wang_hash(unsigned int seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

__global__ void ransacPlaneKernel(
    const float* __restrict__ points,
    int num_points,
    float threshold,
    int max_iterations,
    float max_segmentation_distance,
    int* __restrict__ plane_inliers_counts,
    float4* __restrict__ plane_models,
    unsigned int seed
)
{
    // each block performs one RANSAC iteration
    int iter = blockIdx.x;
    if (iter >= max_iterations) return;

    // select 3 random points
    // use a different seed per iteration/block
    unsigned int s = seed + iter * 199999; 
    
    int idx1 = wang_hash(s) % num_points;
    int idx2 = wang_hash(s + 1) % num_points; // simple increment to get different indices
    int idx3 = wang_hash(s + 2) % num_points;

    // load points (manual indexing for float array x,y,z,i)
    float p1[3] = {points[idx1*4], points[idx1*4+1], points[idx1*4+2]};
    float p2[3] = {points[idx2*4], points[idx2*4+1], points[idx2*4+2]};
    float p3[3] = {points[idx3*4], points[idx3*4+1], points[idx3*4+2]};

    // look for planes only near the sensor (optional heuristic to improve speed and robustness)
    if (p1[0]*p1[0] + p1[1]*p1[1] + p1[2]*p1[2] > (float)max_segmentation_distance) {   // only consider points within maxSegmentationDistance from the sensor
        if (threadIdx.x == 0) {
            plane_inliers_counts[iter] = -1; // mark as invalid
        }
        return;
    }

    // compute plane model (ax + by + cz + d = 0)
    float v1[3] = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]};
    float v2[3] = {p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]};

    // cross product
    float a = v1[1] * v2[2] - v1[2] * v2[1];
    float b = v1[2] * v2[0] - v1[0] * v2[2];
    float c = v1[0] * v2[1] - v1[1] * v2[0];

    float norm = sqrtf(a*a + b*b + c*c);
    
    // check for degenerate triangle
    if (norm < 1e-6f) {
        if (threadIdx.x == 0) {
            plane_inliers_counts[iter] = -1;
        }
        return;
    }

    float inv_norm = 1.0f / norm;
    a *= inv_norm;
    b *= inv_norm;
    c *= inv_norm;
    float d = -(a * p1[0] + b * p1[1] + c * p1[2]);

    // count Inliers
    // each thread counts a subset of points
    int local_count = 0;

    for (int i = threadIdx.x; i < num_points; i += blockDim.x) {
        float x = points[i*4];
        float y = points[i*4+1];
        float z = points[i*4+2];
        
        float dist = fabsf(a * x + b * y + c * z + d);
        if (dist <= threshold) {
            local_count++;
        }
    }

    // block reduction
    __shared__ int s_counts[1024]; // Block dim must be 1024
    // initialize shared mem
    if (threadIdx.x < 1024) s_counts[threadIdx.x] = 0;
    __syncthreads(); // Only needed if we rely on init, but we overwrite

    s_counts[threadIdx.x] = local_count;
    __syncthreads();

    // standard reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_counts[threadIdx.x] += s_counts[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // store result
    if (threadIdx.x == 0) {
        plane_inliers_counts[iter] = s_counts[0];
        plane_models[iter] = make_float4(a, b, c, d);
    }
}

__global__ void markInliersKernel(
    const float* __restrict__ points,
    int num_points,
    int* __restrict__ indices,
    float4 best_plane,
    float threshold
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    float x = points[i*4];
    float y = points[i*4+1];
    float z = points[i*4+2];
    float dist = fabsf(best_plane.x * x + best_plane.y * y + best_plane.z * z + best_plane.w);
    
    // mark as inlier (1) or outlier (0)
    indices[i] = (dist <= threshold) ? 1 : 0;
}

// variant that loads best_plane from a device pointer (avoids D→H sync before launch)
__global__ void markInliersFromDeviceKernel(
    const float* __restrict__ points,
    int num_points,
    int* __restrict__ indices,
    const float4* __restrict__ d_best_plane,
    float threshold
) {
    float4 bp = *d_best_plane;  // all threads in the grid load the same value (L2 cached)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    float x = points[i*4];
    float y = points[i*4+1];
    float z = points[i*4+2];
    float dist = fabsf(bp.x * x + bp.y * y + bp.z * z + bp.w);
    indices[i] = (dist <= threshold) ? 1 : 0;
}

// --------------------
// CUDA SEGMENTATION
// --------------------
CudaSegmentation::CudaSegmentation(segParam_t &params)
{
  segP.distanceThreshold = params.distanceThreshold;
  segP.maxIterations = params.maxIterations;
  segP.probability = params.probability;
  segP.maxSegmentationDistance = params.maxSegmentationDistance * params.maxSegmentationDistance; // store squared distance to avoid sqrt in segmentation

  d_out_count.resize(1);

  #ifdef ENABLE_VERBOSE
      RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "-segmentation kernel info-");
      int minGridSize = 0;
      cudaOccupancyMaxPotentialBlockSize(&minGridSize, &this->compactInliersKernel_block_size, compactInliersKernel, 0, 0);
      RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "  CUDA Occupancy of compactInliersKernel Max Potential Block Size: %d", this->compactInliersKernel_block_size);
  
      cudaOccupancyMaxPotentialBlockSize(&minGridSize, &this->ransacPlaneKernel_block_size, ransacPlaneKernel, 0, 0);
      RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "  CUDA Occupancy of ransacPlaneKernel Max Potential Block Size: %d", this->ransacPlaneKernel_block_size);
  
      cudaOccupancyMaxPotentialBlockSize(&minGridSize, &this->markInliersKernel_block_size, markInliersKernel, 0, 0);
      RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "  CUDA Occupancy of markInliersKernel Max Potential Block Size: %d", this->markInliersKernel_block_size);
  
      cudaOccupancyMaxPotentialBlockSize(&minGridSize, &this->markInliersFromDeviceKernel_block_size, markInliersFromDeviceKernel, 0, 0);
      RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "  CUDA Occupancy of markInliersFromDeviceKernel Max Potential Block Size: %d", this->markInliersFromDeviceKernel_block_size);
  #endif
}

// Funzione principale per segmentare i punti di input
// inputData: array nel host di float (x, y, z, intensità) × nCount
// nCount: numero di punti in input
// out_points: buffer preallocato per restituire gli inlier
// out_num_points: numero effettivo di inlier trovati
void CudaSegmentation::segment(
    float *inputData,
    unsigned int nCount,
    float *out_points,
    unsigned int *out_num_points,
    cudaStream_t stream)
{
      // std::cout << "\n----------- CUDA Segmentation (Custom) ---------------- "
      //           << std::endl;
  #ifdef ENABLE_VERBOSE
      auto t1 = std::chrono::steady_clock::now();
  #endif

  if (nCount < 10) {
      *out_num_points = nCount;
      return; 
  }

  if (d_index.capacity() < nCount) {
      d_index.reserve(nCount);
      d_input.reserve(nCount * 4);
  }
  d_index.resize(nCount);
  d_input.resize(nCount * 4);

  int* raw_index = thrust::raw_pointer_cast(d_index.data());
  float* raw_input = thrust::raw_pointer_cast(d_input.data());

  // copy input within GPU (inputData is already a device pointer from the controller)
  cudaMemcpyAsync(raw_input, inputData, nCount * 4 * sizeof(float), cudaMemcpyDeviceToDevice, stream);

  // ----------------------------------------------------
  // Custom RANSAC Implementation
  // ----------------------------------------------------
  // std::cout << "Launching Custom RANSAC kernel on GPU..." << std::endl;

  int max_iter = segP.maxIterations;
  if (max_iter <= 0) max_iter = 100;
  if (max_iter > 1024) max_iter = 1024;     // limit for memory

  // temp buffers for RANSAC results
  if (int(d_counts.size()) < max_iter) d_counts.resize(max_iter);
  if (int(d_planes.size()) < max_iter) d_planes.resize(max_iter);
  
  int* raw_counts = thrust::raw_pointer_cast(d_counts.data());
  float4* raw_planes = thrust::raw_pointer_cast(d_planes.data());

  // init counts to -1
  thrust::fill(thrust::cuda::par(alloc).on(stream), d_counts.begin(), d_counts.begin() + max_iter, -1);

  // launch RANSAC
  // each block is 1 iteration, using ransacPlaneKernel_block_size threads for inlier counting + reduction
  auto now = std::chrono::high_resolution_clock::now();
  unsigned int seed = (unsigned int)now.time_since_epoch().count();
  ransacPlaneKernel<<<max_iter, this->ransacPlaneKernel_block_size, 0, stream>>>(
      raw_input, nCount, (float)segP.distanceThreshold, max_iter, (float)segP.maxSegmentationDistance, raw_counts, raw_planes, seed
  );

  // find best RANSAC iteration entirely on GPU using thrust::max_element
  auto best_it = thrust::max_element(thrust::cuda::par(alloc).on(stream),
                                     d_counts.begin(), d_counts.begin() + max_iter);
  int best_idx_device = (int)(best_it - d_counts.begin());

  // launch markInliers + compaction unconditionally on GPU (no host sync needed yet).
  // we read the best plane directly from device memory via pointer offset.
  // markInliersKernel reads best_plane by value — we use a small D→D copy into
  // a single-element device buffer so we can pass it to the kernel.

  // copy the winning plane model into a known device location
  if (d_bestPlane.empty()) d_bestPlane.resize(1);
  cudaMemcpyAsync(thrust::raw_pointer_cast(d_bestPlane.data()),
                  raw_planes + best_idx_device, sizeof(float4),
                  cudaMemcpyDeviceToDevice, stream);

  // mark inliers using the best plane (kernel reads plane from device memory)
  {
      int threads = this->markInliersFromDeviceKernel_block_size;
      int blocks = (nCount + threads - 1) / threads;
      // We need to pass float4 by value — launch a wrapper that loads from device ptr
      markInliersFromDeviceKernel<<<blocks, threads, 0, stream>>>(
          raw_input, nCount, raw_index,
          thrust::raw_pointer_cast(d_bestPlane.data()),
          (float)segP.distanceThreshold);
  }

  // compact non-inlier points directly into caller's output buffer (ground removed)
  unsigned int* raw_count = thrust::raw_pointer_cast(d_out_count.data());
  cudaMemsetAsync(raw_count, 0, sizeof(unsigned int), stream);

  {
      int threads = this->compactInliersKernel_block_size;
      int blocks = (nCount + threads - 1) / threads;
      compactInliersKernel<<<blocks, threads, 0, stream>>>(
          raw_input, raw_index, out_points, raw_count, nCount);
  }

  // single sync: only out_num_points is needed on host (for controller's inputSize)
  cudaMemcpyAsync(out_num_points, raw_count, sizeof(unsigned int),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  #ifdef ENABLE_VERBOSE
      auto t2 = std::chrono::steady_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
      std::cout << "Segmentation completed in " << duration / 1e6 << " ms.\n"
                << "  Best plane found with " << *best_it << " inliers out of " << nCount << " points." << std::endl;
  #endif
}