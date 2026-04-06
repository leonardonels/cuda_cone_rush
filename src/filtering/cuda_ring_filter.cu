#include "cuda_cone_rush/filtering/cuda_ring_filter.hpp"

__global__ void passthroughRingFilterKernel(
    const float* __restrict__ input,
    const float* __restrict__ ring_in,
    float* __restrict__ output,
    float* __restrict__ ring_out,
    unsigned int* __restrict__ d_count,
    unsigned int num_points,
    float downX, float upX, bool doX,
    float downY, float upY, bool doY,
    float downZ, float upZ, bool doZ)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_points) return;

    float x = input[tid * 4 + 0];
    float y = input[tid * 4 + 1];
    float z = input[tid * 4 + 2];

    if (doX && (x < downX || x > upX)) return;
    if (doY && (y < downY || y > upY)) return;
    if (doZ && (z < downZ || z > upZ)) return;

    unsigned int write_idx = atomicAdd(d_count, 1);
    output[write_idx * 4 + 0] = x;
    output[write_idx * 4 + 1] = y;
    output[write_idx * 4 + 2] = z;
    output[write_idx * 4 + 3] = input[tid * 4 + 3]; // intensity

    if (ring_in) {
        ring_out[write_idx] = ring_in[tid];
    }
}

CudaRingFilter::CudaRingFilter(float upX, float downX,
                                float upY, float downY,
                                float upZ, float downZ)
{
    this->upLimitX   = upX;
    this->downLimitX = downX;
    this->upLimitY   = upY;
    this->downLimitY = downY;
    this->upLimitZ   = upZ;
    this->downLimitZ = downZ;

    this->filterX = (upX != 1e10f && downX != -1e10f);
    this->filterY = (upY != 1e10f && downY != -1e10f);
    this->filterZ = (upZ != 1e10f && downZ != -1e10f);

    d_count_.resize(1);

    #ifdef ENABLE_VERBOSE
        RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "----ring filter kernel info----");
        int minGridSize = 0;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &this->block_size_, passthroughRingFilterKernel, 0, 0);
        RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "  CUDA Occupancy of passthroughRingFilterKernel Max Potential Block Size: %d", this->block_size_);
    #endif
}

void CudaRingFilter::setRingInput(float* ring_ptr, unsigned int /*size*/)
{
    d_ring_in_ = ring_ptr;
}

void CudaRingFilter::filterPoints(float* inputData, unsigned int inputSize,
                                   float** output, unsigned int* outputSize,
                                   cudaStream_t stream)
{
    #ifdef ENABLE_VERBOSE
        auto t1 = std::chrono::steady_clock::now();
    #endif

    // ensure temp buffer is large enough
    size_t needed = static_cast<size_t>(inputSize) * 4;
    if (d_temp_.capacity() < needed) {
        d_temp_.reserve(needed);
    }
    d_temp_.resize(needed);

    // resize ring temp buffer if ring is active
    if (d_ring_in_) {
        if (d_ring_temp_.capacity() < inputSize) {
            d_ring_temp_.reserve(inputSize);
        }
        d_ring_temp_.resize(inputSize);
    }

    // reset GPU counter to 0
    unsigned int* raw_count = thrust::raw_pointer_cast(d_count_.data());
    cudaMemsetAsync(raw_count, 0, sizeof(unsigned int), stream);

    // launch the single-pass filter + ring compaction kernel
    float* raw_temp = thrust::raw_pointer_cast(d_temp_.data());
    float* raw_ring_out = d_ring_in_ ? thrust::raw_pointer_cast(d_ring_temp_.data()) : nullptr;

    int threads = this->block_size_;
    int blocks  = (inputSize + threads - 1) / threads;

    passthroughRingFilterKernel<<<blocks, threads, 0, stream>>>(
        inputData, d_ring_in_, raw_temp, raw_ring_out, raw_count, inputSize,
        downLimitX, upLimitX, filterX,
        downLimitY, upLimitY, filterY,
        downLimitZ, upLimitZ, filterZ);

    // copy count back to host
    cudaMemcpyAsync(outputSize, raw_count, sizeof(unsigned int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // copy compacted result into the caller's output buffer
    if (*outputSize > 0) {
        cudaMemcpyAsync(*output, raw_temp, (*outputSize) * 4 * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    #ifdef ENABLE_VERBOSE
        auto t2 = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t2 - t1);
        std::cout << "Ring filter completed in " << duration.count() << " ms" << std::endl;
    #endif
}
