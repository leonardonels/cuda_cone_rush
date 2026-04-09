#include "cuda_cone_rush/filtering/cuda_filtering.hpp"

// --------------------------------------------------------------------------
// Custom passthrough filter kernel — replaces libcudafilter.so dependency
// --------------------------------------------------------------------------
__global__ void passthroughFilterKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
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

    // check each enabled axis
    if (doX && (x < downX || x > upX)) return;
    if (doY && (y < downY || y > upY)) return;
    if (doZ && (z < downZ || z > upZ)) return;

    unsigned int write_idx = atomicAdd(d_count, 1);
    output[write_idx * 4 + 0] = x;
    output[write_idx * 4 + 1] = y;
    output[write_idx * 4 + 2] = z;
    output[write_idx * 4 + 3] = input[tid * 4 + 3]; // intensity
}

CudaFilter::CudaFilter(float upX, float downX,
                       float upY, float downY,
                       float upZ, float downZ)
{
    this->upLimitX   = upX;
    this->downLimitX = downX;
    this->upLimitY   = upY;
    this->downLimitY = downY;
    this->upLimitZ   = upZ;
    this->downLimitZ = downZ;

    // determine which axes to filter
    this->filterX = (upX != 1e10f && downX != -1e10f);
    this->filterY = (upY != 1e10f && downY != -1e10f);
    this->filterZ = (upZ != 1e10f && downZ != -1e10f);

    d_count.resize(1);

    #ifdef ENABLE_VERBOSE
        RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "----filter kernel info----");
        int minGridSize = 0;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &this->passthroughFilterKernel_block_size, passthroughFilterKernel, 0, 0);
        RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "  CUDA Occupancy of passthroughFilterKernel Max Potential Block Size: %d", this->passthroughFilterKernel_block_size);
    #endif
}

void CudaFilter::filterPoints(float* inputData, unsigned int inputSize,
                              float** output, unsigned int* outputSize,
                              cudaStream_t stream)
{
    if (inputSize == 0) {
        *outputSize = 0;
        return;
    }

    #ifdef ENABLE_VERBOSE
        auto t1 = std::chrono::steady_clock::now();
    #endif
        // std::cout << "\n------------ CUDA XYZ Filter (Custom) ----------------- " << std::endl;

    // ensure temp buffer is large enough
    size_t needed = static_cast<size_t>(inputSize) * 4;
    if (d_temp.capacity() < needed) {
        d_temp.reserve(needed);
    }
    d_temp.resize(needed);

    // reset GPU counter to 0
    unsigned int* raw_count = thrust::raw_pointer_cast(d_count.data());
    cudaMemsetAsync(raw_count, 0, sizeof(unsigned int), stream);

    // launch the single-pass filter kernel
    float* raw_temp = thrust::raw_pointer_cast(d_temp.data());
    int threads = this->passthroughFilterKernel_block_size;
    int blocks  = (inputSize + threads - 1) / threads;

    passthroughFilterKernel<<<blocks, threads, 0, stream>>>(
        inputData, raw_temp, raw_count, inputSize,
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
        std::cout << "PCL filtered in " << duration.count() << " ms" << std::endl;
    #endif
}