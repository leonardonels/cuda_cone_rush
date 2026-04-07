#include "cuda_cone_rush/utils/cuda_ring_converter.hpp"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

namespace
{

__global__ void convertPointCloud2WithRingKernel(
    const std::uint8_t* __restrict__ input_data,
    float*              __restrict__ output_data,
    float*              __restrict__ ring_data,
    std::uint32_t*      __restrict__ valid_count,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t row_step,
    std::uint32_t point_step)
{
    const std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t r = blockIdx.y * blockDim.y + threadIdx.y;

    if (c >= width || r >= height) return;

    const std::uint8_t* src = input_data
        + static_cast<std::size_t>(r) * row_step
        + static_cast<std::size_t>(c) * point_step;

    float x, y, z, intensity;
    memcpy(&x,         src + 0,  sizeof(float));
    memcpy(&y,         src + 4,  sizeof(float));
    memcpy(&z,         src + 8,  sizeof(float));
    memcpy(&intensity, src + 12, sizeof(float));

    if (x == 0.0f && y == 0.0f && z == 0.0f) return;

    const std::uint32_t idx = atomicAdd(valid_count, 1u);
    const std::uint64_t out_idx = static_cast<std::uint64_t>(idx) * 4;

    output_data[out_idx + 0] = x;
    output_data[out_idx + 1] = y;
    output_data[out_idx + 2] = z;
    output_data[out_idx + 3] = intensity;

    std::uint16_t ring;
    memcpy(&ring, src + 16, sizeof(std::uint16_t));
    ring_data[idx] = static_cast<float>(ring);
}

} // anonymous namespace

CudaRingConverter::CudaRingConverter()
{
    cudaStreamCreate(&stream_);
    cudaMalloc(&d_count_, sizeof(std::uint32_t));
}

CudaRingConverter::~CudaRingConverter()
{
    cudaFree(d_input_);
    cudaFree(d_count_);
    cudaStreamDestroy(stream_);
}

void CudaRingConverter::reserve(std::size_t in_bytes)
{
    if (in_bytes > input_bytes_)
    {
        if (d_input_ != nullptr)
            cudaFree(d_input_);
        d_input_ = nullptr;
        cudaMalloc(&d_input_, in_bytes);
        input_bytes_ = in_bytes;
    }
}

unsigned int CudaRingConverter::convert(
    const sensor_msgs::msg::PointCloud2::SharedPtr& sub_cloud,
    thrust::device_vector<float>& d_out)
{
    const std::uint32_t width      = sub_cloud->width;
    const std::uint32_t height     = sub_cloud->height;
    const std::uint32_t row_step   = sub_cloud->row_step;
    const std::uint32_t point_step = sub_cloud->point_step;

    if (width == 0 || height == 0) return 0;

    const std::size_t in_bytes   = sub_cloud->data.size();
    const std::size_t num_points = static_cast<std::size_t>(width) * height;
    const std::size_t out_floats = num_points * 4;

    reserve(in_bytes);

    if (d_out.size() < out_floats)
        d_out.resize(out_floats);
    if (d_ring_.size() < num_points)
        d_ring_.resize(num_points);

    cudaMemcpyAsync(d_input_, sub_cloud->data.data(),
                    in_bytes, cudaMemcpyHostToDevice, stream_);
    cudaMemsetAsync(d_count_, 0, sizeof(std::uint32_t), stream_);

    const dim3 threads(16, 16);
    const dim3 blocks((width + 15) / 16, (height + 15) / 16);

    convertPointCloud2WithRingKernel<<<blocks, threads, 0, stream_>>>(
        d_input_,
        thrust::raw_pointer_cast(d_out.data()),
        thrust::raw_pointer_cast(d_ring_.data()),
        d_count_,
        width, height,
        row_step, point_step);

    std::uint32_t h_count = 0;
    cudaMemcpyAsync(&h_count, d_count_, sizeof(std::uint32_t),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    return static_cast<unsigned int>(h_count);
}
