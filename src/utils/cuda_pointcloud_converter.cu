#include "cuda_cone_rush/utils/cuda_pointcloud_converter.hpp"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

namespace
{

__global__ void convertPointCloud2Kernel(
    const std::uint8_t* __restrict__ input_data,
    float*              __restrict__ output_data,
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

    const std::uint64_t out_idx = static_cast<std::uint64_t>(r * width + c) * 4;

    output_data[out_idx + 0] = x;
    output_data[out_idx + 1] = y;
    output_data[out_idx + 2] = z;
    output_data[out_idx + 3] = intensity;
}

} // anonymous namespace

CudaConverter::CudaConverter()
{
    cudaStreamCreate(&stream_);
}

CudaConverter::~CudaConverter()
{
    cudaFree(d_input_);
    cudaStreamDestroy(stream_);
}

void CudaConverter::reserve(std::size_t in_bytes)
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

unsigned int CudaConverter::convert(
    const sensor_msgs::msg::PointCloud2::SharedPtr& sub_cloud,
    thrust::device_vector<float>& d_out)
{
    return convert(sub_cloud->data.data(), sub_cloud->data.size(),
                   sub_cloud->width, sub_cloud->height,
                   sub_cloud->row_step, sub_cloud->point_step, d_out);
}

unsigned int CudaConverter::convert(
    const std::uint8_t* data, std::size_t in_bytes,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t row_step, std::uint32_t point_step,
    thrust::device_vector<float>& d_out)
{
    if (width == 0 || height == 0) return 0;

    const std::size_t out_floats = static_cast<std::size_t>(width) * height * 4;

    reserve(in_bytes);

    if (d_out.size() < out_floats)
        d_out.resize(out_floats);

    cudaMemcpyAsync(d_input_, data, in_bytes, cudaMemcpyHostToDevice, stream_);

    const dim3 threads(16, 16);
    const dim3 blocks((width + 15) / 16, (height + 15) / 16);

    convertPointCloud2Kernel<<<blocks, threads, 0, stream_>>>(
        d_input_,
        thrust::raw_pointer_cast(d_out.data()),
        width, height,
        row_step, point_step);

    cudaStreamSynchronize(stream_);
    return static_cast<unsigned int>(width * height);
}
