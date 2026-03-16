#include "cuda_clustering/utils/cuda_pointcloud_converter.hpp"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <pcl/point_types.h>

#include <stdexcept>
#include <string>

namespace pointcloud_utils
{

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

    const std::uint64_t out_idx =
        static_cast<std::uint64_t>(r * width + c) * 4;  // stride 4 always

    float x, y, z;
    memcpy(&x, src + 0, sizeof(float));
    memcpy(&y, src + 4, sizeof(float));
    memcpy(&z, src + 8, sizeof(float));

    output_data[out_idx + 0] = x;
    output_data[out_idx + 1] = y;
    output_data[out_idx + 2] = z;
    output_data[out_idx + 3] = 0.0f;
}

} // namespace

ConverterResources::ConverterResources()
{
    cudaStreamCreate(&stream);
}

ConverterResources::~ConverterResources()
{
    cudaFree(d_input);
    cudaStreamDestroy(stream);
}

void ConverterResources::reserve(std::size_t in_bytes)
{
    if (in_bytes > input_bytes)
    {
        if (d_input != nullptr)  // only free if previously allocated
            cudaFree(d_input);
        d_input = nullptr;
        cudaMalloc(&d_input, in_bytes);
        input_bytes = in_bytes;
    }
}

void convertPointCloud2ToFloatArray(
    const sensor_msgs::msg::PointCloud2::SharedPtr& sub_cloud,
    thrust::device_vector<float>& d_out,
    ConverterResources& res)
{
    const std::uint32_t width      = sub_cloud->width;
    const std::uint32_t height     = sub_cloud->height;
    const std::uint32_t row_step   = sub_cloud->row_step;
    const std::uint32_t point_step = sub_cloud->point_step;

    if (width == 0 || height == 0) return;

    const std::size_t in_bytes   = sub_cloud->data.size();
    const std::size_t out_floats = static_cast<std::size_t>(width) * height * 4;

    res.reserve(in_bytes);

    if (d_out.size() < out_floats)
        d_out.resize(out_floats);

    cudaMemcpyAsync(res.d_input, sub_cloud->data.data(),
                    in_bytes, cudaMemcpyHostToDevice, res.stream);

    const dim3 threads(16, 16);
    const dim3 blocks((width + 15) / 16, (height + 15) / 16);

    convertPointCloud2Kernel<<<blocks, threads, 0, res.stream>>>(
        res.d_input,
        thrust::raw_pointer_cast(d_out.data()),
        width, height,
        row_step, point_step);

    cudaStreamSynchronize(res.stream);
}

} // namespace pointcloud_utils