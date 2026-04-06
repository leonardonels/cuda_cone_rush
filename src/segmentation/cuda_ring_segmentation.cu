#include "cuda_cone_rush/segmentation/cuda_ring_segmentation.hpp"

CudaRingSegmentation::CudaRingSegmentation(segParam_t& params)
{
    segP_.distanceThreshold       = params.distanceThreshold;
    segP_.maxIterations           = params.maxIterations;
    segP_.probability             = params.probability;
    segP_.maxSegmentationDistance  = params.maxSegmentationDistance * params.maxSegmentationDistance;
}

void CudaRingSegmentation::setRingInput(float* ring_ptr, unsigned int /*size*/)
{
    d_ring_in_ = ring_ptr;
}

void CudaRingSegmentation::segment(
    float* inputData,
    unsigned int nCount,
    float* out_points,
    unsigned int* out_num_points,
    cudaStream_t stream)
{
    // TODO: implement ring-aware patchwork-like multi-plane ground segmentation
    // Stub: pass all points through unchanged (no ground removal)
    if (nCount > 0) {
        cudaMemcpyAsync(out_points, inputData, nCount * 4 * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);

        if (d_ring_in_) {
            if (d_ring_out_.size() < nCount)
                d_ring_out_.resize(nCount);
            cudaMemcpyAsync(thrust::raw_pointer_cast(d_ring_out_.data()),
                            d_ring_in_, nCount * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }

        cudaStreamSynchronize(stream);
    }
    *out_num_points = nCount;
}
