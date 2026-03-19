#pragma once
#include <string.h>

#include "cuda_cone_rush/clustering/cuda_clustering.hpp"
#include "cuda_cone_rush/filtering/cuda_filtering.hpp"
#include "cuda_cone_rush/clustering/iclustering.hpp"
#include "cuda_cone_rush/filtering/ifiltering.hpp"
#include "cuda_cone_rush/segmentation/cuda_segmentation.hpp"
#include "cuda_cone_rush/segmentation/isegmentation.hpp"

#ifdef USE_CUDA_POINTCLOUD_CONVERTER
#include "cuda_cone_rush/utils/cuda_pointcloud_converter.hpp"
#else
#include "cuda_cone_rush/utils/pointcloud_converter.hpp"
#endif

#include <cuda_runtime.h>
#include <thrust/memory.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "barq/barq.hpp"

#ifdef USE_PINNED_MEMORY
#include <thrust/system/cuda/experimental/pinned_allocator.h>
template <typename T>
using pinned_host_vector = thrust::host_vector<T, thrust::cuda::experimental::pinned_allocator<T>>;
#else
template <typename T>
using pinned_host_vector = thrust::host_vector<T>;
#endif

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#ifdef LOGGER_PUB
#include <std_msgs/msg/float64.hpp>
#endif
#include <pcl_conversions/pcl_conversions.h>

class ControllerNode : public rclcpp::Node
{
private:
        std::shared_ptr<visualization_msgs::msg::Marker> cones{new visualization_msgs::msg::Marker()};
        std::string input_topic, segmented_topic, filtered_topic, cluster_topic, frame_id;
        bool filterFlag, clusteringFlag, segmentFlag, publishFilteredPc, publishSegmentedPc, publishCluster;
        float downFilterLimitX, upFilterLimitX;
        float downFilterLimitY, upFilterLimitY;
        float downFilterLimitZ, upFilterLimitZ;
        clustering_parameters param;
        segParam_t segP;

        #ifdef ENABLE_VERBOSE
        int limitWarning_ms;
        #endif

        #ifdef NSIGHT_SDK
        int nsight_max_iterations;
        int nsight_iterations = 0;
        #endif

        // BARQ
        rclcpp::TimerBase::SharedPtr timer_;
        std::unique_ptr<BARQ::Reader> reader_;
        size_t barq_max_size_ = 0;
        size_t barq_retry_delay_ms_ = 100;
        bool barq_enabled_ = false;
        int barq_max_retries_ = 5;
        int barq_retries_ = 0;
        int barq_polling_rate_ms_ = 16;  // default to ~60Hz

        // ---------------------------------------------------------------------------
        // using pinned host memory instead of heap-allocated memory
        // ---------------------------------------------------------------------------
        pinned_host_vector<float> h_input;

        thrust::device_vector<float> d_input;
        thrust::device_vector<float> d_output;

        cudaStream_t compute_stream = NULL;
        cudaStream_t copy_stream = NULL;

        #ifdef USE_CUDA_POINTCLOUD_CONVERTER
        // resources for cuda pointcloud converter
        pointcloud_utils::ConverterResources converter_res_;
        #endif

        IFilter *cudaFilter;
        IClustering *clustering;
        Isegmentation *segmentation;

        /* Subscriber */
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub;

        rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr cones_array_pub;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_cp_pub;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr segmented_cp_pub;

        /* Load parameters function */
        void loadParameters();

        /* Initialize BARQ reader */
        void BARQ_reader_init();

        /* Reserve and resize memory before processing */
        void reserveAndResize(size_t inputSize);

        /* PointCloud Callback - ROS2 entry point */
        void scanCallback(const sensor_msgs::msg::PointCloud2::SharedPtr sub_cloud);

        /* Timer callback - BARQ entry point */
        void onTimer();

        /* where the magic happens */
        void runPipeline(unsigned int inputSize);

        /* Publish PointCloud */
        void publishPc(float *points, unsigned int size, rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub);

public:
        ControllerNode();
        ~ControllerNode();

        void getInfo();
};