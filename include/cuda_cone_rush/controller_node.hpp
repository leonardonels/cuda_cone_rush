#pragma once
#include <string.h>

#include "cuda_cone_rush/clustering/cuda_clustering.hpp"
#include "cuda_cone_rush/filtering/cuda_filtering.hpp"
#include "cuda_cone_rush/clustering/iclustering.hpp"
#include "cuda_cone_rush/filtering/ifiltering.hpp"
#include "cuda_cone_rush/segmentation/cuda_segmentation.hpp"
#include "cuda_cone_rush/segmentation/isegmentation.hpp"

#include "cuda_cone_rush/utils/iconverter.hpp"
#include "cuda_cone_rush/utils/cuda_pointcloud_converter.hpp"

#include <cuda_runtime.h>
#include <thrust/memory.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#ifdef ENABLE_BARQ
#include "barq/barq.hpp"
#include "barq/barq_pcl.hpp"
#endif

#ifdef USE_PINNED_MEMORY
#include <thrust/version.h>
#if THRUST_VERSION >= 300000
// CUDA 13 / CCCL >= 3.0: experimental/pinned_allocator.h was removed.
// Use the memory-resource based pinned allocator instead.
#include <thrust/mr/allocator.h>
#include <thrust/system/cuda/memory_resource.h>
template <typename T>
using pinned_host_vector = thrust::host_vector<
    T,
    thrust::mr::stateless_resource_allocator<
        T, thrust::system::cuda::universal_host_pinned_memory_resource>>;
#else
// CUDA 12 and earlier.
#include <thrust/system/cuda/experimental/pinned_allocator.h>
template <typename T>
using pinned_host_vector = thrust::host_vector<T, thrust::cuda::experimental::pinned_allocator<T>>;
#endif
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

#ifdef ENABLE_ZERO_COPY
#include <mmr_base/msg/bounded_pointcloud.hpp>
#endif

#ifdef LATENCY_TESTING
#include <chrono>
#include <mmr_base/msg/latency_sample.hpp>
#endif

class ControllerNode : public rclcpp::Node
{
private:
        std::shared_ptr<visualization_msgs::msg::Marker> cones{new visualization_msgs::msg::Marker()};
        std::string input_topic, segmented_topic, filtered_topic, cluster_topic, frame_id;

        std::string output_frame_id;
        std::string cone_frame_id_;      ///< resolved: output_frame_id, or frame_id if no extrinsics
        bool have_extrinsics_ = false;
        /* lidar2baselink, row-major as written in vehicle_extrinsics.yaml. Plain
           arrays rather than Eigen: this is one 3x3 per scan on a handful of
           points, and the package does not otherwise link Eigen. */
        double lidar_R_[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        double lidar_t_[3] = {0, 0, 0};

        /// Rotate+translate the cone centroids from the lidar frame into cone_frame_id_.
        void transformConesToOutputFrame();
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
        bool barq_enabled_ = false;
        #ifdef ENABLE_BARQ
        std::string barq_topic;
        rclcpp::TimerBase::SharedPtr timer_;
        rclcpp::TimerBase::SharedPtr writer_health_check_timer_;
        std::unique_ptr<BARQ::Reader> reader_;
        size_t barq_max_size_ = 0;
        size_t barq_retry_delay_ms_ = 100;
        int barq_max_retries_ = 5;
        int barq_retries_ = 0;
        double barq_polling_rate_ms_ = 16;  // default to ~60Hz
        int barq_writer_timeout_ms_ = 1000; // timeout for writer health check
        int64_t last_barq_timestamp_ns_ = 0;
        bool barq_initialized_ = false;
        #endif

        // Zero Copy
        bool zero_copy_enabled;

        // ---------------------------------------------------------------------------
        // using pinned host memory instead of heap-allocated memory
        // ---------------------------------------------------------------------------
        pinned_host_vector<float> h_input;

        thrust::device_vector<float> d_input;
        thrust::device_vector<float> d_output;

        cudaStream_t compute_stream = NULL;
        cudaStream_t copy_stream = NULL;

        std::unique_ptr<IConverter> converter_;
        std::unique_ptr<IFilter> cudaFilter_;
        std::unique_ptr<Isegmentation> segmentation_;
        std::unique_ptr<IClustering> clustering_;

        /* Subscriber */
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub;
        #ifdef ENABLE_ZERO_COPY
        rclcpp::Subscription<mmr_base::msg::BoundedPointcloud>::SharedPtr bounded_sub;
        #endif

        rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr cones_array_pub;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_cp_pub;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr segmented_cp_pub;

        #ifdef LATENCY_TESTING
        /* Per-frame latency instrumentation. Compiled out entirely otherwise:
           no publisher, no members, no cost. */
        rclcpp::Publisher<mmr_base::msg::LatencySample>::SharedPtr latency_sample_pub_;
        uint32_t latency_seq_ = 0;
        /* Set at the top of whichever entry point received the frame, read again
           in runPipeline. One variable and not a parameter because the three
           entry points are mutually exclusive at runtime -- the transport is
           chosen at construction -- and the pipeline is synchronous, so there is
           never a second frame in flight to confuse it with. */
        int64_t latency_t_in_ = 0;

        /* CLOCK_MONOTONIC, matching the units of the driver's frame tick. */
        static int64_t monotonicNs()
        {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        }

        void publishLatencySample(const rclcpp::Time& frame_stamp, int64_t t_out);
        #endif

        /* Load parameters function */
        void loadParameters();

        /* Cones topic init */
        void MessageInit();
        
        /* Initialize ROS2 listener */
        void ROSListenerInit();

        #ifdef ENABLE_BARQ
        /* Initialize BARQ reader */
        void BARQReaderInit();

        /* Check if writer is alive */
        void checkForWriterHealth();
        #endif

        /* Reserve and resize memory before processing */
        void reserveAndResize(size_t inputSize);

        /* PointCloud Callbacks - ROS2 entry point */
        void standardPointcloud(const sensor_msgs::msg::PointCloud2::SharedPtr sub_cloud);      // std

        #ifdef ENABLE_ZERO_COPY
        void boundedPointcloud(const mmr_base::msg::BoundedPointcloud::SharedPtr sub_cloud);    // zero copy
        #endif

        #ifdef ENABLE_BARQ
        /* Timer callback - BARQ entry point */
        void onTimer();
        #endif

        /* where the magic happens.
           `capture_stamp` is when the SOURCE cloud was acquired, carried through
           so the published cones carry it instead of the time the pipeline
           happened to finish. Downstream (the EKF in cuda_cone_fused) uses the
           stamp to line each observation up against the pose the car actually
           had when it was taken; stamping at publish time instead folds the
           whole clustering latency into the measurement, which shows up as a
           map offset along the direction of travel that grows with speed. */
        void runPipeline(unsigned int inputSize, const rclcpp::Time& capture_stamp);

        /* Publish PointCloud */
        void publishPc(float *points, unsigned int size, rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub);

public:
        ControllerNode();
        ~ControllerNode();

        void getInfo();
};