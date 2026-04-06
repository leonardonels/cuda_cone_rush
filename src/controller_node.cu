#include "cuda_cone_rush/controller_node.hpp"

#include <csignal>

/**
 *  [ ] change the way points are loaded into the pipeline to keep intensity and ring information (currently lost in the float array conversion)
 */

ControllerNode::ControllerNode() : Node("cuda_cone_rush_node")
{
    this->loadParameters();
    this->getInfo();


    /* Select segmentation class */
    this->segmentation = new CudaSegmentation(segP);
    
    /* Select filtering class */
    this->cudaFilter = new CudaFilter(upFilterLimitX, downFilterLimitX, 
                                   upFilterLimitY, downFilterLimitY, 
                                   upFilterLimitZ, downFilterLimitZ);

    /* Select clustering class */
    this->clustering = new CudaClustering(param);

#ifdef ENABLE_BARQ
    if (this->barq_enabled_) {
        /* Create BARQ subscriber */
        this->BARQ_reader_init();
    }else{
#endif
        /* Define QoS for Best Effort messages transport */
        auto qos = rclcpp::QoS(rclcpp::KeepLast(10), rmw_qos_profile_sensor_data);
        
        /* Create ROS2 subscriber */
        this->cloud_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(this->input_topic, qos,
                                                                                   std::bind(&ControllerNode::scanCallback, this, std::placeholders::_1));
#ifdef ENABLE_BARQ
    }
#endif

    this->cones_array_pub = this->create_publisher<visualization_msgs::msg::Marker>(this->cluster_topic, 100);
    if(this->filterFlag && this->publishFilteredPc)
        this->filtered_cp_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(this->filtered_topic, 100);
    if(this->segmentFlag && this->publishSegmentedPc)
        this->segmented_cp_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(this->segmented_topic, 100);
                                                                               
    /* Cones topic init */
    cones->header.frame_id = this->frame_id;
    cones->ns = "ListaConiRilevati";
    cones->type = visualization_msgs::msg::Marker::SPHERE_LIST;
    cones->action = visualization_msgs::msg::Marker::ADD;
    cones->scale.x = 0.3; // 0.5
    cones->scale.y = 0.2;
    cones->scale.z = 0.2;
    cones->color.a = 1.0; // 1.0
    cones->color.r = 1.0;
    cones->color.g = 0.0;
    cones->color.b = 1.0;
    cones->pose.orientation.x = 0.0;
    cones->pose.orientation.y = 0.0;
    cones->pose.orientation.z = 0.0;
    cones->pose.orientation.w = 1.0;

    /* Cuda streams init*/
    cudaStreamCreate(&copy_stream);
    cudaStreamCreate(&compute_stream);
}

ControllerNode::~ControllerNode()
{
    delete this->segmentation;
    delete this->cudaFilter;
    delete this->clustering;

    if (copy_stream) cudaStreamDestroy(copy_stream);
    if (compute_stream) cudaStreamDestroy(compute_stream);
}

#ifdef ENABLE_BARQ
void ControllerNode::BARQ_reader_init()
{
    RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "Initializing BARQ reader...");

    /* Define BARQ transport layer */
    const size_t kMaxPoints = 300000;
    barq_max_size_ = sizeof(BARQFrameHeader) + kMaxPoints * sizeof(BARQPoint) + 512;

    do{
        this->reader_ = std::make_unique<BARQ::Reader>(this->barq_topic, this->barq_max_size_);
        if (!reader_->init()) {
            RCLCPP_WARN(rclcpp::get_logger("cuda_cone_rush_node"), "Failed to initialize BARQ reader, retrying in %zu ms...", this->barq_retry_delay_ms_);
            std::this_thread::sleep_for(std::chrono::milliseconds(this->barq_retry_delay_ms_));
            this->barq_retries_++;
        } else {
            RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "BARQ reader initialized successfully.");

            // Poll BARQ at 60Hz
            this->timer_ = create_wall_timer(std::chrono::milliseconds(this->barq_polling_rate_ms_), std::bind(&ControllerNode::onTimer, this));
            break;
        }
    } while (this->barq_retries_ < this->barq_max_retries_);
    if (this->barq_retries_ >= this->barq_max_retries_) {
        RCLCPP_ERROR(rclcpp::get_logger("cuda_cone_rush_node"), "Exceeded maximum BARQ initialization retries (%d), exiting.", this->barq_max_retries_);
        // std::raise(SIGINT);  // Gracefully shutdown the node
    }
}
#endif

void ControllerNode::loadParameters()
{
    // ================ Topics and Frames =================
    declare_parameter("input_topic", "/lidar_points");
    declare_parameter("segmented_topic", "/segmented_points");
    declare_parameter("filtered_topic", "/filtered_points");
    declare_parameter("cluster_topic", "/clusters");
    declare_parameter("frame_id", "hesai_lidar");
    
    // ================ BARQ options =================
    declare_parameter("BARQ_enabled", false);
#ifdef ENABLE_BARQ
    declare_parameter("BARQ_topic", "/lidar_points");
    declare_parameter("BARQ_retry_delay_ms", 10);
    declare_parameter("BARQ_max_retries", 5);
    declare_parameter("BARQ_polling_rate_ms", 1);  // 1 ms corresponds to ~1KHz
#endif
    
    // ================ Pipeline options =================
    declare_parameter("filter", false);
    declare_parameter("segment", false);
    declare_parameter("clustering", true);
    
    declare_parameter("publishFilteredPc", false);
    declare_parameter("publishSegmentedPc", false);
    declare_parameter("publishCluster", true);

    // ================ Debug options =================
#ifdef ENABLE_VERBOSE
    declare_parameter("limitWarning_ms", 30);
#endif
#ifdef NSIGHT_SDK
    declare_parameter("nsight_max_iterations", 60);
#endif
    
    // ================ Filtering XYZ options =================
    declare_parameter("downFilterLimitX", -1e10);
    declare_parameter("upFilterLimitX", 1e10);
    declare_parameter("downFilterLimitY", -1e10);
    declare_parameter("upFilterLimitY", 1e10);
    declare_parameter("downFilterLimitZ", 0.0);
    declare_parameter("upFilterLimitZ", 0.0);
    
    // ================ Segmentation options =================
    declare_parameter("distanceThreshold", 0.1);
    declare_parameter("maxSegmentationDistance", 20.0);
    declare_parameter("maxIterations", 80);
    declare_parameter("probability", 0.75);
    
    // ================ Clustering options =================
    declare_parameter("minClusterSize", 1);
    declare_parameter("maxClusterSize", 500);
    declare_parameter("voxelX", 0.8);
    declare_parameter("voxelY", 0.8);
    declare_parameter("voxelZ", 0.8);
    declare_parameter("countThreshold", 5);
    
    declare_parameter("clusterMaxX", 0.4);
    declare_parameter("clusterMaxY", 0.4);
    declare_parameter("clusterMaxZ", 0.4);
    declare_parameter("clusterMinX", -0.1);
    declare_parameter("clusterMinY", -0.1);
    declare_parameter("clusterMinZ", 0.1);
    declare_parameter("maxHeight", 0.4);
    

    // ================ Topics and Frames =================
    get_parameter("input_topic", this->input_topic);
    get_parameter("segmented_topic", this->segmented_topic);
    get_parameter("filtered_topic", this->filtered_topic);
    get_parameter("cluster_topic", this->cluster_topic);
    get_parameter("frame_id", this->frame_id);

    // ================ BARQ options =================
    get_parameter("BARQ_enabled", this->barq_enabled_);
#ifdef ENABLE_BARQ
    get_parameter("BARQ_topic", this->barq_topic);
    get_parameter("BARQ_retry_delay_ms", this->barq_retry_delay_ms_);
    get_parameter("BARQ_max_retries", this->barq_max_retries_);
    get_parameter("BARQ_polling_rate_ms", this->barq_polling_rate_ms_);
#endif

    // ================ Pipeline options =================
    get_parameter("filter", this->filterFlag);
    get_parameter("segment", this->segmentFlag);
    get_parameter("clustering", this->clusteringFlag);

    get_parameter("publishFilteredPc", this->publishFilteredPc);
    get_parameter("publishSegmentedPc", this->publishSegmentedPc);
    get_parameter("publishCluster", this->publishCluster);

    // ================ Debug options =================
#ifdef ENABLE_VERBOSE
    get_parameter("limitWarning_ms", this->limitWarning_ms);
#endif
#ifdef NSIGHT_SDK
    get_parameter("nsight_max_iterations", this->nsight_max_iterations);
#endif

    // ================ Filtering XYZ options =================
    get_parameter("downFilterLimitX", this->downFilterLimitX);
    get_parameter("upFilterLimitX", this->upFilterLimitX);
    get_parameter("downFilterLimitY", this->downFilterLimitY);
    get_parameter("upFilterLimitY", this->upFilterLimitY);
    get_parameter("downFilterLimitZ", this->downFilterLimitZ);
    get_parameter("upFilterLimitZ", this->upFilterLimitZ);

    // ================ Segmentation options =================
    get_parameter("distanceThreshold", this->segP.distanceThreshold);
    get_parameter("maxSegmentationDistance", this->segP.maxSegmentationDistance);
    get_parameter("maxIterations", this->segP.maxIterations);
    get_parameter("probability", this->segP.probability);
    
    // ================ Clustering options =================
    get_parameter("minClusterSize", this->param.clustering.minClusterSize);
    get_parameter("maxClusterSize", this->param.clustering.maxClusterSize);
    get_parameter("voxelX", this->param.clustering.voxelX);
    get_parameter("voxelY", this->param.clustering.voxelY);
    get_parameter("voxelZ", this->param.clustering.voxelZ);
    get_parameter("countThreshold", this->param.clustering.countThreshold);

    get_parameter("clusterMaxX", this->param.filtering.clusterMaxX);
    get_parameter("clusterMaxY", this->param.filtering.clusterMaxY);
    get_parameter("clusterMaxZ", this->param.filtering.clusterMaxZ);
    get_parameter("clusterMinX", this->param.filtering.clusterMinX);
    get_parameter("clusterMinY", this->param.filtering.clusterMinY);
    get_parameter("clusterMinZ", this->param.filtering.clusterMinZ);
    get_parameter("maxHeight", this->param.filtering.maxHeight);
}

void ControllerNode::getInfo()
{
    cudaDeviceProp prop;
    int count = 0;
    cudaGetDeviceCount(&count);
    RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "GPU has cuda devices: %d\n", count);
    for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&prop, i);
        RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "----device id: %d info----", i);
        RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "  GPU : %s", prop.name);
        RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "  Capability: %d.%d", prop.major, prop.minor);
        RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "  Avaiable Streaming Multiprocessors: %d", prop.multiProcessorCount);
        RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "  Global memory: %luMB", prop.totalGlobalMem >> 20);
        RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "  Shared memory in a block: %luKB", prop.sharedMemPerBlock >> 10);
        RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "  warp size: %d", prop.warpSize);
        RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "  threads in a block: %d", prop.maxThreadsPerBlock);
        RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "  block dim: (%d,%d,%d)", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "  grid dim: (%d,%d,%d)", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
}

void ControllerNode::publishPc(float *points, unsigned int size, rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub)
{
    sensor_msgs::msg::PointCloud2 pc;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    pcl_cloud->width = size;
    pcl_cloud->height = 1;
    pcl_cloud->points.resize(size);

    // explicitly cast to void* to silence the object-memory access warning
    memcpy(static_cast<void*>(pcl_cloud->points.data()), points, size * 4 * sizeof(float));

    pcl::toROSMsg(*pcl_cloud, pc);
    pc.header.frame_id = this->frame_id;
    pub->publish(pc);
}

void ControllerNode::reserveAndResize(size_t totalElements)
{
    // ----------------------------------------------------------------------------------
    // removes the malloc logic inside .resize() by reserving enough memory beforehand
    // ----------------------------------------------------------------------------------
    if (h_input.capacity() < totalElements) {
        h_input.reserve(totalElements);
        d_input.reserve(totalElements);
        d_output.reserve(totalElements);
    }
    
    // ----------------------------------------------------------------------------------
    // if the new size is smaller, then resize should be almost instantaneous
    // ----------------------------------------------------------------------------------
    h_input.resize(totalElements);
    d_input.resize(totalElements);
    d_output.resize(totalElements);
}

// ROS2 Entry point for PointCloud2 subscriber callback — not used with BARQ
void ControllerNode::scanCallback(sensor_msgs::msg::PointCloud2::SharedPtr sub_cloud)
{
#ifdef LATENCY_TESTING
    auto now = rclcpp::Clock(RCL_ROS_TIME).now();
    auto timestamp = sub_cloud->header.stamp;
    auto latency = (now - timestamp).nanoseconds() / 1e6;
    std::cout << "ROS2 latency: " << latency << " ms" << std::endl;
#endif

    unsigned int inputSize = sub_cloud->width * sub_cloud->height;
    size_t totalElements = inputSize * 4;

    reserveAndResize(totalElements);

#ifdef USE_CUDA_POINTCLOUD_CONVERTER
    pointcloud_utils::convertPointCloud2ToFloatArray(sub_cloud, d_input, converter_res_);   // converto to device directly with cuda kernel
#else
    // -------------------------------------------------------
    // convert PointCloud2 to float array and copy to device
    // -------------------------------------------------------
    pointcloud_utils::convertPointCloud2ToFloatArray(sub_cloud, h_input.data());
    d_input = h_input;
#endif

    runPipeline(inputSize);
}

#ifdef ENABLE_BARQ
/**
 * struct __attribute__((packed)) BARQPoint {
 *     float    x;
 *     float    y;
 *     float    z;
 *     float    intensity;   // promoted from uint8_t on write
 *     uint16_t ring;
 *     double   timestamp;
 * };  // 26 bytes
 * 
 * struct __attribute__((packed)) BARQFrameHeader {
 *     uint32_t width;       // number of valid points
 *     uint32_t height;      // always 1
 *     uint32_t point_step;  // always sizeof(BARQPoint) = 26
 *     double   timestamp;   // frame start timestamp (seconds)
 * };  // 20 bytes
 */

// BARQ entry point for processing — identical pipeline, different input source
void ControllerNode::onTimer()
{
    size_t sz;
    int64_t ts;

    const void* ptr = reader_->getLatest(sz, ts);

    if (ts == last_barq_timestamp_ns_) {
#ifdef ENABLE_VERBOSE
        std::cout << "[BARQ] No new data (timestamp unchanged)" << std::endl;
#endif
        return;
    }
    last_barq_timestamp_ns_ = ts;

    if (!ptr || sz == 0) {
#ifdef ENABLE_VERBOSE
        std::cout << "[BARQ] No data yet" << std::endl;
#endif
        return;
    }

    // latency depends on polling frequency
#ifdef LATENCY_TESTING
    const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const double barq_ms = static_cast<double>(now_ns - ts) / 1e6;
    std::cout << "BARQ latency: " << barq_ms << " ms" << std::endl;
#endif

    const uint8_t* buf = static_cast<const uint8_t*>(ptr);
    uint32_t width, height, point_step;
    double timestamp;

    size_t off = 0;
    std::memcpy(&width,      buf + off, sizeof(width));      off += sizeof(width);
    std::memcpy(&height,     buf + off, sizeof(height));     off += sizeof(height);
    std::memcpy(&point_step, buf + off, sizeof(point_step)); off += sizeof(point_step);
    std::memcpy(&timestamp,  buf + off, sizeof(timestamp));  off += sizeof(timestamp);

/**
 * This timestamp confirms the exact same result as before, so not useful to add the BARQ header parsing time to the overall latency study
 * 
 #ifdef LATENCY_TESTING
 const double header_ms = static_cast<double>(now_ns - ts) / 1e6;
 std::cout << "BARQ header parsing latency: " << header_ms << " ms" << std::endl;
 #endif
 *
 */

    const uint8_t* points = buf + off;
    
#ifdef ENABLE_VERBOSE
    const size_t points_bytes = width * point_step;
    if (off + points_bytes > sz) {
        std::cout << "[BARQ] Warning: frame size mismatch (expected at least " << off + points_bytes << " bytes, got " << sz << " bytes)" << std::endl;
        return;
    }
#endif

    unsigned int inputSize = width;
    size_t totalElements = inputSize * 4;

    reserveAndResize(totalElements);

    for (uint32_t i = 0; i < width; ++i) {
        const uint8_t* p = points + i * point_step;
        std::memcpy(&h_input[i*4+0], p,      sizeof(float));
        std::memcpy(&h_input[i*4+1], p+4,    sizeof(float));
        std::memcpy(&h_input[i*4+2], p+8,    sizeof(float));
        std::memcpy(&h_input[i*4+3], p+12,   sizeof(float));
    }
    d_input = h_input;

    runPipeline(inputSize);
}
#endif

// Extracted from scanCallback and onTimer since ROS2 and BARQ share the same processing pipeline after input conversion
void ControllerNode::runPipeline(unsigned int inputSize)
{
#if defined(ENABLE_VERBOSE)
    auto t_start = std::chrono::steady_clock::now();
#endif

    // ----------------------------------------------
    // initialize variables and clear cones marker
    // ----------------------------------------------
    cones->points = {};
    unsigned int size = 0;
    float* raw_in  = nullptr;
    float* raw_out = nullptr;

    // -----------------------------------------
    // CUDA Filtering (if enabled)
    // -----------------------------------------
    if (this->filterFlag) {
        raw_in  = thrust::raw_pointer_cast(d_input.data());
        raw_out = thrust::raw_pointer_cast(d_output.data());

        // ----------------------------------------------
        // Call for cudaFilter
        // ----------------------------------------------
        this->cudaFilter->filterPoints(raw_in, inputSize, &raw_out, &size, compute_stream);
        inputSize = size;
        
        // after the swap d_input holds the filtered result
        // while the kernel writes on d_output, 
        // we can start copying the filtered result back to host for publishing
        d_input.swap(d_output);

        if (this->publishFilteredPc) {
            cudaMemcpyAsync(h_input.data(),
                thrust::raw_pointer_cast(d_input.data()),
                size * 4 * sizeof(float), cudaMemcpyDeviceToHost, copy_stream);
            cudaStreamSynchronize(copy_stream);
            this->publishPc(h_input.data(), size, filtered_cp_pub);
        }
    }

    // -----------------------------------------
    // CUDA Segmentation (if enabled)
    // -----------------------------------------
    if (this->segmentFlag) {
        raw_in  = thrust::raw_pointer_cast(d_input.data());
        raw_out = thrust::raw_pointer_cast(d_output.data());
        
        segmentation->segment(raw_in, inputSize, raw_out, &size, compute_stream);
        inputSize = size;
        
        // after the swap d_input holds the segmented result
        // while the kernel writes on d_output, 
        // we can start copying the segmented result back to host for publishing
        d_input.swap(d_output);

        if (this->publishSegmentedPc && size != 0) {
            cudaMemcpyAsync(h_input.data(),
                thrust::raw_pointer_cast(d_input.data()),
                size * 4 * sizeof(float), cudaMemcpyDeviceToHost, copy_stream);
            cudaStreamSynchronize(copy_stream);
            publishPc(h_input.data(), size, segmented_cp_pub);
        }
    }

    // -----------------------------------------
    // CUDA Clustering (if enabled)
    // -----------------------------------------
    if (this->clusteringFlag) {
        raw_in  = thrust::raw_pointer_cast(d_input.data());
        raw_out = thrust::raw_pointer_cast(d_output.data());

        this->clustering->extractClusters(raw_in, inputSize, raw_out, cones, compute_stream);

#if defined(ENABLE_VERBOSE) || defined(LOGGER_PUB)
        auto tend = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::ratio<1,1000>> time_span =
            std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1,1000>>>(tend - t_start);
#endif
#ifdef ENABLE_VERBOSE
        std::cout << "Total processing time: " << time_span.count() << " ms\n" << std::endl;
        if (time_span.count() > this->limitWarning_ms)
            std::cout << "Warning: exceeded " << this->limitWarning_ms << " ms!\n" << std::endl;
#endif
        if (this->publishCluster) {
            cones->header.stamp = this->now();
            cones_array_pub->publish(*cones);
        }

#ifdef NSIGHT_SDK
        nsight_iterations++;
        if (nsight_iterations >= nsight_max_iterations) {
            // Emulate Ctrl+C by sending SIGINT to this process.
            std::raise(SIGINT);
        }
#endif
    }
}