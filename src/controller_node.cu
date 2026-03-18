#include "cuda_cone_rush/controller_node.hpp"

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

    /* Define SAHM transport layer */
    const size_t kMaxPoints = 300000;
    const size_t kPointStep = sizeof(float) * 4 + sizeof(uint16_t) + sizeof(double);
    const size_t kHeaderBytes = sizeof(uint32_t) * 3 + sizeof(double);
    sahm_max_size_ = kMaxPoints * kPointStep + kHeaderBytes + 512;
    RCLCPP_INFO(get_logger(), "[SAHM] reader sahm_max_size_=%zu", sahm_max_size_);
    
    this->reader_ = std::make_unique<SAHM::DirectReader>("/hesai_pointcloud", sahm_max_size_, 30);

    if (!reader_->init()) {
        RCLCPP_ERROR(get_logger(), "[SAHM] Failed to initialize reader!");
        return;
    }
    
    // Poll SAHM at 30Hz — match your lidar publish rate
    this->timer_ = create_wall_timer(
      std::chrono::milliseconds(33),
      std::bind(&ControllerNode::onTimer, this)
    );

    /* Define QoS for Best Effort messages transport */
    // auto qos = rclcpp::QoS(rclcpp::KeepLast(10), rmw_qos_profile_sensor_data);

    this->cones_array_pub = this->create_publisher<visualization_msgs::msg::Marker>(this->cluster_topic, 100);
    if(this->filterFlag && this->publishFilteredPc)
        this->filtered_cp_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(this->filtered_topic, 100);
    if(this->segmentFlag && this->publishSegmentedPc)
        this->segmented_cp_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(this->segmented_topic, 100);
    #ifdef LOGGER_PUB
        this->logger_pub = this->create_publisher<std_msgs::msg::Float64>("/logger/clustering/time", 100);
    #endif

    /* Create subscriber */
    /*this->cloud_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(this->input_topic, qos,
                                                                               std::bind(&ControllerNode::scanCallback, this, std::placeholders::_1));
                                                                               */
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

void ControllerNode::loadParameters()
{
    declare_parameter("input_topic", "/lidar_points");
    declare_parameter("segmented_topic", "/segmented_points");
    declare_parameter("filtered_topic", "/filtered_points");
    declare_parameter("cluster_topic", "/clusters");
    declare_parameter("frame_id", "map");
    declare_parameter("minClusterSize", 0);
    declare_parameter("maxClusterSize", 0);
    declare_parameter("voxelX", 0.0);
    declare_parameter("voxelY", 0.0);
    declare_parameter("voxelZ", 0.0);
    declare_parameter("countThreshold", 0);

    declare_parameter("clusterMaxX", 0.0);
    declare_parameter("clusterMaxY", 0.0);
    declare_parameter("clusterMaxZ", 0.0);
    declare_parameter("clusterMinX", 0.0);
    declare_parameter("clusterMinY", 0.0);
    declare_parameter("clusterMinZ", 0.0);
    declare_parameter("maxHeight", 0.0);

    declare_parameter("downFilterLimitX", -1e10);
    declare_parameter("upFilterLimitX", 1e10);
    declare_parameter("downFilterLimitY", -1e10);
    declare_parameter("upFilterLimitY", 1e10);
    declare_parameter("downFilterLimitZ", 0.0);
    declare_parameter("upFilterLimitZ", 0.0);

    declare_parameter("filter", false);
    declare_parameter("segment", false);
    declare_parameter("publishFilteredPc", false);
    declare_parameter("publishSegmentedPc", false);
    declare_parameter("distanceThreshold", 0.1);
    declare_parameter("maxSegmentationDistance", 20.0);
    declare_parameter("maxIterations", 166);
    declare_parameter("probability", 0.75);
    declare_parameter("clustering", true);
    declare_parameter("publishCluster", true);
    declare_parameter("limitWarning_ms", 30);

    get_parameter("input_topic", this->input_topic);
    get_parameter("segmented_topic", this->segmented_topic);
    get_parameter("filtered_topic", this->filtered_topic);
    get_parameter("cluster_topic", this->cluster_topic);
    get_parameter("frame_id", this->frame_id);
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

    get_parameter("downFilterLimitX", this->downFilterLimitX);
    get_parameter("upFilterLimitX", this->upFilterLimitX);
    get_parameter("downFilterLimitY", this->downFilterLimitY);
    get_parameter("upFilterLimitY", this->upFilterLimitY);
    get_parameter("downFilterLimitZ", this->downFilterLimitZ);
    get_parameter("upFilterLimitZ", this->upFilterLimitZ);
    
    get_parameter("filter", this->filterFlag);
    get_parameter("segment", this->segmentFlag);
    get_parameter("publishFilteredPc", this->publishFilteredPc);
    get_parameter("publishSegmentedPc", this->publishSegmentedPc);
    get_parameter("distanceThreshold", this->segP.distanceThreshold);
    get_parameter("maxSegmentationDistance", this->segP.maxSegmentationDistance);
    get_parameter("maxIterations", this->segP.maxIterations);
    get_parameter("probability", this->segP.probability);
    get_parameter("clustering", this->clusteringFlag);
    get_parameter("publishCluster", this->publishCluster);
    get_parameter("limitWarning_ms", this->limitWarning_ms);
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

void ControllerNode::scanCallback(sensor_msgs::msg::PointCloud2::SharedPtr sub_cloud)
{
    // -----------------------------------------
    // Start timing
    // -----------------------------------------
    #if defined(ENABLE_VERBOSE) || defined(LOGGER_PUB)
        std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();
    #endif

    // ----------------------------------------------
    // initialize variables and clear cones marker
    // ----------------------------------------------
    cones->points = {};
    unsigned int size = 0;
    float* raw_in = nullptr;
    float* raw_out = nullptr;

    unsigned int inputSize = sub_cloud->width * sub_cloud->height;
    
    size_t totalElements = inputSize * 4;      // input size times number of fields for each point
    
    #ifdef ENABLE_VERBOSE
        auto t1 = std::chrono::steady_clock::now();
    #endif

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
    
    // -------------------------------------------------------
    // convert PointCloud2 to float array and copy to device
    // -------------------------------------------------------
    #ifdef USE_CUDA_POINTCLOUD_CONVERTER
        pointcloud_utils::convertPointCloud2ToFloatArray(sub_cloud, d_input, converter_res_);   // converto to device directly with cuda kernel
    #else
        pointcloud_utils::convertPointCloud2ToFloatArray(sub_cloud, h_input.data());
        d_input = h_input;
    #endif

    #ifdef ENABLE_VERBOSE
    auto t2 = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t2 - t1);
        std::cout << "PointCloud2 conversion and copy to device: " << duration.count() << " ms" << std::endl;
    #endif

    // -----------------------------------------
    // CUDA Filtering (if enabled)
    // -----------------------------------------
    if (this->filterFlag)
    {
        // ----------------------------------------------
        // Get raw pointers for your custom CUDA kernel
        // ----------------------------------------------
        raw_in = thrust::raw_pointer_cast(d_input.data());
        raw_out = thrust::raw_pointer_cast(d_output.data());

        // ----------------------------------------------
        // Call for cudaFilter
        // ----------------------------------------------
        this->cudaFilter->filterPoints(raw_in, inputSize, &raw_out, &size, compute_stream);
        inputSize = size;

        d_input.swap(d_output);

        if (this->publishFilteredPc)
        {
            // After swap, d_input holds the filtered result
            float* filtered_data = thrust::raw_pointer_cast(d_input.data());
            cudaMemcpyAsync(h_input.data(), filtered_data, size * 4 * sizeof(float), cudaMemcpyDeviceToHost, copy_stream);
            cudaStreamSynchronize(copy_stream);
            this->publishPc(h_input.data(), size, filtered_cp_pub);
        }

    }

    // -----------------------------------------
    // CUDA Segmentation (if enabled)
    // -----------------------------------------
    if (this->segmentFlag)
    {
        raw_in = thrust::raw_pointer_cast(d_input.data());
        raw_out = thrust::raw_pointer_cast(d_output.data());

        segmentation->segment(raw_in, inputSize, raw_out, &size, compute_stream);
        inputSize = size;

        d_input.swap(d_output);

        if (this->publishSegmentedPc && size != 0)
        {
            // After swap, d_input holds the segmented result
            float* segmented_data = thrust::raw_pointer_cast(d_input.data());
            cudaMemcpyAsync(h_input.data(), segmented_data, size * 4 * sizeof(float), cudaMemcpyDeviceToHost, copy_stream);
            cudaStreamSynchronize(copy_stream);
            publishPc(h_input.data(), size, segmented_cp_pub);
        }
    }

    // -----------------------------------------
    // CUDA Clustering (if enabled)
    // -----------------------------------------
    if (this->clusteringFlag)
    {
        raw_in = thrust::raw_pointer_cast(d_input.data());
        raw_out = thrust::raw_pointer_cast(d_output.data());

        // -----------------------------
        // call to extractClusters()
        // -----------------------------
        this->clustering->extractClusters(raw_in, inputSize, raw_out, cones, compute_stream);
        
        #if defined(ENABLE_VERBOSE) || defined(LOGGER_PUB)
            std::chrono::steady_clock::time_point tend = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::ratio<1, 1000>> time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(tend - t_start);
        #endif
        #ifdef ENABLE_VERBOSE
            std::cout << "Total processing time for this callback: " << time_span.count() << " ms\n" << std::endl;
            if (time_span.count() > this->limitWarning_ms) {
                std::cout << "----------------------------------------------------------------------------\n"
                          << "Warning: Processing time exceeded " << this->limitWarning_ms << " ms! Actual time: " << time_span.count() << " ms\n" 
                          << "----------------------------------------------------------------------------\n" << std::endl;
            } 
        #endif

        if (this->publishCluster)
        {
            cones->header.stamp = this->now();
            cones_array_pub->publish(*cones);
        }

        #ifdef LOGGER_PUB
            // Publish timing information to a ROS topic
            processing_time_ms += time_span.count();
            count++;
            if(count == 100)
            {
                std_msgs::msg::Float64 time_msg;
                time_msg.data = processing_time_ms / 100;  // average processing time
                logger_pub->publish(time_msg);
                processing_time_ms = 0.0;  // reset for next batch
                count = 0;
            }
        #endif
    }
}

void ControllerNode::onTimer()
{
    size_t sz = 0;
    const void* ptr = reader_->getLatest(sz);

    if (!ptr || sz == 0) {
        RCLCPP_WARN(get_logger(), "[SAHM] No data yet");
        return;
    }

    const uint8_t* buf = static_cast<const uint8_t*>(ptr);
    uint32_t width, height, point_step;
    double timestamp;

    size_t off = 0;
    std::memcpy(&width,      buf + off, sizeof(width));      off += sizeof(width);
    std::memcpy(&height,     buf + off, sizeof(height));     off += sizeof(height);
    std::memcpy(&point_step, buf + off, sizeof(point_step)); off += sizeof(point_step);
    std::memcpy(&timestamp,  buf + off, sizeof(timestamp));  off += sizeof(timestamp);

    const uint8_t* points = buf + off;
    const size_t points_bytes = width * point_step;

    if (off + points_bytes > sz) {
        RCLCPP_WARN(get_logger(), "[SAHM] Malformed frame, skipping");
        return;
    }

    // Call the processing pipeline directly
    this->processPointCloud(points, width, point_step, timestamp);
}

// Replaces scanCallback — same pipeline, different input source
void ControllerNode::processPointCloud(
    const uint8_t* points,
    uint32_t width,
    uint32_t point_step,
    double timestamp)
{
    #if defined(ENABLE_VERBOSE) || defined(LOGGER_PUB)
        auto t_start = std::chrono::steady_clock::now();
    #endif

    cones->points = {};
    unsigned int size = 0;
    float* raw_in = nullptr;
    float* raw_out = nullptr;

    unsigned int inputSize = width;
    size_t totalElements = inputSize * 4;

    #ifdef ENABLE_VERBOSE
        auto t1 = std::chrono::steady_clock::now();
    #endif

    if (h_input.capacity() < totalElements) {
        h_input.reserve(totalElements);
        d_input.reserve(totalElements);
        d_output.reserve(totalElements);
    }
    h_input.resize(totalElements);
    d_input.resize(totalElements);
    d_output.resize(totalElements);

    // Convert packed SAHM bytes → float array [x, y, z, intensity]
    // point_step = 4+4+4+4+2+8 = 26 bytes, but we only need x,y,z,intensity (first 16)
    for (uint32_t i = 0; i < width; ++i) {
        const uint8_t* p = points + i * point_step;
        std::memcpy(&h_input[i * 4 + 0], p,      sizeof(float));  // x
        std::memcpy(&h_input[i * 4 + 1], p + 4,  sizeof(float));  // y
        std::memcpy(&h_input[i * 4 + 2], p + 8,  sizeof(float));  // z
        std::memcpy(&h_input[i * 4 + 3], p + 12, sizeof(float));  // intensity
    }
    d_input = h_input;

    #ifdef ENABLE_VERBOSE
        auto t2 = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t2 - t1);
        std::cout << "SAHM conversion and copy to device: " << duration.count() << " ms" << std::endl;
    #endif

    // CUDA Filtering
    if (this->filterFlag) {
        raw_in  = thrust::raw_pointer_cast(d_input.data());
        raw_out = thrust::raw_pointer_cast(d_output.data());
        this->cudaFilter->filterPoints(raw_in, inputSize, &raw_out, &size, compute_stream);
        inputSize = size;
        d_input.swap(d_output);

        if (this->publishFilteredPc) {
            float* filtered_data = thrust::raw_pointer_cast(d_input.data());
            cudaMemcpyAsync(h_input.data(), filtered_data, size * 4 * sizeof(float), cudaMemcpyDeviceToHost, copy_stream);
            cudaStreamSynchronize(copy_stream);
            this->publishPc(h_input.data(), size, filtered_cp_pub);
        }
    }

    // CUDA Segmentation
    if (this->segmentFlag) {
        raw_in  = thrust::raw_pointer_cast(d_input.data());
        raw_out = thrust::raw_pointer_cast(d_output.data());
        segmentation->segment(raw_in, inputSize, raw_out, &size, compute_stream);
        inputSize = size;
        d_input.swap(d_output);

        if (this->publishSegmentedPc && size != 0) {
            float* segmented_data = thrust::raw_pointer_cast(d_input.data());
            cudaMemcpyAsync(h_input.data(), segmented_data, size * 4 * sizeof(float), cudaMemcpyDeviceToHost, copy_stream);
            cudaStreamSynchronize(copy_stream);
            publishPc(h_input.data(), size, segmented_cp_pub);
        }
    }

    // CUDA Clustering
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
            std::cout << "Total processing time: " << time_span.count() << " ms\n";
            if (time_span.count() > this->limitWarning_ms)
                std::cout << "Warning: exceeded " << this->limitWarning_ms << " ms!\n";
        #endif

        if (this->publishCluster) {
            cones->header.stamp = this->now();
            cones_array_pub->publish(*cones);
        }

        #ifdef LOGGER_PUB
            processing_time_ms += time_span.count();
            count++;
            if (count == 100) {
                std_msgs::msg::Float64 time_msg;
                time_msg.data = processing_time_ms / 100;
                logger_pub->publish(time_msg);
                processing_time_ms = 0.0;
                count = 0;
            }
        #endif
    }
}