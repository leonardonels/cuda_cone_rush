#include "cuda_cone_rush/controller_node.hpp"

#include <csignal>
#include <memory>
#include <cstdint>

/**
 * TODO:
 * [ ] remove the first host copy from ros -> host -> device to ros -> device 
 */
ControllerNode::ControllerNode() : Node("cuda_cone_rush_node")
{
    this->loadParameters();
    this->getInfo();
    this->MessageInit();

    this->converter_ = std::make_unique<CudaConverter>();

    this->cudaFilter_ = std::make_unique<CudaFilter>(upFilterLimitX, downFilterLimitX,
                                   upFilterLimitY, downFilterLimitY,
                                   upFilterLimitZ, downFilterLimitZ);

    this->segmentation_ = std::make_unique<CudaSegmentation>(segP);

    this->clustering_ = std::make_unique<CudaClustering>(param);

#ifdef ENABLE_BARQ
    if (this->barq_enabled_) {

        /* Create BARQ subscriber */
        this->BARQReaderInit();
    }else {
        this->ROSListenerInit();
    }
#else
    this->ROSListenerInit();
#endif

    if(this->clusteringFlag && this->publishCluster)
        this->cones_array_pub = this->create_publisher<visualization_msgs::msg::Marker>(this->cluster_topic, 100);
    if(this->filterFlag && this->publishFilteredPc)
        this->filtered_cp_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(this->filtered_topic, 100);
    if(this->segmentFlag && this->publishSegmentedPc)
        this->segmented_cp_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(this->segmented_topic, 100);

#ifdef LATENCY_TESTING
    /* Reliable and deep, for the same reason the driver's frame tick is: this is
       a ~28 byte message and a dropped one is a measurement that can never be
       taken, which would show up as a missing sample rather than as an error. */
    this->latency_sample_pub_ = this->create_publisher<mmr_base::msg::LatencySample>(
        "/latency/sample/cone_rush", rclcpp::QoS(rclcpp::KeepLast(100)).reliable());
    RCLCPP_INFO(this->get_logger(),
        "LATENCY_TESTING: per-frame samples on /latency/sample/cone_rush");
#endif
                                                                               
    /* Cuda streams init*/
    cudaStreamCreate(&copy_stream);
    cudaStreamCreate(&compute_stream);
}

ControllerNode::~ControllerNode()
{
    if (copy_stream) cudaStreamDestroy(copy_stream);
    if (compute_stream) cudaStreamDestroy(compute_stream);
}

void ControllerNode::transformConesToOutputFrame()
{
    /* No-op when the extrinsics were unavailable: cone_frame_id_ is then the
       lidar frame and the points are already correct in it. */
    if (!this->have_extrinsics_) return;

    for (auto & p : this->cones->points) {
        const double x = p.x, y = p.y, z = p.z;
        /* R is lidar2baselink as written in rows -- it maps a vector in the
           lidar frame into base_link, which is precisely this direction. See
           the note in vehicle_extrinsics.yaml before "fixing" the indexing:
           transposing does not cancel the mounting tilt, it doubles it. */
        p.x = lidar_R_[0] * x + lidar_R_[1] * y + lidar_R_[2] * z + lidar_t_[0];
        p.y = lidar_R_[3] * x + lidar_R_[4] * y + lidar_R_[5] * z + lidar_t_[1];
        p.z = lidar_R_[6] * x + lidar_R_[7] * y + lidar_R_[8] * z + lidar_t_[2];
    }
}

void ControllerNode::MessageInit()
{
    cones->header.frame_id = this->cone_frame_id_;
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
}

void ControllerNode::ROSListenerInit(){
    auto qos = rclcpp::QoS(rclcpp::SensorDataQoS());
    // depth is already 5 from the profile, override if needed:
    // qos.keep_last(10);
        
    /* Create ROS2 subscriber */
#ifdef ENABLE_ZERO_COPY
    if(this->zero_copy_enabled){
        this->bounded_sub = this->create_subscription<mmr_base::msg::BoundedPointcloud>(this->input_topic, qos,
                                                                                       std::bind(&ControllerNode::boundedPointcloud, this, std::placeholders::_1));
        RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "Subscribed to bounded topic: %s", this->input_topic.c_str());
    }else{
#endif
        this->cloud_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(this->input_topic, qos,
                                                                                       std::bind(&ControllerNode::standardPointcloud, this, std::placeholders::_1));
        RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "Subscribed to topic: %s", this->input_topic.c_str());
#ifdef ENABLE_ZERO_COPY
    }
#endif
}

#ifdef ENABLE_BARQ
void ControllerNode::BARQReaderInit()
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
            /* duration<double, milli>, NOT std::chrono::milliseconds: the latter
               has an integral rep (int64_t), so its converting constructor is
               disabled for a floating-point argument and sub-millisecond periods
               are not merely truncated, they do not compile. create_wall_timer
               takes any duration and converts to nanoseconds internally. */
            this->timer_ = create_wall_timer(
                std::chrono::duration<double, std::milli>(this->barq_polling_rate_ms_),
                std::bind(&ControllerNode::onTimer, this));
            RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "BARQ reader initialized successfully.");
            
            this->writer_health_check_timer_ = create_wall_timer(std::chrono::milliseconds(this->barq_writer_timeout_ms_), std::bind(&ControllerNode::checkForWriterHealth, this));
            RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "BARQ writer health check initialized successfully."); 

            this->barq_retries_= 0 ;
            break;
        }
    } while (this->barq_retries_ < this->barq_max_retries_);
    if (this->barq_retries_ >= this->barq_max_retries_) {
        RCLCPP_ERROR(rclcpp::get_logger("cuda_cone_rush_node"), "Exceeded maximum BARQ initialization retries (%d), switching to ROS2 subscriber.", this->barq_max_retries_);
        // std::raise(SIGINT);  // Gracefully shutdown the node
        if (this->timer_) this->timer_->cancel();
        if (this->writer_health_check_timer_) this->writer_health_check_timer_->cancel();
        this->ROSListenerInit();    // Fallback to ROS2 subscriber if BARQ fails to initialize
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
    declare_parameter("frame_id", "lidar_link");
    declare_parameter("output_frame_id", "base_link");
    /* From the shared vehicle_extrinsics.yaml (wildcard node key), the same
       block fast_limo and vehicle_tf_node read -- docs/FRAMES.md rule 5. Empty
       default so a missing file is detectable rather than silently identity. */
    declare_parameter<std::vector<double>>("extrinsics.lidar.t", std::vector<double>{});
    declare_parameter<std::vector<double>>("extrinsics.lidar.R", std::vector<double>{});

    // ================ Zero Copy options =================
    declare_parameter("zero_copy_enabled", true);
    
    // ================ BARQ options =================
    declare_parameter("BARQ_enabled", false);
#ifdef ENABLE_BARQ
    declare_parameter("BARQ_topic", "/lidar_points");
    declare_parameter("BARQ_retry_delay_ms", 10);
    declare_parameter("BARQ_max_retries", 5);
    /* Declared as a DOUBLE (note the 1.0). Declared as an int, rcl rejects a
       fractional override outright -- "BARQ_polling_rate_ms: 0.1" then aborts
       the node at startup, which as a forked child of as_demo is silent: the
       node is simply absent. The corollary is that every params file must now
       write a float literal; a bare `1` is an integer override against a double
       parameter and is rejected just as hard. */
    declare_parameter("BARQ_polling_rate_ms", 1.0);  // [ms]; 1.0 -> ~1 kHz, 0.1 -> ~10 kHz
    declare_parameter("BARQ_writer_timeout_ms", 1000);
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
    get_parameter("output_frame_id", this->output_frame_id);

    {
        const auto t = get_parameter("extrinsics.lidar.t").as_double_array();
        const auto R = get_parameter("extrinsics.lidar.R").as_double_array();
        this->have_extrinsics_ = (t.size() == 3 && R.size() == 9);
        if (this->have_extrinsics_) {
            for (int i = 0; i < 3; ++i) this->lidar_t_[i] = t[i];
            for (int i = 0; i < 9; ++i) this->lidar_R_[i] = R[i];
            this->cone_frame_id_ = this->output_frame_id;
            RCLCPP_INFO(this->get_logger(),
                        "publishing %s in '%s' (transformed from '%s')",
                        this->cluster_topic.c_str(), this->cone_frame_id_.c_str(),
                        this->frame_id.c_str());
        } else {
            /* Publish in the frame the data is actually in. Labelling untransformed
               lidar-frame cones as base_link would be exactly the silent lie
               rule 8 exists to prevent. */
            this->cone_frame_id_ = this->frame_id;
            RCLCPP_WARN(this->get_logger(),
                        "extrinsics.lidar.{t,R} missing -- is vehicle_extrinsics.yaml "
                        "passed to this node? Publishing %s in '%s' untransformed.",
                        this->cluster_topic.c_str(), this->cone_frame_id_.c_str());
        }
    }

    // ================ Zero Copy options =================
    get_parameter("zero_copy_enabled", this->zero_copy_enabled);

    // ================ BARQ options =================
    get_parameter("BARQ_enabled", this->barq_enabled_);
#ifdef ENABLE_BARQ
    get_parameter("BARQ_topic", this->barq_topic);
    get_parameter("BARQ_retry_delay_ms", this->barq_retry_delay_ms_);
    get_parameter("BARQ_max_retries", this->barq_max_retries_);
    get_parameter("BARQ_polling_rate_ms", this->barq_polling_rate_ms_);
    get_parameter("BARQ_writer_timeout_ms", this->barq_writer_timeout_ms_);
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
        RCLCPP_INFO(rclcpp::get_logger("cuda_cone_rush_node"), "-------------------------");
    }
}

#ifdef ENABLE_BARQ
void ControllerNode::checkForWriterHealth()
{   if (reader_->isWriterAlive()) return;
    RCLCPP_ERROR(rclcpp::get_logger("cuda_cone_rush_node"), "BARQ writer is not alive. Attempting to reinitialize...");
    this->timer_->cancel();
    this->reader_->destroy();
    this->BARQReaderInit();
}
#endif

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

void ControllerNode::standardPointcloud(sensor_msgs::msg::PointCloud2::SharedPtr sub_cloud)
{
#ifdef LATENCY_TESTING
    /* FIRST statement of the callback: after the middleware has delivered and
       deserialised, which is where it belongs. Deserialisation is a cost plain
       ROS pays and zero copy does not, so it has to fall INSIDE the delivery
       interval being compared, not outside it. */
    latency_t_in_ = monotonicNs();
#endif
#ifdef ENABLE_VERBOSE
    /* Capture-to-receive, NOT transport latency: it includes the driver's whole
       decode. It used to live under LATENCY_TESTING, where it printed a line per
       frame from the hot callback -- blocking stdout I/O inside the very
       interval being measured, and more expensive than everything the sampling
       above costs. Kept here for bench debugging; never enable it for a run
       whose numbers you will quote. */
    {
        const auto now = rclcpp::Clock(RCL_ROS_TIME).now();
        const auto latency = (now - rclcpp::Time(sub_cloud->header.stamp)).nanoseconds() / 1e6;
        std::cout << "ROS2 capture-to-receive: " << latency << " ms" << std::endl;
    }
#endif

    unsigned int inputSize = sub_cloud->width * sub_cloud->height;
    size_t totalElements = inputSize * 4;   // XYZI

    reserveAndResize(totalElements);

    inputSize = converter_->convert(sub_cloud, d_input);

    runPipeline(inputSize, rclcpp::Time(sub_cloud->header.stamp));
}

#ifdef ENABLE_ZERO_COPY
void ControllerNode::boundedPointcloud(mmr_base::msg::BoundedPointcloud::SharedPtr sub_cloud)
{
#ifdef LATENCY_TESTING
    // See standardPointcloud: first statement, deliberately after deserialisation.
    latency_t_in_ = monotonicNs();
#endif
#ifdef ENABLE_VERBOSE
    /* NOTE: BoundedPointcloud has no header -- the old block here read
       sub_cloud->header.stamp, which only compiled because it never did. The
       frame time is the `stamp` field. */
    {
        const auto now = rclcpp::Clock(RCL_ROS_TIME).now();
        const auto latency = (now - rclcpp::Time(sub_cloud->stamp)).nanoseconds() / 1e6;
        std::cout << "zero-copy capture-to-receive: " << latency << " ms" << std::endl;
    }
#endif

    unsigned int inputSize = sub_cloud->width;
    size_t totalElements = inputSize * 4;   // XYZI

    reserveAndResize(totalElements);

    /* The length is width*26, NOT data.size(). BoundedPointcloud::data is a
       fixed std::array<uint8_t, 3407872> -- data.size() is that constant no
       matter how full the frame is -- and convert() memcpys exactly in_bytes to
       the device. At the ~64.7k points these clouds actually carry that copied
       3.41 MB per frame where 1.68 MB was live: 2.03x the bytes, ~35 MB/s of
       waste at 20 Hz, charged to this node and to the zero-copy path only. The
       PointCloud2 overload is unaffected because there data.size() == row_step *
       height, i.e. already exact. */
    const std::size_t live_bytes = std::size_t(26) * sub_cloud->width;
    inputSize = converter_->convert(sub_cloud->data.data(), live_bytes,
                                    sub_cloud->width, std::uint32_t(1),
                                    std::uint32_t(26)*sub_cloud->width, std::uint32_t(26),
                                    d_input);

    /* BoundedPointcloud has no header; its `stamp` field IS the sensor
       frame-start time (see the msg definition). */
    runPipeline(inputSize, rclcpp::Time(sub_cloud->stamp));
}
#endif

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

#ifdef LATENCY_TESTING
    /* The BARQ arm's t_in is the POLL THAT FOUND THE DATA, not the instant the
       data arrived -- there is no callback here to be woken. So its delivery
       figure carries up to BARQ_polling_rate_ms of poll (1 ms as shipped,
       ~0.5 ms on average) that is not transport. Say so wherever it is plotted;
       it is the one arm whose number is not purely the transport's. */
    latency_t_in_ = monotonicNs();
#endif
#ifdef ENABLE_VERBOSE
    {
        const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        std::cout << "BARQ capture-to-receive: "
                  << static_cast<double>(now_ns - ts) / 1e6 << " ms" << std::endl;
    }
#endif

    const uint8_t* buf = static_cast<const uint8_t*>(ptr);
    uint32_t width, height, point_step;
    double timestamp;

    size_t off = 0;
    std::memcpy(&width,      buf + off, sizeof(width));      off += sizeof(width);
    std::memcpy(&height,     buf + off, sizeof(height));     off += sizeof(height);
    std::memcpy(&point_step, buf + off, sizeof(point_step)); off += sizeof(point_step);
    std::memcpy(&timestamp,  buf + off, sizeof(timestamp));  off += sizeof(timestamp);

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

    inputSize = converter_->convert(points, static_cast<size_t>(width) * point_step,
                                    width, 1, width * point_step, point_step, d_input);

    /* BARQ frame header carries the frame START time in seconds (see the layout
       comment above) — that is the acquisition time, not the writer's publish
       time, so it is the right stamp to propagate. */
    runPipeline(inputSize, rclcpp::Time(static_cast<int64_t>(timestamp * 1e9), RCL_ROS_TIME));
}
#endif

#ifdef LATENCY_TESTING
/*
 * One sample per frame: which frame, when this node got it, when it was ready to
 * publish. Nothing derived, nothing averaged -- as_demo's monitor owns every
 * statistic, so this stays a fixed-size fill and a publish with no allocation
 * and no branching on the data.
 *
 * `frame_stamp` is the LiDAR capture stamp, which is the same value under all
 * three transports (PointCloud2::header.stamp, BoundedPointcloud::stamp, and the
 * BARQ frame header's timestamp all carry it), so a sample joins against the
 * driver's tick and against the downstream nodes' samples with no per-transport
 * special casing.
 */
void ControllerNode::publishLatencySample(const rclcpp::Time& frame_stamp, int64_t t_out)
{
    if (!latency_sample_pub_) return;

    mmr_base::msg::LatencySample m;
    m.frame_stamp = frame_stamp;
    m.t_in = latency_t_in_;
    m.t_out = t_out;
    m.seq = latency_seq_++;
    latency_sample_pub_->publish(m);
}
#endif

void ControllerNode::runPipeline(unsigned int inputSize, const rclcpp::Time& capture_stamp)
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

        this->cudaFilter_->filterPoints(raw_in, inputSize, &raw_out, &size, compute_stream);
        inputSize = size;

        // after the swap d_input holds the filtered result
        d_input.swap(d_output);

        if (this->publishFilteredPc) {
            cudaMemcpyAsync(thrust::raw_pointer_cast(h_input.data()),
                thrust::raw_pointer_cast(d_input.data()),
                size * 4 * sizeof(float), cudaMemcpyDeviceToHost, copy_stream);
            cudaStreamSynchronize(copy_stream);
            this->publishPc(thrust::raw_pointer_cast(h_input.data()), size, filtered_cp_pub);
        }
    }

    // -----------------------------------------
    // CUDA Segmentation (if enabled)
    // -----------------------------------------
    if (this->segmentFlag) {
        raw_in  = thrust::raw_pointer_cast(d_input.data());
        raw_out = thrust::raw_pointer_cast(d_output.data());

        segmentation_->segment(raw_in, inputSize, raw_out, &size, compute_stream);
        inputSize = size;

        // after the swap d_input holds the segmented result
        d_input.swap(d_output);

        if (this->publishSegmentedPc && size != 0) {
            cudaMemcpyAsync(thrust::raw_pointer_cast(h_input.data()),
                thrust::raw_pointer_cast(d_input.data()),
                size * 4 * sizeof(float), cudaMemcpyDeviceToHost, copy_stream);
            cudaStreamSynchronize(copy_stream);
            publishPc(thrust::raw_pointer_cast(h_input.data()), size, segmented_cp_pub);
        }
    }

    // -----------------------------------------
    // CUDA Clustering (if enabled)
    // -----------------------------------------
    if (this->clusteringFlag) {
        raw_in  = thrust::raw_pointer_cast(d_input.data());
        raw_out = thrust::raw_pointer_cast(d_output.data());

        this->clustering_->extractClusters(raw_in, inputSize, raw_out, cones, compute_stream);

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
            /* Clustering produced these in the lidar frame; move them into the
               published frame before anyone sees them. */
            this->transformConesToOutputFrame();
            /* Stamp with the SOURCE cloud's acquisition time, not now(): the
               cones describe the world as it was when the scan was taken, and
               downstream needs that to fuse them against the matching pose. */
            cones->header.stamp = capture_stamp;
#ifdef LATENCY_TESTING
            const int64_t latency_t_out = monotonicNs();
#endif
            cones_array_pub->publish(*cones);
#ifdef LATENCY_TESTING
            publishLatencySample(capture_stamp, latency_t_out);
#endif
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