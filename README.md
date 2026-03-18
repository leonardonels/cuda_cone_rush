# cuCONERUSH

cuCONERUSH is a GPU-first ROS 2 package for real-time LiDAR cones detection.
The runtime package name is `cuda_cone_rush`.

The processing pipeline stays on CUDA buffers across stages:

```text
PointCloud2 -> optional XYZ filter -> optional ground segmentation -> optional voxel clustering -> cone markers
```

## Tested on
| Lidar Points | Board | avg exec (ms) |
| --- | --- | --- |
| 128K | Orin Nano Super | 10ms |
| 128K | Orin Dev 2048 cuda | 5ms |
| 128K | RTX 4060 Mobile 3072 cuda | 1.1ms |

## What Is Included

- CUDA passthrough filtering (`cuda_filtering`)
- CUDA RANSAC ground segmentation (`cuda_segmentation`)
- CUDA voxel clustering + cone dimension filtering (`cuda_clustering` + `cluster_filtering`)
- Two point cloud conversion backends:
    - CUDA backend (`src/utils/cuda_pointcloud_converter.cu`)
    - CPU backend (`src/utils/pointcloud_converter.cpp`)
- Optional performance logging to ROS topic (`LOGGER_PUB`)

## Requirements

- ROS 2 Humble
- CUDA Toolkit
- PCL
- `ament_cmake`

## Build

From your workspace root:

```bash
cd ~/ros2_ws
colcon build --packages-select cuda_cone_rush
source install/setup.bash
```

## CMake Options

These options are available in `CMakeLists.txt`:

| Option | Default | Description |
|---|---|---|
| `ENABLE_VERBOSE` | `OFF` | Prints processing timings to stdout |
| `LOGGER_PUB` | `OFF` | Publishes average processing time to `/logger/clustering/time` |
| `USE_CUDA_POINTCLOUD_CONVERTER` | `ON` | Uses CUDA converter instead of CPU converter |
| `USE_PINNED_MEMORY` | `ON` on x86, `OFF` otherwise | Enables pinned host memory for host/device transfers |

Example:

```bash
colcon build --packages-select cuda_cone_rush --cmake-args -DENABLE_VERBOSE=ON -DLOGGER_PUB=ON
```

## Run

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch cuda_cone_rush cudaconerush.launch.py
```

## Configuration

Parameters are loaded from `config/config.yaml` under `/cuda_cone_rush_node`.

### Topics

- `input_topic`
- `filtered_topic`
- `segmented_topic`
- `cluster_topic`
- `frame_id`

### Clustering

- `minClusterSize`
- `maxClusterSize`
- `voxelX`
- `voxelY`
- `voxelZ`
- `countThreshold`

### Cluster Dimension Filter

- `clusterMaxX`, `clusterMaxY`, `clusterMaxZ`
- `clusterMinX`, `clusterMinY`, `clusterMinZ`
- `maxHeight`

### Passthrough Filter Bounds

- `downFilterLimitX`, `upFilterLimitX`
- `downFilterLimitY`, `upFilterLimitY`
- `downFilterLimitZ`, `upFilterLimitZ`

### Segmentation

- `distanceThreshold`
- `maxSegmentationDistance`
- `maxIterations`
- `probability`

### Pipeline Toggles and Timing Guard

- `filter`
- `publishFilteredPc`
- `segment`
- `publishSegmentedPc`
- `clustering`
- `publishCluster`
- `limitWarning_ms`

## ROS 2 Interface

Subscriptions:

- `input_topic` (`sensor_msgs/msg/PointCloud2`)

Publications:

- `filtered_topic` (`sensor_msgs/msg/PointCloud2`) when `publishFilteredPc=true`
- `segmented_topic` (`sensor_msgs/msg/PointCloud2`) when `publishSegmentedPc=true`
- `cluster_topic` (`visualization_msgs/msg/Marker`) when `publishCluster=true`
- `/logger/clustering/time` (`std_msgs/msg/Float64`) when compiled with `LOGGER_PUB=ON`

## Package Layout

```text
src/
    main.cpp
    controller_node.cu
    filtering/cuda_filtering.cu
    segmentation/cuda_segmentation.cu
    clustering/cuda_clustering.cu
    clustering/cluster_filtering/dimension_filter.cpp
    utils/cuda_pointcloud_converter.cu
    utils/pointcloud_converter.cpp

include/cuda_clustering/
    controller_node.hpp
    filtering/
    segmentation/
    clustering/
    utils/
```

## License

Apache-2.0
