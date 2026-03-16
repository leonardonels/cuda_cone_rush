#pragma once
#include <geometry_msgs/msg/point.hpp>
#include <optional>
struct cluster_filter
{
    float clusterMaxX, clusterMaxY, clusterMaxZ, clusterMinX, clusterMinY, clusterMinZ, maxHeight;
};

class IClusterFiltering
{
    public:
        virtual ~IClusterFiltering() = default;
        virtual std::optional<geometry_msgs::msg::Point> analiseCluster(float* cluster, unsigned int points_num) = 0;
};