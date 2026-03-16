#include "cuda_clustering/utils/pointcloud_converter.hpp"

namespace pointcloud_utils
{

    void convertPointCloud2ToFloatArray(
        const sensor_msgs::msg::PointCloud2::SharedPtr &sub_cloud,
        float *output_data)
    {
        // assegno variabili per migliorare leggibilità
        const std::uint32_t width = sub_cloud->width;
        const std::uint32_t height = sub_cloud->height;

        std::uint8_t *cloud_data = reinterpret_cast<std::uint8_t *>(output_data);

        const std::uint8_t *input_data = sub_cloud->data.data();
        const std::uint32_t row_step = sub_cloud->row_step;
        const std::uint32_t point_step = sub_cloud->point_step;

        for (std::uint32_t r = 0; r < height; ++r)
        {
            const std::uint8_t *row_data = input_data + r * row_step;

            for (std::uint32_t c = 0; c < width; ++c)
            {

                const std::uint8_t *pt_data = row_data + c * point_step;
                std::memcpy(cloud_data, pt_data, 3 * sizeof(float));
                cloud_data += sizeof(pcl::PointXYZ);
            }
        }
    }
}
