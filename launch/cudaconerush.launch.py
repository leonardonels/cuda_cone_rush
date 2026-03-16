from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os


def generate_launch_description():
    ld = LaunchDescription()

    config_node = os.path.join(
        get_package_share_directory('cuda_cone_rush'),
        'config',
        'config.yaml'
        )

    node=Node(
            package='cuda_cone_rush',
            name='cuda_cone_rush_node',
            executable='cuda_cone_rush_node',
            output='screen',
            parameters=[config_node]
        )

    ld.add_action(node)
    return ld