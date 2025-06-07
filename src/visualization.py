import open3d as o3d
import numpy as np
import os

from src.utils import create_transform_matrix, transform_points


all_points = []

def visualize_all_lidars(trucksc, sample):

    for sensor in ['LIDAR_LEFT', 'LIDAR_RIGHT', 'LIDAR_TOP_FRONT', 'LIDAR_REAR']:
        # Get sample_data and file path
        lidar_data = trucksc.get('sample_data', sample['data'][sensor])
        filepath = os.path.join(trucksc.dataroot, lidar_data['filename'])

        # Load point cloud
        pcd = o3d.io.read_point_cloud(filepath)
        points = np.asarray(pcd.points)

        # Get calibration transform to ego frame
        calib = trucksc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        transform = create_transform_matrix(calib['translation'], calib['rotation'])

        # Apply transform to ego frame
        points_ego = transform_points(points, transform)
        all_points.append(points_ego)

    # Merge all points
    combined_points = np.vstack(all_points)

    # Visualize
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(combined_points)
    o3d.visualization.draw_geometries([pcd_all])
