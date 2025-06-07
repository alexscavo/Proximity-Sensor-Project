from scipy.spatial.transform import Rotation as R
import numpy as np

def create_transform_matrix(translation, rotation):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(rotation).as_matrix()
    T[:3, 3] = translation
    return T

def transform_points(points, transform):
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))  # Nx4
    transformed = (transform @ points_h.T).T
    return transformed[:, :3]