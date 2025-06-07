import numpy as np
from truckscenes import TruckScenes



def init_dataset(version: str, root: str, verbose=True):
    return TruckScenes(version, root, verbose)



def get_sample_data(trucksc, sample_token: str, sensor: str):
    """
    Gets data and metadata for a given sensor at a sample.

    Args:
        trucksc: TruckScenes object
        sample_token: token of the sample
        sensor: sensor name string (e.g., 'LIDAR_LEFT' or 'RADAR_REAR')

    Returns:
        dict with data, sensor pose, and calibration
    """

    sample = trucksc.get('sample', sample_token)
    sensor_data = trucksc.get('sample_data', sample['data'][sensor])
    calibrated_sensor = trucksc.get('calibrated_sensor', sensor_data['calibrated_sensor_token'])
    ego_pose = trucksc.get('ego_pose', sensor_data['ego_pose_token'])
    return {
        'data': sensor_data,
        'calibration': calibrated_sensor,
        'ego_pose': ego_pose
    }