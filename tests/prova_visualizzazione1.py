import sys
import os

# Add the path to src/ to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'truckscenes', 'src')))

from truckscenes import TruckScenes
import matplotlib.pyplot as plt
import numpy as np




trucksc = TruckScenes('v1.0-mini', 'data\mini_dataset\man-truckscenes', True)

print('-'*50)
trucksc.list_scenes()
print('-'*50)

my_scene = trucksc.scene[0]
print(my_scene)
print('-'*50)

first_sample_token = my_scene['first_sample_token']
# trucksc.render_sample_radar_lidar(first_sample_token)      # camera + radar + lidar
# plt.show()

my_sample = trucksc.get('sample', first_sample_token)
print('-'*50)
trucksc.list_sample(my_sample['token'])
print('-'*50)

sensor = 'CAMERA_RIGHT_FRONT'
cam_front_data = trucksc.get('sample_data', my_sample['data'][sensor])
# trucksc.render_sample_data(cam_front_data['token'])     # camera
# plt.show()


my_annotation_token = my_sample['anns'][14]
my_annotation_metadata = trucksc.get('sample_annotation', my_annotation_token)
# trucksc.render_annotation(my_annotation_token)
# plt.show()

my_instance = trucksc.get('instance', my_annotation_metadata['instance_token'])
instance_token = my_instance['token']
# trucksc.render_instance(instance_token)
# plt.show()

