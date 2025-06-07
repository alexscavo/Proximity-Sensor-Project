
import sys
import os
import cv2
from matplotlib.patches import Rectangle

# Add truckscenes/src to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'truckscenes', 'src'))
import argparse
import matplotlib.pyplot as plt
from src.io import init_dataset, get_sample_data
import os
import open3d as o3d
import numpy as np
from pypcd4 import pypcd4 as pypcd
from scipy.spatial import ConvexHull
from truckscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from truckscenes.utils.geometry_utils import view_points, transform_matrix, \
    BoxVisibility
from pyquaternion import Quaternion
from ultralytics import YOLO
from src.utils import create_transform_matrix, transform_points

VEHICLE_HEIGHTS = {
    'car': 1.5,
    'truck': 3.0,
    'bus': 3.2,
    'van': 2.0,
    'motorcycle': 1.2
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main(config):

    # Load YOLO model
    model = YOLO('yolov8m.pt')  # or 'yolov8n.pt'
    scene_number = 3
    window_size = 4     # window size for computing the velocity of the bb from camera sensor
    track_id_to_plot = 4
    MAX_MISSING_FRAMES = 3
    MIN_POSITIVE_COUNT = 3  # min number of consecutive last frames to check if the vehicle is approaching

    # Initialize the TruckScenes dataset
    trucksc = init_dataset('v1.0-mini', 'data/mini_dataset/man-truckscenes')

    # Select scene and sample
    scene = trucksc.scene[scene_number]
    first_sample_token = scene['first_sample_token']
    my_sample = trucksc.get('sample', first_sample_token)
    # new_token = my_sample['next']
    # my_sample = trucksc.get('sample', new_token)

    # Sensor names
    radar_sensor = 'RADAR_LEFT_BACK'
    camera_sensor = 'CAMERA_LEFT_BACK'


    #######################################################################
    #######################################################################
    #######################################################################

    sample_token = scene['first_sample_token']
    sample_tokens = []
    slope = 0.38      # adjust to match your filtering logic
    intercept = 200  # adjust to where the cutoff starts

    tracked_objects = {}
    archived_tracks = {}

    while sample_token:
        
        sample_tokens.append(sample_token)
        sample = trucksc.get('sample', sample_token)

        # Get sample data
        radar_data = trucksc.get('sample_data', sample['data'][radar_sensor])
        camera_data = trucksc.get('sample_data', sample['data'][camera_sensor])

        # Get file paths
        radar_file = os.path.join('data/mini_dataset/man-truckscenes', radar_data['filename'])
        camera_file = os.path.join('data/mini_dataset/man-truckscenes', camera_data['filename'])

        # === Get focal lengths fx and fy ===
        calibrated_sensor_token = camera_data['calibrated_sensor_token']
        calibrated_sensor = trucksc.get('calibrated_sensor', calibrated_sensor_token)
        intrinsic = calibrated_sensor['camera_intrinsic']
        fx = intrinsic[0][0]
        fy = intrinsic[1][1]

        # print(intrinsic)
        # print(f"Focal lengths - fx: {fx}, fy: {fy}")

        orig_width, orig_height = 1920, 1040
        new_width, new_height = 1980, 943

        scale_x = new_width / orig_width     # ≈ 1.03125
        scale_y = new_height / orig_height   # ≈ 0.9067

        fx = fx * scale_x          # ≈ 660.0
        fy = fy * scale_y          # ≈ 580.3

        # print(f"Focal lengths - fx: {fx_scaled}, fy: {fy_scaled}")

        # Load camera image
        img = cv2.imread(camera_file)
        height, width = img.shape[:2]

        # print(f"Image dimensions: width={width}, height={height}")

        # Run YOLOv8 detection
        results = model.track(
            source=camera_file,
            tracker='bytetrack.yaml',
            persist=True,
            stream=False,
            verbose=False
        )    # keep track of IDs
        boxes = results[0].boxes

        # Load radar point cloud
        pc = pypcd.PointCloud.from_path(radar_file)
        points_raw = pc.pc_data

        # ego pose
        timestamp = camera_data['timestamp']  
        # print('timestamp radar: ', timestamp)

        # Find the closest ego_motion_chassis entry
        ego_chassis = min(
            trucksc.ego_motion_chassis,
            key=lambda em: abs(em['timestamp'] - timestamp)
        )

        # print(f"Ego speed at time {ego_chassis['timestamp']}: {ego_chassis['vx']:.2f} m/s → {ego_chassis['vx']*3.6:.2f} km/h")


        points_2d, coloring, img, ax1, ax2, mask = trucksc.render_pointcloud_in_image_subplot(     
            sample_token=sample['token'],
            pointsensor_channel='RADAR_LEFT_BACK',
            camera_channel='CAMERA_LEFT_BACK',
            render_intensity=True,
            dot_size=4
        )

        # association between radar points (projected in 2D) and camera boxes
        projected_uv = points_2d.T  # shape (N, 2)
        u = projected_uv[:, 0]
        v = projected_uv[:, 1]

        geo_filter = v >= slope * u + intercept

        visible_points = points_raw[mask]              # apply mask first
        vrel_x_visible = visible_points['vrel_x']  # relative velocity along x axis of the radar points mapped on the image  

        visible_points_filtered = visible_points[geo_filter]           # shape (N_filtered,)
        projected_uv_filtered = projected_uv[geo_filter]               # shape (N_filtered, 2)

        ax1.scatter(projected_uv_filtered[:, 0], projected_uv_filtered[:, 1], marker='o', c=coloring[geo_filter],
                   s=4, edgecolors='none')

        vx = visible_points_filtered['vrel_x']
        vy = visible_points_filtered['vrel_y']
        vz = visible_points_filtered['vrel_z']
        v_rel_local = np.stack([vx, vy, vz], axis=1)  # shape (N, 3)


        # === we need radar data and camera data in the same ref system to then fuse them ===
        # Get radar calibration rotation matrix (radar → ego frame)
        radar_calib = trucksc.get('calibrated_sensor', radar_data['calibrated_sensor_token'])
        R_radar_to_ego = Quaternion(radar_calib['rotation']).rotation_matrix

        # Rotate radar velocity vectors into ego vehicle frame
        v_rel_ego = v_rel_local @ R_radar_to_ego.T  # shape (N, 3)
        
        boxes = results[0].boxes

        
        matched_points_uv = []  # list of (u, v) for radar points inside boxes
        matched_speeds = []     # optional: store vrel_x values
        box_matches = {}        
        
        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            corners = np.array([
                [x1, y1],
                [x2, y1],
                [x1, y2],
                [x2, y2]
            ])
            class_id = int(boxes.cls[i].item())
            class_name = model.names[class_id]
            conf = boxes.conf[i].item()

            # Get track_id (may be None)
            track_id = int(box.id.item()) if box.id is not None else -1

            #Apply geometric filter (check how many corners are above the filter line)
            corner_u = corners[:, 0]
            corner_v = corners[:, 1]
            corner_mask = corner_v < slope * corner_u + intercept

            # If 3 or more corners are above the line, skip this box
            if np.sum(corner_mask) >= 3:
                continue

            if class_id in [2, 5, 7] and conf > 0.5:

                # Find radar points inside this box
                in_box_mask = (
                    (projected_uv_filtered[:, 0] >= x1) &
                    (projected_uv_filtered[:, 0] <= x2) &
                    (projected_uv_filtered[:, 1] >= y1) &
                    (projected_uv_filtered[:, 1] <= y2)
                )

                matched_vx = v_rel_ego[in_box_mask, 0]    # look for the velocities (along x) of the points corresponding to the vehicle
                matched_uv = projected_uv_filtered[in_box_mask][:, :2]     # se non funziona, togliere [:, :2]

                box_matches[i] = {
                    'bbox': [x1, y1, x2, y2],
                    'class_name': class_name,
                    'class_id': class_id,
                    'track_id': track_id,
                    'conf': conf,
                    'points': matched_uv,
                    'vrel_x': matched_vx
                }
                matched_points_uv.append(matched_uv)
                matched_speeds.append(matched_vx)


        slope = 0.38      # adjust to match your filtering logic
        intercept = 200  # adjust to where the cutoff starts
        img_np = np.array(img)

        # Generate u (horizontal image coords)
        u_vals = np.linspace(0, img_np.shape[1], 500)
        v_vals = slope * u_vals + intercept

        ax1.plot(u_vals, v_vals, color='red', linestyle='--', linewidth=2, label='Filter boundary')
        ax1.legend(loc='lower right')

        # plot showing found bounding boxes with the relative radar points inside it
        ax2.scatter(projected_uv_filtered[:, 0], projected_uv_filtered[:, 1], s=2, color='yellow', label='Radar points')

        #######################################
        ax1.imshow(img)
        ax1.axis('off')
        ax2.imshow(img)
        plt.tight_layout()
        #######################################

        # === for plotting ===
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            corners = np.array([
                [x1, y1],
                [x2, y1],
                [x1, y2],
                [x2, y2]
            ])
            #Apply geometric filter (check how many corners are above the filter line)
            corner_u = corners[:, 0]
            corner_v = corners[:, 1]
            corner_mask = corner_v < slope * corner_u + intercept

            # If 3 or more corners are above the line, skip this box
            if np.sum(corner_mask) >= 3:
                continue

            cls_id = int(boxes.cls[i].item())
            conf = boxes.conf[i].item()
            track_id = box_matches.get(i, {}).get('track_id', -1)
            if cls_id in [2, 5, 7] and conf > 0.5:
                rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='lime', facecolor='none')
                rect1 = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='lime', facecolor='none')
                ax2.add_patch(rect)
                ax1.add_patch(rect1)
                ax1.text(x1, y1 - 10, f"{model.names[cls_id]} ({conf:.2f})", color='lime', fontsize=11)
                ax2.text(x1, y1 - 10, f"ID {track_id}", color='lime', fontsize=11)

                
        # === MAIN CYCLE ===
        # 2. Draw radar points that matched vehicles — BIGGER & RED
        for box_id, data in box_matches.items():
            x1, y1, x2, y2 = data['bbox']
            points_uv = data['points']       # shape (N, 2) or empty
            vrel_x = data['vrel_x']         # array of radar-relative velocities (possibly empty)
            cls_name = data['class_name']
            conf = data['conf']
            track_id = data['track_id']
            class_id = data['class_id']
            matched_uv = data['points']  # 2D points for plottin

            v_z_kmh = None

            corners = np.array([
                [x1, y1],
                [x2, y1],
                [x1, y2],
                [x2, y2]
            ])
            #Apply geometric filter (check how many corners are above the filter line)
            corner_u = corners[:, 0]
            corner_v = corners[:, 1]
            corner_mask = corner_v < slope * corner_u + intercept

            # If 3 or more corners are above the line, skip this box
            if np.sum(corner_mask) >= 3:
                continue

            # 2a) Plot radar points if any
            if len(points_uv) > 0:
                ax2.scatter(
                    points_uv[:, 0], points_uv[:, 1],
                    s=30, color='red',
                    label='Matched radar' if box_id == 0 else ""
                )

            # 2b) Compute radar-based average v_rel_x only if vrel_x is non-empty
            if len(vrel_x) > 0:
                avg_vx = - np.mean(vrel_x) * 3.6
                # absolute_speed_kmh = (ego_chassis['vx'] - avg_vx) * 3.6

            else:
                # No radar points matched this box: skip or set to zero
                avg_vx = 0.0

            # 2c) Update persistence for bounding-box velocity
            if track_id != -1:
                if track_id not in tracked_objects:
                    tracked_objects[track_id] = {
                        'class_name': cls_name,
                        'class_id': class_id,
                        'history': [],
                        'missing_frames': 0
                    }

                # compute bounding-box center
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                history_list = tracked_objects[track_id]['history']
                tracked_objects[track_id]['missing_frames'] = 0     # I saw the vehicle, reset the missing-frames counter to 0

                # Append this sample’s data
                history_list.append({
                    'timestamp': timestamp,
                    'bbox':    [x1, y1, x2, y2],
                    'center':  [cx, cy],
                    'vrel_x':  avg_vx,
                    'points_2d': matched_uv.tolist(),
                })

                # Compute bbox velocity if there’s a previous sample
                if len(history_list) >= 2:
                    class_name = tracked_objects[track_id]['class_name']
                    H_real = VEHICLE_HEIGHTS.get(class_name, None)

                    if H_real is not None and fx > 0:
                        # Compute Z for each entry in the history (if not already present)
                        for entry in history_list:
                            bbox = entry['bbox']
                            h = bbox[3] - bbox[1]  # Always recompute h
                            if 'Z' not in entry and h > 0:
                                entry['Z'] = (H_real * fx) / h
                                entry['h_px'] = h  # store h as part of the data
                            elif h <= 0:
                                entry['Z'] = None

                        # Use the most recent N entries that have valid Z
                        recent_entries = [e for e in history_list[-window_size:] if e.get('Z') is not None]

                        velocities = []
                        
                        for i in range(len(recent_entries) - 1):
                            Z0 = recent_entries[i]['Z']
                            Z1 = recent_entries[i + 1]['Z']
                            dt = 0.1    # f = 10 Hz

                            if dt > 0:
                                v_z = (Z0 - Z1) / dt
                                velocities.append(v_z)

                        if velocities:
                            v_z_m_s = np.mean(velocities)
                            v_z_kmh = v_z_m_s * 3.6
                            converted_str = f"{v_z_kmh:.2f} km/h"
                        else:
                            v_z_m_s = None
                            v_z_kmh = None
                            converted_str = "N/A"

                        history_list[-1].update({
                            'v_bbox_z_rel_m_s': v_z_m_s,
                            'v_bbox_z_rel_kmh': v_z_kmh
                        })
                    else:
                        print(f"[DEBUG] dt ≤ 0 (dt={dt:.6f}); skipping bbox height velocity")


                # === velocity fusion ===
                k = 5.0  # radar influence constant
                num_radar_points = len(vrel_x)

                if len(history_list) < 2 and num_radar_points < 4:
                    alpha = None
                else:
                    alpha = num_radar_points / (num_radar_points + k)
                
                # compute the radar contribution/importance                
                if alpha is not None:
                    if v_z_kmh is not None:
                        vehicle_rel_vel = alpha * avg_vx + (1 - alpha) * v_z_kmh
                        print(f"[Track {track_id}] v_fused = {vehicle_rel_vel:.2f} km/h | alpha = {alpha:.2f} | radar = {avg_vx:.2f} km/h | bbox = {v_z_kmh:.2f} km/h | n_radar = {num_radar_points}")
                    else:
                        vehicle_rel_vel = avg_vx
                        print(f"[Track {track_id}] v_fused = {vehicle_rel_vel:.2f} km/h | alpha = {alpha:.2f} | radar = {avg_vx:.2f} km/h | bbox = - km/h | n_radar = {num_radar_points}")
                else:
                    vehicle_rel_vel = None
                    print(f"[Track {track_id}] ❗ velocity not computed | radar pts = {num_radar_points} | bbox frames = {len(history_list)}")


                history_list[-1].update({'v_fused_kmh': vehicle_rel_vel})   # add the computed velocity to the saved track in this timestamp
                

            # 2d) Annotate image with radar-based velocity (if desired)
            if vehicle_rel_vel == None:
                ax1.text(x1, y2 + 30, f"v_rel: - km/h", color='cyan', fontsize=10)   
            else:
                ax1.text(x1, y2 + 30, f"v_rel: {vehicle_rel_vel:.2f} km/h", color='cyan', fontsize=10)

            
            # for plotting
            if track_id != -1 and track_id in tracked_objects:
                if track_id not in archived_tracks:
                    archived_tracks[track_id] = tracked_objects[track_id]['history']




        for track_id, obj in tracked_objects.items():
            if obj['missing_frames'] > 0:
                continue  # skip stale/uncertain tracks

            history = obj['history']
            vels = [entry.get('v_fused_kmh') for entry in history if entry.get('v_fused_kmh') is not None]

            if len(vels) >= MIN_POSITIVE_COUNT:  # at least 3 entries
                recent = vels[-MIN_POSITIVE_COUNT:]  
                if all(v > 5 for v in recent):  # threshold of min 5 km/h
                    print(f"⚠️ Vehicle {track_id} ({obj['class_name']}) is approaching")
                    history[-1]['is_approaching'] = True

    

        # === missing vehicles ===
        current_track_ids = set(box_matches[i]['track_id'] for i in box_matches if box_matches[i]['track_id'] != -1)

        # Increment missing frame counter for those NOT seen
        for tid in tracked_objects:
            if tid not in current_track_ids:
                tracked_objects[tid]['missing_frames'] += 1

        to_delete = [tid for tid, obj in tracked_objects.items() if obj['missing_frames'] > MAX_MISSING_FRAMES]
        for tid in to_delete:
            del tracked_objects[tid]        

        ax1.legend()
        ax2.legend()
        plt.show()
        sample_token = sample['next']
        print('='*50)

    plt.close('all')

    # === PLOTS ===
    if track_id_to_plot in archived_tracks:
        history = archived_tracks[track_id_to_plot]
        
        frame_indices = []
        fused_velocities = []
        bbox_velocities = []
        radar_velocities = []
        radar_points_count = []
        approaching_status = []

        for i, entry in enumerate(history):
            fused_vel = entry.get('v_fused_kmh')
            bbox_vel = entry.get('v_bbox_z_rel_kmh')
            radar_vel = entry.get('vrel_x')
            radar_points = len(entry.get('points_2d', []))
            is_approaching = entry.get('is_approaching', False)

            if fused_vel is not None:
                frame_indices.append(i)
                fused_velocities.append(fused_vel)
                bbox_velocities.append(bbox_vel)
                radar_velocities.append(radar_vel)
                radar_points_count.append(radar_points)
                approaching_status.append(is_approaching)

        # === First Figure: Fused Velocity and Radar Points ===
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(frame_indices, fused_velocities, color='tab:blue', alpha=0.5, linewidth=2, label='Fused Velocity')
        for i, is_approach in zip(frame_indices, approaching_status):
            if is_approach:
                ax1.axvspan(i - 0.5, i + 0.5, color='green', alpha=0.1)
        ax1.scatter([i for i, a in zip(frame_indices, approaching_status) if a],
                    [v for v, a in zip(fused_velocities, approaching_status) if a],
                    color='green', s=100, marker='^', label='Approaching')
        ax1.scatter([i for i, a in zip(frame_indices, approaching_status) if not a],
                    [v for v, a in zip(fused_velocities, approaching_status) if not a],
                    color='gray', s=60, marker='s', label='Not Approaching')
        ax1.set_title('Fused Velocity Over Frames')
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel('Velocity (km/h)')
        ax1.grid(True)
        ax1.legend()

        ax2.plot(frame_indices, radar_points_count, color='tab:red', marker='x')
        ax2.set_title('Radar Points Used Over Frames')
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('Radar Points Count')
        ax2.grid(True)

        plt.suptitle(f'Track ID {track_id_to_plot} - Fused Vel. & Radar Points')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # === Second Figure: BBox vs Radar Velocity ===
        fig2, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(frame_indices, bbox_velocities, label='BBox Velocity', color='tab:orange', linewidth=2)
        ax3.plot(frame_indices, radar_velocities, label='Radar Velocity', color='tab:purple', linewidth=2, linestyle='--')

        ax3.set_title(f'Track ID {track_id_to_plot} - BBox vs Radar Velocity Over Frames')
        ax3.set_xlabel('Frame Index')
        ax3.set_ylabel('Velocity (km/h)')
        ax3.grid(True)
        ax3.legend()
        plt.tight_layout()
        plt.show()


    else:
        print(f"No data available for track ID {track_id_to_plot}")
    

if __name__ == "__main__":
    # Config dictionary for data path etc.
    config = {
        "data_root": "./data"  # change to your actual path
    }
    main(config)
