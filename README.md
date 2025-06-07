# Proximity Sensor Project

A sensor fusion system that combines radar point clouds and YOLOv8-based camera detections to estimate vehicle velocities and detect approaching vehicles. Designed for applications such as blind spot monitoring and rear collision warning.

## Overview

This project processes multimodal sensor data from the [TruckScenes Devkit](https://truckscenes.com/) using:

- **Radar** for precise velocity estimation
- **Camera (YOLOv8 + ByteTrack)** for object detection and tracking
- **Sensor fusion** to produce robust vehicle proximity detection

## Features

- Real-time object detection and multi-object tracking
- Radar-to-camera point cloud projection and filtering
- Velocity estimation using radar points and bounding box changes
- Adaptive fusion of velocity estimates
- Approaching vehicle alert system

## Dataset

Utilizes the `v1.0-mini` version of the TruckScenes dataset (10 scenes). Only radar and camera data from the rear-left side of the truck were used.

## Dependencies

- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- NumPy, OpenCV, Matplotlib
- Additional devkit dependencies (see dataset SDK)
