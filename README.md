Real-Time CCTV People Analytics System
YOLOv8 + ByteTrack + ResNet50 Gender Classification

📌 Overview

This project extends a YOLOv8-based people tracking system into a real-time CCTV analytics engine with:
🔍 Multi-person detection (YOLOv8 ONNX)
🧠 Real-time tracking (ByteTrack)
👤 Gender classification (Custom-trained ResNet50)
📊 Analytical reporting (Template-based structured report)
🎥 Video processing with output saving
📄 Automatic .txt analytics report generation

The system supports:
Webcam inference
Offline video processing
Batch analytics export

Video/Webcam
     ↓
YOLOv8 (ONNX Detection)
     ↓
ByteTrack (ID Assignment)
     ↓
ResNet50 Gender Classification
     ↓
Voting Stabilization per Track ID
     ↓
Video Output + Analytics Report

Key Features
1. Real-Time People Tracking
YOLOv8 ONNX inference
ByteTrack multi-object tracking
Stable ID assignment across frames

2. Custom Gender Classification Model
ResNet50 pretrained backbone
Fine-tuned for binary gender classification
Majority-vote stabilization per tracked ID
Confidence threshold filtering

3. Analytics Report Generation
Automatically generates structured report:
Example:
Video Analysis Report
----------------------
Total unique people detected: 7
Gender distribution: 4 Male, 3 Female
Processed duration: 18.2 seconds

4. Output Artifacts
Processed tracking video
Text analytics report
Overlay with ID + Gender (M/F)

Usage:
1. Real-Time Webcam
python main.py --source 0 --weights weights/yolov8n.onnx --gender_weights weights/best_resnet50_gender_model.pth --view --report_sec 10

Optional parameters:
--gender_every 5
--gender_conf 0.7
--report_sec 10
--report_overlay

2. Offline Video Analytics
python video_gender_test.py

Outputs:
output_gender_tracking.mp4
output_report.txt



Gender Model Details
Backbone: ResNet50 (ImageNet pretrained)
Input size: 224x224
Loss: CrossEntropyLoss
Optimizer: Adam
Voting mechanism per track ID
Confidence filtering threshold


Author
Developed and extended by Sherzod Abdumalikov
AI Engineer | Computer Vision | CCTV Analytics Systems

Quick way to confirm what you actually need
Run this once in your project env:
python -c "import cv2, numpy, torch, torchvision, onnxruntime; from PIL import Image; print('OK')"

