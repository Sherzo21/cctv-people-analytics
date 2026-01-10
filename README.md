# People Tracker

[![Downloads](https://img.shields.io/github/downloads/yakhyo/people-tracker/total)](https://github.com/yakhyo/people-tracker/releases)
[![GitHub License](https://img.shields.io/github/license/yakhyo/people-tracker)](https://github.com/yakhyo/people-tracker/blob/main/LICENSE)

Real-time multi-person tracking using [YOLOv8-CrowdHuman](https://github.com/yakhyo/yolov8-crowdhuman) + [**ByteTrack**](https://github.com/yakhyo/bytetrack-tracker).

<p align="center">
  <img src="assets/out_video.gif" alt="People Tracker Demo">
</p>


<details>
<summary>Click to see higher quality video</summary>

<video src="https://github.com/user-attachments/assets/1a658a67-9ce8-4c66-999a-b36391994abc" controls width="640"></video>

</details>

## Installation

```bash
git clone https://github.com/yakhyo/people-tracker.git
cd people-tracker
pip install -r requirements.txt
```

### Download Weights

```bash
mkdir -p weights
wget -P weights https://github.com/yakhyo/people-tracker/releases/download/weights/yolov8n.onnx
```

## Usage

```bash
python main.py --source 0 --view                    # Webcam
python main.py --source video.mp4 --save            # Video (auto output name)
python main.py --source video.mp4 --save out.mp4   # Video (custom output)
python main.py --source image.jpg --save            # Image
```

**Controls:** `q` quit, `r` reset tracker

## Project Structure

```
people-tracker/
├── assets/          # Sample videos
├── bytetrack/       # ByteTrack tracker
├── models/          # YOLOv8 ONNX detector
├── weights/         # Model weights
├── utils.py         # Visualization
├── main.py          # Entry point
└── requirements.txt
```

## References

- [YOLOv8-CrowdHuman](https://github.com/yakhyo/yolov8-crowdhuman) - YOLOv8 trained on CrowdHuman dataset
- [ByteTrack](https://github.com/yakhyo/bytetrack-tracker) - Multi-object tracking
