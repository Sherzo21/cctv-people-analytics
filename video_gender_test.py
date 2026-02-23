import cv2
import numpy as np
import time
from pathlib import Path

from models import YOLOv8
from bytetrack import BYTETracker
from gender_model import GenderClassifier
from utils import draw_tracks, generate_colors


VIDEO_PATH = "assets/test_video_1.mp4"
YOLO_WEIGHTS = "weights/yolov8n.onnx"
GENDER_WEIGHTS = "weights/best_resnet50_gender_model.pth"
OUTPUT_VIDEO = "output_gender_tracking.mp4"
OUTPUT_REPORT = "output_report.txt"


def generate_template_report(stats: dict) -> str:
    window = stats.get("window_seconds", 0)
    total_unique = stats.get("total_unique_people", 0)
    male = stats.get("male", 0)
    female = stats.get("female", 0)
    unknown = stats.get("unknown", 0)

    caution = ""
    if unknown > 0:
        caution = f" ({unknown} unclassified)"

    return (
        f"Video Analysis Report\n"
        f"----------------------\n"
        f"Total unique people detected: {total_unique}\n"
        f"Gender distribution: {male} Male, {female} Female{caution}\n"
        f"Processed duration: {window:.1f} seconds\n"
    )


def main():

    print("Initializing models...")
    detector = YOLOv8(YOLO_WEIGHTS)
    tracker = BYTETracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
    gender_model = GenderClassifier(GENDER_WEIGHTS)
    colors = generate_colors()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    fps_input = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_input,
        (width, height),
    )

    frame_idx = 0
    seen_ids = set()
    trajectories = {}
    track_gender = {}

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        boxes, scores, _ = detector(frame)
        dets = np.hstack([boxes, scores.reshape(-1, 1)]) if len(boxes) else np.empty((0, 5))
        tracks = tracker.update(dets)

        active_ids = set()

        for trk in tracks:
            x1, y1, x2, y2, tid = trk
            tid = int(tid)

            active_ids.add(tid)
            seen_ids.add(tid)

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            info = track_gender.get(tid)
            if info is None:
                info = {
                    "votes": {"male": 0, "female": 0},
                    "label": "unknown",
                }
                track_gender[tid] = info

            label, conf, _ = gender_model.predict(crop)

            if conf >= 0.6 and label in ("male", "female"):
                info["votes"][label] += 1

                if info["votes"]["male"] > info["votes"]["female"]:
                    info["label"] = "male"
                elif info["votes"]["female"] > info["votes"]["male"]:
                    info["label"] = "female"

        id_to_gender = {tid: v["label"] for tid, v in track_gender.items()}

        draw_tracks(frame, tracks, colors, trajectories=None,
                    fps=0.0, total_ids=len(seen_ids),
                    id_to_gender=id_to_gender)

        writer.write(frame)

        print(f"\rProcessing frame {frame_idx}", end="")

    cap.release()
    writer.release()

    duration = time.time() - start_time

    male = sum(1 for v in track_gender.values() if v["label"] == "male")
    female = sum(1 for v in track_gender.values() if v["label"] == "female")
    unknown = sum(1 for v in track_gender.values() if v["label"] == "unknown")

    stats = {
        "window_seconds": duration,
        "total_unique_people": len(seen_ids),
        "male": male,
        "female": female,
        "unknown": unknown,
    }

    report = generate_template_report(stats)

    with open(OUTPUT_REPORT, "w") as f:
        f.write(report)

    print("\n\nProcessing complete.")
    print(report)


if __name__ == "__main__":
    main()