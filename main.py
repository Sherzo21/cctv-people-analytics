#!/usr/bin/env python3
"""People Tracker - YOLOv8 + ByteTrack"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from bytetrack import BYTETracker
from models import YOLOv8
from utils import draw_tracks, generate_colors

VID_FORMATS = {"mp4", "avi", "mov", "mkv", "webm", "m4v"}
IMG_FORMATS = {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="People Tracker")
    parser.add_argument("--source", type=str, default="0", help="Video/image path or webcam index")
    parser.add_argument("--weights", type=str, default="weights/yolov8n.onnx", help="ONNX model path")
    parser.add_argument("--save", type=str, nargs="?", const="", help="Save results (optional: output path)")
    parser.add_argument("--view", action="store_true", help="Display results")
    return parser.parse_args()


def get_source_type(source: str) -> str:
    if source.isdigit():
        return "webcam"
    ext = Path(source).suffix[1:].lower()
    if ext in VID_FORMATS:
        return "video"
    elif ext in IMG_FORMATS:
        return "image"
    raise ValueError(f"Unsupported format: {ext}")


def main() -> None:
    args = parse_args()

    # Initialize
    detector = YOLOv8(args.weights)
    tracker = BYTETracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
    colors = generate_colors()

    source_type = get_source_type(args.source)
    cap = None
    writer = None
    save = args.save is not None

    # Setup capture
    if source_type == "webcam":
        cap = cv2.VideoCapture(int(args.source))
        args.view = True
    elif source_type == "video":
        cap = cv2.VideoCapture(args.source)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup writer
    if save and cap:
        output_path = args.save if args.save else f"output_{Path(args.source).stem}.mp4"
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w, h = int(cap.get(3)), int(cap.get(4))
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Process
    frame_idx = 0
    fps_buffer: list[float] = []
    seen_ids: set[int] = set()
    trajectories: dict[int, list[tuple[int, int]]] = {}

    try:
        while True:
            t0 = time.perf_counter()

            # Read frame
            if source_type == "image":
                frame = cv2.imread(args.source)
                if frame is None:
                    break
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            frame_idx += 1

            # Detect & Track
            boxes, scores, _ = detector(frame)
            dets = np.hstack([boxes, scores.reshape(-1, 1)]) if len(boxes) else np.empty((0, 5))
            tracks = tracker.update(dets)

            # Update trajectories (only for active tracks)
            active_ids = set()
            for trk in tracks:
                tid = int(trk[4])
                active_ids.add(tid)
                seen_ids.add(tid)
                cx, cy = int((trk[0] + trk[2]) / 2), int((trk[1] + trk[3]) / 2)
                trajectories.setdefault(tid, []).append((cx, cy))
                if len(trajectories[tid]) > 50:
                    trajectories[tid] = trajectories[tid][-50:]

            # Remove trajectories for tracks no longer in frame
            for tid in list(trajectories.keys()):
                if tid not in active_ids:
                    del trajectories[tid]

            # FPS
            fps_buffer.append(1.0 / (time.perf_counter() - t0))
            if len(fps_buffer) > 30:
                fps_buffer.pop(0)
            fps = sum(fps_buffer) / len(fps_buffer)

            # Draw
            draw_tracks(frame, tracks, colors, trajectories, fps=fps, total_ids=len(seen_ids))

            # Status
            if source_type == "video":
                print(f"\rFrame {frame_idx}/{total_frames} | FPS: {fps:.1f}", end="")
            elif source_type == "webcam":
                print(f"\rFrame {frame_idx} | FPS: {fps:.1f}", end="")

            # Save/Display
            if writer:
                writer.write(frame)
            if args.view:
                cv2.imshow("People Tracker", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    tracker.reset()
                    trajectories.clear()
                    seen_ids.clear()

            if source_type == "image":
                if save:
                    out = args.save if args.save else f"output_{Path(args.source).name}"
                    cv2.imwrite(out, frame)
                if args.view:
                    cv2.waitKey(0)
                break

    finally:
        if cap:
            cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
