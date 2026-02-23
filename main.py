"""People Tracker - YOLOv8 + ByteTrack"""

import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from bytetrack import BYTETracker
from models import YOLOv8
from utils import draw_tracks, generate_colors
from gender_model import GenderClassifier

VID_FORMATS = {"mp4", "avi", "mov", "mkv", "webm", "m4v"}
IMG_FORMATS = {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="People Tracker")
    parser.add_argument("--source", type=str, default="0", help="Video/image path or webcam index")
    parser.add_argument("--weights", type=str, default="weights/yolov8n.onnx", help="ONNX model path")
    parser.add_argument("--save", type=str, nargs="?", const="", help="Save results (optional: output path)")
    parser.add_argument("--view", action="store_true", help="Display results")
    parser.add_argument("--gender_weights", type=str, default="weights/best_resnet50_gender_model.pth",
                    help="Gender model weights (.pth)")
    parser.add_argument("--gender_every", type=int, default=10,
                        help="Run gender inference every N frames per track")
    parser.add_argument("--gender_conf", type=float, default=0.70,
                        help="Confidence threshold to accept gender prediction")
    parser.add_argument("--report_sec", type=float, default=10.0,
                        help="Generate report every N seconds (template-based, no LLM)")
    parser.add_argument("--report_overlay", action="store_true",
                        help="Show the report text on the video frame")
    
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



def generate_template_report(stats: dict) -> str:
    window = stats.get("window_seconds", 10)
    total_unique = stats.get("total_unique_people", 0)
    in_frame = stats.get("currently_in_frame", 0)
    male = stats.get("male", 0)
    female = stats.get("female", 0)
    unknown = stats.get("unknown", 0)
    fps = stats.get("fps", 0.0)

    caution = ""
    if unknown > 0:
        caution = f" ({unknown} unclassified)"

    # Short, professional, “LLM-like” text
    return (
        f"Last {window:.0f}s: {total_unique} unique people, {in_frame} in frame. "
        f"Gender: {male} M, {female} F{caution}. FPS ~{fps:.1f}."
    )




def main() -> None:
    args = parse_args()

    # Initializing
    detector = YOLOv8(args.weights)
    tracker = BYTETracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
    colors = generate_colors()
    # Gender model
    gender_model = GenderClassifier(args.gender_weights)

    # track_id -> gender state (votes + last inference info)
    track_gender: dict[int, dict] = {}
    
    # Report state (template-based)
    last_report_time = time.time()
    last_report_text = ""

    source_type = get_source_type(args.source)
    cap = None
    writer = None
    save = args.save is not None

    # Setup capture
    if source_type == "webcam":
        cap = cv2.VideoCapture(int(args.source))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open webcam source={args.source}")
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

            # Reading the frame
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
            # --- Gender classification per track_id (cached + voting) ---
            for trk in tracks:
                x1, y1, x2, y2, tid = trk
                tid = int(tid)

                # Clip bbox to frame
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                w = x2 - x1
                h = y2 - y1

                # Skip tiny crops (unreliable)
                if w < 20 or h < 40:
                    continue

                info = track_gender.get(tid)
                if info is None:
                    info = {
                        "votes": {"male": 0, "female": 0},
                        "label": "unknown",
                        "last_frame": 0,
                        "conf": 0.0,
                    }
                    track_gender[tid] = info

                # Only run inference every N frames for this track
                if (frame_idx - info["last_frame"]) < args.gender_every:
                    continue

                crop = frame[y1:y2, x1:x2]
                label, conf, _ = gender_model.predict(crop)

                info["last_frame"] = frame_idx
                info["conf"] = conf

                # Vote only if confident enough
                if conf >= args.gender_conf and label in ("male", "female"):
                    info["votes"][label] += 1

                    # Majority vote decides stable label
                    if info["votes"]["male"] > info["votes"]["female"]:
                        info["label"] = "male"
                    elif info["votes"]["female"] > info["votes"]["male"]:
                        info["label"] = "female"
                    else:
                        info["label"] = "unknown"

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
            # --- Template-based report (no LLM/API) ---
            now = time.time()
            if (now - last_report_time) >= args.report_sec:
                male = sum(1 for v in track_gender.values() if v["label"] == "male")
                female = sum(1 for v in track_gender.values() if v["label"] == "female")
                unknown = sum(1 for v in track_gender.values() if v["label"] == "unknown")

                stats = {
                    "window_seconds": args.report_sec,
                    "total_unique_people": len(seen_ids),
                    "currently_in_frame": len(tracks),
                    "male": male,
                    "female": female,
                    "unknown": unknown,
                    "fps": float(fps),
                }

                last_report_text = generate_template_report(stats)

                # Print to console (like an LLM report)
                print("\n========== REPORT ==========")
                print(last_report_text)
                print("===========================\n")

                last_report_time = now

            id_to_gender = {tid: v["label"] for tid, v in track_gender.items()}
            # Draw
            draw_tracks(frame, tracks, colors, trajectories, fps=fps, total_ids=len(seen_ids), id_to_gender=id_to_gender)

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
                    track_gender.clear()

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
