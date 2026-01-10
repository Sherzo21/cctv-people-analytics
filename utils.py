# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from typing import Tuple

import cv2
import numpy as np


def generate_colors(num_colors: int = 100, seed: int = 42) -> list[Tuple[int, int, int]]:
    """Generate random colors for track visualization.

    Args:
        num_colors: Number of colors to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of BGR color tuples.
    """
    np.random.seed(seed)
    return [(int(r), int(g), int(b)) for r, g, b in np.random.rand(num_colors, 3) * 255]


def draw_bbox(
    image: np.ndarray,
    bbox: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    corner_thickness: int = 2,
    proportion: float = 0.12,
) -> None:
    """Draw a bounding box with fancy corners on an image.

    Args:
        image: Input image to draw on (modified in-place).
        bbox: Bounding box coordinates [x1, y1, x2, y2].
        color: Color of the bounding box in BGR. Defaults to green.
        corner_thickness: Thickness of the corner lines. Defaults to 2.
        proportion: Proportion of corner length to box dimensions. Defaults to 0.12.
    """
    x1, y1, x2, y2 = map(int, bbox)
    width = x2 - x1
    height = y2 - y1

    corner_length = int(proportion * min(width, height))

    # Draw the thin rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

    # Top-left corner (bolder)
    cv2.line(image, (x1, y1), (x1 + corner_length, y1), color, corner_thickness)
    cv2.line(image, (x1, y1), (x1, y1 + corner_length), color, corner_thickness)

    # Top-right corner (bolder)
    cv2.line(image, (x2, y1), (x2 - corner_length, y1), color, corner_thickness)
    cv2.line(image, (x2, y1), (x2, y1 + corner_length), color, corner_thickness)

    # Bottom-left corner (bolder)
    cv2.line(image, (x1, y2), (x1, y2 - corner_length), color, corner_thickness)
    cv2.line(image, (x1, y2), (x1 + corner_length, y2), color, corner_thickness)

    # Bottom-right corner (bolder)
    cv2.line(image, (x2, y2), (x2, y2 - corner_length), color, corner_thickness)
    cv2.line(image, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)


def draw_trajectory(
    image: np.ndarray,
    trajectories: dict[int, list[Tuple[int, int]]],
    colors: list[Tuple[int, int, int]],
    max_length: int = 40,
) -> None:
    """Draw trajectory lines for tracked objects.

    Args:
        image: Input image to draw on (modified in-place).
        trajectories: Dict mapping track_id to list of (cx, cy) center points.
        colors: List of colors for different track IDs.
        max_length: Maximum number of points to draw per trajectory.
    """
    for track_id, points in trajectories.items():
        if len(points) < 2:
            continue

        color = colors[track_id % len(colors)]
        # Draw only the last max_length points
        pts = points[-max_length:]

        # Draw lines with fading effect (older = thinner)
        for i in range(1, len(pts)):
            thickness = max(1, int((i / len(pts)) * 2))
            cv2.line(image, pts[i - 1], pts[i], color, thickness, lineType=cv2.LINE_AA)


def draw_tracks(
    image: np.ndarray,
    tracks: np.ndarray,
    colors: list[Tuple[int, int, int]],
    trajectories: dict[int, list[Tuple[int, int]]] | None = None,
    show_id: bool = True,
    show_info: bool = True,
    fps: float = 0.0,
    total_ids: int = 0,
) -> None:
    """Draw tracked objects on image.

    Args:
        image: Input image to draw on (modified in-place).
        tracks: Array of tracks [N, 5] with [x1, y1, x2, y2, track_id].
        colors: List of colors for different track IDs.
        trajectories: Dict mapping track_id to list of center points for trajectory lines.
        show_id: Whether to show track ID labels.
        show_info: Whether to show info overlay (FPS, counts).
        fps: Current frames per second.
        total_ids: Total unique IDs tracked so far.
    """
    # Draw trajectories first (behind boxes)
    if trajectories:
        draw_trajectory(image, trajectories, colors)

    for x1, y1, x2, y2, track_id in tracks.astype(int):
        color = colors[track_id % len(colors)]

        # Draw bounding box with fancy corners
        draw_bbox(image, (x1, y1, x2, y2), color=color)

        # Draw track ID label
        if show_id:
            label = f"{track_id}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)

            # Label background
            cv2.rectangle(
                image,
                (x1, y1 - label_h - 4),
                (x1 + label_w + 4, y1),
                color,
                -1,
            )

            # Label text
            cv2.putText(
                image,
                label,
                (x1 + 2, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )

    # Draw info overlay (single line)
    if show_info:
        info_text = f"FPS: {fps:.1f} | In Frame: {len(tracks)} | Total: {total_ids}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        padding = 6

        (tw, th), _ = cv2.getTextSize(info_text, font, font_scale, thickness)

        # Semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (4, 4), (10 + tw + padding, 10 + th + padding), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

        # Draw text
        cv2.putText(image, info_text, (8, 8 + th), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
