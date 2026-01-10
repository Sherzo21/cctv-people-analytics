# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from typing import Tuple

import cv2
import numpy as np
import onnxruntime
import torch
import torchvision


class YOLOv8:
    """YOLOv8 ONNX inference class for human detection.

    Supports GPU (CUDA) and CPU inference with configurable confidence
    and NMS thresholds.

    Args:
        model_path: Path to ONNX model file.
        conf_thres: Confidence threshold for detections. Defaults to 0.25.
        iou_thres: IoU threshold for NMS. Defaults to 0.45.
        max_det: Maximum number of detections per image. Defaults to 300.
        nms_mode: NMS method ('torchvision' or 'numpy'). Defaults to 'torchvision'.

    Example:
        >>> detector = YOLOv8("weights/yolov8n.onnx", conf_thres=0.5)
        >>> boxes, scores, class_ids = detector(image)
    """

    def __init__(
        self,
        model_path: str,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        nms_mode: str = "torchvision",
    ) -> None:
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.nms_mode = nms_mode

        self._initialize_model(model_path)

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference on input image.

        Args:
            image: Input image in BGR format (H, W, C).

        Returns:
            Tuple containing:
                - boxes: [N, 4] bounding boxes in (x1, y1, x2, y2) format
                - scores: [N] confidence scores
                - class_ids: [N] class indices
        """
        if not isinstance(image, np.ndarray) or len(image.shape) != 3:
            raise ValueError("Input must be a 3D numpy array (H, W, C)")

        self.orig_height, self.orig_width = image.shape[:2]

        # Preprocess with letterbox
        img, self.ratio, (self.pad_w, self.pad_h) = self._letterbox(image, self.img_size)

        return self._detect(img)

    def _initialize_model(self, model_path: str) -> None:
        """Initialize ONNX Runtime session."""
        try:
            self.session = onnxruntime.InferenceSession(
                model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )

            self.output_names = [x.name for x in self.session.get_outputs()]
            self.input_names = [x.name for x in self.session.get_inputs()]

            # Get input shape from model
            input_shape = self.session.get_inputs()[0].shape
            self.img_size = (
                input_shape[2] if isinstance(input_shape[2], int) else 640,
                input_shape[3] if isinstance(input_shape[3], int) else 640,
            )

            # Get model metadata
            metadata = self.session.get_modelmeta().custom_metadata_map
            self.stride = int(metadata.get("stride", 32))
            self.names = eval(metadata.get("names", "{0: 'person'}"))

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

    @staticmethod
    def _letterbox(
        image: np.ndarray,
        target_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114),
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Resize and pad image to target shape while preserving aspect ratio.

        Args:
            image: Input image (BGR format).
            target_shape: Target (height, width).
            color: Padding color (BGR).

        Returns:
            Tuple of (resized_image, scale, (pad_w, pad_h)).
        """
        height, width = image.shape[:2]

        scale = min(target_shape[0] / height, target_shape[1] / width)
        new_size = (int(width * scale), int(height * scale))

        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

        dw = (target_shape[1] - new_size[0]) / 2
        dh = (target_shape[0] - new_size[1]) / 2
        top, bottom = int(dh), int(target_shape[0] - new_size[1] - int(dh))
        left, right = int(dw), int(target_shape[1] - new_size[0] - int(dw))

        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return image, scale, (dw, dh)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for inference."""
        img = img[:, :, ::-1]  # BGR to RGB
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = np.expand_dims(img, axis=0)
        return np.ascontiguousarray(img)

    def _postprocess(self, prediction: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process model output to get detections."""
        # YOLOv8 output: [1, 4+num_classes, num_boxes] -> [num_boxes, 4+num_classes]
        outputs = np.squeeze(prediction[0]).T

        boxes = outputs[:, :4]  # x_center, y_center, width, height
        class_scores = outputs[:, 4:]

        # Get max class score and class id
        if class_scores.ndim == 1:
            scores = class_scores
            class_ids = np.zeros(len(scores), dtype=np.int32)
        else:
            scores = np.max(class_scores, axis=1)
            class_ids = np.argmax(class_scores, axis=1)

        # Apply confidence threshold
        mask = scores > self.conf_thres
        boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]

        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        # Convert xywh to xyxy
        boxes = self._xywh2xyxy(boxes)

        # Scale boxes to original image size
        boxes = self._scale_boxes(boxes)

        # Apply NMS
        if self.nms_mode == "torchvision":
            indices = torchvision.ops.nms(
                torch.tensor(boxes, dtype=torch.float32),
                torch.tensor(scores, dtype=torch.float32),
                self.iou_thres,
            ).numpy()
        else:
            indices = self._nms_numpy(boxes, scores, self.iou_thres)

        if len(indices) == 0:
            return np.array([]), np.array([]), np.array([])

        indices = indices[: self.max_det]
        return boxes[indices], scores[indices], class_ids[indices]

    def _scale_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """Scale boxes from letterbox coords to original image coords."""
        if len(boxes) == 0:
            return boxes

        boxes[:, [0, 2]] -= self.pad_w
        boxes[:, [1, 3]] -= self.pad_h
        boxes[:, :4] /= self.ratio

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, self.orig_width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, self.orig_height)

        return boxes

    @staticmethod
    def _xywh2xyxy(x: np.ndarray) -> np.ndarray:
        """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    @staticmethod
    def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        """Non-Maximum Suppression (NumPy fallback)."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return np.array(keep)

    def _detect(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run detection pipeline."""
        input_tensor = self._preprocess(img)
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return self._postprocess(outputs)
