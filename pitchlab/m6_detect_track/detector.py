from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pitchlab.core.config import SAHIConfig, YoloConfig
from pitchlab.core.types import BBox, EntityClass, Track2D

if TYPE_CHECKING:
    from ultralytics import YOLO


class YOLOSAHIDetector:
    """YOLO + optional SAHI sliced inference detector.

    Supports lazy loading of the underlying YOLO model weights.
    """

    def __init__(self, yolo_cfg: YoloConfig, sahi_cfg: SAHIConfig) -> None:
        self.yolo_cfg = yolo_cfg
        self.sahi_cfg = sahi_cfg
        self._model: YOLO | None = None

    def _get_model(self) -> YOLO:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.yolo_cfg.weights)
            # Send to device if available
            try:
                self._model.to(self.yolo_cfg.device)
            except Exception:
                pass
        return self._model

    def detect(self, frame: np.ndarray, frame_idx: int) -> list[Track2D]:
        """Runs detection on a single frame.

        Returns unassigned Track2D objects (track_id=-1).
        """
        if self.sahi_cfg.enabled:
            raw_dets = self._detect_sahi(frame)
        else:
            raw_dets = self._detect_full(frame)

        results: list[Track2D] = []
        for box, conf, cls_enum in raw_dets:
            results.append(
                Track2D(
                    track_id=-1,
                    frame_idx=frame_idx,
                    bbox_xyxy=box,
                    conf=float(conf),
                    cls=cls_enum,
                )
            )
        return results

    def _detect_full(self, frame: np.ndarray) -> list[tuple[BBox, float, EntityClass]]:
        model = self._get_model()
        preds = model.predict(
            source=frame,
            conf=self.yolo_cfg.conf_thresh,
            iou=self.yolo_cfg.iou_thresh,
            imgsz=self.yolo_cfg.imgsz,
            device=self.yolo_cfg.device,
            verbose=False,
        )

        detections: list[tuple[BBox, float, EntityClass]] = []
        if not preds:
            return detections

        pred = preds[0]
        if pred.boxes is None:
            return detections

        boxes_xyxy = pred.boxes.xyxy.cpu().numpy()
        confs = pred.boxes.conf.cpu().numpy()
        classes = pred.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes_xyxy, confs, classes):
            cls_enum = self._map_class(cls_id)
            if cls_enum is not None:
                detections.append(
                    (
                        (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        float(conf),
                        cls_enum,
                    )
                )
        return detections

    def _detect_sahi(self, frame: np.ndarray) -> list[tuple[BBox, float, EntityClass]]:
        h, w = frame.shape[:2]
        tile_h, tile_w = self.sahi_cfg.slice_height, self.sahi_cfg.slice_width
        ol_h = int(tile_h * self.sahi_cfg.overlap_height_ratio)
        ol_w = int(tile_w * self.sahi_cfg.overlap_width_ratio)
        step_y = max(1, tile_h - ol_h)
        step_x = max(1, tile_w - ol_w)

        all_boxes: list[list[float]] = []
        all_confs: list[float] = []
        all_classes: list[EntityClass] = []

        # Full frame run for global context
        full_dets = self._detect_full(frame)
        for box, conf, cls_enum in full_dets:
            all_boxes.append(list(box))
            all_confs.append(conf)
            all_classes.append(cls_enum)

        # Slices
        for y1 in range(0, max(1, h - tile_h + step_y), step_y):
            for x1 in range(0, max(1, w - tile_w + step_x), step_x):
                y2 = min(h, y1 + tile_h)
                x2 = min(w, x1 + tile_w)
                crop = frame[y1:y2, x1:x2]

                crop_dets = self._detect_full(crop)
                for box, conf, cls_enum in crop_dets:
                    # Map box back to original coordinates
                    gx1 = box[0] + x1
                    gy1 = box[1] + y1
                    gx2 = box[2] + x1
                    gy2 = box[3] + y1
                    all_boxes.append([gx1, gy1, gx2, gy2])
                    all_confs.append(conf)
                    all_classes.append(cls_enum)

        if not all_boxes:
            return []

        # Class-aware Non-Maximum Suppression (NMS)
        return self._nms_per_class(all_boxes, all_confs, all_classes, iou_thresh=0.5)

    def _nms_per_class(
        self,
        boxes: list[list[float]],
        confs: list[float],
        classes: list[EntityClass],
        iou_thresh: float,
    ) -> list[tuple[BBox, float, EntityClass]]:
        out: list[tuple[BBox, float, EntityClass]] = []
        unique_classes = set(classes)

        for c in unique_classes:
            idx = [i for i, cl in enumerate(classes) if cl == c]
            if not idx:
                continue

            c_boxes = np.array([boxes[i] for i in idx], dtype=np.float32)
            c_confs = np.array([confs[i] for i in idx], dtype=np.float32)

            keep_idx = self._box_nms(c_boxes, c_confs, iou_thresh)
            for k in keep_idx:
                b = c_boxes[k]
                out.append(
                    (
                        (float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                        float(c_confs[k]),
                        c,
                    )
                )

        return out

    @staticmethod
    def _box_nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]

        return keep

    @staticmethod
    def _map_class(class_id: int) -> EntityClass | None:
        """Map class index to EntityClass.
        0: player, 32: sports ball (standard COCO)
        """
        if class_id == 0:
            return EntityClass.PLAYER
        if class_id == 32:
            return EntityClass.BALL
        return None
