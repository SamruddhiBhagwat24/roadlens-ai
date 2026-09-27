import torch
"""Modular Road Object and Region Detector for RoadLens AI.

Extracts candidate regions for Mini Challenge 2 targets:
- STOP_SIGN (octagonal red geometric targets)
- SPEED_LIMIT_SIGN (vertical rectangular white/black regulatory targets)
- ADVISORY_SPEED (square/rectangular yellow warning plaques)
- WARNING_SIGN (diamond yellow/orange hazard signs)
- WORK_ZONE_SIGN (orange construction signs)
- LICENSE_PLATE (horizontal rectangular vehicle license plates)
- VEHICLE (car, truck, bus, motorcycle contextual targets)

Architecture:
1. BaseDetector: Abstract interface for all object detection backends.
2. YOLOv8RoadDetector: Deep-learning object detector loading local weights (models/yolov8s.pt).
   - Reports real confidence from model inference.
   - Maps COCO classes (e.g. 'stop sign' -> STOP_SIGN, vehicle classes -> VEHICLE).
   - Designed to seamlessly support custom fine-tuned RoadLens weights when trained.
3. HeuristicRoadDetector: Computer vision baseline using multi-spectral HSV color segmentation,
   morphological gradients, and contour polygon approximation (runs anywhere without C/GPU dependencies).
4. RoadDetectorService: Unified interface that automatically chooses the optimal detector
   and provides intelligent fallback.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import os
import numpy as np
import cv2
from backend.app.schemas import DetectionItem


class BaseDetector(ABC):
    """Abstract interface for RoadLens object detectors."""

    @abstractmethod
    def name(self) -> str:
        """Returns the unique name of the detector."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Checks if the required model weights and runtime are available."""
        pass

    @abstractmethod
    def detect(self, image_np: np.ndarray) -> List[DetectionItem]:
        """Detects road targets and returns DetectionItem objects with real bounding boxes and confidence."""
        pass


class HeuristicRoadDetector(BaseDetector):
    """Built-in OpenCV contour, HSV color-segmentation, and morphology detector.

    Detects geometric sign shapes (octagons, diamonds, rectangles) and license plate contours.
    Guarantees offline baseline operation on any system.
    """

    def __init__(self, min_area_ratio: float = 0.001, max_area_ratio: float = 0.60):
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def name(self) -> str:
        return "HeuristicContourCV"

    def is_available(self) -> bool:
        return True

    def detect(self, image_np: np.ndarray) -> List[DetectionItem]:
        if image_np is None or image_np.size == 0:
            return []

        h, w = image_np.shape[:2]
        total_pixels = w * h
        min_area = int(total_pixels * self.min_area_ratio)
        max_area = int(total_pixels * self.max_area_ratio)

        # Convert to HSV for color-guided segmentation
        if len(image_np.shape) == 3:
            hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
        else:
            hsv = cv2.cvtColor(cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)

        detections: List[DetectionItem] = []

        # 1. Red Sign Detection (Stop Signs, Regulatory borders)
        red_mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        detections.extend(self._find_candidates(red_mask, image_np, "STOP_SIGN", min_area, max_area, target_aspect=1.0))

        # 2. Yellow & Orange Sign Detection (Warning, Advisory Speed, Work Zone)
        yellow_mask = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([35, 255, 255]))
        orange_mask = cv2.inRange(hsv, np.array([8, 120, 100]), np.array([20, 255, 255]))

        detections.extend(self._find_candidates(yellow_mask, image_np, "WARNING_SIGN", min_area, max_area, target_aspect=1.0))
        detections.extend(self._find_candidates(orange_mask, image_np, "WORK_ZONE_SIGN", min_area, max_area, target_aspect=1.2))

        # 3. High-Contrast Rectangles (Speed Limit Signs, License Plates)
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if len(image_np.shape) == 3 else image_np
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # License plates (aspect ratio approx 1.8 - 2.5) and Speed Signs (approx 0.6 - 0.9)
        detections.extend(self._find_rectangular_candidates(thresh, image_np, min_area, max_area))

        # Non-Maximum Suppression to eliminate overlapping duplicates
        return self._apply_nms(detections, iou_threshold=0.35)

    def _find_candidates(
        self, mask: np.ndarray, orig: np.ndarray, class_name: str, min_area: int, max_area: int, target_aspect: float
    ) -> List[DetectionItem]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = w / float(h)
                aspect_diff = abs(aspect - target_aspect)

                if aspect_diff < 0.6:
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                    sides = len(approx)

                    extent = float(area) / float(w * h + 1e-6)
                    aspect_fit = 1.0 - min(1.0, aspect_diff / 0.6)
                    poly_fit = 1.0 if (sides >= 6 and class_name == "STOP_SIGN") or (sides == 4 and class_name != "STOP_SIGN") else 0.70

                    # Dynamic confidence derived from geometric and shape fidelity
                    score = float(np.clip(0.40 + 0.30 * extent + 0.20 * aspect_fit + 0.10 * poly_fit, 0.40, 0.98))
                    results.append(
                        DetectionItem(
                            class_name=class_name,
                            confidence=round(score, 2),
                            bounding_box=[int(x), int(y), int(x + w), int(y + h)],
                            label=class_name.replace("_", " ").title(),
                        )
                    )
        return results

    def _find_rectangular_candidates(
        self, thresh: np.ndarray, orig: np.ndarray, min_area: int, max_area: int
    ) -> List[DetectionItem]:
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = w / float(h)
                extent = float(area) / float(w * h + 1e-6)

                # US License Plate aspect ratio is roughly 2.0 (12" x 6")
                if 1.6 <= aspect <= 2.6:
                    aspect_fit = 1.0 - min(1.0, abs(aspect - 2.0) / 2.0)
                    score = float(np.clip(0.40 + 0.35 * extent + 0.25 * aspect_fit, 0.45, 0.98))
                    results.append(
                        DetectionItem(
                            class_name="LICENSE_PLATE",
                            confidence=round(score, 2),
                            bounding_box=[int(x), int(y), int(x + w), int(y + h)],
                            label="License Plate",
                        )
                    )
                # Regulatory Speed Limit Sign aspect ratio is roughly 0.7 to 0.9 (24" x 30")
                elif 0.6 <= aspect <= 0.95:
                    aspect_fit = 1.0 - min(1.0, abs(aspect - 0.80) / 0.80)
                    score = float(np.clip(0.40 + 0.35 * extent + 0.25 * aspect_fit, 0.45, 0.98))
                    results.append(
                        DetectionItem(
                            class_name="SPEED_LIMIT_SIGN",
                            confidence=round(score, 2),
                            bounding_box=[int(x), int(y), int(x + w), int(y + h)],
                            label="Speed Limit Sign",
                        )
                    )
        return results

    @staticmethod
    def _apply_nms(
        detections: List[DetectionItem],
        iou_threshold: float = 0.35,
        class_aware: bool = True,
    ) -> List[DetectionItem]:
        if not detections:
            return []

        dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep = []

        while dets:
            current = dets.pop(0)
            keep.append(current)

            remaining = []
            for d in dets:
                if class_aware and current.class_name != d.class_name:
                    cross_iou = HeuristicRoadDetector._calc_iou(current.bounding_box, d.bounding_box)
                    if cross_iou >= 0.85:
                        continue
                    remaining.append(d)
                    continue

                iou = HeuristicRoadDetector._calc_iou(current.bounding_box, d.bounding_box)
                if iou < iou_threshold:
                    remaining.append(d)
            dets = remaining

        return keep

    @staticmethod
    def _calc_iou(boxA: List[int], boxB: List[int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return float(iou)


class YOLOv8RoadDetector(BaseDetector):
    """Deep learning detector loading YOLOv8 weights (e.g. models/yolov8s.pt).

    Capabilities & Class Scope:
    - Pretrained COCO weights contain: 'stop sign' (class 11), vehicles ('car', 'truck', 'bus', 'motorcycle').
    - Pretrained COCO weights do NOT contain: 'LICENSE_PLATE', 'SPEED_LIMIT_SIGN', 'ADVISORY_SPEED',
      'WARNING_SIGN', or 'WORK_ZONE_SIGN'.
    - When fine-tuned or custom RoadLens weights are loaded, custom classes are mapped automatically.
    - Confidence scores are obtained directly from actual model inference (zero fabrication).
    """

    def __init__(
        self,
        weights_path: str = "models/yolov8s.pt",
        conf_threshold: float = 0.25,
        detector_name: str = "YOLOv8s",
    ):
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.detector_name = detector_name
        self._model = None

    def name(self) -> str:
        return self.detector_name

    def is_available(self) -> bool:
        if not os.path.exists(self.weights_path):
            return False
        try:
            import ultralytics
            return True
        except ImportError:
            return False

    def _get_model(self):
        if self._model is None and self.is_available():
            from ultralytics import YOLO
            self._model = YOLO(self.weights_path)
        return self._model

    def detect(self, image_np: np.ndarray) -> List[DetectionItem]:
        if not self.is_available() or image_np is None or image_np.size == 0:
            return []

        model = self._get_model()
        if model is None:
            return []

        from backend.app.performance.device import device_manager
        # Prefer CPU for low-overhead local execution or device_manager device
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        try:
            results = model.predict(image_np, conf=self.conf_threshold, verbose=False, device=device)
        except Exception:
            # Fallback to pure CPU if accelerator prediction encounters an issue
            results = model.predict(image_np, conf=self.conf_threshold, verbose=False, device="cpu")

        detections: List[DetectionItem] = []
        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        names = model.names

        for box in boxes:
            cls_id = int(box.cls[0])
            raw_name = names.get(cls_id, str(cls_id)).lower()
            conf = float(box.conf[0])
            xyxy = [int(v) for v in box.xyxy[0]]

            mapped_class, label = self._map_class(raw_name)
            if mapped_class:
                detections.append(
                    DetectionItem(
                        class_name=mapped_class,
                        confidence=round(conf, 2),
                        bounding_box=xyxy,
                        label=label,
                    )
                )

        return detections

    @staticmethod
    def _map_class(raw_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Maps model class names to RoadLens detection schema."""
        name_clean = raw_name.replace("_", " ").strip().lower()
        if name_clean == "stop sign":
            return "STOP_SIGN", "Stop Sign"
        elif name_clean in ["license plate", "plate", "license_plate"]:
            return "LICENSE_PLATE", "License Plate"
        elif "speed limit" in name_clean:
            return "SPEED_LIMIT_SIGN", "Speed Limit Sign"
        elif "advisory" in name_clean:
            return "ADVISORY_SPEED", "Advisory Speed Sign"
        elif "warning" in name_clean:
            return "WARNING_SIGN", "Warning Sign"
        elif "work zone" in name_clean or "construction" in name_clean:
            return "WORK_ZONE_SIGN", "Work Zone Sign"
        elif name_clean in ["car", "truck", "bus", "motorcycle"]:
            return "VEHICLE", name_clean.title()
        elif name_clean == "traffic light":
            return "TRAFFIC_LIGHT", "Traffic Light"
        return None, None


class RoadDetectorService:
    """Orchestrator and registry for modular object detection backends."""

    def __init__(self, preferred_detector: str = "auto"):
        self.detectors: Dict[str, BaseDetector] = {
            "yolo": YOLOv8RoadDetector(weights_path="models/yolov8s.pt", detector_name="YOLOv8s"),
            "plate_yolo": YOLOv8RoadDetector(
                weights_path="models/plate_yolov8.pt",
                conf_threshold=0.25,
                detector_name="plate_yolo",
            ),
            "sign_yolo": YOLOv8RoadDetector(
                weights_path="models/roadlens_sign_yolov8n.pt",
                conf_threshold=0.25,
                detector_name="RoadLensSignYOLO",
            ),
            "heuristic": HeuristicRoadDetector(),
        }
        self.preferred_detector = preferred_detector.lower().strip()

    def get_active_detector(self) -> BaseDetector:
        if self.preferred_detector in self.detectors and self.detectors[self.preferred_detector].is_available():
            return self.detectors[self.preferred_detector]
        if self.detectors["yolo"].is_available():
            return self.detectors["yolo"]
        if "plate_yolo" in self.detectors and self.detectors["plate_yolo"].is_available():
            return self.detectors["plate_yolo"]
        return self.detectors["heuristic"]

    def detect(self, image_np: np.ndarray) -> List[DetectionItem]:
        """Detects road targets.

        Perception flow:
        1. General YOLOv8s: Detects vehicles (car, truck, bus, motorcycle), stop signs, etc.
        2. Dedicated plate_yolo: Detects LICENSE_PLATE targets with dedicated trained model.
        3. Heuristic fallback: Supplements speed signs (unsupported by COCO) or acts as
           baseline if YOLO models are unavailable.
        4. Class-aware NMS merge: Deduplicates boxes without suppressing plates on vehicles.
        """
        if self.preferred_detector == "heuristic":
            return self.detectors["heuristic"].detect(image_np)
        if self.preferred_detector == "plate_yolo":
            return self.detectors["plate_yolo"].detect(image_np)

        detections: List[DetectionItem] = []

        # 1. General YOLOv8s for vehicles and signs
        if self.detectors["yolo"].is_available():
            general_dets = self.detectors["yolo"].detect(image_np)
            detections.extend(general_dets)
        else:
            heuristic_dets = self.detectors["heuristic"].detect(image_np)
            detections.extend(heuristic_dets)

        # 2. Dedicated sign_yolo for trained RoadLens sign classes
        if "sign_yolo" in self.detectors and self.detectors["sign_yolo"].is_available():
            sign_dets = self.detectors["sign_yolo"].detect(image_np)
            detections.extend(sign_dets)

        # 3. Dedicated plate_yolo for LICENSE_PLATE
        if "plate_yolo" in self.detectors and self.detectors["plate_yolo"].is_available():
            plate_dets = self.detectors["plate_yolo"].detect(image_np)
            # Dedicated detector is used specifically for license-plate detection
            dedicated_plates = [d for d in plate_dets if d.class_name == "LICENSE_PLATE"]
            detections.extend(dedicated_plates)

        # 3. Fallback proposals from HeuristicContourCV
        # Only fall back to heuristic for plates if dedicated plate_yolo is unavailable
        has_plate = any(d.class_name == "LICENSE_PLATE" for d in detections)
        sign_yolo_available = "sign_yolo" in self.detectors and self.detectors["sign_yolo"].is_available()
        if not has_plate or not sign_yolo_available:
            heuristic_proposals = self.detectors["heuristic"].detect(image_np)
            for prop in heuristic_proposals:
                if prop.class_name == "LICENSE_PLATE" and not has_plate:
                    # Only fallback if plate_yolo is not available
                    if not (self.detectors.get("plate_yolo") and self.detectors["plate_yolo"].is_available()):
                        detections.append(prop)
                elif prop.class_name == "SPEED_LIMIT_SIGN" and not sign_yolo_available:
                    detections.append(prop)

        return HeuristicRoadDetector._apply_nms(detections, iou_threshold=0.35)

    def warmup(self) -> None:
        """Pre-loads model weights to ensure steady-state inference without cold-start penalties."""
        for det in self.detectors.values():
            if hasattr(det, "_get_model") and det.is_available():
                det._get_model()


# Backward compatibility alias
RoadObjectDetector = HeuristicRoadDetector

# Global detector singleton
road_detector = RoadDetectorService()
