#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random

import cv2
import numpy as np
from typing import List, Dict
from pathlib import Path

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. Please install with: pip install ultralytics")


class YOLOPOIDetector:

    def __init__(self, model_path: str, confidence_threshold: float = 0.5):

        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package is required. Install with: pip install ultralytics")

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model file not found: {model_path}")

        self.confidence_threshold = confidence_threshold
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            self.model = YOLO(str(self.model_path))
            print(f"Successfully loaded YOLO model from {self.model_path}")

            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
                print(f"Model classes: {list(self.class_names.values())}")
            else:
                self.class_names = {}
                print("Warning: Could not retrieve class names from model")

        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def detect_pois(self, image: np.ndarray) -> List[Dict]:

        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:

            results = self.model(image, conf=self.confidence_threshold, verbose=False)

            detections = []

            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes

                    for i in range(len(boxes)):
                        bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                        confidence = boxes.conf[i].cpu().numpy()
                        class_id = int(boxes.cls[i].cpu().numpy())

                        center_x = (bbox[0] + bbox[2]) / 2
                        center_y = (bbox[1] + bbox[3]) / 2

                        class_name = self.class_names.get(class_id, f"class_{class_id}")

                        detection = {
                            'class_id': class_id,
                            'class_name': class_name,
                            'confidence': float(confidence),
                            'bbox': bbox.tolist(),
                            'center': [float(center_x), float(center_y)]
                        }

                        detections.append(detection)

            print(f"Detected {len(detections)} POIs")
            for det in detections:
                print(f"  - {det['class_name']}: {det['confidence']:.3f}")

            return detections

        except Exception as e:
            print(f"Error during POI detection: {e}")
            return []

    def visualize_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:

        vis_image = image.copy()

        for det in detections:
            bbox = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']

            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]


            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1), color, -1)

            cv2.putText(vis_image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return vis_image


def test_yolo_detector(model_path: str, image_path: str):

    try:

        detector = YOLOPOIDetector(model_path)


        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return

        print(f"Testing with image: {image_path}")
        print(f"Image shape: {image.shape}")

        detections = detector.detect_pois(image)

        if detections:
            print(f"\nDetection results:")
            for i, det in enumerate(detections):
                print(f"  {i + 1}. {det['class_name']} (confidence: {det['confidence']:.3f})")
                print(f"     bbox: {det['bbox']}")
                print(f"     center: {det['center']}")

            vis_image = detector.visualize_detections(image, detections)


            output_path = f"yolo_detection_result.jpg"
            cv2.imwrite(output_path, vis_image)
            print(f"\nVisualization saved to: {output_path}")
        else:
            print("No POIs detected in the image")

    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python yolo_poi_detector.py <model_path> <image_path>")
        print("Example: python yolo_poi_detector.py yolo_poi.pt test_image.jpg")
    else:
        model_path = sys.argv[1]
        image_path = sys.argv[2]
        test_yolo_detector(model_path, image_path)