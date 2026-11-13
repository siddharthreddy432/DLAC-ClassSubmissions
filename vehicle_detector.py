from pathlib import Path

import cv2
from ultralytics import YOLO


MODEL_PATH = Path(__file__).parent / "yolov8n.pt"
MODEL = YOLO(str(MODEL_PATH))

VEHICLE_LABELS = {"car", "motorcycle", "bus", "truck"}


def detect(
    frame,
    labels: set[str] | None = None,
    confidence: float = 0.25,
):
    results = MODEL(frame, conf=confidence, verbose=False)[0]
    detections = []

    for box in results.boxes:
        name = results.names[int(box.cls[0])]
        if labels is not None and name not in labels:
            continue
        x1, y1, x2, y2 = [round(float(value)) for value in box.xyxy[0]]
        detections.append(
            {
                "name": name,
                "confidence": round(float(box.conf[0]), 3),
                "box": [x1, y1, x2, y2],
            }
        )

    annotated = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        caption = f'{detection["name"]} {detection["confidence"]:.2f}'
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            annotated,
            caption,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 0),
            2,
        )
    return annotated, detections


def density_for(count: int) -> str:
    if count <= 5:
        return "Low"
    if count <= 12:
        return "Medium"
    return "High"
