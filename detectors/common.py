from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MODEL = YOLO(str(Path(__file__).parent.parent / "yolov8n.pt"))


def detect_objects(
    frame: np.ndarray, labels: set[str] | None = None
) -> tuple[np.ndarray, list[dict]]:
    result = MODEL(frame, verbose=False)[0]
    detections = []
    names = result.names

    for box in result.boxes:
        label = names[int(box.cls[0])]
        if labels is not None and label not in labels:
            continue
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [round(float(value)) for value in box.xyxy[0]]
        detections.append({
            "name": label,
            "confidence": round(confidence, 3),
            "box": [x1, y1, x2, y2],
        })

    annotated = result.plot()
    if labels is not None:
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
