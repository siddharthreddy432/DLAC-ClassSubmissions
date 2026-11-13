import argparse
from pathlib import Path

import cv2

from vehicle_detector import VEHICLE_LABELS, detect


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect multiple vehicle types.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("multiple_vehicle_detection.mp4"))
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            annotated, _ = detect(frame, VEHICLE_LABELS)
            writer.write(annotated)
            cv2.imshow("Multiple Vehicle Detection", annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()
    print(f"Multiple vehicle detection video saved to {args.output}")


if __name__ == "__main__":
    main()
