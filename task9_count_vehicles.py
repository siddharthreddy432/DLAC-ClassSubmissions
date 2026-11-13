import argparse
from pathlib import Path

import cv2

from vehicle_detector import VEHICLE_LABELS, detect


def main() -> None:
    parser = argparse.ArgumentParser(description="Count vehicles in video frames.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--step", type=int, default=10)
    args = parser.parse_args()
    if args.step < 1:
        raise ValueError("--step must be at least 1.")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_number % args.step == 0:
                _, detections = detect(frame, VEHICLE_LABELS)
                print(f"Frame {frame_number}: {len(detections)} vehicles")
            frame_number += 1
    finally:
        capture.release()


if __name__ == "__main__":
    main()
