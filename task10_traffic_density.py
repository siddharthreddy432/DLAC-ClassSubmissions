import argparse
from pathlib import Path

import cv2

from vehicle_detector import VEHICLE_LABELS, density_for, detect


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate traffic density from vehicle count.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--step", type=int, default=10)
    args = parser.parse_args()
    if args.step < 1:
        raise ValueError("--step must be at least 1.")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")
    maximum = 0
    sampled = 0
    total = 0
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_number % args.step == 0:
                _, detections = detect(frame, VEHICLE_LABELS)
                count = len(detections)
                maximum = max(maximum, count)
                total += count
                sampled += 1
            frame_number += 1
    finally:
        capture.release()

    average = total / sampled if sampled else 0
    print(f"Maximum vehicles: {maximum}")
    print(f"Average vehicles: {average:.2f}")
    print(f"Traffic density: {density_for(maximum)}")


if __name__ == "__main__":
    main()
