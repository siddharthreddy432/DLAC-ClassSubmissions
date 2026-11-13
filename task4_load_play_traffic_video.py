import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a traffic video.")
    parser.add_argument("video", type=Path, default=Path("traffic_sample.mp4"), nargs="?")
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    delay = max(1, round(1000 / (capture.get(cv2.CAP_PROP_FPS) or 25)))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            cv2.imshow("Traffic Video", frame)
            if cv2.waitKey(delay) & 0xFF == 27:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
