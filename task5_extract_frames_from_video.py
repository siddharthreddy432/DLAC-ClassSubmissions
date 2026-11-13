import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from a traffic video.")
    parser.add_argument("video", type=Path, default=Path("traffic_sample.mp4"), nargs="?")
    parser.add_argument("--output", type=Path, default=Path("extracted_frames"))
    parser.add_argument("--step", type=int, default=10)
    args = parser.parse_args()
    if args.step < 1:
        raise ValueError("--step must be at least 1.")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")
    args.output.mkdir(parents=True, exist_ok=True)

    count = 0
    saved = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if count % args.step == 0:
                output_path = args.output / f"frame_{count:06d}.jpg"
                if not cv2.imwrite(str(output_path), frame):
                    raise RuntimeError(f"Could not save {output_path}.")
                saved += 1
            count += 1
    finally:
        capture.release()
    print(f"Extracted {saved} frames to {args.output}")


if __name__ == "__main__":
    main()
