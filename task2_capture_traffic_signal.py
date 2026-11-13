import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture an image from a camera.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("traffic_signal.jpg"))
    args = parser.parse_args()

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}.")

    print("Press SPACE to capture an image, or ESC to cancel.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the camera.")
            cv2.imshow("Traffic Signal Capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                return
            if key == 32:
                if not cv2.imwrite(str(args.output), frame):
                    raise RuntimeError(f"Could not save {args.output}.")
                print(f"Traffic image saved to {args.output}")
                return
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
