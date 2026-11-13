import argparse
from pathlib import Path

import cv2

from vehicle_detector import VEHICLE_LABELS, detect


def main() -> None:
    parser = argparse.ArgumentParser(description="Display vehicle names and confidence scores.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("labelled_detection.jpg"))
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    annotated, detections = detect(image, VEHICLE_LABELS)
    if not cv2.imwrite(str(args.output), annotated):
        raise RuntimeError(f"Could not save {args.output}.")
    for item in detections:
        print(f'{item["name"]}: {item["confidence"]:.3f}')
    cv2.imshow("Labelled Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
