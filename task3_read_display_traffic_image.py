import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Read and display a traffic image.")
    parser.add_argument("image", type=Path, nargs="?", default=Path("traffic_signal.jpg"))
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    cv2.imshow("Traffic Image", image)
    print("Press any key in the image window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
