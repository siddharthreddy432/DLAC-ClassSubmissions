import importlib
import sys


REQUIRED_LIBRARIES = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "ultralytics": "ultralytics",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
}


def main() -> int:
    print("=" * 60)
    print("VisionPort Task 1: Environment and Library Verification")
    print("=" * 60)
    print(f"Python version: {sys.version.split()[0]}")

    if sys.version_info < (3, 10):
        print("WARNING: Python 3.10 or higher is recommended.")

    all_installed = True
    print("\nChecking installed packages:")
    for module_name, display_name in REQUIRED_LIBRARIES.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "installed")
            print(f"[PASS] {display_name}: {version}")
        except ImportError:
            print(f"[MISSING] {display_name}")
            all_installed = False

    print("\nResult:", "Environment is ready." if all_installed else "Install missing packages.")
    return 0 if all_installed else 1


if __name__ == "__main__":
    raise SystemExit(main())
