from .common import detect_objects

LABELS = {"bus"}


def detect(frame):
    return detect_objects(frame, LABELS)
