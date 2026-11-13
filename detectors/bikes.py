from .common import detect_objects

LABELS = {"bicycle", "motorcycle"}


def detect(frame):
    return detect_objects(frame, LABELS)
