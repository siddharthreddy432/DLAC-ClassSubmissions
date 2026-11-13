from .common import detect_objects

LABELS = {"car"}


def detect(frame):
    return detect_objects(frame, LABELS)
