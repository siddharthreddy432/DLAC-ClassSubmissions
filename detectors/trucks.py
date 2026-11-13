from .common import detect_objects

LABELS = {"truck"}


def detect(frame):
    return detect_objects(frame, LABELS)
