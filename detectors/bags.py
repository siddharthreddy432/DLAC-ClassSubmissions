from .common import detect_objects

LABELS = {"backpack", "handbag", "suitcase"}


def detect(frame):
    return detect_objects(frame, LABELS)
