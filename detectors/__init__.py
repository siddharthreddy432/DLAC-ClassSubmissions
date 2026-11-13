from .bikes import LABELS as BIKE_LABELS
from .buses import LABELS as BUS_LABELS
from .cars import LABELS as CAR_LABELS
from .common import detect_objects
from .trucks import LABELS as TRUCK_LABELS
from .bags import LABELS as BAG_LABELS

DETECTORS = {
    "all": None,
    "car": CAR_LABELS,
    "bike": BIKE_LABELS,
    "bus": BUS_LABELS,
    "truck": TRUCK_LABELS,
    "bag": BAG_LABELS,
}

__all__ = ["DETECTORS", "detect_objects"]
