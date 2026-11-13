import base64
import io
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from detectors import DETECTORS, detect_objects

ROOT = Path(__file__).parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
app = FastAPI(title="VisionPort Traffic Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["null", "http://localhost:5500", "http://127.0.0.1:5500"],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=OUTPUTS), name="outputs")


def density_for(count: int) -> str:
    if count <= 5:
        return "Low"
    if count <= 12:
        return "Medium"
    return "High"


def analyze_frame(
    frame: np.ndarray, category: str = "all"
) -> tuple[np.ndarray, list[dict]]:
    labels = DETECTORS.get(category)
    if category not in DETECTORS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown detector '{category}'. Choose from: {', '.join(DETECTORS)}.",
        )
    return detect_objects(frame, labels)


def image_response(image: np.ndarray) -> StreamingResponse:
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode the processed image.")
    return StreamingResponse(io.BytesIO(encoded.tobytes()), media_type="image/jpeg")


async def read_image(file: UploadFile) -> np.ndarray:
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
    return image


@app.get("/")
async def serve_frontend():
    return FileResponse(ROOT / "visionport.html")


@app.get("/health")
async def health():
    return {"status": "ok", "model": "yolov8n.pt"}


@app.post("/detect/")
async def detect_objects_endpoint(file: UploadFile = File(...), category: str = "all"):
    image = await read_image(file)
    annotated, detections = analyze_frame(image, category)
    return image_response(annotated)


@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...), category: str = "all"):
    image = await read_image(file)
    annotated, detections = analyze_frame(image, category)
    ok, encoded = cv2.imencode(".jpg", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode the processed image.")
    return {
        "image": base64.b64encode(encoded).decode("ascii"),
        "detections": detections,
        "count": len(detections),
        "density": density_for(len(detections)),
    }


@app.post("/analyze-video/")
async def analyze_video(
    file: UploadFile = File(...), frame_step: int = 10, category: str = "all"
):
    if frame_step < 1:
        raise HTTPException(status_code=400, detail="frame_step must be at least 1.")
    if category not in DETECTORS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown detector '{category}'. Choose from: {', '.join(DETECTORS)}.",
        )

    job_id = uuid.uuid4().hex
    job_dir = OUTPUTS / job_id
    frames_dir = job_dir / "frames"
    frames_dir.mkdir(parents=True)
    source_path = job_dir / "source.mp4"
    with source_path.open("wb") as target:
        shutil.copyfileobj(file.file, target)

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable video.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = job_dir / "annotated.mp4"
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    frame_number = 0
    sampled_frames = []
    total_detections = 0
    max_count = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            annotated, detections = analyze_frame(frame, category)
            writer.write(annotated)
            if frame_number % frame_step == 0:
                frame_path = frames_dir / f"frame_{frame_number:06d}.jpg"
                cv2.imwrite(str(frame_path), annotated)
                sampled_frames.append(f"/outputs/{job_id}/frames/{frame_path.name}")
                total_detections += len(detections)
                max_count = max(max_count, len(detections))
            frame_number += 1
    finally:
        capture.release()
        writer.release()

    return {
        "video": f"/outputs/{job_id}/annotated.mp4",
        "frames": sampled_frames,
        "frame_count": frame_number,
        "sampled_frame_count": len(sampled_frames),
        "max_vehicles_in_frame": max_count,
        "average_vehicles_in_sampled_frames": round(
            total_detections / len(sampled_frames), 2
        ) if sampled_frames else 0,
        "density": density_for(max_count),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
