import io
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from ultralytics import YOLO

# Initialize FastAPI and load the object detection model
app = FastAPI(title="Indian Traffic Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["null", "http://localhost:5500", "http://127.0.0.1:5500"],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
model = YOLO("yolov8n.pt")  # Swap with custom-trained weights (e.g., 'best.pt') for specific Indian vehicles


@app.get("/")
async def serve_frontend():
    return FileResponse(Path(__file__).parent / "visionport.html")


@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    # 1. Read the uploaded image bytes
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    # 2. Run Object Detection inference
    results = model(img)

    # 3. Annotate the image with bounding boxes and labels
    annotated_img = results[0].plot()

    # 4. Stream the processed image back in real time
    _, encoded_img = cv2.imencode(".jpg", annotated_img)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
