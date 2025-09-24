from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
import os
from dotenv import load_dotenv

from utils import load_models, preprocess_image_bytes, choose_model_key_from_crop

# Load .env
load_dotenv()

# Environment variables
APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", 8000))
STATIC_DIR = os.getenv("STATIC_DIR", "static")
MODEL_CONFIG_PATH = os.getenv("MODEL_CONFIG_PATH", "models/model_config.json")

app = FastAPI(title="Crop Disease Prediction API")

# Load models at startup
MODELS = load_models(MODEL_CONFIG_PATH)

# Serve static files
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root():
    return {"message": "Crop Disease Prediction API running. Use /docs to test."}


@app.post("/predict")
async def predict(
    crop: str = Form(..., description="Crop name (rice/banana/plant)"),
    file: UploadFile = File(..., description="Image file"),
):
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    # Choose model based on crop
    key = choose_model_key_from_crop(crop)
    if key not in MODELS:
        raise HTTPException(status_code=400, detail=f"No model available for crop '{crop}'.")

    model_info = MODELS[key]

    # Preprocess image
    img_bytes = await file.read()
    x = preprocess_image_bytes(img_bytes, target_size=model_info["target_size"])

    # Make prediction
    preds = model_info["model"].predict(x)
    preds = np.array(preds).squeeze()
    idx = int(preds.argmax())
    confidence = float(preds[idx])
    label = model_info["labels"][idx]

    return {
        "crop": crop,
        "model_used": key,
        "predicted_disease": label,
        "confidence": round(confidence, 4),
    }


@app.get("/upload", response_class=HTMLResponse)
def upload_form():
    """
    Serve the styled frontend (upload.html).
    Place your upload.html inside the STATIC_DIR.
    """
    file_path = os.path.join(STATIC_DIR, "upload.html")
    if not os.path.exists(file_path):
        return HTMLResponse(
            content=f"<h3>Upload page not found. Please add upload.html to '{STATIC_DIR}'.</h3>",
            status_code=404,
        )
    return FileResponse(file_path)


if __name__ == "__main__":
    uvicorn.run("main:app", host=APP_HOST, port=APP_PORT, reload=True)
