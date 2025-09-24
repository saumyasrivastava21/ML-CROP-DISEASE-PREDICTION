import json
import numpy as np
from io import BytesIO
from PIL import Image
import os
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Default paths from .env or fallback
MODEL_CONFIG_PATH = os.getenv("MODEL_CONFIG_PATH", "models/model_config.json")


def load_models(config_path=None):
    """
    Load models and label files based on a configuration JSON.
    """
    if config_path is None:
        config_path = MODEL_CONFIG_PATH

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[ERROR] Config file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    models = {}
    for key, info in cfg.items():
        model_path = os.path.normpath(info.get("path"))
        labels_path = os.path.normpath(info.get("labels"))
        target_size = tuple(info.get("target_size", [224, 224]))

        # Check model existence
        if not model_path or not os.path.exists(model_path):
            print(f"[WARN] Model file missing for '{key}': {model_path}")
            continue

        # Check labels existence
        if not labels_path or not os.path.exists(labels_path):
            print(f"[WARN] Labels file missing for '{key}': {labels_path}")
            continue

        # Load Keras model
        try:
            model = load_model(model_path)
            print(f"[INFO] Loaded '{key}' model from {model_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load model for '{key}': {e}")
            continue

        # Load labels JSON
        try:
            with open(labels_path, "r") as lf:
                labels_json = json.load(lf)
                if isinstance(labels_json, dict):
                    labels = [labels_json[str(i)] for i in range(len(labels_json))]
                elif isinstance(labels_json, list):
                    labels = labels_json
                else:
                    raise ValueError("Labels JSON must be a list or dict.")
        except Exception as e:
            print(f"[ERROR] Failed to load labels for '{key}': {e}")
            continue

        # Store in models dict
        models[key] = {"model": model, "labels": labels, "target_size": target_size}

    if not models:
        raise RuntimeError("[ERROR] No models loaded. Check paths in config JSON.")

    return models


def preprocess_image_bytes(img_bytes, target_size=(224, 224)):
    """
    Convert raw image bytes to a preprocessed numpy array for prediction.
    """
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img = img.resize((target_size[1], target_size[0]))  # PIL: (width, height)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def choose_model_key_from_crop(crop_name: str) -> str:
    """
    Map crop name (user input) to a model key.
    Defaults to 'plant' if not matched.
    """
    crop_name = str(crop_name).lower().strip()
    if "rice" in crop_name:
        return "rice"
    elif "banana" in crop_name:
        return "banana"
    else:
        return "plant"
