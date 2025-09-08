
import joblib, numpy as np
from .features import extract_features
from PIL import Image

MODEL_PATH = None
MODEL = None
FEATURE_ORDER = None

def load_model(path: str):
    global MODEL_PATH, MODEL, FEATURE_ORDER
    MODEL_PATH = path
    obj = joblib.load(path)
    MODEL = obj["model"]
    FEATURE_ORDER = obj["feature_order"]
    return MODEL

def predict_image(image_path: str):
    if MODEL is None:
        raise RuntimeError("Modelo no cargado. Llama load_model(path).")
    img = Image.open(image_path).convert("RGB")
    feats = extract_features(img)
    x = np.array([feats.get(k, 0.0) for k in FEATURE_ORDER], dtype=np.float32).reshape(1,-1)
    proba_ai = float(MODEL.predict_proba(x)[0,1])
    # Umbral un poco más alto para reducir falsos positivos de IA
    y_pred = int(proba_ai >= 0.6)
    return {"prob_ia": proba_ai, "pred_label": "IA" if y_pred==1 else "Real", "features": feats}

def explain_top_features(n=8):
    if hasattr(MODEL, "feature_importances_"):
        import numpy as np
        importances = MODEL.feature_importances_
        inds = np.argsort(importances)[::-1][:n]
        items = [(FEATURE_ORDER[i], float(importances[i])) for i in inds]
        return items
    return []
