
import argparse, os, glob, joblib
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier
from .features import extract_features

def load_dataset(data_dir: str):
    ai_paths, real_paths = [], []
    for e in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
        ai_paths += glob.glob(os.path.join(data_dir, "ai", e))
        real_paths += glob.glob(os.path.join(data_dir, "real", e))
    X, y, feat_keys = [], [], None
    for label, paths in [(1, ai_paths), (0, real_paths)]:
        for p in tqdm(paths, desc=f"Procesando {'AI' if label==1 else 'Real'}"):
            try:
                img = Image.open(p).convert("RGB")
                feats = extract_features(img)
                if feat_keys is None: feat_keys = sorted(feats.keys())
                X.append([feats.get(k, 0.0) for k in feat_keys])
                y.append(label)
            except Exception:
                pass
    return np.array(X, dtype=np.float32), np.array(y), feat_keys

def train(data_dir: str, out_path: str):
    X, y, feat_keys = load_dataset(data_dir)
    if len(X) == 0:
        raise RuntimeError("No se encontraron imágenes en data_dir.")
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X, y)
    obj = {"model": model, "feature_order": feat_keys}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(obj, out_path)
    acc = float(model.score(X, y))
    print(f"Modelo guardado en {out_path}")
    print(f"Accuracy (train set, sin val split): {acc:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="models/model_baseline.pkl")
    args = ap.parse_args()
    train(args.data_dir, args.out)

if __name__ == "__main__":
    main()
