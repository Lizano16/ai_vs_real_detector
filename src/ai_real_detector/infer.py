
import argparse, os, json, glob
from .model import load_model, predict_image, explain_top_features

def main():
    parser = argparse.ArgumentParser(description="Clasificador IA vs Real (CLI)")
    parser.add_argument("--image", type=str, help="Ruta a una imagen")
    parser.add_argument("--folder", type=str, help="Carpeta con imágenes")
    parser.add_argument("--model", type=str, default="models/model_baseline.pkl", help="Ruta al modelo")
    args = parser.parse_args()

    load_model(args.model)

    paths = []
    if args.image: paths.append(args.image)
    if args.folder:
        exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
        for e in exts:
            paths += glob.glob(os.path.join(args.folder, e))

    if not paths:
        print("Nada que clasificar. Usa --image o --folder.")
        return

    results = []
    for p in paths:
        try:
            r = predict_image(p); r["path"] = p; results.append(r)
            print(f"{p}: {r['pred_label']} (prob_ia={r['prob_ia']:.3f})")
        except Exception as e:
            print(f"Error con {p}: {e}")

    print("\nTop features del modelo:")
    for name, imp in explain_top_features(10):
        print(f"- {name}: {imp:.4f}")

    with open("predicciones.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nResultados guardados en predicciones.json")

if __name__ == "__main__":
    main()
