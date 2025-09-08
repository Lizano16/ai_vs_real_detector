import gradio as gr
import os
from PIL import Image
from src.ai_real_detector.model import load_model, predict_image, explain_top_features

MODEL_PATH = os.getenv("MODEL_PATH", "models/model_baseline.pkl")
load_model(MODEL_PATH)

def explain_human(readable_features, prob_ai):
    reasons = []
    if readable_features.get("fft_ratio_high_center", 0) < 0.08 and prob_ai >= 0.5:
        reasons.append("Colores muy uniformes y poco detalle fino.")
    if readable_features.get("edges_std", 0) < 15 and prob_ai >= 0.5:
        reasons.append("Bordes y texturas suaves (apariencia digital).")
    if readable_features.get("hist_entropy_r", 0) < 0.10 and prob_ai >= 0.5:
        reasons.append("Poca variedad de tonos de color.")

    if prob_ai < 0.5:
        if readable_features.get("edges_std", 0) >= 10:
            reasons.append("Texturas y bordes presentes como en una foto.")
        if readable_features.get("fft_ratio_high_center", 0) >= 0.06:
            reasons.append("Detalle fino consistente con fotografía.")
        reasons.append("Variaciones naturales de color/ruido de cámara.")

    if not reasons:
        reasons.append("Los patrones globales coinciden con ejemplos del entrenamiento.")

    label = "IA" if prob_ai >= 0.60 else "Real"
    icon = "❌" if label == "IA" else "✅"
    return f"{icon} Predicción: {label}\nProbabilidad IA: {prob_ai:.3f}\n\nMotivos:\n- " + "\n- ".join(reasons)

def predict_ui(img: Image.Image):
    # Si no hay imagen, muestra aviso en Resultado y deja vacía la explicación técnica
    if img is None:
        return " Primero sube o toma una imagen.", ""
    tmp_path = "tmp_upload.png"
    img.save(tmp_path)
    res = predict_image(tmp_path)
    human = explain_human(res["features"], res["prob_ia"])
    top_feats = explain_top_features(8)
    exp_lines = [f"{name}: {imp:.4f}" for name, imp in top_feats]
    technical = "\n".join(exp_lines) if exp_lines else "Importancias no disponibles."
    return human, technical

with gr.Blocks(title="Detector de imágenes: IA vs Real") as demo:
    gr.Markdown("## Detector de imágenes: IA vs Real")
    gr.Markdown("Sube una imagen y te diré si parece **IA** o **Real**. Explicación en lenguaje simple.")

    with gr.Row():
        with gr.Column(scale=6):
            inp = gr.Image(type="pil", label="Imagen", sources=["upload", "webcam"], height=420)
            # Botón deshabilitado hasta que haya imagen
            btn = gr.Button("Analizar", variant="primary", interactive=False)
        with gr.Column(scale=5):
            out_label = gr.Textbox(label="Resultado", lines=6, interactive=False)
            out_exp = gr.Textbox(label="Explicación (Técnica opcional)", lines=10, interactive=False)

    # Limpia y deshabilita botón si no hay imagen; habilita si sí hay
    def _clear_if_none(img):
        if img is None:
            return "", "", gr.update(interactive=False)
        return gr.update(), gr.update(), gr.update(interactive=True)

    # Ahora actualizamos 3 salidas: resultado, explicación y botón
    inp.change(fn=_clear_if_none, inputs=[inp], outputs=[out_label, out_exp, btn])
    btn.click(fn=predict_ui, inputs=[inp], outputs=[out_label, out_exp])

if __name__ == "__main__":
    demo.launch()

