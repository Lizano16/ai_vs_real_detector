# Image Detector: AI or Real? — Enhanced UI Version

Application to detect if an image was **AI-generated** or a **real photograph**.  
Includes **Gradio interface**, explanation in **natural language**, and scripts to **train your own model**.

> The included model is for demonstration (baseline). For better accuracy, **retrain** with your real images and AI examples.

---

##  Features
- **Professional interface** with:
  - **✅ Real** / **❌ AI** and AI probability.
  - Explanation in simple language.
  - Optional technical explanation (top features).
- **Automatic cleaning** of results when there is no image.
- Quick training with your own data.
- Dataset folder included (empty).

---

## 🗂 Project Structure
```
ai_vs_real_detector/
│
├── app/                         # Graphic interface code
│   └── main.py                   # Main Gradio UI script
│
├── src/
│   └── ai_real_detector/         # Detector logic
│       ├── features.py           # Image feature extraction
│       ├── model.py              # Loading, prediction, and technical explanation
│       ├── infer.py              # Command-line classification
│       └── train.py              # Model training
│
├── datasets/                     # Dataset folder
│   └── sample/                   # Sample dataset
│       ├── ai/                   # AI-generated images
│       └── real/                 # Real photographs
│
├── models/                       # Trained models
│   └── model_baseline.pkl        # Included baseline model
│
├── tests/                        # Unit tests
├── README.md                     # This file
└── requirements.txt              # Project dependencies
```

---

##  Installation
1. Have Python 3.10 or higher.
2. (Optional) Create a virtual environment:
```bash
py -m venv .venv
.venv\Scripts\activate
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```
3. Install dependencies:
```bash
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

---

##  Quick Usage
Start the interface:
```bash
py -m app.main
```
Open in browser:
```
http://127.0.0.1:7860
```

Steps:
1. Upload image.
2. Press **Analyze**.
3. View result and explanation.

---

##  CLI
Classify a single image:
```bash
py -m src.ai_real_detector.infer --image "path/image.jpg"
```

Classify a folder:
```bash
py -m src.ai_real_detector.infer --folder "path/folder"
```

---

 **If the commands don't work, replace `py` with `python`.**

---

##  Train the Model
1. Place images in:
```
datasets/sample/ai/
datasets/sample/real/
```
2. Train:
```bash
py -m src.ai_real_detector.train --data-dir datasets/sample --out models/model_baseline.pkl
```
3. Run interface:
```bash
py -m app.main
```

---

##  Tips
- Use a variety of images.
- Keep balance between AI and real.
- Adjust AI threshold in `model.py` if needed.

---

##  Privacy
- Local usage: nothing is uploaded to the Internet.
- If deployed online, use HTTPS and access control.

---

##  Common Issues
- **ModuleNotFoundError**:  
  `py -m pip install -r requirements.txt`  
  If `py` doesn't work, use:  
  `python -m pip install -r requirements.txt`
- **No images found**: Place files in the correct folders.
- **Model won't load**: Train or copy `model_baseline.pkl`.

---

##  License
MIT License  
Copyright (c) 2025 Erick Alejandro González Lizano
