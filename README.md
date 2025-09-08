# Detector de Imágenes: ¿IA o Real? — Versión UI Mejorada

Aplicación para detectar si una imagen fue **generada por IA** o es **fotografía real**.  
Incluye **interfaz en Gradio**, explicación en **lenguaje natural**, y scripts para **entrenar tu propio modelo**.

> El modelo incluido es de demostración (baseline). Para mejor precisión, **reentrena** con tus imágenes reales y ejemplos IA.

---

##  Características
- **Interfaz profesional** con:
  - **✅ Real** / **❌ IA** y probabilidad de IA.
  - Explicación en lenguaje simple.
  - Explicación técnica opcional (top features).
- **Limpieza automática** de resultados cuando no hay imagen.
- Entrenamiento rápido con tus datos.
- Carpeta de dataset incluida (vacía).

---

## 🗂 Estructura del proyecto
```
ai_vs_real_detector/
│
├── app/                         # Código de la interfaz gráfica
│   └── main.py                   # Script principal de la UI con Gradio
│
├── src/
│   └── ai_real_detector/         # Lógica del detector
│       ├── features.py           # Extracción de características de la imagen
│       ├── model.py              # Carga, predicción y explicación técnica
│       ├── infer.py              # Clasificación desde línea de comandos
│       └── train.py              # Entrenamiento del modelo
│
├── datasets/                     # Carpeta para datasets
│   └── sample/                   # Dataset de ejemplo
│       ├── ai/                   # Imágenes generadas por IA
│       └── real/                 # Fotografías reales
│
├── models/                       # Modelos entrenados
│   └── model_baseline.pkl        # Modelo base incluido
│
├── tests/                        # Pruebas unitarias
├── README.md                     # Este archivo
└── requirements.txt              # Dependencias del proyecto
```

---

##  Instalación
1. Tener Python 3.10 o superior.
2. (Opcional) Crear entorno virtual:
```bash
py -m venv .venv
.venv\Scripts\activate
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```
3. Instalar dependencias:
```bash
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

---

##  Uso rápido
Iniciar interfaz:
```bash
py -m app.main  (O usa "python -m app.main")


```
Abrir en navegador:
```
http://127.0.0.1:7860
```

Pasos:
1. Cargar imagen.
2. Presionar **Analizar**.
3. Ver resultado y explicación.

---

##  CLI
Clasificar una imagen:
```bash
py -m src.ai_real_detector.infer --image "ruta/imagen.jpg"
```

Clasificar carpeta:
```bash
py -m src.ai_real_detector.infer --folder "ruta/carpeta"
```


##  Entrenar modelo
1. Colocar imágenes en:
```
datasets/sample/ai/
datasets/sample/real/
```
2. Entrenar:
```bash
py -m src.ai_real_detector.train --data-dir datasets/sample --out models/model_baseline.pkl
```
3. Ejecutar interfaz:
```bash
py -m app.main
```

---

##  Consejos
- Usar imágenes variadas.
- Mantener balance entre IA y reales.
- Ajustar umbral de IA en `model.py` si es necesario.

---

##  Privacidad
- En local no se sube nada a Internet.
- Si publicas online, usar HTTPS y controles de acceso.

---

##  Problemas comunes
- **ModuleNotFoundError**: `py -m pip install -r requirements.txt` Yo uso py pero si no sirve usa `python -m pip install -r requirements.txt`
- **No se encontraron imágenes**: colocar archivos en carpetas correspondientes.
- **No carga modelo**: entrenar o copiar `model_baseline.pkl`.
- **En caso de que los comandos no sirvan remplaza  `py` por `python`.**
--- 

##  Licencia
MIT License  
Copyright (c) 2025 Erick Alejandro González Lizano

