# Satori AI - Neural Ink Engine 🎨🧠

An immersive deep learning workspace demonstrating high-performance handwritten character recognition. Draw any digit (`0-9`) or uppercase letter (`A-Z`) on the interactive glassmorphic canvas to witness real-time neural classification (under 5ms latency).

---

## ⚡ Live Web App
Access the live deployment here: **[⚡ Live Web App Demo](https://somurex.github.io/CodeAlpha_Handwriten_charecter_recognition/)**

---

## ✨ Features

- **Full 36-Class Recognition:** Accurately classifies digits `0-9` and uppercase letters `A-Z`.
- **Robust Preprocessing Pipeline:** 
  - Canvas transparency alpha channel extraction to isolate stroke ink.
  - Dynamic bounding-box cropping to center and normalize characters.
  - Aspect-ratio preserving pad-to-square with 15% margin offsets.
- **High Performance:** Inference is executed using an optimized ONNX CPU runtime session, bringing prediction latencies below **5ms**.
- **Modern Interactive UI:** 
  - Responsive canvas with mouse and touch drawing support.
  - Floating predictions panel displaying real-time top-5 classification probabilities.
  - Double-buffered brush and eraser drawing modes with undo stack capabilities.
  - Interactive tilt card specifications and dynamic central core visuals.

---

## 🛠️ Technology Stack

- **Deep Learning Framework:** PyTorch (Model definition & training)
- **Model Interoperability:** ONNX (Open Neural Network Exchange)
- **Inference Runtime:** ONNX Runtime (CPU Execution Provider)
- **Backend API:** FastAPI (served via Uvicorn)
- **Image Processing:** OpenCV & Pillow (for synthetic generation & prep)
- **Frontend Engine:** Semantic HTML5, Vanilla CSS3 (Custom Variables, Transitions), Modern JavaScript (Canvas API)

---

## 📂 Project Structure

```text
├── app.py              # FastAPI server & image preprocessing pipeline
├── train.py            # PyTorch model training & ONNX export script
├── model.onnx          # Exported 36-class neural weights
├── requirements.txt    # Python dependencies file
├── static/
│   ├── index.html      # Main user interface
│   ├── app.js          # Canvas drawing, UI handlers & API calls
│   └── style.css       # Premium custom variables & style system
└── .gitignore          # File exclusions (excludes local datasets)
```

---

## 🚀 Getting Started

Follow these steps to run the Satori AI engine locally:

### 1. Clone the Repository
```bash
git clone https://github.com/SOMUREX/CodeAlpha_Handwriten_charecter_recognition.git
cd CodeAlpha_Handwriten_charecter_recognition
```

### 2. Set Up Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate    # On macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train and Export the Model (Optional)
A pre-trained `model.onnx` is already included. To retrain the 36-class CNN model on a newly generated synthetic dataset:
```bash
python -X utf8 train.py
```

### 5. Start the Server
Run the FastAPI application locally:
```bash
python -m uvicorn app:app --port 8000 --reload
```
Open **`http://127.0.0.1:8000`** in your browser to start drawing!

---

## 🧬 Model Architecture

The custom CNN architecture consists of:
1. **Conv2D Block 1:** 32 filters, $3 \times 3$ kernel + BatchNorm + ReLU
2. **Conv2D Block 2:** 64 filters, $3 \times 3$ kernel + BatchNorm + ReLU + MaxPool2D ($2 \times 2$) + Dropout (0.25)
3. **Fully Connected Block:** Dense Layer (128 units) + BatchNorm + ReLU + Dropout (0.5)
4. **Output Classifier:** Softmax dense layer with 36 outputs.
