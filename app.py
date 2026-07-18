import base64
# pyrefly: ignore [missing-import]
import numpy as np
import cv2
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import os

app = FastAPI(title="Satori AI - Neural Ink Backend")

# Model path and base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.onnx")
ort_session = None

# Labels mapping (EMNIST 36 classes: 10 digits + 26 uppercase letters)
EMNIST_CLASSES = [str(i) for i in range(10)] + [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Cache session loading
def get_session():
    global ort_session
    if ort_session is None:
        if os.path.exists(MODEL_PATH):
            # Load with CPU provider
            ort_session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        else:
            ort_session = None
    return ort_session

class PredictRequest(BaseModel):
    image: str # Base64 data URL

def preprocess_image(base64_str: str) -> np.ndarray:
    # 1. Decode base64 to image
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    # The canvas output has an alpha channel, decode as IMREAD_UNCHANGED to keep it
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError("Could not decode image.")

    # 2. Extract drawing from canvas
    # The user draws with orange ink on transparent/dark canvas.
    # If the image has 4 channels (RGBA), the drawing is in the RGB channels and alpha channel represents opacity.
    # Let's extract the alpha channel if available, as it represents exactly what was drawn!
    if img.shape[2] == 4:
        # Alpha channel represents the brush stroke opacity (0 = background, 255 = drawn ink)
        gray = img[:, :, 3]
    else:
        # Fallback to standard grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    # 3. Find bounding box of ink to crop and auto-center
    coords = cv2.findNonZero(gray)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Crop the drawing
        cropped = gray[y:y+h, x:x+w]
        
        # Add padding to make it square
        max_dim = max(w, h)
        padded = np.zeros((max_dim, max_dim), dtype=np.uint8)
        
        # Center the cropped image in the padded square
        pad_x = (max_dim - w) // 2
        pad_y = (max_dim - h) // 2
        padded[pad_y:pad_y+h, pad_x:pad_x+w] = cropped
        
        # Add extra border padding (e.g. 15% margin) to avoid drawing touching the edges
        border = int(max_dim * 0.15)
        if border > 0:
            padded = cv2.copyMakeBorder(padded, border, border, border, border, cv2.BORDER_CONSTANT, value=0)
            
        gray_processed = padded
    else:
        # Fallback if canvas is empty
        gray_processed = cv2.resize(gray, (28, 28))

    # 4. Resize to 28x28
    img_resized = cv2.resize(gray_processed, (28, 28), interpolation=cv2.INTER_AREA)

    # 5. MNIST specific: no transpose needed as training data is upright
    img_float = img_resized.astype(np.float32) / 255.0
    
    # 6. MNIST dataset normalizations
    img_normalized = (img_float - 0.1307) / 0.3081
    
    # Reshape to (1, 1, 28, 28)
    img_final = np.expand_dims(np.expand_dims(img_normalized, axis=0), axis=0)
    return img_final

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1)

@app.post("/predict")
async def predict(req: PredictRequest):
    session = get_session()
    if session is None:
        return {
            "success": False,
            "error": "Model not trained yet. Please run train.py first to generate model.onnx."
        }
    
    try:
        input_tensor = preprocess_image(req.image)
        ort_inputs = {session.get_inputs()[0].name: input_tensor}
        ort_outs = session.run(None, ort_inputs)
        logits = ort_outs[0][0]
        
        # Apply softmax to get probabilities
        probs = softmax(logits)
        
        # Sort predictions
        top_indices = np.argsort(probs)[::-1][:5]
        
        predictions = []
        for idx in top_indices:
            # Map index to EMNIST label
            label = EMNIST_CLASSES[idx] if idx < len(EMNIST_CLASSES) else str(idx)
            predictions.append({
                "label": label,
                "confidence": float(probs[idx])
            })
            
        return {
            "success": True,
            "prediction": predictions[0]["label"],
            "confidence": predictions[0]["confidence"],
            "top5": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Serve static files from root directory
@app.get("/")
async def get_index():
    path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return HTMLResponse("<h1>Satori AI Frontend is loading...</h1>")

@app.get("/style.css")
async def get_style():
    return FileResponse(os.path.join(BASE_DIR, "style.css"))

@app.get("/app.js")
async def get_js():
    return FileResponse(os.path.join(BASE_DIR, "app.js"))

@app.get("/model.onnx")
async def get_model():
    return FileResponse(os.path.join(BASE_DIR, "model.onnx"))

@app.get("/model.onnx.data")
async def get_model_data():
    return FileResponse(os.path.join(BASE_DIR, "model.onnx.data"))
