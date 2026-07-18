document.addEventListener('DOMContentLoaded', () => {
    // === ONNX Runtime Web Initialization ===
    let ortSession = null;
    async function loadModel() {
        try {
            console.log("Loading ONNX model and weights dynamically...");
            const modelBuf = await fetch('./model.onnx').then(r => r.arrayBuffer());
            const dataBuf = await fetch('./model.onnx.data').then(r => r.arrayBuffer());
            ortSession = await ort.InferenceSession.create(new Uint8Array(modelBuf), {
                externalData: [
                    {
                        data: new Uint8Array(dataBuf),
                        path: 'model.onnx.data'
                    }
                ]
            });
            console.log("ONNX Runtime Web loaded successfully. Client-side inference enabled.");
        } catch (e) {
            console.warn("ONNX Runtime Web failed to load model.onnx. Falling back to API server.", e);
        }
    }
    loadModel();

    // === Canvas Drawing Logic ===
    const canvas = document.getElementById('paint-canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const canvasWrapper = document.querySelector('.canvas-wrapper');
    
    let isDrawing = false;
    let currentMode = 'draw'; // 'draw' or 'erase'
    let drawingHistory = [];
    
    // Line style configuration
    const BRUSH_COLOR = '#FF6B35';
    const BRUSH_SIZE = 24;
    const ERASER_SIZE = 36;
    
    // Set canvas dimensions explicitly to match its display size for high resolution drawing
    function resizeCanvas() {
        // Draw logic uses coordinates relative to the canvas internal width/height.
        // We hardcode the internal size to 450x450 to match the backend expectations,
        // and CSS scaling handles the display representation.
        canvas.width = 450;
        canvas.height = 450;
        
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        clearCanvas();
        saveState(); // Save initial empty state
    }
    
    function clearCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        canvasWrapper.classList.remove('drawing');
    }
    
    // Save state for undo functionality
    function saveState() {
        if (drawingHistory.length >= 20) {
            drawingHistory.shift();
        }
        drawingHistory.push(canvas.toDataURL());
    }
    
    function undo() {
        if (drawingHistory.length > 1) {
            drawingHistory.pop(); // Remove current state
            const previousStateUrl = drawingHistory[drawingHistory.length - 1];
            
            const img = new Image();
            img.onload = () => {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0);
            };
            img.src = previousStateUrl;
        } else if (drawingHistory.length === 1) {
            clearCanvas();
        }
    }
    
    // Get mouse/touch position relative to canvas
    function getCoordinates(e) {
        const rect = canvas.getBoundingClientRect();
        // Calculate coordinate scale factor in case CSS has resized the display width
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        
        let clientX, clientY;
        if (e.touches && e.touches.length > 0) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        } else {
            clientX = e.clientX;
            clientY = e.clientY;
        }
        
        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        };
    }
    
    function startDrawing(e) {
        isDrawing = true;
        canvasWrapper.classList.add('drawing');
        
        const coords = getCoordinates(e);
        ctx.beginPath();
        ctx.moveTo(coords.x, coords.y);
        
        // Configure brush based on active mode
        if (currentMode === 'draw') {
            ctx.globalCompositeOperation = 'source-over';
            ctx.strokeStyle = BRUSH_COLOR;
            ctx.lineWidth = BRUSH_SIZE;
            ctx.shadowBlur = 4;
            ctx.shadowColor = BRUSH_COLOR;
        } else {
            ctx.globalCompositeOperation = 'destination-out';
            ctx.lineWidth = ERASER_SIZE;
            ctx.shadowBlur = 0;
        }
        
        // Draw a single dot on click/tap
        ctx.lineTo(coords.x, coords.y);
        ctx.stroke();
        
        e.preventDefault();
    }
    
    function draw(e) {
        if (!isDrawing) return;
        
        const coords = getCoordinates(e);
        ctx.lineTo(coords.x, coords.y);
        ctx.stroke();
        
        e.preventDefault();
    }
    
    function stopDrawing() {
        if (isDrawing) {
            isDrawing = false;
            saveState();
            
            // Auto trigger inference in real-time
            autoTriggerInference();
        }
    }
    
    // Mouse Event Listeners
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseleave', stopDrawing);
    
    // Touch Event Listeners (Mobile support)
    canvas.addEventListener('touchstart', startDrawing, { passive: false });
    canvas.addEventListener('touchmove', draw, { passive: false });
    canvas.addEventListener('touchend', stopDrawing);
    
    // Initialize
    resizeCanvas();
    
    // Controls
    const btnDraw = document.getElementById('btn-draw');
    const btnErase = document.getElementById('btn-erase');
    const btnUndo = document.getElementById('btn-undo');
    const btnClear = document.getElementById('btn-clear');
    const btnPredict = document.getElementById('btn-predict');
    const predictionsPanel = document.getElementById('predictions-box');
    
    btnDraw.addEventListener('click', () => {
        currentMode = 'draw';
        btnDraw.classList.add('active');
        btnErase.classList.remove('active');
    });
    
    btnErase.addEventListener('click', () => {
        currentMode = 'erase';
        btnErase.classList.add('active');
        btnDraw.classList.remove('active');
    });
    
    btnUndo.addEventListener('click', undo);
    btnClear.addEventListener('click', () => {
        clearCanvas();
        predictionsPanel.style.display = 'none';
    });
    
    btnPredict.addEventListener('click', analyzeInk);
    
    // Debounce timer for real-time predictions
    let inferenceTimeout;
    function autoTriggerInference() {
        clearTimeout(inferenceTimeout);
        inferenceTimeout = setTimeout(() => {
            analyzeInk();
        }, 800); // Trigger after 800ms of inactivity
    }
    
    // Helper: Softmax activation function
    function softmax(arr) {
        const maxVal = Math.max(...arr);
        const exps = arr.map(x => Math.exp(x - maxVal));
        const sumExps = exps.reduce((a, b) => a + b, 0);
        return arr.map((_, i) => exps[i] / sumExps);
    }

    // Helper: Preprocess canvas drawing to float32 input tensor [1, 1, 28, 28]
    function preprocessCanvas() {
        const buffer = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = buffer.data;
        
        let minX = canvas.width, maxX = 0, minY = canvas.height, maxY = 0;
        let hasInk = false;
        for (let y = 0; y < canvas.height; y++) {
            for (let x = 0; x < canvas.width; x++) {
                const alpha = data[(y * canvas.width + x) * 4 + 3];
                if (alpha > 10) {
                    hasInk = true;
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }
        
        if (!hasInk) return null;
        
        // Dynamic bounding box cropping with 15% margins
        const w = maxX - minX + 1;
        const h = maxY - minY + 1;
        const maxDim = Math.max(w, h);
        
        const padding = Math.floor(maxDim * 0.15);
        const tempSize = maxDim + 2 * padding;
        
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = tempSize;
        tempCanvas.height = tempSize;
        const tempCtx = tempCanvas.getContext('2d');
        
        const offsetX = padding + Math.floor((maxDim - w) / 2);
        const offsetY = padding + Math.floor((maxDim - h) / 2);
        
        tempCtx.drawImage(
            canvas,
            minX, minY, w, h,
            offsetX, offsetY, w, h
        );
        
        // Resize to 28x28
        const resizeCanvas = document.createElement('canvas');
        resizeCanvas.width = 28;
        resizeCanvas.height = 28;
        const resizeCtx = resizeCanvas.getContext('2d');
        resizeCtx.drawImage(tempCanvas, 0, 0, tempSize, tempSize, 0, 0, 28, 28);
        
        // Extract 28x28 normalized grayscale values (via alpha channel)
        const resizeData = resizeCtx.getImageData(0, 0, 28, 28).data;
        const floatData = new Float32Array(28 * 28);
        for (let i = 0; i < 28 * 28; i++) {
            const val = resizeData[i * 4 + 3] / 255.0;
            // Normalize: (val - 0.1307) / 0.3081
            floatData[i] = (val - 0.1307) / 0.3081;
        }
        
        return new ort.Tensor('float32', floatData, [1, 1, 28, 28]);
    }

    // API Call / ONNX Runtime Web Inference
    async function analyzeInk() {
        // Check if canvas is completely empty (transparent)
        const buffer = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = buffer.data;
        let hasInk = false;
        for (let i = 3; i < data.length; i += 4) {
            if (data[i] > 10) { // Check if alpha channel has drawing
                hasInk = true;
                break;
            }
        }
        
        if (!hasInk) {
            predictionsPanel.style.display = 'none';
            return;
        }

        // Try client-side ONNX Web Runtime inference first
        if (ortSession) {
            try {
                const tensor = preprocessCanvas();
                if (tensor) {
                    const feeds = { [ortSession.inputNames[0]]: tensor };
                    const results = await ortSession.run(feeds);
                    const output = results[ortSession.outputNames[0]];
                    const logits = Array.from(output.data);
                    
                    const probs = softmax(logits);
                    
                    // 36 Classes
                    const classes = [
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                        'U', 'V', 'W', 'X', 'Y', 'Z'
                    ];
                    
                    const indexedProbs = probs.map((p, i) => ({ label: classes[i], confidence: p }));
                    indexedProbs.sort((a, b) => b.confidence - a.confidence);
                    
                    const top5 = indexedProbs.slice(0, 5);
                    
                    const result = {
                        success: true,
                        prediction: top5[0].label,
                        confidence: top5[0].confidence,
                        top5: top5
                    };
                    
                    renderPredictions(result);
                    return;
                }
            } catch (err) {
                console.warn("Client-side ONNX Web Runtime execution failed, falling back to API:", err);
            }
        }
        
        // Fallback to backend API
        const dataUrl = canvas.toDataURL('image/png');
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: dataUrl })
            });
            
            const result = await response.json();
            
            if (result.success) {
                renderPredictions(result);
            } else {
                console.error(result.error);
            }
        } catch (err) {
            console.error('Error during prediction API call:', err);
        }
    }
    
    function renderPredictions(data) {
        predictionsPanel.style.display = 'block';
        
        // Render main prediction
        const predMain = document.getElementById('pred-main');
        const predConf = document.getElementById('pred-conf');
        
        predMain.textContent = data.prediction;
        predConf.textContent = `${(data.confidence * 100).toFixed(1)}% Confidence`;
        
        // Render top 5
        const top5Container = document.getElementById('pred-top5');
        top5Container.innerHTML = '';
        
        data.top5.forEach((item, index) => {
            const row = document.createElement('div');
            row.className = 'pred-row';
            
            const label = document.createElement('span');
            label.className = 'pred-row-label';
            label.textContent = item.label;
            
            const barWrap = document.createElement('div');
            barWrap.className = 'pred-row-bar-wrap';
            
            const bar = document.createElement('div');
            bar.className = 'pred-row-bar';
            
            const pct = document.createElement('span');
            pct.className = 'pred-row-pct';
            pct.textContent = `${(item.confidence * 100).toFixed(0)}%`;
            
            barWrap.appendChild(bar);
            row.appendChild(label);
            row.appendChild(barWrap);
            row.appendChild(pct);
            top5Container.appendChild(row);
            
            // Animate width progress bar shortly after append
            setTimeout(() => {
                bar.style.width = `${item.confidence * 100}%`;
            }, 50);
        });
    }
    
    // === Perspective Tilt Card Effect ===
    const cards = document.querySelectorAll('.tilt-card-wrapper');
    
    cards.forEach(wrapper => {
        const card = wrapper.querySelector('.tilt-card');
        
        wrapper.addEventListener('mousemove', (e) => {
            const rect = wrapper.getBoundingClientRect();
            // Mouse coordinates relative to card element
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // Calculate offsets from the center of the card
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            
            // Degrees of rotation (-5deg to 5deg)
            const rotateX = ((centerY - y) / centerY) * 6;
            const rotateY = ((x - centerX) / centerX) * 6;
            
            // Apply transformations
            card.style.transform = `rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale(1.02)`;
        });
        
        wrapper.addEventListener('mouseleave', () => {
            // Smooth reset of card transformation
            card.style.transform = 'rotateX(0deg) rotateY(0deg) scale(1)';
        });
    });
    
    // === Feedback / Calibration Form Submission ===
    const feedbackForm = document.getElementById('feedback-form');
    const successMsg = document.getElementById('form-success-msg');
    
    feedbackForm.addEventListener('submit', (e) => {
        e.preventDefault();
        
        const username = document.getElementById('user-name').value;
        const drawnChar = document.getElementById('drawn-char').value;
        
        // In real project, we would POST this to a database endpoint.
        // For demonstration, we simulate feedback success banner.
        feedbackForm.classList.add('hidden');
        successMsg.classList.remove('hidden');
        
        // Reset form after a timeout to allow subsequent entries
        setTimeout(() => {
            feedbackForm.reset();
            feedbackForm.classList.remove('hidden');
            successMsg.classList.add('hidden');
        }, 5000);
    });
    
    // === Scroll trigger animations & visual enhancements ===
    const scrollBtn = document.getElementById('scroll-to-canvas');
    if (scrollBtn) {
        scrollBtn.addEventListener('click', (e) => {
            e.preventDefault();
            document.getElementById('playground').scrollIntoView({ behavior: 'smooth' });
            
            // Pulsate canvas border to draw attention
            const wrapper = document.querySelector('.canvas-wrapper');
            wrapper.style.borderColor = '#FFD700';
            setTimeout(() => {
                wrapper.style.borderColor = 'rgba(255, 107, 53, 0.25)';
            }, 1500);
        });
    }
});
