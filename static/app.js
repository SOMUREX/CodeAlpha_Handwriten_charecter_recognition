document.addEventListener('DOMContentLoaded', () => {
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
    
    // API Call to Backend `/predict`
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
