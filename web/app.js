/**
 * YOLO Tank Detection - Web Application
 *
 * Browser-based inference using ONNX Runtime Web
 */

// =====================================================
// Configuration
// =====================================================

const CONFIG = {
    modelPath: 'models/student_quantized.onnx',
    inputSize: 640,
    confThreshold: 0.15,  // Lowered for webcam testing (was 0.25)
    iouThreshold: 0.45,
    classNames: ['Airplane', 'Helicopter', 'Person', 'Tank', 'Vehicle'],
    colors: [
        '#FF6B6B',  // Airplane - Red
        '#4ECDC4',  // Helicopter - Teal
        '#45B7D1',  // Person - Blue
        '#96CEB4',  // Tank - Green
        '#FFEAA7'   // Vehicle - Yellow
    ]
};

// =====================================================
// Global Variables
// =====================================================

let session = null;
let isModelLoaded = false;

// Webcam variables
let webcamStream = null;
let isWebcamRunning = false;
let animationFrameId = null;
let lastFrameTime = 0;
let frameCount = 0;
let fps = 0;

// DOM Elements
const uploadSection = document.getElementById('upload-section');
const fileInput = document.getElementById('file-input');
const uploadBtn = document.getElementById('upload-btn');
const loadingDiv = document.getElementById('loading');
const resultSection = document.getElementById('result-section');
const resultCanvas = document.getElementById('result-canvas');
const detectionResults = document.getElementById('detection-results');
const newImageBtn = document.getElementById('new-image-btn');
const statusBar = document.getElementById('status-bar');

// Webcam DOM Elements
const webcamSection = document.getElementById('webcam-section');
const webcamVideo = document.getElementById('webcam-video');
const webcamCanvas = document.getElementById('webcam-canvas');
const startWebcamBtn = document.getElementById('start-webcam-btn');
const stopWebcamBtn = document.getElementById('stop-webcam-btn');
const fpsDisplay = document.getElementById('fps-display');
const imageModeBtn = document.getElementById('image-mode-btn');
const webcamModeBtn = document.getElementById('webcam-mode-btn');

// =====================================================
// Model Loading
// =====================================================

async function loadModel() {
    try {
        updateStatus('Loading model...', 'loading');

        // Create ONNX Runtime session
        session = await ort.InferenceSession.create(CONFIG.modelPath, {
            executionProviders: ['wasm'],  // WebAssembly backend
            graphOptimizationLevel: 'all'
        });

        isModelLoaded = true;
        updateStatus('Model ready', 'ready');
        console.log('Model loaded successfully');

    } catch (error) {
        console.error('Model loading error:', error);
        updateStatus('Model loading failed: ' + error.message, 'error');
    }
}

// =====================================================
// Image Processing
// =====================================================

function preprocessImage(img) {
    // Draw image on canvas (resize to 640x640)
    const canvas = document.createElement('canvas');
    canvas.width = CONFIG.inputSize;
    canvas.height = CONFIG.inputSize;
    const ctx = canvas.getContext('2d');

    // Get actual dimensions (handle both Image and Video elements)
    const imgWidth = img.videoWidth || img.width;
    const imgHeight = img.videoHeight || img.height;

    // Maintain aspect ratio and center
    const scale = Math.min(
        CONFIG.inputSize / imgWidth,
        CONFIG.inputSize / imgHeight
    );
    const scaledWidth = imgWidth * scale;
    const scaledHeight = imgHeight * scale;
    const offsetX = (CONFIG.inputSize - scaledWidth) / 2;
    const offsetY = (CONFIG.inputSize - scaledHeight) / 2;

    // Fill background (letterbox)
    ctx.fillStyle = '#808080';
    ctx.fillRect(0, 0, CONFIG.inputSize, CONFIG.inputSize);

    // Draw image
    ctx.drawImage(img, offsetX, offsetY, scaledWidth, scaledHeight);

    // Extract image data
    const imageData = ctx.getImageData(0, 0, CONFIG.inputSize, CONFIG.inputSize);
    const data = imageData.data;

    // Normalize RGB and convert to CHW format
    const float32Data = new Float32Array(3 * CONFIG.inputSize * CONFIG.inputSize);

    for (let i = 0; i < CONFIG.inputSize * CONFIG.inputSize; i++) {
        float32Data[i] = data[i * 4] / 255.0;                                    // R
        float32Data[i + CONFIG.inputSize * CONFIG.inputSize] = data[i * 4 + 1] / 255.0;  // G
        float32Data[i + 2 * CONFIG.inputSize * CONFIG.inputSize] = data[i * 4 + 2] / 255.0;  // B
    }

    return {
        tensor: float32Data,
        scale: scale,
        offsetX: offsetX,
        offsetY: offsetY,
        originalWidth: imgWidth,
        originalHeight: imgHeight
    };
}

// =====================================================
// Inference
// =====================================================

async function runInference(imageData) {
    if (!isModelLoaded || !session) {
        throw new Error('Model is not loaded');
    }

    // Create input tensor
    const inputTensor = new ort.Tensor(
        'float32',
        imageData.tensor,
        [1, 3, CONFIG.inputSize, CONFIG.inputSize]
    );

    // Run inference
    const feeds = { images: inputTensor };
    const results = await session.run(feeds);

    return results;
}

// =====================================================
// Post-processing
// =====================================================

function postprocess(outputs, imageData) {
    // Parse YOLO output
    const output = outputs[Object.keys(outputs)[0]];
    const data = output.data;
    const dims = output.dims;  // [1, 9, 8400] for 5 classes (4 + 5)

    console.log('Output dims:', dims);

    const detections = [];
    const numClasses = CONFIG.classNames.length;  // 5
    const numBoxes = dims[2];  // 8400

    // YOLOv8 output: [1, 4+numClasses, 8400]
    // Rows 0-3: x_center, y_center, width, height
    // Rows 4-8: class scores (5 classes)

    for (let i = 0; i < numBoxes; i++) {
        // Box coordinates (normalized values)
        const x_center = data[0 * numBoxes + i];
        const y_center = data[1 * numBoxes + i];
        const width = data[2 * numBoxes + i];
        const height = data[3 * numBoxes + i];

        // Class scores
        let maxScore = 0;
        let classId = 0;
        for (let c = 0; c < numClasses; c++) {
            const score = data[(4 + c) * numBoxes + i];
            if (score > maxScore) {
                maxScore = score;
                classId = c;
            }
        }

        // Confidence threshold filtering
        if (maxScore < CONFIG.confThreshold) continue;

        // Convert to original image coordinates
        const x1 = (x_center - width / 2 - imageData.offsetX) / imageData.scale;
        const y1 = (y_center - height / 2 - imageData.offsetY) / imageData.scale;
        const x2 = (x_center + width / 2 - imageData.offsetX) / imageData.scale;
        const y2 = (y_center + height / 2 - imageData.offsetY) / imageData.scale;

        // Clipping
        const clippedX1 = Math.max(0, Math.min(x1, imageData.originalWidth));
        const clippedY1 = Math.max(0, Math.min(y1, imageData.originalHeight));
        const clippedX2 = Math.max(0, Math.min(x2, imageData.originalWidth));
        const clippedY2 = Math.max(0, Math.min(y2, imageData.originalHeight));

        detections.push({
            x1: clippedX1,
            y1: clippedY1,
            x2: clippedX2,
            y2: clippedY2,
            confidence: maxScore,
            classId: classId
        });
    }

    // Apply NMS
    if (detections.length > 0) {
        const boxes = detections.map(d => ({ x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2 }));
        const scores = detections.map(d => d.confidence);
        const nmsIndices = nonMaxSuppression(boxes, scores, CONFIG.iouThreshold);
        return nmsIndices.map(idx => detections[idx]);
    }

    return detections;
}

function nonMaxSuppression(boxes, scores, iouThreshold) {
    // NMS algorithm implementation
    const indices = [];
    const sortedIndices = scores
        .map((score, idx) => ({ score, idx }))
        .sort((a, b) => b.score - a.score)
        .map(item => item.idx);

    while (sortedIndices.length > 0) {
        const current = sortedIndices.shift();
        indices.push(current);

        const remainingIndices = [];
        for (const idx of sortedIndices) {
            const iou = calculateIoU(boxes[current], boxes[idx]);
            if (iou < iouThreshold) {
                remainingIndices.push(idx);
            }
        }
        sortedIndices.length = 0;
        sortedIndices.push(...remainingIndices);
    }

    return indices;
}

function calculateIoU(box1, box2) {
    const x1 = Math.max(box1.x1, box2.x1);
    const y1 = Math.max(box1.y1, box2.y1);
    const x2 = Math.min(box1.x2, box2.x2);
    const y2 = Math.min(box1.y2, box2.y2);

    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    const area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    const union = area1 + area2 - intersection;

    return intersection / union;
}

// =====================================================
// Drawing
// =====================================================

function drawDetections(img, detections) {
    const ctx = resultCanvas.getContext('2d');

    // Set canvas size
    resultCanvas.width = img.width;
    resultCanvas.height = img.height;

    // Draw original image
    ctx.drawImage(img, 0, 0);

    // Draw detection results
    for (const det of detections) {
        const color = CONFIG.colors[det.classId] || '#FFFFFF';

        // Bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);

        // Label background
        const label = `${CONFIG.classNames[det.classId]} ${(det.confidence * 100).toFixed(1)}%`;
        ctx.font = 'bold 14px Arial';
        const textWidth = ctx.measureText(label).width;

        ctx.fillStyle = color;
        ctx.fillRect(det.x1, det.y1 - 22, textWidth + 10, 22);

        // Label text
        ctx.fillStyle = '#000000';
        ctx.fillText(label, det.x1 + 5, det.y1 - 6);
    }

    // Update results list
    updateDetectionList(detections);
}

function updateDetectionList(detections) {
    if (detections.length === 0) {
        detectionResults.innerHTML = '<p style="color: #888;">No objects detected</p>';
        return;
    }

    let html = '';
    for (const det of detections) {
        html += `
            <div class="detection-item">
                <span class="detection-class" style="color: ${CONFIG.colors[det.classId]}">
                    ${CONFIG.classNames[det.classId]}
                </span>
                <span class="detection-confidence">
                    ${(det.confidence * 100).toFixed(1)}%
                </span>
            </div>
        `;
    }

    detectionResults.innerHTML = html;
}

// =====================================================
// UI Helpers
// =====================================================

function updateStatus(message, type) {
    statusBar.textContent = message;
    statusBar.className = 'status-bar ' + type;
}

function showLoading() {
    loadingDiv.classList.add('active');
    uploadSection.style.display = 'none';
    resultSection.classList.remove('active');
}

function showResult() {
    loadingDiv.classList.remove('active');
    uploadSection.style.display = 'none';
    resultSection.classList.add('active');
}

function resetUI() {
    loadingDiv.classList.remove('active');
    uploadSection.style.display = 'block';
    resultSection.classList.remove('active');
}

// =====================================================
// Event Handlers
// =====================================================

async function handleImageUpload(file) {
    if (!file || !file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
    }

    showLoading();

    try {
        // Load image
        const img = await loadImage(file);

        // Preprocessing
        const imageData = preprocessImage(img);

        // Inference
        updateStatus('Running inference...', 'loading');
        const outputs = await runInference(imageData);

        // Post-processing
        const detections = postprocess(outputs, imageData);

        // Display results
        drawDetections(img, detections);
        showResult();
        updateStatus('Detection complete', 'ready');

    } catch (error) {
        console.error('Detection error:', error);
        updateStatus('Detection failed: ' + error.message, 'error');
        resetUI();
    }
}

function loadImage(file) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = URL.createObjectURL(file);
    });
}

// =====================================================
// Event Listeners
// =====================================================

// File select button
uploadBtn.addEventListener('click', () => {
    fileInput.click();
});

// On file select
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleImageUpload(e.target.files[0]);
    }
});

// Drag and drop
uploadSection.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadSection.classList.add('dragover');
});

uploadSection.addEventListener('dragleave', () => {
    uploadSection.classList.remove('dragover');
});

uploadSection.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadSection.classList.remove('dragover');

    if (e.dataTransfer.files.length > 0) {
        handleImageUpload(e.dataTransfer.files[0]);
    }
});

// New image button
newImageBtn.addEventListener('click', () => {
    resetUI();
    fileInput.value = '';
});

// =====================================================
// Webcam Functions
// =====================================================

async function startWebcam() {
    if (!isModelLoaded) {
        alert('Please wait for the model to load');
        return;
    }

    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'environment'
            }
        });

        webcamVideo.srcObject = webcamStream;
        await webcamVideo.play();

        // Set canvas size
        webcamCanvas.width = webcamVideo.videoWidth;
        webcamCanvas.height = webcamVideo.videoHeight;

        isWebcamRunning = true;
        startWebcamBtn.style.display = 'none';
        stopWebcamBtn.style.display = 'inline-block';

        updateStatus('Webcam running - Real-time detection', 'ready');

        // Start detection loop
        lastFrameTime = performance.now();
        frameCount = 0;
        detectFrame();

    } catch (error) {
        console.error('Webcam error:', error);
        updateStatus('Webcam access denied: ' + error.message, 'error');
    }
}

function stopWebcam() {
    isWebcamRunning = false;

    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }

    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }

    webcamVideo.srcObject = null;
    startWebcamBtn.style.display = 'inline-block';
    stopWebcamBtn.style.display = 'none';
    fpsDisplay.textContent = 'FPS: --';

    // Clear canvas
    const ctx = webcamCanvas.getContext('2d');
    ctx.clearRect(0, 0, webcamCanvas.width, webcamCanvas.height);

    updateStatus('Model ready', 'ready');
}

async function detectFrame() {
    if (!isWebcamRunning || !isModelLoaded) return;

    try {
        // Preprocess video frame
        const imageData = preprocessImage(webcamVideo);

        // Run inference
        const outputs = await runInference(imageData);

        // Post-process (disable verbose logging for webcam)
        const detections = postprocessWebcam(outputs, imageData);

        // Draw results on canvas
        drawWebcamDetections(detections);

        // Calculate FPS
        frameCount++;
        const elapsed = performance.now() - lastFrameTime;
        if (elapsed >= 1000) {
            fps = Math.round(frameCount * 1000 / elapsed);
            fpsDisplay.textContent = `FPS: ${fps} | Detections: ${detections.length}`;
            frameCount = 0;
            lastFrameTime = performance.now();
        }

    } catch (error) {
        console.error('Detection error:', error);
        fpsDisplay.textContent = `Error: ${error.message}`;
    }

    // Continue loop
    if (isWebcamRunning) {
        animationFrameId = requestAnimationFrame(detectFrame);
    }
}

// Webcam-specific postprocess (no console.log for performance)
function postprocessWebcam(outputs, imageData) {
    const output = outputs[Object.keys(outputs)[0]];
    const data = output.data;
    const dims = output.dims;

    const detections = [];
    const numClasses = CONFIG.classNames.length;
    const numBoxes = dims[2];

    for (let i = 0; i < numBoxes; i++) {
        const x_center = data[0 * numBoxes + i];
        const y_center = data[1 * numBoxes + i];
        const width = data[2 * numBoxes + i];
        const height = data[3 * numBoxes + i];

        let maxScore = 0;
        let classId = 0;
        for (let c = 0; c < numClasses; c++) {
            const score = data[(4 + c) * numBoxes + i];
            if (score > maxScore) {
                maxScore = score;
                classId = c;
            }
        }

        if (maxScore < CONFIG.confThreshold) continue;

        const x1 = (x_center - width / 2 - imageData.offsetX) / imageData.scale;
        const y1 = (y_center - height / 2 - imageData.offsetY) / imageData.scale;
        const x2 = (x_center + width / 2 - imageData.offsetX) / imageData.scale;
        const y2 = (y_center + height / 2 - imageData.offsetY) / imageData.scale;

        const clippedX1 = Math.max(0, Math.min(x1, imageData.originalWidth));
        const clippedY1 = Math.max(0, Math.min(y1, imageData.originalHeight));
        const clippedX2 = Math.max(0, Math.min(x2, imageData.originalWidth));
        const clippedY2 = Math.max(0, Math.min(y2, imageData.originalHeight));

        detections.push({
            x1: clippedX1,
            y1: clippedY1,
            x2: clippedX2,
            y2: clippedY2,
            confidence: maxScore,
            classId: classId
        });
    }

    if (detections.length > 0) {
        const boxes = detections.map(d => ({ x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2 }));
        const scores = detections.map(d => d.confidence);
        const nmsIndices = nonMaxSuppression(boxes, scores, CONFIG.iouThreshold);
        return nmsIndices.map(idx => detections[idx]);
    }

    return detections;
}

function drawWebcamDetections(detections) {
    const ctx = webcamCanvas.getContext('2d');

    // Draw video frame
    ctx.drawImage(webcamVideo, 0, 0);

    // Draw detections
    for (const det of detections) {
        const color = CONFIG.colors[det.classId] || '#FFFFFF';

        // Bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);

        // Label background
        const label = `${CONFIG.classNames[det.classId]} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = 'bold 16px Arial';
        const textWidth = ctx.measureText(label).width;

        ctx.fillStyle = color;
        ctx.fillRect(det.x1, det.y1 - 24, textWidth + 10, 24);

        // Label text
        ctx.fillStyle = '#000000';
        ctx.fillText(label, det.x1 + 5, det.y1 - 6);
    }
}

// =====================================================
// Mode Switching
// =====================================================

function switchToImageMode() {
    stopWebcam();
    imageModeBtn.classList.add('active');
    webcamModeBtn.classList.remove('active');
    uploadSection.style.display = 'block';
    webcamSection.style.display = 'none';
    resultSection.classList.remove('active');
}

function switchToWebcamMode() {
    imageModeBtn.classList.remove('active');
    webcamModeBtn.classList.add('active');
    uploadSection.style.display = 'none';
    webcamSection.style.display = 'block';
    resultSection.classList.remove('active');
}

// =====================================================
// Initialization
// =====================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('YOLO Tank Detection - Initializing...');
    loadModel();

    // Mode switching
    if (imageModeBtn) {
        imageModeBtn.addEventListener('click', switchToImageMode);
    }
    if (webcamModeBtn) {
        webcamModeBtn.addEventListener('click', switchToWebcamMode);
    }

    // Webcam controls
    if (startWebcamBtn) {
        startWebcamBtn.addEventListener('click', startWebcam);
    }
    if (stopWebcamBtn) {
        stopWebcamBtn.addEventListener('click', stopWebcam);
    }
});
