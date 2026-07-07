let cameraActive = false;
let modelLoaded = false;
let activeHandLandmarks = null;
let customGesturesDb = [];
let lastPrediction = null;
let stableFramesCount = 0;
const STABILITY_THRESHOLD = 8; // Consec. frames to confirm input
let sentenceText = "";
let autoSpeak = true;
let autoAppend = true;
let fpsCount = 0;
let lastFpsUpdate = Date.now();

// Elements
const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output-canvas');
const canvasCtx = canvasElement.getContext('2d');
const cameraPlaceholder = document.getElementById('camera-placeholder');
const toggleCameraBtn = document.getElementById('toggle-camera-btn');
const systemStatus = document.getElementById('system-status');
const fpsCounter = document.getElementById('fps-counter');
const handCountDisplay = document.getElementById('hand-count');
const currentLetterText = document.getElementById('current-letter');
const confidenceFill = document.getElementById('confidence-fill');
const confidencePercentage = document.getElementById('confidence-percentage');
const accumulatedTextBox = document.getElementById('accumulated-text');
const clearTextBtn = document.getElementById('clear-text-btn');
const speakSentenceBtn = document.getElementById('speak-sentence-btn');
const voiceSelect = document.getElementById('voice-select');
const rateRange = document.getElementById('voice-rate');
const pitchRange = document.getElementById('voice-pitch');
const rateVal = document.getElementById('rate-val');
const pitchVal = document.getElementById('pitch-val');
const autoSpeakToggle = document.getElementById('auto-speak-toggle');
const autoAppendToggle = document.getElementById('auto-append-toggle');
const addLetterBtn = document.getElementById('add-letter-btn');
const addSpaceBtn = document.getElementById('add-space-btn');
const backspaceBtn = document.getElementById('backspace-btn');
const onscreenKeyboard = document.getElementById('onscreen-keyboard');

const gestureLabelInput = document.getElementById('gesture-label');
const recordSampleBtn = document.getElementById('record-sample-btn');
const resetSignBtn = document.getElementById('reset-sign-btn');
const resetAllBtn = document.getElementById('reset-all-btn');
const downloadDataBtn = document.getElementById('download-data-btn');
const gesturesList = document.getElementById('gestures-list');
const sampleCountSpan = document.getElementById('sample-count');
const gestureCountBadge = document.getElementById('gesture-count');

// Voice Synthesizer
const synth = window.speechSynthesis;
let voices = [];

function populateVoiceList() {
    voices = synth.getVoices();
    voiceSelect.innerHTML = '';
    voices.forEach((voice, i) => {
        const option = document.createElement('option');
        option.textContent = `${voice.name} (${voice.lang})`;
        option.value = i;
        if (voice.default) option.selected = true;
        voiceSelect.appendChild(option);
    });
}
populateVoiceList();
if (speechSynthesis.onvoiceschanged !== undefined) {
    speechSynthesis.onvoiceschanged = populateVoiceList;
}

rateRange.addEventListener('input', () => { rateVal.textContent = rateRange.value; });
pitchRange.addEventListener('input', () => { pitchVal.textContent = pitchRange.value; });
autoSpeakToggle.addEventListener('change', () => { autoSpeak = autoSpeakToggle.checked; });

function speakText(text) {
    if (!text || synth.speaking) return;
    const utter = new SpeechSynthesisUtterance(text);
    if (voices[voiceSelect.value]) utter.voice = voices[voiceSelect.value];
    utter.rate = parseFloat(rateRange.value);
    utter.pitch = parseFloat(pitchRange.value);
    synth.speak(utter);
}

// Local Storage + Pretrained data loader
function loadCustomGestures() {
    // 1. First, load your exported python dataset (if the file exists)
    if (typeof PRETRAINED_DATA !== 'undefined') {
        customGesturesDb = [...PRETRAINED_DATA];
        console.log(`Loaded ${PRETRAINED_DATA.length} pre-trained samples from Python!`);
    } else {
        customGesturesDb = [];
        console.log("No pre-trained Python data found. Operating in custom training mode.");
    }

    // 2. Load any additional custom signs recorded directly in the browser
    const saved = localStorage.getItem('signvox_custom_gestures');
    if (saved) {
        const browserSamples = JSON.parse(saved);
        customGesturesDb = [...customGesturesDb, ...browserSamples];
        console.log(`Loaded ${browserSamples.length} browser-recorded samples!`);
    }
    
    updateGesturesDatabaseUI();
}

function saveCustomGestures() {
    localStorage.setItem('signvox_custom_gestures', JSON.stringify(customGesturesDb));
    updateGesturesDatabaseUI();
}

function updateGesturesDatabaseUI() {
    gesturesList.innerHTML = '';
    gestureCountBadge.textContent = `${customGesturesDb.length} samples registered`;
    if (customGesturesDb.length === 0) {
        gesturesList.innerHTML = '<p class="empty-list-text">No custom signs recorded yet.</p>';
        return;
    }
    const counts = {};
    customGesturesDb.forEach(item => { counts[item.label] = (counts[item.label] || 0) + 1; });
    Object.keys(counts).forEach(label => {
        const itemEl = document.createElement('div');
        itemEl.className = 'db-item';
        itemEl.innerHTML = `
            <span class="db-item-name">${label}</span>
            <span class="db-item-count">${counts[label]} samples</span>
            <i class="fa-solid fa-circle-xmark db-item-delete" title="Delete"></i>
        `;
        itemEl.querySelector('.db-item-delete').addEventListener('click', () => {
            customGesturesDb = customGesturesDb.filter(item => item.label !== label);
            saveCustomGestures();
        });
        gesturesList.appendChild(itemEl);
    });
}

// MediaPipe Setup
const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});
hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7
});

function resizeCanvas() {
    canvasElement.width = canvasElement.parentElement.clientWidth;
    canvasElement.height = canvasElement.parentElement.clientHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// Math: Landmark Normalization (2D compatible: X and Y only)
function normalizeLandmarks(landmarks) {
    const wrist = landmarks[0];
    let translated = [];
    for (let i = 0; i < landmarks.length; i++) {
        translated.push({
            x: landmarks[i].x - wrist.x,
            y: landmarks[i].y - wrist.y
        });
    }
    const scale = Math.sqrt(translated[9].x**2 + translated[9].y**2);
    const scaleFactor = scale > 0 ? scale : 1.0;
    
    let normalized = [];
    for (let i = 0; i < translated.length; i++) {
        normalized.push(translated[i].x / scaleFactor);
        normalized.push(translated[i].y / scaleFactor);
    }
    return normalized;
}

// KNN Classifier (Calculates 2D Euclidean distance to templates)
function classifyKNN(normalizedFeatures) {
    if (customGesturesDb.length === 0) return null;
    let distances = [];
    customGesturesDb.forEach(sample => {
        let diffSum = 0;
        // Loop through 42 coordinates (21 landmarks * 2 dimensions)
        for (let i = 0; i < 42; i++) {
            diffSum += (normalizedFeatures[i] - sample.landmarks[i])**2;
        }
        distances.push({ label: sample.label, distance: Math.sqrt(diffSum) });
    });
    distances.sort((a, b) => a.distance - b.distance);
    
    if (distances[0].distance > 0.75) return null; // Distance threshold check

    // Vote weighted KNN
    const k = Math.min(3, distances.length);
    const votes = {};
    for(let i=0; i<k; i++) {
        const n = distances[i];
        votes[n.label] = (votes[n.label] || 0) + (1.0 / (n.distance + 0.001));
    }
    let bestLabel = null; let maxVotes = -1;
    Object.keys(votes).forEach(lbl => {
        if(votes[lbl] > maxVotes) { maxVotes = votes[lbl]; bestLabel = lbl; }
    });
    return { label: bestLabel, confidence: Math.max(0, 1 - (distances[0].distance / 0.8)) };
}

function predictGesture(landmarks) {
    const norm = normalizeLandmarks(landmarks);
    // ONLY use your custom trained gestures! Pre-defined geometric rules are completely disabled.
    return classifyKNN(norm); 
}

// Process camera feed results
function onResults(results) {
    fpsCount++;
    const now = Date.now();
    if (now - lastFpsUpdate >= 1000) {
        fpsCounter.textContent = fpsCount;
        fpsCount = 0; lastFpsUpdate = now;
    }
    
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const landmarks = results.multiHandLandmarks[0];
        activeHandLandmarks = landmarks;
        handCountDisplay.textContent = "1 Hand";
        recordSampleBtn.removeAttribute('disabled');

        drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: '#18e3eb', lineWidth: 3 });
        drawLandmarks(canvasCtx, landmarks, { color: '#a855f7', lineWidth: 1, radius: 5 });

        const prediction = predictGesture(landmarks);
        if (prediction) {
            currentLetterText.textContent = prediction.label;
            const pct = Math.round(prediction.confidence * 100);
            confidencePercentage.textContent = `${pct}%`;
            confidenceFill.style.width = `${pct}%`;

            if (prediction.label === lastPrediction) {
                stableFramesCount++;
                if (stableFramesCount === STABILITY_THRESHOLD) handleConfirmedLetter(prediction.label);
            } else {
                lastPrediction = prediction.label;
                stableFramesCount = 0;
            }
        } else {
            currentLetterText.textContent = "Unknown";
            confidencePercentage.textContent = "0%";
            confidenceFill.style.width = "0%";
            stableFramesCount = 0;
        }
    } else {
        activeHandLandmarks = null;
        handCountDisplay.textContent = "No hands";
        recordSampleBtn.setAttribute('disabled', 'true');
        currentLetterText.textContent = "—";
        confidencePercentage.textContent = "0%";
        confidenceFill.style.width = "0%";
        
        if (lastPrediction !== null) {
            stableFramesCount++;
            if (stableFramesCount > 30) {
                lastPrediction = null; stableFramesCount = 0;
                addSpaceToWord();
            }
        }
    }
    canvasCtx.restore();
}
hands.onResults(onResults);

// Single source of truth for the sentence: `sentenceText`.
// Every add/space/backspace mutates this string, then renderSentence() reflects it in the DOM.
function renderSentence() {
    if (sentenceText.length === 0) {
        accumulatedTextBox.innerHTML = '<p id="accumulated-text" class="placeholder-text">Translated words will appear here...</p>';
    } else {
        accumulatedTextBox.innerHTML = '<p id="accumulated-text"></p>';
        document.getElementById('accumulated-text').textContent = sentenceText;
    }
}

// Appends a letter/word to the sentence. Works the same whether it came from
// the camera (auto-detect) or a manual tap on the on-screen keyboard —
// no more silently dropping repeated letters (e.g. the double L in "HELLO").
function addLetterToSentence(letter, { speak = true } = {}) {
    if (!letter) return;
    sentenceText += letter;
    renderSentence();
    if (speak && autoSpeak) speakText(letter);
}

function addSpaceToSentence({ speak = true } = {}) {
    if (sentenceText.length === 0 || sentenceText.endsWith(' ')) return;
    sentenceText += ' ';
    renderSentence();
    if (speak && autoSpeak) {
        const words = sentenceText.trim().split(' ');
        const lastWord = words[words.length - 1];
        if (lastWord) speakText(lastWord);
    }
}

function backspaceSentence() {
    if (sentenceText.length === 0) return;
    sentenceText = sentenceText.slice(0, -1);
    renderSentence();
}

function clearWordAccumulator() {
    sentenceText = "";
    lastPrediction = null;
    renderSentence();
}
clearTextBtn.addEventListener('click', clearWordAccumulator);
speakSentenceBtn.addEventListener('click', () => {
    if (sentenceText.length > 0) speakText(sentenceText);
});

// Called automatically once the camera holds a stable detection for
// STABILITY_THRESHOLD frames. Only runs when "Auto-add from camera" is on.
function handleConfirmedLetter(letter) {
    if (!autoAppend) return;
    addLetterToSentence(letter);
}

function addSpaceToWord() {
    addSpaceToSentence();
}

// --- Manual controls ---
addLetterBtn.addEventListener('click', () => {
    const letter = currentLetterText.textContent;
    if (letter && letter !== '—' && letter !== 'Unknown') addLetterToSentence(letter);
});
addSpaceBtn.addEventListener('click', () => addSpaceToSentence());
backspaceBtn.addEventListener('click', () => backspaceSentence());
autoAppendToggle.addEventListener('change', () => { autoAppend = autoAppendToggle.checked; });

// Build the on-screen A-Z keyboard once
'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('').forEach(letter => {
    const key = document.createElement('button');
    key.type = 'button';
    key.className = 'key-btn';
    key.textContent = letter;
    key.addEventListener('click', () => addLetterToSentence(letter));
    onscreenKeyboard.appendChild(key);
});
const spaceKey = document.createElement('button');
spaceKey.type = 'button';
spaceKey.className = 'key-btn key-space';
spaceKey.textContent = 'Space';
spaceKey.addEventListener('click', () => addSpaceToSentence());
onscreenKeyboard.appendChild(spaceKey);

// Record samples
recordSampleBtn.addEventListener('click', () => {
    const label = gestureLabelInput.value.trim().toUpperCase();
    if (!label) { alert("Enter label!"); return; }
    if (!activeHandLandmarks) return;
    
    customGesturesDb.push({ label: label, landmarks: normalizeLandmarks(activeHandLandmarks) });
    saveCustomGestures();
    
    const count = customGesturesDb.filter(i => i.label === label).length;
    sampleCountSpan.textContent = count;
});

gestureLabelInput.addEventListener('input', () => {
    const label = gestureLabelInput.value.trim().toUpperCase();
    sampleCountSpan.textContent = customGesturesDb.filter(i => i.label === label).length;
});

resetSignBtn.addEventListener('click', () => {
    const label = gestureLabelInput.value.trim().toUpperCase();
    if (confirm(`Delete samples for "${label}"?`)) {
        customGesturesDb = customGesturesDb.filter(i => i.label !== label);
        saveCustomGestures();
        sampleCountSpan.textContent = 0;
    }
});

resetAllBtn.addEventListener('click', () => {
    if (confirm("Delete all data?")) {
        customGesturesDb = []; saveCustomGestures();
        sampleCountSpan.textContent = 0; gestureLabelInput.value = "";
    }
});

downloadDataBtn.addEventListener('click', () => {
    if (customGesturesDb.length === 0) return;
    const str = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(customGesturesDb));
    const dl = document.createElement('a');
    dl.setAttribute("href", str); dl.setAttribute("download", "custom_gestures.json");
    document.body.appendChild(dl); dl.click(); dl.remove();
});

// Camera control
async function startCamera() {
    cameraPlaceholder.querySelector('p').textContent = "Accessing camera...";
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: "user" },
            audio: false
        });
        videoElement.srcObject = stream;
        videoElement.play();
        cameraActive = true;
        cameraPlaceholder.style.display = 'none';
        toggleCameraBtn.textContent = 'Stop Camera';
        toggleCameraBtn.className = 'btn btn-danger btn-sm';
        
        async function loop() {
            if (cameraActive) {
                if (videoElement.readyState === 4) await hands.send({ image: videoElement });
                requestAnimationFrame(loop);
            }
        }
        requestAnimationFrame(loop);
    } catch (e) {
        cameraPlaceholder.querySelector('p').textContent = "Camera Error: " + e.message;
    }
}

function stopCamera() {
    if (videoElement.srcObject) videoElement.srcObject.getTracks().forEach(t => t.stop());
    cameraActive = false;
    cameraPlaceholder.style.display = 'flex';
    cameraPlaceholder.querySelector('p').textContent = "Camera stopped.";
    toggleCameraBtn.textContent = 'Start Camera';
    toggleCameraBtn.className = 'btn btn-secondary btn-sm';
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
}

toggleCameraBtn.addEventListener('click', () => { cameraActive ? stopCamera() : startCamera(); });

window.addEventListener('load', () => {
    loadCustomGestures();
    setTimeout(() => {
        modelLoaded = true;
        systemStatus.innerHTML = '<span class="status-dot online"></span><span class="status-text">AI Ready</span>';
        toggleCameraBtn.removeAttribute('disabled');
        cameraPlaceholder.querySelector('p').textContent = "Click Start Camera to begin.";
    }, 1000);
});