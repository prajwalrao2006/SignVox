# 🤟 SignVox — AI Sign Language to Speech Translator

<div align="center">

**Real-time Sign Language Recognition & Speech Output**

*Bridging the communication gap between sign language users and the hearing world.*

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange?style=for-the-badge)
![ML](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-red?style=for-the-badge)

</div>

---

## 📖 About

**SignVox** is an AI-powered application that detects **American Sign Language (ASL)** hand signs in real-time using a webcam, recognizes the letters (A-Z), and converts them into spoken words using text-to-speech technology.

### ✨ Key Features

- 🖐️ **Real-time Hand Detection** — Tracks hand landmarks using MediaPipe
- 🔤 **Letter Recognition (A-Z)** — Machine learning model classifies hand signs
- 🔊 **Text-to-Speech** — Automatically speaks detected letters and words
- 📝 **Sentence Building** — Build full sentences with space support
- 🎨 **Beautiful Dark Mode GUI** — Premium interface with live camera feed
- 📊 **Confidence Display** — Shows prediction confidence in real-time
- 📜 **History Log** — Tracks all spoken words/sentences

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.8+** | Core programming language |
| **OpenCV** | Camera capture & image processing |
| **MediaPipe** | Hand landmark detection (21 points) |
| **Scikit-learn** | Random Forest classifier for sign recognition |
| **pyttsx3** | Offline text-to-speech engine |
| **Tkinter + Pillow** | GUI framework |
| **NumPy** | Numerical computations |

---

## 📁 Project StructureLet's create a **professional README.md** for your project — essential for college submission! 🎓

Create a new file **`README.md`** in your signvox folder and paste this:

```markdown
# 🤟 SignVox — AI Sign Language to Speech Translator

<div align="center">

**Real-time Sign Language Recognition & Speech Output**

*Bridging the communication gap between sign language users and the hearing world.*

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange?style=for-the-badge)
![ML](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-red?style=for-the-badge)

</div>

---

## 📖 About

**SignVox** is an AI-powered application that detects **American Sign Language (ASL)** hand signs in real-time using a webcam, recognizes the letters (A-Z), and converts them into spoken words using text-to-speech technology.

### ✨ Key Features

- 🖐️ **Real-time Hand Detection** — Tracks hand landmarks using MediaPipe
- 🔤 **Letter Recognition (A-Z)** — Machine learning model classifies hand signs
- 🔊 **Text-to-Speech** — Automatically speaks detected letters and words
- 📝 **Sentence Building** — Build full sentences with space support
- 🎨 **Beautiful Dark Mode GUI** — Premium interface with live camera feed
- 📊 **Confidence Display** — Shows prediction confidence in real-time
- 📜 **History Log** — Tracks all spoken words/sentences

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.8+** | Core programming language |
| **OpenCV** | Camera capture & image processing |
| **MediaPipe** | Hand landmark detection (21 points) |
| **Scikit-learn** | Random Forest classifier for sign recognition |
| **pyttsx3** | Offline text-to-speech engine |
| **Tkinter + Pillow** | GUI framework |
| **NumPy** | Numerical computations |

---

## 📁 Project Structure

```
signvox/
├── hand_detector.py      # Hand detection module (MediaPipe Tasks API)
├── collect_data.py       # Data collection script (record hand signs)
├── train_model.py        # Model training script (Random Forest)
├── signvox.py            # Basic version (OpenCV window)
├── gui.py                # Beautiful GUI version (main app)
├── requirements.txt      # Python dependencies
├── hand_landmarker.task  # MediaPipe hand landmark model
├── data/
│   ├── landmarks.npy     # Collected hand landmark data
│   └── labels.npy        # Corresponding letter labels
└── models/
    ├── signvox_model.pkl  # Trained ML model
    └── labels.pkl         # Label list
```

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/signvox.git
cd signvox
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows CMD
# OR
.\venv\Scripts\Activate.ps1  # Windows PowerShell
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Hand Landmarker Model
Download `hand_landmarker.task` from [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models) and place it in the project root.

---

## 🚀 Usage

### Step 1: Collect Training Data
```bash
python collect_data.py
```
- Show ASL signs to the camera for each letter
- Press **SPACE** to start recording, **N** for next letter
- Collects 200 samples per letter

### Step 2: Train the Model
```bash
python train_model.py
```
- Trains a Random Forest classifier
- Displays accuracy and per-letter report

### Step 3: Run SignVox
```bash
python gui.py
```

### Controls

| Key | Action |
|-----|--------|
| **Show hand sign** | Auto-detects and speaks the letter |
| **S** | Add space (for building sentences) |
| **SPACE** | Speak the full word/sentence |
| **Backspace** | Delete last letter |
| **C** | Clear all text |
| **ESC** | Quit |

---

## 🧠 How It Works

```
┌──────────┐    ┌──────────────┐    ┌────────────┐    ┌──────────┐
│  Webcam   │───▶│  MediaPipe    │───▶│  ML Model   │───▶│  Speech   │
│  Input    │    │  Hand Detect  │    │  Predict    │    │  Output   │
└──────────┘    └──────────────┘    └────────────┘    └──────────┘
                       │                    │
                 21 Landmarks         Letter (A-Z)
                 (42 features)       + Confidence %
```

1. **Capture** — Webcam captures video frames in real-time
2. **Detect** — MediaPipe extracts 21 hand landmarks (x, y coordinates)
3. **Normalize** — Landmarks are normalized relative to hand position
4. **Classify** — Random Forest model predicts the ASL letter
5. **Speak** — pyttsx3 converts the letter/word to speech

---

## 👥 Team

| Name |
|------|
| **Aadithya Kumar** 

**Institution:** Sapthagiri NPS University, Bangalore
**Course:** B.Tech — 1st Year

| Name | Role |
|------|------|
| **A L Prajwal Rao** 

**Institution:** Sapthagiri NPS University, Bangalore
**Course:** B.Tech — 1st Year

| Name | Role |
|------|------|
| **Abhay Surya S** 

**Institution:** Sapthagiri NPS University, Bangalore
**Course:** B.Tech — 1st Year

| Name | Role |
|------|------|
| **Adithya Raj** 

**Institution:** Sapthagiri NPS University, Bangalore
**Course:** B.Tech — 1st Year

| Name | Role |
|------|------|
| **Ajit Suresh Patgar** 

**Institution:** Sapthagiri NPS University, Bangalore
**Course:** B.Tech — 1st Year

---

## 📄 License

This project is developed for educational purposes as part of the Python programming course project.

---

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) by Google — Hand landmark detection
- [Scikit-learn](https://scikit-learn.org/) — Machine learning framework
- [OpenCV](https://opencv.org/) — Computer vision library

---

<div align="center">

**Made with ❤️ by Team SignVox**

*Sapthagiri NPS University, Bangalore — 2026*

</div>
```

---

Save it! Now your project has a **professional README** that looks amazing on GitHub! 🎓✅

### What's left:

| Task | Status |
|------|--------|
| ✅ Hand Detection | Done |
| ✅ Data Collection | Done |
| ✅ Model Training | Done |
| ✅ Live Detection + Speech | Done |
| ✅ Beautiful GUI | Done |
| ✅ Space for Sentences | Done |
| ✅ README.md | Done |

**Your SignVox project is COMPLETE!** 🎉🤟🔥
