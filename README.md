# Real-Time American Sign Language Recognition System

A production-style **AI system that detects American Sign Language (ASL) hand gestures in real time using computer vision and deep learning**.

The system converts hand signs into letters, builds words, and optionally speaks the generated sentence using text-to-speech.

This project demonstrates an **end-to-end machine learning pipeline**, including dataset preparation, model training, real-time inference, backend APIs, and a web interface.

---

# Features

* Real-time ASL letter recognition (A-Z)
* Hand landmark detection using MediaPipe
* Deep learning classification model
* Sentence builder from predicted letters
* Optional text-to-speech output
* FastAPI backend for inference
* Browser-based webcam interface
* GPU-accelerated training

---

# Model Performance

Model trained on the **ASL Alphabet Dataset**.

| Metric           | Value      |
| ---------------- | ---------- |
| Test Accuracy    | **99.40%** |
| Classes          | 26 (A-Z)   |
| Training Samples | ~60,000    |
| Test Samples     | ~12,000    |

Average Scores:

* Precision: **0.9939**
* Recall: **0.9939**
* F1 Score: **0.9939**

---

# System Architecture

```
Webcam
   down
OpenCV Frame Capture
   down
MediaPipe Hand Detection
   down
21 Hand Landmarks Extraction
   down
Neural Network Classifier
   down
Predicted Letter (A-Z)
   down
Sentence Builder
   down
Text-to-Speech
   down
FastAPI Backend + Web UI
```

---

# Project Structure

```
asl-sign-language-ai/
|
|-- backend/
|   |-- app.py
|
|-- data_collection/
|   |-- collect_data.py
|
|-- inference/
|   |-- realtime_prediction.py
|
|-- training/
|   |-- train_model.py
|
|-- utils/
|   |-- feature_extraction.py
|
|-- frontend/
|   |-- index.html
|
|-- dataset/
|   |-- .gitkeep
|
|-- models/
|   |-- .gitkeep
|
|-- requirements.txt
|-- README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/asl-sign-language-ai.git
cd asl-sign-language-ai
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Training the Model

If you want to retrain the model:

```bash
python training/train_model.py --epochs 80 --batch_size 32
```

This will generate:

```
models/
   asl_model.h5
   labels.json
```

---

# Real-Time Webcam Inference

Run:

```bash
python inference/realtime_prediction.py
```

This opens the webcam and performs real-time sign recognition.

---

# Run the Web Application

Start the backend:

```bash
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

Open in browser:

```text
http://127.0.0.1:8000
```

---

# Demo

### Web Interface

![Web Interface](docs/screenshots/web-interface.png)

### Real-Time CLI

![Realtime CLI](docs/screenshots/realtime-cli.png)

---

# Dataset

Training dataset: **ASL Alphabet Dataset**

Approximately **87,000 images across 29 classes**.

Classes used for training:

```text
A - Z
```

Other dataset classes (`del`, `space`, `nothing`) are excluded during training.

---

# Tech Stack

* Python
* TensorFlow
* OpenCV
* MediaPipe
* FastAPI
* JavaScript / HTML
* pyttsx3 (Text-to-Speech)

---

# Future Improvements

* Word-level sign recognition
* Sentence-level translation
* Mobile deployment
* Transformer-based gesture modeling
* Multi-hand gesture recognition

---

# License

This project is open-source and available under the MIT License.

---

# Author

**Mehul Gupta**

Machine Learning and Software Engineering

If you find this project useful, consider giving it a star on GitHub.
