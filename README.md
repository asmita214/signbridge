# SignBridge 🤟

### Real-Time ASL Sign Language to Text & Speech Translator

SignBridge is an end-to-end machine learning system that detects American Sign Language hand gestures in real-time through a webcam and converts them into text and spoken audio. Built with a multi-model ensemble architecture combining Random Forest, Neural Network, and LSTM, it achieves 99.53% accuracy on static sign classification and supports both left and right-handed users.

---

## Demo

📹 [Watch Full Demo](your-drive-link-here)

---

## The Problem

Over 5 million deaf and mute people in India alone face communication barriers every day — at hospitals, schools, banks, and everyday situations. They cannot communicate with hearing people who do not know sign language. Existing solutions require a human interpreter or are limited to static dictionaries with no real-time translation capability.

SignBridge solves this by acting as a real-time bridge — a deaf person signs in front of a webcam and the system instantly converts it to readable and spoken English.

---

## How It Works
Webcam Feed
↓
MediaPipe Hand Detection (21 landmark points)
↓
Feature Extraction (63 coordinate values per frame)
↓
┌─────────────────────────────────────┐
│  LETTER MODE        │  WORD MODE    │
│  Random Forest      │  LSTM         │
│  Neural Network     │  (sequences)  │
│  Ensemble           │               │
└─────────────────────────────────────┘
↓
Stability Buffer + Majority Voting
↓
Word Builder → Sentence Builder
↓
Gemini 1.5 Flash (Grammar Correction)
↓
Text Output + Speech Output
---

## Model Architecture & Accuracy

| Model | Purpose | Architecture | Accuracy |
|---|---|---|---|
| Random Forest | Static A-Z letters | 300 trees, log2 features | 99.53% |
| Neural Network | Static A-Z letters | 4 Dense layers, BatchNorm, Dropout | 99.21% |
| LSTM | Dynamic word signs | 3 LSTM layers + 2 Dense layers | 96.67% |

**Ensemble Strategy:** Random Forest (40% weight) + Neural Network (60% weight) combined via probability averaging for static sign prediction. LSTM runs in parallel on 30-frame sequences for dynamic word detection.

---

## Features

- **Real-time letter detection** — A to Z with 99.53% accuracy
- **Dynamic word recognition** — 20 ASL words including HELLO, THANKYOU, SORRY, PLEASE, HELP, YES, NO, STOP and more using LSTM on movement sequences
- **Gemini AI grammar correction** — raw sign output converted to natural grammatically correct English
- **Both hands supported** — trained on original + mirrored landmark data for left and right hand users
- **Letter Mode / Word Mode toggle** — switch between spelling letters and recognising full words
- **Stability buffer with majority voting** — 8 out of 10 consecutive frames must agree before a letter registers, eliminating false positives
- **Keyboard controls** — SPACE to push word to sentence, BACKSPACE to delete, G for grammar correction, M to switch mode, R to restart
- **Real-time confidence display** — top 3 predictions shown with confidence bars live on screen

---

## Tech Stack

| Layer | Technology |
|---|---|
| Hand Detection | MediaPipe Hands |
| Computer Vision | OpenCV |
| ML Models | scikit-learn, TensorFlow/Keras |
| AI Grammar Fix | Google Gemini 1.5 Flash API |
| Interface | Python + OpenCV desktop window |
| Environment | Python 3.10, venv |

---

## Dataset

**Static Signs (A-Z):**
- ASL Alphabet Dataset from Kaggle — 87,000 images, 29 classes
- Self-collected personal webcam data — 200 samples per letter × 24 letters
- All data mirrored horizontally to create left-hand variants
- Total training samples after augmentation: 130,000+

**Dynamic Signs (Words):**
- Entirely self-collected — 30 sequences × 20 signs = 600 sequences
- Each sequence is 30 frames × 63 landmark values
- Collected in real environment matching deployment conditions

**Why self-collected data matters:** The Kaggle dataset was recorded in controlled studio conditions. Real webcams have different lighting, backgrounds, and angles. Adding 200 personal samples per class improved real-world accuracy from ~15% confidence to 99%+ confidence — a direct demonstration of domain adaptation.

---

## Project Structure
signbridge/
├── app.py                          ← main application
├── data_collection/
│   ├── extract_landmarks.py        ← processes Kaggle images to CSV
│   ├── record_dynamic_signs.py     ← records dynamic sign sequences
│   └── collect_my_signs.py         ← collects personal webcam data
├── training/
│   ├── train_all.py                ← trains RF + NN + LSTM locally
│   └── retrain_with_mydata.py      ← retrains with personal + Kaggle data
├── models/
│   ├── rf_model.pkl                ← trained Random Forest
│   ├── label_encoder.pkl           ← class label mapping
│   ├── le_dynamic.pkl              ← dynamic sign label mapping
│   ├── nn_weights.weights.h5       ← Neural Network weights
│   └── lstm_weights.weights.h5     ← LSTM weights
├── backend/
│   ├── main.py                     ← FastAPI + WebSocket server
│   ├── predictor.py                ← model inference
│   └── gemini_service.py           ← Gemini API integration
└── requirements.txt
---

## Setup & Run

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/signbridge.git
cd signbridge
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add Gemini API key**

Create a `.env` file in the root folder:
Get a free key at [aistudio.google.com](https://aistudio.google.com)

**5. Download model files**

Model files are not included in the repo due to size. Either:
- Train from scratch using `python training/train_all.py` after downloading the [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) from Kaggle
- Or contact me for the pre-trained model files

**6. Run**
```bash
python app.py
```

---

## Controls

| Key | Action |
|---|---|
| SPACE | Push current word to sentence |
| M | Toggle between Letter Mode and Word Mode |
| G | Fix grammar with Gemini AI |
| BACKSPACE | Delete last letter |
| R | Restart everything |
| C | Clear sentence |
| Q | Quit |

---

## Supported Dynamic Signs (Word Mode)

HELLO · THANKYOU · SORRY · PLEASE · HELP · YES · NO · STOP · MORE · WATER · GOOD · BAD · WANT · COME · GO · I · YOU · WHAT · WHERE · NAME

---

## Key Technical Decisions

**Why Random Forest over KNN?**
KNN at 98.48% had slightly higher accuracy but requires storing all 50,000+ training samples in memory and calculates distance to every sample at prediction time — too slow for real-time 30fps inference. Random Forest predicts in milliseconds from a compact trained model.

**Why self-collected data on top of Kaggle?**
The Kaggle dataset uses studio lighting and green screen. Real webcams have different conditions. Adding 200 personal samples per class with domain-matched data dramatically improved real-world performance — this is domain adaptation in practice.

**Why LSTM for dynamic signs?**
Static signs are positional — one frame is enough. Dynamic signs like HELLO involve movement across time. LSTM has memory across 30 consecutive frames (1 second of video) making it the right architecture for sequence classification.

**Why majority voting in stability buffer?**
Single-frame predictions flicker between similar signs. Requiring 8 out of 10 consecutive frames to agree eliminates false positives while keeping response time under 1 second.

---

## Future Scope

- **ISL (Indian Sign Language) support** using the IIT Bombay INCLUDE dataset (263 signs, 4,287 videos) — directly relevant for India's 5 million deaf population
- **Two-handed sign support** using MediaPipe's dual hand detection
- **Mobile deployment** using TensorFlow Lite model quantization
- **Real-time web version** using FastAPI WebSocket backend already implemented in `/backend`

---

## Author

**Asmita Gupta**
B.Tech CSE, Bhagwan Parshuram Institute of Technology (BPIT)

---

*Built with the goal of making communication accessible for the deaf and mute community.*
