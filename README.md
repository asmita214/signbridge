# SignBridge — Real-time ASL Sign Language Translator

Real-time American Sign Language detection system that converts hand signs 
to text and uses Gemini AI for grammar correction.

## Demo
[Watch Demo Video](your-youtube-link)

## Tech Stack
- MediaPipe — real-time hand landmark detection
- Random Forest + Neural Network ensemble — static sign classification  
- LSTM — dynamic gesture recognition
- Gemini API — grammar correction
- FastAPI + WebSocket — real-time backend
- React + Vite — frontend

## Model Accuracy
| Model | Task | Accuracy |
|---|---|---|
| Random Forest | Static signs A-Z | 99.53% |
| Neural Network | Static signs A-Z | 99.21% |
| LSTM | Dynamic word signs | 96.67% |

## Features
- Real-time A-Z letter detection
- 20 dynamic word signs (HELLO, SORRY, THANKYOU etc.)
- Gemini AI grammar correction
- Text to speech output
- Works for both left and right hand
- Letter mode and Word mode toggle

## How to Run
1. Clone the repo
2. Install dependencies: pip install -r requirements.txt
3. Add your Gemini API key to .env file
4. Run: python app.py

## Dataset
- ASL Alphabet Dataset (Kaggle) — 87,000 images
- Self-collected personal landmark data — 200 samples per class
- Both datasets mirrored for left-hand support