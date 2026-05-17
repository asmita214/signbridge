import cv2
import numpy as np
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

from backend.predictor import extract_landmarks, predict_static, predict_dynamic
from backend.gemini_service import correct_grammar

app = FastAPI(title="SignBridge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SEQUENCE_LENGTH = 30
STATIC_CONFIDENCE_THRESHOLD = 0.60
DYNAMIC_CONFIDENCE_THRESHOLD = 0.80
DYNAMIC_COOLDOWN_FRAMES = 60

@app.get("/")
def root():
    return {"message": "SignBridge API is running"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    sequence_buffer = []
    dynamic_cooldown = 0

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            frame_base64 = payload["frame"]

            img_bytes = base64.b64decode(frame_base64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            landmarks = extract_landmarks(frame_rgb)

            if landmarks is not None:
                sequence_buffer.append(landmarks.tolist())
                if len(sequence_buffer) > SEQUENCE_LENGTH:
                    sequence_buffer.pop(0)

                # Static prediction
                static_label, static_conf = predict_static(landmarks)
                if static_conf < STATIC_CONFIDENCE_THRESHOLD:
                    static_label = None

                # Dynamic prediction with cooldown
                dynamic_label = None
                dynamic_conf = 0.0

                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    if dynamic_cooldown > 0:
                        dynamic_cooldown -= 1
                    else:
                        dynamic_label, dynamic_conf = predict_dynamic(sequence_buffer)
                        if dynamic_conf < DYNAMIC_CONFIDENCE_THRESHOLD:
                            dynamic_label = None
                        elif dynamic_label is not None:
                            dynamic_cooldown = DYNAMIC_COOLDOWN_FRAMES
                            sequence_buffer = []

                await websocket.send_text(json.dumps({
                    "type": "prediction",
                    "static_label": static_label,
                    "static_confidence": static_conf,
                    "dynamic_label": dynamic_label,
                    "dynamic_confidence": dynamic_conf,
                    "hand_detected": True
                }))
            else:
                sequence_buffer = []
                dynamic_cooldown = 0
                await websocket.send_text(json.dumps({
                    "type": "prediction",
                    "static_label": None,
                    "static_confidence": 0.0,
                    "dynamic_label": None,
                    "dynamic_confidence": 0.0,
                    "hand_detected": False
                }))

    except WebSocketDisconnect:
        print("Client disconnected")

class GrammarRequest(BaseModel):
    text: str

@app.post("/correct-grammar")
async def grammar_endpoint(request: GrammarRequest):
    corrected = correct_grammar(request.text)
    return {"original": request.text, "corrected": corrected}