from fastapi import FastAPI, File, UploadFile, Response
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import cv2
import base64
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager

model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    model["predict"]  = YOLO("/home/conte/code/Soumiabenhamou90/Brain-Tumor-Detection/brainTumorDetection/ml_logic/models/YOLO100epoch/best100epoch.pt")
    yield
    # Clean up the ML models and release the resources
    model.clear()

app = FastAPI(lifespan=lifespan)


# Configurer les origines autorisées (dans ce cas, localhost:3000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Autoriser uniquement Next.js en développement
    allow_credentials=True,
    allow_methods=["*"],  # Autoriser toutes les méthodes HTTP
    allow_headers=["*"],  # Autoriser tous les en-têtes
)

# # Charger le modèle YOLOv8
# model = YOLO("/home/conte/code/Soumiabenhamou90/Brain-Tumor-Detection/brainTumorDetection/ml_logic/models/YOLO100epoch/best100epoch.pt")
# imagePath = "/home/conte/code/Soumiabenhamou90/Brain-Tumor-Detection/brainTumorDetection/API/Tr-pi_0015.jpg"

class PredictionResult(BaseModel):
    label: str
    confidence: float
    bbox: list

# @app.get("/predict", response_model=dict) #tester directement avec une image dans le project

@app.post("/predict", response_model=dict)
async def predict(file: UploadFile = File(...)):
    # Charger l'image directement depuis le chemin
    # image = Image.open(imagePath).convert("RGB")
    # image_np = np.array(image)

    #Charger l'image depuis un formulaire
    # Lire l'image depuis le fichier envoyé
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image_np = np.array(image)

    # Faire la prédiction
    results = model["predict"](image_np)

    # Extraire les résultats de prédiction
    predictions = []
    classIndex = int(results[0].boxes.cls[0].item())
    classPredict = results[0].names[classIndex]

    for result in results:
        predictions.append({
            "class": classPredict,
            "confidence": float(result.boxes.conf[0]) if len(result.boxes.conf) > 0 else 0.0,
            "bbox": result.boxes.xyxy[0].tolist() if len(result.boxes.xyxy) > 0 else []
        })

    # Annoter l'image
    annotated_image = np.array(image)
    if len(results[0].boxes.xyxy) > 0:
        x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[0])
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_image, f"{classPredict}: {results[0].boxes.conf[0]:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Encoder l'image annotée en base64 pour l'affichage
    _, buffer = cv2.imencode(".jpg", annotated_image)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "predictions": predictions,
        "annotated_image": img_base64
    }

# Lancer l'API avec uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
