# medical_ai_server.py - Complete Multi-Disease Detection API
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import onnxruntime as ort
from PIL import Image
import io
import numpy as np
import json

app = FastAPI(title="SwasthaSetu Multi-Disease Detection API")

# CORS for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
pneumonia_model = None
tb_model = None
malaria_session = None

# Image preprocessing transforms (match your training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.on_event("startup")
async def load_models():
    global pneumonia_model, tb_model, malaria_session
    
    try:
        # Load PyTorch models
        print("Loading PyTorch models...")
        
        # Pneumonia model (EfficientNet-B0)
        from efficientnet_pytorch import EfficientNet
        pneumonia_model = EfficientNet.from_pretrained('efficientnet-b0')
        pneumonia_model._fc = nn.Linear(pneumonia_model._fc.in_features, 2)
        pneumonia_model.load_state_dict(torch.load('swasthsetu_pneumonia_model.pth', map_location='cpu'))
        pneumonia_model.eval()
        
        # TB model (EfficientNet-B0)
        tb_model = models.efficientnet_b0(pretrained=False)
        tb_model.classifier[1] = nn.Linear(tb_model.classifier[1].in_features, 2)
        tb_model.load_state_dict(torch.load('best_tb_model.pth', map_location='cpu'))
        tb_model.eval()
        
        # Malaria model (ONNX)
        print("Loading ONNX model...")
        malaria_session = ort.InferenceSession('malaria_resnet18.onnx')
        
        print("✅ All models loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")

@app.get("/")
async def root():
    return {
        "message": "SwasthaSetu Multi-Disease Detection API",
        "status": "online",
        "models": ["pneumonia", "tuberculosis", "malaria"],
        "version": "1.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "pneumonia": pneumonia_model is not None,
            "tb": tb_model is not None,
            "malaria": malaria_session is not None
        }
    }

@app.post("/predict")
async def predict_disease(image: UploadFile = File(...)):
    try:
        # Read and preprocess image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        input_tensor = transform(pil_image).unsqueeze(0)
        
        predictions = {}
        
        with torch.no_grad():
            # Pneumonia prediction
            pneumonia_output = torch.softmax(pneumonia_model(input_tensor), dim=1)
            pneumonia_prob = float(pneumonia_output[0][1])  # Pneumonia class
            predictions['Pneumonia'] = pneumonia_prob * 100
            
            # TB prediction
            tb_output = torch.softmax(tb_model(input_tensor), dim=1)
            tb_prob = float(tb_output[0][1])  # TB class
            predictions['Tuberculosis'] = tb_prob * 100
            
            # Malaria prediction (ONNX)
            input_array = input_tensor.numpy()
            malaria_output = malaria_session.run(None, {'input': input_array})[0]
            malaria_prob = float(malaria_output[0][0])  # Malaria probability
            predictions['Malaria'] = malaria_prob * 100
            
            # Normal probability
            max_disease_prob = max(pneumonia_prob, tb_prob, malaria_prob)
            predictions['Normal'] = (1.0 - max_disease_prob) * 100
        
        # Determine final diagnosis
        max_disease = max(predictions, key=predictions.get)
        confidence = predictions[max_disease]
        
        if max_disease == 'Normal':
            diagnosis = 'NORMAL'
            has_disease = False
        else:
            diagnosis = f'{max_disease.upper()} DETECTED'
            has_disease = True
        
        return {
            'success': True,
            'diagnosis': diagnosis,
            'confidence': 'High' if confidence > 70 else 'Medium' if confidence > 50 else 'Low',
            'hasDisease': has_disease,
            'predictions': predictions,
            'confidenceScore': confidence / 100,
            'modelSource': 'SwasthaSetu Trained Models',
            'processingTime': 'Real-time'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
