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
import traceback

app = FastAPI(title="SwasthaSetu Multi-Disease Detection API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
pneumonia_model = None
tb_model = None
malaria_session = None

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_pneumonia_model():
    """Load pneumonia model with multiple fallback strategies"""
    global pneumonia_model
    try:
        print("🔄 Loading pneumonia model...")
        
        # Strategy 1: Try loading with EfficientNet-PyTorch
        try:
            from efficientnet_pytorch import EfficientNet
            pneumonia_model = EfficientNet.from_name('efficientnet-b0')
            pneumonia_model._fc = nn.Linear(pneumonia_model._fc.in_features, 2)
            state_dict = torch.load('swasthsetu_pneumonia_model.pth', map_location='cpu')
            pneumonia_model.load_state_dict(state_dict)
            pneumonia_model.eval()
            print("✅ Pneumonia model loaded with EfficientNet-PyTorch")
            return True
        except Exception as e:
            print(f"⚠️ EfficientNet-PyTorch failed: {e}")
        
        # Strategy 2: Try loading with torchvision EfficientNet
        try:
            pneumonia_model = models.efficientnet_b0(pretrained=False)
            pneumonia_model.classifier[1] = nn.Linear(pneumonia_model.classifier[1].in_features, 2)
            state_dict = torch.load('swasthsetu_pneumonia_model.pth', map_location='cpu')
            pneumonia_model.load_state_dict(state_dict)
            pneumonia_model.eval()
            print("✅ Pneumonia model loaded with torchvision")
            return True
        except Exception as e:
            print(f"⚠️ Torchvision loading failed: {e}")
        
        # Strategy 3: Try loading as complete model
        try:
            pneumonia_model = torch.load('swasthsetu_pneumonia_model.pth', map_location='cpu')
            pneumonia_model.eval()
            print("✅ Pneumonia model loaded as complete model")
            return True
        except Exception as e:
            print(f"⚠️ Complete model loading failed: {e}")
            
        return False
        
    except Exception as e:
        print(f"❌ Failed to load pneumonia model: {e}")
        return False

def load_tb_model():
    """Load TB model with multiple fallback strategies"""
    global tb_model
    try:
        print("🔄 Loading TB model...")
        
        # Strategy 1: Try with torchvision EfficientNet
        try:
            tb_model = models.efficientnet_b0(pretrained=False)
            tb_model.classifier[1] = nn.Linear(tb_model.classifier[1].in_features, 2)
            state_dict = torch.load('best_tb_model.pth', map_location='cpu')
            tb_model.load_state_dict(state_dict)
            tb_model.eval()
            print("✅ TB model loaded with torchvision")
            return True
        except Exception as e:
            print(f"⚠️ Torchvision TB loading failed: {e}")
        
        # Strategy 2: Try loading as complete model
        try:
            tb_model = torch.load('best_tb_model.pth', map_location='cpu')
            tb_model.eval()
            print("✅ TB model loaded as complete model")
            return True
        except Exception as e:
            print(f"⚠️ Complete TB model loading failed: {e}")
            
        return False
        
    except Exception as e:
        print(f"❌ Failed to load TB model: {e}")
        return False

def load_malaria_model():
    """Load malaria ONNX model"""
    global malaria_session
    try:
        print("🔄 Loading malaria ONNX model...")
        malaria_session = ort.InferenceSession('malaria_resnet18.onnx')
        print("✅ Malaria model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to load malaria model: {e}")
        return False

@app.on_event("startup")
async def load_models():
    """Load all models with comprehensive error handling"""
    print("🚀 Starting SwasthaSetu model loading...")
    
    # Load each model independently
    pneumonia_loaded = load_pneumonia_model()
    tb_loaded = load_tb_model()
    malaria_loaded = load_malaria_model()
    
    # Report loading status
    print(f"📊 Model Loading Summary:")
    print(f"   Pneumonia: {'✅' if pneumonia_loaded else '❌'}")
    print(f"   TB: {'✅' if tb_loaded else '❌'}")
    print(f"   Malaria: {'✅' if malaria_loaded else '❌'}")
    
    if not any([pneumonia_loaded, tb_loaded, malaria_loaded]):
        print("⚠️ Warning: No models loaded successfully")
    else:
        print("🎉 SwasthaSetu AI ready for predictions!")

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
        print("🔄 Starting prediction...")
        
        # Check if at least one model is loaded
        if pneumonia_model is None and tb_model is None and malaria_session is None:
            raise HTTPException(
                status_code=503, 
                detail="No AI models are currently loaded. Please try again later."
            )
        
        # Read and preprocess image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        input_tensor = transform(pil_image).unsqueeze(0)
        
        predictions = {}
        
        with torch.no_grad():
            # Pneumonia prediction (if model loaded)
            if pneumonia_model is not None:
                try:
                    pneumonia_output = torch.softmax(pneumonia_model(input_tensor), dim=1)
                    pneumonia_prob = float(pneumonia_output[0][1])
                    predictions['Pneumonia'] = pneumonia_prob * 100
                    print(f"✅ Pneumonia prediction: {pneumonia_prob:.3f}")
                except Exception as e:
                    print(f"⚠️ Pneumonia prediction failed: {e}")
                    predictions['Pneumonia'] = 0.0
            else:
                predictions['Pneumonia'] = 0.0
            
            # TB prediction (if model loaded)
            if tb_model is not None:
                try:
                    tb_output = torch.softmax(tb_model(input_tensor), dim=1)
                    tb_prob = float(tb_output[0][1])
                    predictions['Tuberculosis'] = tb_prob * 100
                    print(f"✅ TB prediction: {tb_prob:.3f}")
                except Exception as e:
                    print(f"⚠️ TB prediction failed: {e}")
                    predictions['Tuberculosis'] = 0.0
            else:
                predictions['Tuberculosis'] = 0.0
            
            # Malaria prediction (if model loaded)
            if malaria_session is not None:
                try:
                    input_array = input_tensor.numpy()
                    malaria_output = malaria_session.run(None, {'input': input_array})[0]
                    malaria_prob = float(malaria_output[0][0])
                    predictions['Malaria'] = malaria_prob * 100
                    print(f"✅ Malaria prediction: {malaria_prob:.3f}")
                except Exception as e:
                    print(f"⚠️ Malaria prediction failed: {e}")
                    predictions['Malaria'] = 0.0
            else:
                predictions['Malaria'] = 0.0
            
            # Calculate normal probability
            max_disease_prob = max(
                predictions['Pneumonia'] / 100,
                predictions['Tuberculosis'] / 100,
                predictions['Malaria'] / 100
            )
            predictions['Normal'] = max(0, (1.0 - max_disease_prob) * 100)
        
        # Determine final diagnosis
        max_disease = max(predictions, key=predictions.get)
        confidence = predictions[max_disease]
        
        if max_disease == 'Normal' or confidence < 50:
            diagnosis = 'NORMAL'
            has_disease = False
        else:
            diagnosis = f'{max_disease.upper()} DETECTED'
            has_disease = True
        
        # Determine confidence level
        confidence_level = 'High' if confidence > 70 else 'Medium' if confidence > 50 else 'Low'
        
        print(f"🎯 Final diagnosis: {diagnosis} ({confidence_level})")
        
        return {
            'success': True,
            'diagnosis': diagnosis,
            'confidence': confidence_level,
            'hasDisease': has_disease,
            'predictions': predictions,
            'confidenceScore': confidence / 100,
            'modelSource': 'SwasthaSetu Trained Models',
            'processingTime': 'Real-time'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
