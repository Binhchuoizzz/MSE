from fastapi import FastAPI, HTTPException, status
from datetime import datetime
from models import IrisFeatures, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse
from model_manager import get_model_manager
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MLFlow Tutorial 04",
    description="This is a simple API that uses MLFlow to track the model",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    logger.info("Loading model...")
    model_manager = get_model_manager()
    if not model_manager.model:
        raise Exception("Failed to load model")
    logger.info("Model loaded successfully")
    logger.info(f"Model info: {model_manager.get_model_info()}")

@app.get("/")
async def read_root():
    return {
        "message": "MLFlow Tutorial 04",
        "timestamp": datetime.now().isoformat(),
        "status": "running"
        }

@app.get("/health")
async def health_check():
    return {
        "timestamp": datetime.now().isoformat(),
        "status": "healthy"
        }

@app.get("/model/info")
async def model_info():
    model_manager = get_model_manager()
    if model_manager.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    return {
        "model_info": model_manager.get_model_info(),
        "mlflow_uri": model_manager.mlflow_uri,
        "status": "success"
        }

@app.get("/info")
async def info():
    return {
        "message": "MLFlow Tutorial 04 - ML Model API Demo",
        "timestamp": datetime.now().isoformat(),
        "status": "running"
        }

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures):
    # Make a prediction
    manager = get_model_manager()
    if manager.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    features_array = np.array([features.sepal_length,
                               features.sepal_width,
                               features.petal_length,
                               features.petal_width]).reshape(1, -1)

    prediction = manager.predict(features_array)
    return PredictionResponse(
        prediction=prediction['prediction'],
        probability=prediction['probability'],
        all_posibilities=prediction['all_posibilities'],
        model_version=prediction['model_version'],
        timestamp=datetime.now().isoformat()
    )

@app.post('/model/reload')
async def reload_model():
    manager = get_model_manager()
    success = manager.load_model()
    if success:
        return {
            'status': 'success',
            'message': 'Model reloaded successfully',
            'model_info': manager.get_model_info()
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         app, 
#         host="0.0.0.0", 
#         port=8000,
#     )