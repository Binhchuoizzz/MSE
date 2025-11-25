from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime

class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, le=10,
                                description="The length of the sepal")
    sepal_width: float = Field(..., ge=0, le=10,
                               description="The width of the sepal")
    petal_length: float = Field(..., ge=0, le=10,
                                description="The length of the petal")
    petal_width: float = Field(..., ge=0, le=10,
                               description="The width of the petal")

class Config:
    json_schema_extra = {
        "examples": [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        ]
    }
    @validator('*')
    def check_positive(cls, v):
        if v < 0:
            raise ValueError("Value must be positive")
        return v

class PredictionResponse(BaseModel):
    prediction: str
    probability: float = Field(..., ge=0, le=1.0)
    all_posibilities: dict
    timestamp: datetime = Field(..., description="The timestamp of the prediction")
    model_version: Optional[str] = None

class BatchPredictionRequest(BaseModel):
    features: List[IrisFeatures] = Field(...,
                                         min_items=1,
                                         max_items=100)

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    batch_size: int
    total_time: float