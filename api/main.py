from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="LifePass ML API",
    description="Document classification API for LifePass",
    version="1.0"
)

# Load trained model
model = joblib.load("models/lifepass_classifier.joblib")


class DocumentRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "LifePass ML API is running"
    }


@app.post("/predict")
def predict(request: DocumentRequest):

    prediction = model.predict([request.text])[0]

    # Get prediction probabilities
    probabilities = model.predict_proba([request.text])[0]

    classes = model.classes_

    confidence = max(probabilities)

    return {
        "document_type": prediction,
        "confidence": round(float(confidence), 4)
    }