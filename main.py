from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import csv
from datetime import datetime

# Initialize the API
app = FastAPI(title="Group 22 AI Core - MLOps Backend")

# Allow the Next.js frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to localhost:3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define file paths
MODEL_PATH = "models/decision_tree_v1.joblib"
ENCODER_PATH = "models/brand_encoder.joblib"
FEEDBACK_LOG_PATH = "data/processed/live_feedback.csv"

# Load the trained model and encoder
try:
    dt_model = joblib.load(MODEL_PATH)
    brand_encoder = joblib.load(ENCODER_PATH)
    print("✅ Model and Encoder loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading models: {e}")

# --- DATA MODELS (Pydantic) ---
class SessionData(BaseModel):
    session_duration: float
    product_views: int
    cart_additions: int
    max_price: float
    brand_affinity: str

class FeedbackData(BaseModel):
    prediction_probability: float
    recommended_bundle: str
    user_feedback: str # 'up' or 'down'

# --- ENDPOINTS ---

@app.get("/")
def read_root():
    return {"status": "Active", "system": "Hybrid MLOps Backend Running"}

@app.post("/api/predict")
def predict_purchase(data: SessionData):
    try:
        # 1. Handle Brand Encoding securely (fallback to 'unknown' if unseen brand)
        brand = data.brand_affinity
        if brand not in brand_encoder.classes_:
            brand = 'unknown'
        brand_encoded = brand_encoder.transform([brand])[0]

        # 2. Format features for the Decision Tree
        features = pd.DataFrame([{
            'Session_Duration': data.session_duration,
            'Product_Views': data.product_views,
            'Cart_Additions': data.cart_additions,
            'Max_Price': data.max_price,
            'Brand_Encoded': brand_encoded
        }])

        # 3. Execute Model Prediction
        probability = dt_model.predict_proba(features)[0][1] # Probability of Class 1 (Purchase)
        
        # 4. Generate Business Logic (Bundle Recommendation)
        bundle_type = "None"
        if probability > 0.70:
            bundle_type = "High-End Peripherals (e.g., Gaming Mouse)"
        elif probability > 0.40:
            bundle_type = "Targeted 10% Discount Code"

        return {
            "purchase_probability": round(float(probability), 4),
            "bundle_action": bundle_type,
            "explainability_note": f"Triggered by {data.product_views} views and ${data.max_price} price focus."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/feedback")
def log_feedback(feedback: FeedbackData):
    """
    MLOps Endpoint: Captures live user feedback to trigger future retraining.
    """
    os.makedirs(os.path.dirname(FEEDBACK_LOG_PATH), exist_ok=True)
    
    # Check if we need to write CSV headers
    write_header = not os.path.exists(FEEDBACK_LOG_PATH)
    
    with open(FEEDBACK_LOG_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["timestamp", "probability", "bundle", "feedback"])
        
        writer.writerow([
            datetime.now().isoformat(),
            feedback.prediction_probability,
            feedback.recommended_bundle,
            feedback.user_feedback
        ])
    
    return {"status": "success", "message": "Feedback securely logged to MLOps pipeline."}