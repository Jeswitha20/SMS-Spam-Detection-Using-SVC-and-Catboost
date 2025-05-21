from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel
import numpy as np
import csv
from datetime import datetime
from langdetect import detect
from deep_translator import GoogleTranslator
import re

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained models & vectorizer
svc_model = joblib.load("backend/models/svc_model.pkl")
catboost_model = joblib.load("backend/models/catboost_model.pkl")
vectorizer = joblib.load("backend/models/vectorizer.pkl")  # Load the text vectorizer

# Define input data models
class TextData(BaseModel):
    text: str  # Match frontend input

class FeedbackData(BaseModel):
    text: str
    original_prediction: int
    correct_label: int
    timestamp: str

# Define obvious spam patterns
OBVIOUS_SPAM_PATTERNS = {
    'winning': r'(?i)(won|win|winner|winning|prize|reward)',
    'urgent': r'(?i)(urgent|immediate|hurry|limited time|expires)',
    'money': r'(?i)(\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|rs|rupees?|inr|usd))',
    'links': r'(?i)(click|link|http|www\.)',
    'personal_info': r'(?i)(password|account|bank|credit card|ssn|social security)',
    'lottery': r'(?i)(lottery|jackpot|lucky|draw)',
    'inheritance': r'(?i)(inheritance|unclaimed|funds|transfer)',
    'gift_cards': r'(?i)(gift card|voucher|coupon)',
    'verification': r'(?i)(verify|verification|confirm|account suspended)',
    'lottery_prize': r'(?i)(lottery|prize|jackpot|lucky|draw)',
    'urgent_action': r'(?i)(urgent|immediate|hurry|limited time|expires)',
    'money_amount': r'(?i)(\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|rs|rupees?|inr|usd))',
    'click_link': r'(?i)(click|link|http|www\.)',
    'personal_data': r'(?i)(password|account|bank|credit card|ssn|social security)'
}

def check_obvious_spam(text: str) -> float:
    """Check for obvious spam patterns in the text"""
    text_lower = text.lower()
    score = 0.0
    
    # Check each pattern category
    for pattern in OBVIOUS_SPAM_PATTERNS.values():
        if re.search(pattern, text_lower):
            score += 0.2  # Add 20% for each matching pattern
    
    return min(score, 1.0)  # Cap at 100%

def get_spam_precautions():
    return [
        "Do not click on any links in suspicious messages - they may lead to phishing sites.",
        "Never share personal information (bank details, passwords, SSN) via SMS.",
        "Be wary of urgent or time-sensitive offers - scammers often create false urgency.",
        "Report suspicious messages to your mobile carrier or spam reporting services.",
        "Block numbers that consistently send spam messages to prevent future attempts.",
        "If the message mentions winning something you never entered, it's likely a scam."
    ]

def translate_text(text: str) -> str:
    """Translate non-English text to English using Google Translate"""
    try:
        # Detect the language
        detected_lang = detect(text)
        
        # If the language is not English, translate it
        if detected_lang != 'en':
            translator = GoogleTranslator(source=detected_lang, target='en')
            translated_text = translator.translate(text)
            return translated_text
        return text
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text  # Return original text if translation fails

@app.post("/predict")
def predict(data: TextData):
    try:
        # Translate text if it's not in English
        processed_text = translate_text(data.text)
        
        # Check for obvious spam patterns
        obvious_spam_score = check_obvious_spam(processed_text)
        
        # Transform text input into numerical features using vectorizer
        text_vectorized = vectorizer.transform([processed_text])
        
        # Predict with both models
        svc_probability = svc_model.predict_proba(text_vectorized).tolist()[0][1]  # Get spam probability
        catboost_probability = catboost_model.predict_proba(text_vectorized).tolist()[0][1]  # Get spam probability

        # Adjust weights for the models and pattern matching
        # Give more weight to pattern matching for obvious spam
        if obvious_spam_score > 0.6:  # If strong spam patterns are found
            svc_weight = 0.3
            catboost_weight = 0.2
            pattern_weight = 0.5
        else:  # For less obvious cases, rely more on ML models
            svc_weight = 0.45
            catboost_weight = 0.45
            pattern_weight = 0.1

        # Compute weighted probability including pattern matching
        weighted_probability = (
            svc_weight * svc_probability + 
            catboost_weight * catboost_probability +
            pattern_weight * obvious_spam_score
        )

        # Adjust threshold based on pattern score
        # If strong spam patterns are found, use a lower threshold
        # If no obvious patterns, use a higher threshold to avoid false positives
        threshold = 0.35 if obvious_spam_score > 0.4 else 0.5
        final_prediction = 1 if weighted_probability > threshold else 0

        # Provide precautions if message is spam
        precautions = get_spam_precautions() if final_prediction == 1 else []

        return {
            "prediction": final_prediction,
            "weighted_probability": weighted_probability,
            "svc_probability": svc_probability,
            "catboost_probability": catboost_probability,
            "pattern_score": obvious_spam_score,
            "threshold_used": threshold,
            "precautions": precautions,
            "original_language": detect(data.text),
            "translated_text": processed_text if processed_text != data.text else None
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/feedback")
def collect_feedback(data: FeedbackData):
    try:
        # Save feedback to CSV file with timestamp
        with open("backend/data/user_feedback.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                data.text,
                data.original_prediction,
                data.correct_label,
                data.timestamp
            ])
        return {"message": "Feedback recorded successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recording feedback: {str(e)}")

@app.get("/")
def home():
    return {"message": "SMS Spam Detection API (Weighted Ensemble Model)"}
