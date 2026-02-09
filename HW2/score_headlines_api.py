
"""
This module exposes a FastAPI service to score headlines using a pre-trained
SentenceTransformer model and an SVM classifier.
"""

import logging
import sys
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = FastAPI()

# --- GLOBAL MODEL LOADING ---
logging.info("Attempting to load models...")
try:
    # Load the embedding model and the SVM classifier
    model = SentenceTransformer("/opt/huggingface_models/all-MiniLM-L6-v2")#Using hugging face LLM in remote server instead of downloading LLM everytime
    clf = joblib.load('svm.joblib')
    logging.info("Models loaded successfully.")
except FileNotFoundError:
    logging.critical("Fatal Error: 'svm.joblib' file not found. Server cannot start.")
    sys.exit(1)
except Exception as startup_error:  # pylint: disable=broad-except
    logging.critical("Fatal Error loading models: %s", startup_error)
    sys.exit(1)

# --- Pydantic Model for Request Body ---
class HeadlineList(BaseModel):
    """
    Pydantic model defining the expected JSON body for the
    /score_headlines endpoint.
    """
    headlines: list[str]

# --- API Endpoints ---

@app.get('/status')
def status():
    """
    Returns a simple JSON object to confirm the service is up.
    """
    logging.info("Status check endpoint called.")
    return {'status': 'OK'}

@app.post('/score_headlines')
def score_headlines(data: HeadlineList):
    """
    Accepts a list of headlines, encodes them using the global model,
    predicts sentiment, and returns the labels.
    """
    # Log general info: A request came in
    count = len(data.headlines)
    logging.info("Received request to score %s headlines.", count)

    # Log detailed info: content of the request
    logging.debug("Headlines content: %s", data.headlines)

    try:
        # 1. Encode the incoming headlines
        embeddings = model.encode(data.headlines)

        # 2. Predict sentiment using the loaded classifier
        predictions = clf.predict(embeddings)

        logging.info("Prediction successful.")

        # 3. Return only the labels
        return {'labels': predictions.tolist()}

    except Exception as processing_error:  # pylint: disable=broad-except
        # Log the specific error if something goes wrong during prediction
        logging.error("Error during prediction: %s", processing_error)
        return {"error": "An error occurred during processing."}
