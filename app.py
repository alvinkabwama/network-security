import sys, os  # System-specific parameters and functions, OS utilities

import certifi  # Provides CA bundle for SSL certificate verification
ca = certifi.where()  # Path to the CA certificate bundle

from dotenv import load_dotenv  # To load environment variables from a .env file

load_dotenv()  # Load environment variables into process environment

# Get MongoDB connection URL from environment variables
mongo_db_url = os.getenv("MONGODB_URL_KEY")
print(mongo_db_url)  # Debug print to verify MongoDB URL is loaded correctly

import pymongo  # MongoDB client library
from networksecurity.logging import logger  # Project logger (imported but not directly used below)
import os, sys  # Re-imported (kept as-is)
from networksecurity.exception.exception import NetworkSecurityException  # Custom exception type
from networksecurity.pipeline.training_pipeline import TrainingPipeline  # Orchestrates training pipeline

from fastapi.middleware.cors import CORSMiddleware  # Middleware to handle CORS
from fastapi import FastAPI, File, UploadFile, Request  # FastAPI core classes and request/file handling
from uvicorn import run as app_run  # ASGI server runner
from fastapi.responses import Response  # Basic HTTP Response
from starlette.responses import RedirectResponse  # Response to redirect client
import pandas as pd  # Data manipulation and CSV handling

from networksecurity.utils.main_utils.utils import load_object  # Utility to load serialized objects
from networksecurity.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME, DATA_INGESTION_COLLECTION_NAME  # DB constants
from networksecurity.utils.ml_utils.model.estimator import NetworkModel  # Wrapper model combining preprocessor + model

# Create MongoDB client using loaded URL and certificate for TLS
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

# Select the database and collection used during data ingestion
database = client[DATA_INGESTION_DATABASE_NAME]
collection = client[DATA_INGESTION_COLLECTION_NAME]

# Initialize FastAPI application
app = FastAPI()
origins = ["*"]  # Allow all origins (for CORS)

# Configure CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # Origins allowed to access this API
    allow_credentials=True,      # Allow cookies/credentials
    allow_methods=["*"],         # Allow all HTTP methods
    allow_headers=["*"]          # Allow all headers
)

from fastapi.templating import Jinja2Templates  # Template rendering for HTML responses
templates = Jinja2Templates(directory="./templates")  # Directory where HTML templates are stored


# Root route: redirect to the interactive API docs (/docs)
@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


# Route to trigger model training pipeline
@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()  # Create training pipeline instance
        train_pipeline.run_pipeline()        # Run the full training process
        return Response("Training is successful")  # Simple text response
    except Exception as e:
        # Wrap any error into custom NetworkSecurityException for consistent handling/logging
        raise NetworkSecurityException(e, sys)
    

# Route to handle prediction via file upload
@app.get("/predict")
async def predict_route(request:Request, file:UploadFile=File(...)):
    try:
        # Read uploaded CSV file into a pandas DataFrame
        df = pd.read_csv(file.file)

        # Load preprocessor and final trained model from disk
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")

        # Wrap preprocessor and model into NetworkModel for consistent prediction pipeline
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

        # Run prediction on uploaded data
        y_pred = network_model.predict(df)

        # Add prediction results as a new column in the DataFrame
        df["predicted_column"] = y_pred
        print(df["predicted_column"])  # Debug print of predictions

        # Save prediction results to CSV file
        df.to_csv("prediction_output/output.csv")

        # Convert DataFrame to HTML table for rendering in a template
        table_html = df.to_html(classes='table table-striped')

        # Render table.html template with generated table HTML
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})

    except Exception as e:
        # Any exception during prediction is wrapped in NetworkSecurityException
        raise NetworkSecurityException

# Entry point to run the FastAPI app using Uvicorn
if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8888)
