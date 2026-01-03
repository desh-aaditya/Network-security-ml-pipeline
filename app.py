import sys
import os
import certifi
import pandas as pd
import pymongo

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.constant.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME
)

# ------------------ ENV SETUP ------------------
load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")

if not mongo_db_url:
    raise ValueError("MongoDB URL not found in environment variables")

# ------------------ DB CONNECTION ------------------
ca = certifi.where()
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# ------------------ FASTAPI APP ------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# ------------------ ROUTES ------------------
@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        return Response("Training completed successfully")
    except Exception as e:
        raise NetworkSecurityException(e, sys)

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        preprocessor = load_object("final_models/preprocessor.pkl")
        model = load_object("final_models/model.pkl")

        network_model = NetworkModel(
            preprocessor=preprocessor,
            model=model
        )

        predictions = network_model.predict(df)
        df["predicted_column"] = predictions

        os.makedirs("prediction_output", exist_ok=True)
        df.to_csv("prediction_output/output.csv", index=False)

        table_html = df.to_html(classes="table table-striped")
        return templates.TemplateResponse(
            "table.html",
            {"request": request, "table": table_html}
        )

    except Exception as e:
        raise NetworkSecurityException(e, sys)
