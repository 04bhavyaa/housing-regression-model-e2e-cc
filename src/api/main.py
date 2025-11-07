"""
FastAPI web service to serve the trained housing regression model.

Features:
- Loads model and data artifacts securely from S3 using environment variables.
- Exposes REST endpoints for prediction, health checks, and batch runs.
- Handles local caching to avoid re-downloading on every startup.
"""

from fastapi import FastAPI
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import os
import boto3
from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

# ---- Internal imports ----
from src.inference_pipeline.inference import predict
from src.batch.run_monthly import run_monthly_predictions

# ----------------------------
# 1. Config & Environment Setup
# ----------------------------
load_dotenv()  # Load AWS creds + config from .env

S3_BUCKET = os.getenv("S3_BUCKET_NAME", "housing-data-regression-cc")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-west-2")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Initialize boto3 S3 client
try:
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )
except (NoCredentialsError, PartialCredentialsError):
    raise RuntimeError("❌ AWS credentials not found. Check your .env file.")

# ----------------------------
# 2. Local Cache / Paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_best_model.pkl"
TRAIN_FE_PATH = PROJECT_ROOT / "data" / "processed" / "feature_engineered_train.csv"


def load_from_s3(key: str, local_path: Path) -> str:
    """Download file from S3 if not cached locally."""
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            print(f"📥 Downloading {key} from S3 bucket {S3_BUCKET}...")
            s3.download_file(S3_BUCKET, key, str(local_path))
            print(f"✅ Downloaded {key}")
        except NoCredentialsError:
            raise RuntimeError("❌ Missing AWS credentials for S3 access.")
        except Exception as e:
            raise RuntimeError(f"⚠️ Failed to download {key} from S3: {e}")
    return str(local_path)


# Download model + features if not already present
load_from_s3("models/xgb_best_model.pkl", MODEL_PATH)
load_from_s3("processed/feature_engineered_train.csv", TRAIN_FE_PATH)

# ----------------------------
# 3. Load training feature schema
# ----------------------------
if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = [c for c in _train_cols.columns if c != "price"]
else:
    TRAIN_FEATURE_COLUMNS = None

# ----------------------------
# 4. Initialize FastAPI App
# ----------------------------
app = FastAPI(title="🏠 Housing Regression API")

# ----------------------------
# 5. Endpoints
# ----------------------------

@app.get("/")
def root():
    """Basic landing route."""
    return {"message": "Housing Regression API is running 🚀"}


@app.get("/health")
def health() -> Dict[str, Any]:
    """Health check for model and feature schema."""
    status: Dict[str, Any] = {"model_path": str(MODEL_PATH)}

    if not MODEL_PATH.exists():
        status["status"] = "unhealthy"
        status["error"] = "Model not found locally"
    else:
        status["status"] = "healthy"
        if TRAIN_FEATURE_COLUMNS:
            status["n_features_expected"] = len(TRAIN_FEATURE_COLUMNS)

    return status


@app.post("/predict")
def predict_batch(data: List[dict]):
    """Make predictions for a batch of JSON records."""
    if not MODEL_PATH.exists():
        return {"error": f"Model not found at {MODEL_PATH}"}

    df = pd.DataFrame(data)
    if df.empty:
        return {"error": "No data provided"}

    preds_df = predict(df, model_path=MODEL_PATH)
    resp = {"predictions": preds_df["predicted_price"].astype(float).tolist()}

    if "actual_price" in preds_df.columns:
        resp["actuals"] = preds_df["actual_price"].astype(float).tolist()

    return resp


@app.post("/run_batch")
def run_batch():
    """Trigger a full monthly batch prediction job."""
    preds = run_monthly_predictions()
    return {
        "status": "success",
        "rows_predicted": int(len(preds)),
        "output_dir": "data/predictions/",
    }


@app.get("/latest_predictions")
def latest_predictions(limit: int = 5):
    """Return preview of latest prediction CSV."""
    pred_dir = PROJECT_ROOT / "data" / "predictions"
    files = sorted(pred_dir.glob("preds_*.csv"))

    if not files:
        return {"error": "No predictions found"}

    latest_file = files[-1]
    df = pd.read_csv(latest_file)
    return {
        "file": latest_file.name,
        "rows": int(len(df)),
        "preview": df.head(limit).to_dict(orient="records"),
    }


# ----------------------------
# Run locally (if needed)
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
