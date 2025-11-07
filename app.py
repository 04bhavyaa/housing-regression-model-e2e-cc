"""
Streamlit app to visualize housing price predictions
and interact with the FastAPI model endpoint.

Features:
- Securely fetches input data from S3 (via .env credentials)
- Sends filtered batches to FastAPI `/predict`
- Displays predictions vs. actuals, metrics, and monthly trends
"""

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import boto3
import os
from pathlib import Path
from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

# ============================
# 1. Environment & Config
# ============================
load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "housing-data-regression-cc")
REGION = os.getenv("AWS_DEFAULT_REGION", "eu-west-2")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Initialize S3 client
try:
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=REGION,
    )
except (NoCredentialsError, PartialCredentialsError):
    st.error("❌ AWS credentials not found. Check your .env file.")
    st.stop()

# ============================
# 2. S3 Loader
# ============================
def load_from_s3(key: str, local_path: str) -> str:
    """Download file from S3 if not cached locally."""
    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        st.info(f"📥 Downloading `{key}` from S3 bucket `{S3_BUCKET}`...")
        try:
            s3.download_file(S3_BUCKET, key, str(local_path))
            st.success(f"✅ Downloaded {key}")
        except NoCredentialsError:
            st.error("❌ Missing AWS credentials for S3 access.")
            st.stop()
        except Exception as e:
            st.error(f"⚠️ Failed to download {key} from S3: {e}")
            st.stop()
    return str(local_path)

# Paths (cached locally if missing)
HOLDOUT_ENGINEERED_PATH = load_from_s3(
    "processed/feature_engineered_holdout.csv",
    "data/processed/feature_engineered_holdout.csv",
)
HOLDOUT_META_PATH = load_from_s3(
    "processed/cleaning_holdout.csv",
    "data/processed/cleaning_holdout.csv",
)

# ============================
# 3. Data Loading
# ============================
@st.cache_data
def load_data():
    """Load engineered and meta holdout datasets."""
    fe = pd.read_csv(HOLDOUT_ENGINEERED_PATH)
    meta = pd.read_csv(HOLDOUT_META_PATH, parse_dates=["date"])[["date", "city_full"]]

    if len(fe) != len(meta):
        st.warning("⚠️ Engineered and meta holdout lengths differ. Aligning by index.")
        min_len = min(len(fe), len(meta))
        fe = fe.iloc[:min_len].copy()
        meta = meta.iloc[:min_len].copy()

    disp = pd.DataFrame(index=fe.index)
    disp["date"] = meta["date"]
    disp["region"] = meta["city_full"]
    disp["year"] = disp["date"].dt.year
    disp["month"] = disp["date"].dt.month
    disp["actual_price"] = fe["price"]

    return fe, disp

fe_df, disp_df = load_data()

# ============================
# 4. UI
# ============================
st.title("🏠 Housing Price Prediction — Holdout Explorer")

years = sorted(disp_df["year"].unique())
months = list(range(1, 13))
regions = ["All"] + sorted(disp_df["region"].dropna().unique())

col1, col2, col3 = st.columns(3)
with col1:
    year = st.selectbox("Select Year", years, index=0)
with col2:
    month = st.selectbox("Select Month", months, index=0)
with col3:
    region = st.selectbox("Select Region", regions, index=0)

if st.button("Show Predictions 🚀"):
    mask = (disp_df["year"] == year) & (disp_df["month"] == month)
    if region != "All":
        mask &= disp_df["region"] == region

    idx = disp_df.index[mask]

    if len(idx) == 0:
        st.warning("No data found for these filters.")
    else:
        st.write(f"📅 Running predictions for **{year}-{month:02d}** | Region: **{region}**")
        payload = fe_df.loc[idx].to_dict(orient="records")

        try:
            resp = requests.post(API_URL, json=payload, timeout=90)
            resp.raise_for_status()
            out = resp.json()
            preds = out.get("predictions", [])
            actuals = out.get("actuals", None)

            view = disp_df.loc[idx, ["date", "region", "actual_price"]].copy()
            view = view.sort_values("date")
            view["prediction"] = pd.Series(preds, index=view.index).astype(float)

            if actuals is not None and len(actuals) == len(view):
                view["actual_price"] = pd.Series(actuals, index=view.index).astype(float)

            # Metrics
            mae = (view["prediction"] - view["actual_price"]).abs().mean()
            rmse = ((view["prediction"] - view["actual_price"]) ** 2).mean() ** 0.5
            avg_pct_error = (
                ((view["prediction"] - view["actual_price"]).abs() / view["actual_price"]).mean() * 100
            )

            st.subheader("Predictions vs Actuals")
            st.dataframe(
                view[["date", "region", "actual_price", "prediction"]].reset_index(drop=True),
                use_container_width=True,
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("MAE", f"{mae:,.0f}")
            with c2:
                st.metric("RMSE", f"{rmse:,.0f}")
            with c3:
                st.metric("Avg % Error", f"{avg_pct_error:.2f}%")

            # ============================
            # Yearly Trend Chart
            # ============================
            yearly_data = (
                disp_df[(disp_df["year"] == year) if region == "All" else (disp_df["year"] == year) & (disp_df["region"] == region)]
                .copy()
            )
            idx_all = yearly_data.index
            payload_all = fe_df.loc[idx_all].to_dict(orient="records")

            resp_all = requests.post(API_URL, json=payload_all, timeout=90)
            resp_all.raise_for_status()
            preds_all = resp_all.json().get("predictions", [])

            yearly_data["prediction"] = pd.Series(preds_all, index=yearly_data.index).astype(float)
            monthly_avg = yearly_data.groupby("month")[["actual_price", "prediction"]].mean().reset_index()

            highlight_month = month
            fig = px.line(
                monthly_avg,
                x="month",
                y=["actual_price", "prediction"],
                markers=True,
                labels={"value": "Price", "month": "Month"},
                title=f"Yearly Trend — {year}{'' if region=='All' else f' — {region}'}",
            )

            fig.add_vrect(
                x0=highlight_month - 0.5,
                x1=highlight_month + 0.5,
                fillcolor="red",
                opacity=0.1,
                layer="below",
                line_width=0,
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"API call failed: {e}")
            st.exception(e)

else:
    st.info("Choose filters and click **Show Predictions** to compute.")
