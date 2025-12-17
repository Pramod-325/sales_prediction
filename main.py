from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from prisma import Prisma
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import pickle
import os
import lightgbm as lgb
from fastapi.middleware.cors import CORSMiddleware
from blob_storage import get_storage_backend
import pandas as pd
import numpy as np
from prophet import Prophet

# origins = [
#     "http://localhost:5173",
#     "http://127.0.0.1:5173"
# ]


models = {}

class SalesRecord(BaseModel):
    product_code: str
    ds: datetime
    y: float

class ForecastRecord(BaseModel):
    product_code: str
    forecast_date: datetime
    predicted_sales: float

class MetricsResponse(BaseModel):
    model_version: str
    wmape: Optional[float]
    accuracy: Optional[float]
    last_trained: datetime

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(">>> 🟢 STARTUP: Initializing resources...")
    
    # A. Initialize Storage
    try:
        app.state.storage = get_storage_backend()
        print(f"   ✅ Storage loaded: {app.state.storage}")
    except Exception as e:
        print(f"   ❌ Storage failed: {e}")
        # We don't raise here to let the app start, but endpoints will fail later
        app.state.storage = None

    # B. Connect Database
    try:
        db = Prisma()
        await db.connect()
        app.state.db = db
        print("   ✅ Database connected")
    except Exception as e:
        print(f"   ❌ Database connection failed: {e}")

    # C. Load Global Models
    models = {}
    if app.state.storage:
        try:
            # Try to load models, handle errors gracefully if files missing
            # NOTE: Ensure 'lgb_model.txt' exists in your models_local folder!
            try:
                models['lgb'] = app.state.storage.load_lgbm("lgb_model.txt")
                models['encoder'] = app.state.storage.download_pickle("product_encoder.pkl")
                print("   ✅ ML Models loaded")
            except Exception as load_err:
                print(f"   ⚠️ ML Models missing (Live inference will fail): {load_err}")
        except Exception as e:
            print(f"   ❌ Model Loader failed: {e}")
    
    app.state.models = models

    yield  # <--- App runs here

    # D. Cleanup
    print(">>> 🔴 SHUTDOWN: Cleaning up...")
    if hasattr(app.state, 'db') and app.state.db.is_connected():
        await app.state.db.disconnect()

app = FastAPI(title="Sales Forecasting API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/sales/history/{product_id}", response_model=List[SalesRecord])
async def get_history(product_id: str):
    """Fetch actual sales history for a product."""
    db = app.state.db
    records = await db.saleshistory.find_many(
        where={'product_code': product_id},
        order={'ds': 'asc'}
    )
    if not records:
        raise HTTPException(status_code=404, detail="Product history not found")
    return records


@app.get("/sales/forecast/{product_id}", response_model=List[ForecastRecord])
async def get_forecast(product_id: str):
    """Fetch pre-calculated 2-year forecast."""
    db = app.state.db
    records = await db.salesforecast.find_many(
        where={'product_code': product_id},
        order={'forecast_date': 'asc'}
    )
    if not records:
        raise HTTPException(status_code=404, detail="Forecasts not found for this product")
    return records


@app.get("/metrics/model", response_model=MetricsResponse)
async def get_metrics():
    """Get the latest model performance metrics."""
    db = app.state.db
    # Get the latest entry
    metric = await db.modelmetric.find_first(
        order={'training_run_date': 'desc'}
    )
    
    if not metric:
        return MetricsResponse(
            model_version="None", wmape=0, accuracy=0, last_trained=datetime.now()
        )

    return MetricsResponse(
        model_version=metric.model_version,
        wmape=metric.wmape,
        accuracy=metric.accuracy,
        last_trained=metric.training_run_date
    )

@app.post("/sales/forecast/live/{product_id}")
async def generate_live_forecast(product_id: str, background_tasks: BackgroundTasks):
    """
    1. Retrains/Creates model for specific product immediately.
    2. Returns the new 2-year forecast in the response.
    3. Uploads model to Blob Storage in the background (to save time).
    """
    db = app.state.db
    storage = app.state.storage
    # lgbm_model = models.get('lgb')
    # encoder = models.get('encoder')
    lgbm_model = app.state.models.get('lgb')
    encoder = app.state.models.get('encoder')

    # --- 1. FETCH LATEST HISTORY ---
    history_records = await db.saleshistory.find_many(
        where={'product_code': product_id},
        order={'ds': 'asc'}
    )
    
    if not history_records:
        raise HTTPException(status_code=404, detail="No sales history found. Add data to 'sales_history' first.")

    df_history = pd.DataFrame([vars(r) for r in history_records])
    df_history = df_history.rename(columns={'ds': 'ds', 'y': 'y'})

    if pd.api.types.is_datetime64_any_dtype(df_history['ds']):
         df_history['ds'] = df_history['ds'].dt.tz_localize(None)
    else:
         # Safety: If it came as string, convert then strip
         df_history['ds'] = pd.to_datetime(df_history['ds']).dt.tz_localize(None)

    df_history['y_log'] = np.log1p(df_history['y'])

    # --- 2. PROPHET: LOAD OR CREATE ---
    filename = f"{product_id}.pkl"
    m = storage.download_pickle(filename)

    # Logic: If model exists, fit on new data. If not, create new.
    # Note: For single product, this takes ~1-3 seconds, which is safe for HTTP request.
    seasonality = True if len(df_history) > 52 else False
    
    if m:
        # Update existing
        m = Prophet(yearly_seasonality=seasonality, weekly_seasonality=False, daily_seasonality=False)
        m.fit(df_history)
    else:
        # Create New
        m = Prophet(yearly_seasonality=seasonality, weekly_seasonality=False, daily_seasonality=False)
        m.fit(df_history)

    # --- 3. GENERATE TREND & FEATURES ---
    future = m.make_future_dataframe(periods=104, freq='W')
    forecast = m.predict(future)
    
    df_features = forecast[['ds', 'yhat']].rename(columns={'yhat': 'prophet_pred_log'})
    df_features['Product_Code'] = product_id

    # Feature Engineering (Standardized)
    df_features['month'] = df_features['ds'].dt.month
    df_features['week'] = df_features['ds'].dt.isocalendar().week.astype(int)
    df_features['year'] = df_features['ds'].dt.year
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    df_features['week_sin'] = np.sin(2 * np.pi * df_features['week'] / 52)
    df_features['week_cos'] = np.cos(2 * np.pi * df_features['week'] / 52)

    # Encoding
    if product_id in encoder.classes_:
        encoded_id = encoder.transform([product_id])[0]
    else:
        encoded_id = -1 
    df_features['Product_Code_Encoded'] = encoded_id

    # --- 4. PREDICT (LightGBM) ---
    features = ['prophet_pred_log', 'Product_Code_Encoded', 
                'month_sin', 'month_cos', 'week_sin', 'week_cos', 'year']
    
    lgb_preds_log = lgbm_model.predict(df_features[features])
    df_features['predicted_sales'] = np.expm1(lgb_preds_log).clip(min=0)

    # --- 5. FILTER & SAVE ---
    
    # A. Save Model (Background Task to keep API fast)
    background_tasks.add_task(storage.upload_pickle, m, filename)

    # B. Filter Future Data Only
    last_history_date = df_history['ds'].max()
    df_final = df_features[df_features['ds'] > last_history_date].copy()

    # C. Save to DB (We await this so DB is consistent before return)
    forecast_records = []
    response_data = [] # List to return to user
    
    for _, row in df_final.iterrows():
        # Create record for DB
        record = {
            'product_code': product_id,
            'forecast_date': row['ds'],
            'predicted_sales': float(row['predicted_sales'])
        }
        forecast_records.append(record)
        
        # Create simplified record for JSON Response
        response_data.append({
            'date': row['ds'].strftime('%Y-%m-%d'),
            'sales': round(float(row['predicted_sales']), 2)
        })

    await db.salesforecast.delete_many(where={'product_code': product_id})
    await db.salesforecast.create_many(data=forecast_records)

    # --- 6. RETURN RESULT ---
    return {
        "status": "success", 
        "message": f"Forecast updated for {product_id}. Model upload queued in background.",
        "forecast": response_data  # <--- The requested data
    }