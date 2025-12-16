from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from prisma import Prisma
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import pickle
import os
import lightgbm as lgb

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
    # DB Connection
    db = Prisma()
    await db.connect()
    app.state.db = db
    print("✅ Database Connected")

    # Load ML Models (For live inference endpoint)
    if os.path.exists('./ml_bin/global_lgbm/lgb_model.txt'):
        print("✅ Loading LightGBM...")
        models['lgb'] = lgb.Booster(model_file='./ml_bin/global_lgbm/lgb_model.txt')
        with open('./ml_bin/global_lgbm/product_encoder.pkl', 'rb') as f:
            models['encoder'] = pickle.load(f)
    else:
        print("⚠️ ML Artifacts not found. Live inference endpoint will fail.")

    yield
    
    # Disconnect DB
    if app.state.db.is_connected():
        await app.state.db.disconnect()
        print("🛑 Database Disconnected")


app = FastAPI(title="Sales Forecasting API", lifespan=lifespan)

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
async def generate_live_forecast(product_id: str):
    """
    Triggers an on-demand re-forecast using the loaded .pkl and LGBM models.
    (Simplified Logic for Demonstration)
    """
    if 'lgb' not in models:
        raise HTTPException(status_code=503, detail="ML Models not loaded on server")

    prophet_path = f"./ml_bin/prophet_individual/{product_id}.pkl"
    if not os.path.exists(prophet_path):
        raise HTTPException(status_code=404, detail="Individual Prophet model not found")

    try:
        # Load Prophet
        with open(prophet_path, 'rb') as f:
            m = pickle.load(f)
        
        # 1. Prophet Future
        future = m.make_future_dataframe(periods=104, freq='W')
        forecast = m.predict(future)
        
        # 2. Extract needed trend
        # In a real app, you would run the full feature engineering function here
        # For this example, we return a success message confirming the pipeline is ready
        
        return {
            "status": "success", 
            "message": f"Live model loaded for {product_id}. Pipeline ready for inference calculation.",
            "steps_executed": ["Prophet Loaded", "Future DataFrame Created", "Trend Predicted"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))