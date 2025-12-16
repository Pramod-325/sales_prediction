from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from prisma import Prisma
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import pickle
import os
import lightgbm as lgb
import io
from abc import ABC, abstractmethod

# --- Abstract Interface (The Contract) ---
class StorageBackend(ABC):
    @abstractmethod
    def download_pickle(self, filename: str): pass
    @abstractmethod
    def upload_pickle(self, obj, filename: str): pass
    @abstractmethod
    def load_lgbm(self, filename: str): pass

# --- Option A: Local Disk ---
class LocalStorage(StorageBackend):
    def __init__(self, base_dir="models_local"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        print(f"📂 Storage initialized: Local Disk ({self.base_dir})")

    def _get_path(self, filename):
        return os.path.join(self.base_dir, filename)

    def download_pickle(self, filename: str):
        path = self._get_path(filename)
        if not os.path.exists(path):
            print(f"⚠️ Local file not found: {path}")
            return None
        with open(path, 'rb') as f:
            return pickle.load(f)

    def upload_pickle(self, obj, filename: str):
        path = self._get_path(filename)
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        print(f"✅ Saved local file: {path}")

    def load_lgbm(self, filename: str):
        # We assume the global models are in 'models/global_lgbm' locally
        # You might need to adjust this path based on where you saved Phase 4 output
        potential_paths = [
            self._get_path(filename),
            f"models/global_lgbm/{filename}", # Standard pipeline path
            filename
        ]
        
        for p in potential_paths:
            if os.path.exists(p):
                print(f"✅ Loaded LGBM from: {p}")
                return lgb.Booster(model_file=p)
        
        raise FileNotFoundError(f"LGBM model not found in: {potential_paths}")

# --- Option B: Azure Blob (For Production) ---
class AzureBlobStorage(StorageBackend):
    def __init__(self):
        try:
            from azure.storage.blob import BlobServiceClient
            conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            if not conn_str:
                raise ValueError("AZURE_STORAGE_CONNECTION_STRING is missing")
            
            self.blob_service = BlobServiceClient.from_connection_string(conn_str)
            self.container_name = "models"
            self.container_client = self.blob_service.get_container_client(self.container_name)
            print("☁️ Storage initialized: Azure Blob")
        except ImportError:
            raise ImportError("Run 'uv add azure-storage-blob' for production mode")

    def download_pickle(self, filename: str):
        try:
            blob_client = self.container_client.get_blob_client(filename)
            stream = io.BytesIO()
            blob_client.download_blob().readinto(stream)
            stream.seek(0)
            return pickle.load(stream)
        except Exception as e:
            print(f"⚠️ Azure Blob read error ({filename}): {e}")
            return None

    def upload_pickle(self, obj, filename: str):
        stream = io.BytesIO()
        pickle.dump(obj, stream)
        stream.seek(0)
        blob_client = self.container_client.get_blob_client(filename)
        blob_client.upload_blob(stream, overwrite=True)

    def load_lgbm(self, filename: str):
        blob_client = self.container_client.get_blob_client(filename)
        download_path = f"/tmp/{filename}"
        with open(download_path, "wb") as f:
            blob_client.download_blob().readinto(f)
        return lgb.Booster(model_file=download_path)

def get_storage_backend():
    # If explicit PROD flag or Azure connection string exists, use Azure
    if os.getenv("APP_ENV") == "production":
        return AzureBlobStorage()
    else:
        # Default to local
        return LocalStorage()

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

# main.py
from blob_storage import get_storage_backend

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize Storage (Local or Azure based on Env)
    try:
        app.state.storage = get_storage_backend()
    except Exception as e:
        print(f"❌ CRITICAL STORAGE ERROR: {e}")
        # Assign a dummy or fail hard, so you see the error immediately
        raise e 

    # 2. Connect DB
    db = Prisma()
    await db.connect()
    app.state.db = db
    
    # 3. Load Global Models via the Storage Interface
    storage = app.state.storage
    try:
        # Using the unified method 'load_lgbm'
        models['lgb'] = storage.load_lgbm("lgb_model.txt")
        
        # Using 'download_pickle' (works for local file read too)
        # Ensure 'product_encoder.pkl' is in your 'models_local' folder or 'models/global_lgbm'
        models['encoder'] = storage.download_pickle("product_encoder.pkl")
        
        print("✅ Global Models Loaded Successfully")
    except Exception as e:
        print(f"⚠️ Model Loading Failed: {e}")
        print("   (Live endpoints will fail until models are fixed)")

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
async def generate_live_forecast(product_id: str, background_tasks: BackgroundTasks):
    """
    1. Retrains/Creates model for specific product immediately.
    2. Returns the new 2-year forecast in the response.
    3. Uploads model to Blob Storage in the background (to save time).
    """
    try:
        db = app.state.db
        storage = app.state.storage
        lgbm_model = models.get('lgb')
        encoder = models.get('encoder')

        # --- 1. FETCH LATEST HISTORY ---
        history_records = await db.saleshistory.find_many(
            where={'product_code': product_id},
            order={'ds': 'asc'}
        )
        
        if not history_records:
            raise HTTPException(status_code=404, detail="No sales history found. Add data to 'sales_history' first.")

        df_history = pd.DataFrame([vars(r) for r in history_records])
        df_history = df_history.rename(columns={'ds': 'ds', 'y': 'y'})
        df_history['y_log'] = np.log1p(df_history['y'])

        # --- 2. PROPHET: LOAD OR CREATE ---
        filename = f"prophet_{product_id}.pkl"
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))