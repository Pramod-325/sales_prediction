import os
import pickle
import io
import lightgbm as lgb
import tempfile  # <--- NEW IMPORT
from abc import ABC, abstractmethod

# --- Abstract Interface ---
class StorageBackend(ABC):
    @abstractmethod
    def download_pickle(self, filename: str): pass
    @abstractmethod
    def upload_pickle(self, obj, filename: str): pass
    @abstractmethod
    def load_lgbm(self, filename: str): pass

# --- Option A: Local Disk (Windows/Mac) ---
class LocalStorage(StorageBackend):
    def __init__(self, base_dir="./ml_bin"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        print(f"📂 Storage initialized: Local Disk ({self.base_dir})")

    def _get_path(self, filename):
        return os.path.join(self.base_dir, filename)

    def download_pickle(self, filename: str):
        path = self._get_path(filename)
        
        # Fallback search logic
        if not os.path.exists(path):
            potential_paths = [
                f"./ml_bin/global_lgbm/{filename}", 
                f"./ml_bin/{filename}",
                filename
            ]
            for p in potential_paths:
                if os.path.exists(p):
                    path = p
                    break
        
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
        potential_paths = [
            self._get_path(filename),
            f"./ml_bin/global_lgbm/{filename}",
            filename
        ]
        
        for p in potential_paths:
            if os.path.exists(p):
                print(f"✅ Loaded LGBM from: {p}")
                return lgb.Booster(model_file=p)
        
        raise FileNotFoundError(f"LGBM model not found. Searched in: {potential_paths}")

# --- Option B: Azure Blob (Production) ---
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
        
        # --- FIX FOR WINDOWS COMPATIBILITY ---
        # 1. Get the OS-specific temp directory
        temp_dir = tempfile.gettempdir() 
        # 2. Join correctly using OS separator
        download_path = os.path.join(temp_dir, filename) 
        # -------------------------------------

        print(f"⬇️ Downloading LGBM to temp: {download_path}")
        with open(download_path, "wb") as f:
            blob_client.download_blob().readinto(f)
            
        return lgb.Booster(model_file=download_path)

# --- Factory Function ---
def get_storage_backend():
    if os.getenv("APP_ENV") == "production":
        return AzureBlobStorage()
    else:
        return LocalStorage(base_dir="./ml_bin")
