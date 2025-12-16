import asyncio
import pandas as pd
from prisma import Prisma
import numpy as np
import os

async def seed_data():
    db = Prisma()
    await db.connect()

    print(">>> 🌱 Seeding Database...")

    # --- 1. Seed History (from sales_data_final.csv) ---
    try:
        print("   -> Loading History...")
        df = pd.read_csv('./ml_bin/sales_data_final.csv') # Path to your source data
        
        # Melt Logic
        id_vars = ['Product_Code']
        date_cols = [c for c in df.columns if c not in id_vars]
        df_long = df.melt(id_vars=id_vars, value_vars=date_cols, var_name='ds', value_name='y')
        df_long['ds'] = pd.to_datetime(df_long['ds'])

        # Prepare batch for Prisma
        history_records = []
        for _, row in df_long.iterrows():
            history_records.append({
                'product_code': str(row['Product_Code']),
                'ds': row['ds'],
                'y': float(row['y'])
            })

        # Bulk Create (Chunking recommended for large datasets)
        # Using create_many is much faster
        await db.saleshistory.create_many(data=history_records, skip_duplicates=True)
        print(f"   ✅ Inserted {len(history_records)} history records.")

    except Exception as e:
        print(f"   ❌ Error seeding history: {e}")

    # --- 2. Seed Forecasts (from the training output csv) ---
    try:
        print("   -> Loading Forecasts...")
        # Load the CSV you generated in Phase 4 of training pipeline
        df_forecast = pd.read_csv('./ml_bin/final_forecast_backup.csv') 
        df_forecast['forecast_date'] = pd.to_datetime(df_forecast['forecast_date'])

        forecast_records = []
        for _, row in df_forecast.iterrows():
            forecast_records.append({
                'product_code': str(row['product_code']),
                'forecast_date': row['forecast_date'],
                'predicted_sales': float(row['predicted_sales'])
            })

        # Clear old forecasts first (Pipeline logic: replace)
        await db.salesforecast.delete_many()
        
        await db.salesforecast.create_many(data=forecast_records, skip_duplicates=True)
        print(f"   ✅ Inserted {len(forecast_records)} forecast records.")

    except Exception as e:
        print(f"   ❌ Error seeding forecasts: {e}")

    # --- 3. Seed Metrics (Dummy or Read from log) ---
    await db.modelmetric.create(data={
        'wmape': 0.185,
        'accuracy': 0.815,
        'rmse': 124.5,
        'description': 'Initial Seed Run'
    })
    print("   ✅ Inserted initial metrics.")

    await db.disconnect()
    print(">>> 🌱 Seeding Complete!")

if __name__ == '__main__':
    asyncio.run(seed_data())