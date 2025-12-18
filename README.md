#  Sale Prediction – Hybrid Sales Forecasting System

A production-ready **Sales Prediction system** that forecasts future sales for multiple products using a **hybrid machine learning approach**.  
The system is designed to deliver **accurate, business-oriented forecasts** even with **limited historical data**.

---

##  Project Overview

This project predicts **weekly sales for 800+ products** over a **2-year forecast horizon**, using only **52 values/product of weekly of historical data**.  
To overcome data constraints, a **Hybrid Ensemble Model** is used:

- **Facebook's Prophet** for trend (adaptive Linear & Logistic curves) and seasonality (recurring patterns) detection
- **LightGBM** for learning global correction patterns and Final Ensemble Model

This approach combines **time-series forecasting of Prophet** with **machine learning intelligence of LightGBM**.

---

##  Objectives

- Forecast sales accurately with short historical data
- Handle zero and intermittent sales
- Provide business-relevant evaluation metrics
- Support batch forecasting and live retraining (Only at training phase)
- Enable scalable deployment using APIs

---

##  Model Architecture
<img src="https://github.com/Pramod-325/sales_prediction/blob/main/aspyr_sales_prediction_architecture.png">


### Level 1: Facebook Prophet
- Captures yearly seasonality and trend
- Handles missing values and noisy data
- Suitable for short time-series datasets

### Level 2: LightGBM
- Uses Prophet predictions as features
- Learns cross-product and global sales patterns
- Corrects Prophet prediction errors

---

##  Data Pipeline

1. Convert wide-format sales data to long format
2. Apply log transformation to stabilize variance
3. Generate cyclical time features (week/month)
4. Encode product identifiers
5. Train hybrid model
6. Store forecasts and metrics in database

---

## Evaluation Metrics

- **WMAPE (Weighted Mean Absolute Percentage Error)**
- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Square Error)**
- **Business Accuracy = 1 − WMAPE**

Metrics are selected to reflect **real business impact**.

---

##  Tech Stack

### Backend
- Python 3.11
- FastAPI
- Facebook Prophet
- LightGBM
- PostgreSQL
- Prisma ORM

### Frontend
- React
- Material UI
- Recharts

### Storage
- Local filesystem (development)
- Azure Blob Storage (production)

---

##  API Endpoints

| Method | Endpoint | Description |
|------|--------|------------|
| GET | `/sales/history/{id}` | Fetch historical sales |
| GET | `/sales/forecast/{id}` | Fetch forecasted sales |
| GET | `/metrics/model/` | Fetch metrics of the trained model |
| POST | `/sales/forecast/live/{id}` | Live retraining & update |

---

##  Dashboard Features

- Sales history visualization
- Forecast charts (weekly / monthly / yearly)
- KPI cards (Revenue, Growth, Accuracy)
- Product heatmap & leaderboard
- Export forecasts to CSV

---

##  Project Status

- [x] Data pipeline implemented  
- [x] Hybrid model trained and validated  
- [x] Backend API developed  
- [x] Frontend dashboard integrated  
- [x] Ready for deployment  
- [x] We were able to achieve more than 84% accuracy with new Ensemble model from the initial 71% accuracy by only using LightGBM
---

##  Use Cases

- Demand forecasting
- Inventory planning
- Supply chain optimization
- Business decision support

---

## 📄 License

This project is developed for **academic and learning purposes Under MIT LICENSE**.
