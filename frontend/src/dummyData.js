export const dummySalesHistory = [
  { product_code: "P12", ds: "2025-01-07", y: 3 },
  { product_code: "P12", ds: "2025-01-14", y: 4 },
  { product_code: "P12", ds: "2025-01-21", y: 1 },
  { product_code: "P12", ds: "2025-01-28", y: 6 },
  { product_code: "P12", ds: "2025-02-04", y: 4 },
];

export const dummyForecast = [
  { product_code: "P12", forecast_date: "2026-01-04", predicted_sales: 2.52 },
  { product_code: "P12", forecast_date: "2026-01-11", predicted_sales: 1.01 },
  { product_code: "P12", forecast_date: "2026-01-18", predicted_sales: 0.79 },
  { product_code: "P12", forecast_date: "2026-01-25", predicted_sales: 0.96 },
  { product_code: "P12", forecast_date: "2026-02-01", predicted_sales: 1.73 },
];

export const dummyMetrics = {
  model_version: "Hybrid-Prophet-LGBM-v1",
  accuracy: 84.89,
  wmape: 15.11,
  last_trained: "2025-12-16T17:08:22",
};

export const dummyLiveForecast = {
  status: "success",
  message: "Live model loaded for P12. Pipeline ready for inference calculation.",
  steps_executed: [
    "Prophet Loaded",
    "Future DataFrame Created",
    "Trend Predicted",
  ],
};
