import React, { useState } from "react";
import {
  Box,
  Button,
  TextField,
  Typography,
  Paper,
  Stack,
} from "@mui/material";
import { DataGrid } from "@mui/x-data-grid";
import { BarChart } from "@mui/x-charts/BarChart";
import { dummyForecast } from "../dummyData.js";

export default function SalesForecast() {
  const [rows, setRows] = useState([]);

  const loadDummyForecast = () => {
    setRows(
      dummyForecast.map((r, i) => ({
        id: i,
        date: r.forecast_date,
        prediction: r.predicted_sales,
      }))
    );
  };

  return (
    <Box
      sx={{
        minHeight: "100vh",
        background: "linear-gradient(135deg, #e8f5e9, #e3f2fd)",
        p: 4,
      }}
    >
      {/* Page Header */}
      <Typography variant="h4" fontWeight="bold" gutterBottom>
        Sales Forecast
      </Typography>
      <Typography color="text.secondary" mb={3}>
        Predicted sales for the next two years using ML models
      </Typography>

      {/* Card */}
      <Paper elevation={4} sx={{ p: 3, borderRadius: 3 }}>
        {/* Controls */}
        <Stack direction="row" spacing={2} mb={4}>
          <TextField
            label="Product ID"
            value="P12"
            disabled
            size="small"
          />
          <Button
            variant="contained"
            onClick={loadDummyForecast}
            sx={{ px: 4 }}
          >
            Load Dummy Forecast
          </Button>
        </Stack>

        {/* Chart */}
        {rows.length > 0 && (
          <Box mb={4}>
            <Typography variant="h6" mb={1}>
              Forecast Trend
            </Typography>
            <BarChart
              xAxis={[{ scaleType: "band", data: rows.map(r => r.date) }]}
              series={[{ data: rows.map(r => r.prediction) }]}
              height={300}
            />
          </Box>
        )}

        {/* Table */}
        <Typography variant="h6" mb={1}>
          Forecast Table
        </Typography>
        <DataGrid
          rows={rows}
          columns={[
            { field: "date", headerName: "Forecast Date", width: 220 },
            { field: "prediction", headerName: "Predicted Sales", width: 180 },
          ]}
          autoHeight
          pageSizeOptions={[5]}
          disableRowSelectionOnClick
          sx={{
            borderRadius: 2,
            "& .MuiDataGrid-columnHeaders": {
              backgroundColor: "#f5f5f5",
              fontWeight: "bold",
            },
          }}
        />
      </Paper>
    </Box>
  );
}
