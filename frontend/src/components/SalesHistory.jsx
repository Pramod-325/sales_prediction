import React, { useState } from "react";
// import api from "../api";
import { dummySalesHistory } from "../dummyData.js";
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

export default function SalesHistory() {
  const [productId, setProductId] = useState("");
  const [rows, setRows] = useState([]);

  const USE_DUMMY = true; // toggle switch

  const fetchHistory = async () => {
    if (USE_DUMMY) {
      setRows(
        dummySalesHistory.map((r, i) => ({
          id: i,
          date: r.ds,
          sales: r.y,
        }))
      );
    } else {
      // const res = await api.get(`/sales/history/${productId}`);
      // setRows(res.data);
    }
  };

  return (
    <Box
      sx={{
        minHeight: "100vh",
        background: "linear-gradient(135deg, #e3f2fd, #f3e5f5)",
        p: 4,
      }}
    >
      {/* Page Title */}
      <Typography variant="h4" fontWeight="bold" gutterBottom>
        Sales History
      </Typography>
      <Typography color="text.secondary" mb={3}>
        View historical sales data for selected products
      </Typography>

      {/* Card Container */}
      <Paper elevation={4} sx={{ p: 3, borderRadius: 3 }}>
        {/* Controls */}
        <Stack direction="row" spacing={2} mb={4}>
          <TextField
            label="Product ID"
            placeholder="e.g. P12"
            value={productId}
            onChange={(e) => setProductId(e.target.value)}
            size="small"
          />
          <Button
            variant="contained"
            size="medium"
            onClick={fetchHistory}
            sx={{ px: 4 }}
          >
            Fetch Data
          </Button>
        </Stack>

        {/* Chart */}
        {rows.length > 0 && (
          <Box mb={4}>
            <Typography variant="h6" mb={1}>
              Sales Trend
            </Typography>
            <BarChart
              xAxis={[{ scaleType: "band", data: rows.map((r) => r.date) }]}
              series={[{ data: rows.map((r) => r.sales) }]}
              height={300}
            />
          </Box>
        )}

        {/* Grid */}
        <Typography variant="h6" mb={1}>
          Sales Table
        </Typography>
        <DataGrid
          rows={rows}
          columns={[
            { field: "date", headerName: "Date", width: 200 },
            { field: "sales", headerName: "Sales", width: 150 },
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
