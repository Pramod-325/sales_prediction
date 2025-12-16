import React from "react";
import { Box, Paper, Typography, Divider } from "@mui/material";
import { dummyMetrics } from "../dummyData.js";

export default function ModelMetrics() {
  return (
    <Box
      sx={{
        minHeight: "100vh",
        background: "linear-gradient(135deg, #fffde7, #fff3e0)",
        p: 4,
      }}
    >
      {/* Page Header */}
      <Typography variant="h4" fontWeight="bold" gutterBottom>
        Model Metrics
      </Typography>
      <Typography color="text.secondary" mb={3}>
        Performance and training details of the ML model
      </Typography>

      {/* Card */}
      <Paper
        elevation={4}
        sx={{
          p: 4,
          borderRadius: 3,
          maxWidth: 500,
        }}
      >
        <Typography variant="h6" fontWeight="bold" gutterBottom>
          Model Performance
        </Typography>

        <Divider sx={{ mb: 2 }} />

        <Typography sx={{ mb: 1 }}>
          <b>Model Version:</b> {dummyMetrics.model_version}
        </Typography>

        <Typography sx={{ mb: 1 }}>
          <b>Accuracy:</b> {dummyMetrics.accuracy}%
        </Typography>

        <Typography sx={{ mb: 1 }}>
          <b>WMAPE:</b> {dummyMetrics.wmape}
        </Typography>

        <Typography sx={{ mb: 1 }}>
          <b>Last Trained:</b> {dummyMetrics.last_trained}
        </Typography>
      </Paper>
    </Box>
  );
}
