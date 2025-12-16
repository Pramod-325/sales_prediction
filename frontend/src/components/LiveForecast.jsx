import React, { useState } from "react";
import {
  Box,
  Button,
  TextField,
  Typography,
  List,
  ListItem,
  Paper,
  Stack,
} from "@mui/material";
import { dummyLiveForecast } from "../dummyData.js";

export default function LiveForecast() {
  const [result, setResult] = useState(null);

  const runDummyForecast = () => {
    setResult(dummyLiveForecast);
  };

  return (
    <Box
      sx={{
        minHeight: "100vh",
        background: "linear-gradient(135deg, #f3e5f5, #ede7f6)",
        p: 4,
      }}
    >
      {/* Page Header */}
      <Typography variant="h4" fontWeight="bold" gutterBottom>
        Live Forecast
      </Typography>
      <Typography color="text.secondary" mb={3}>
        Trigger live prediction pipeline for selected product
      </Typography>

      {/* Card */}
      <Paper elevation={4} sx={{ p: 3, borderRadius: 3, maxWidth: 600 }}>
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
            color="secondary"
            onClick={runDummyForecast}
            sx={{ px: 4 }}
          >
            Run Dummy Forecast
          </Button>
        </Stack>

        {/* Result */}
        {result && (
          <Box>
            <Typography
              variant="subtitle1"
              fontWeight="bold"
              color="success.main"
              mb={2}
            >
              {result.message}
            </Typography>

            <Typography variant="subtitle2" mb={1}>
              Pipeline Steps Executed:
            </Typography>

            <List
              sx={{
                backgroundColor: "#f9f9f9",
                borderRadius: 2,
                px: 2,
              }}
            >
              {result.steps_executed.map((step, i) => (
                <ListItem key={i}>• {step}</ListItem>
              ))}
            </List>
          </Box>
        )}
      </Paper>
    </Box>
  );
}
