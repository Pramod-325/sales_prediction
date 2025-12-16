import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import SalesHistory from "./components/SalesHistory";
import SalesForecast from "./components/SalesForecast";
import LiveForecast from "./components/LiveForecast";
import ModelMetrics from "./components/ModelMetrics";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/history" element={<SalesHistory />} />
        <Route path="/forecast" element={<SalesForecast />} />
        <Route path="/live" element={<LiveForecast />} />
        <Route path="/metrics" element={<ModelMetrics />} />
      </Routes>
    </BrowserRouter>
  );
}
