import React, { useState } from "react";
import { api, endpoints } from "../services/api.js";
import { aggregateData } from "../utils/dataHelpers.js";
import { TrendingUp, Calendar, BarChart3 } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

export default function SalesForecast() {
  const [productId, setProductId] = useState("");
  const [rawData, setRawData] = useState([]);
  const [view, setView] = useState("weekly");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchData = async () => {
    if (!productId.trim()) {
      setError("Please enter a Product ID");
      return;
    }
    
    setLoading(true);
    setError(null);
    setRawData([]);
    
    try {
      const res = await api.get(endpoints.forecast(productId.toUpperCase()));
      const formatted = res.data.map((item) => ({
        date: item.forecast_date.split("T")[0],
        sales: item.predicted_sales,
      }));
      setRawData(formatted);
    } catch (err) {
      if (err.response && err.response.status === 404) {
        setError("Product not found. Please check the Product ID.");
      } else {
        setError("Error loading forecast. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  const chartData = aggregateData(rawData, view);

  return (
    <div className="min-h-screen bg-[#003f66] relative overflow-hidden">
      
      {/* Animated background */}
      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiNmZmYiIGZpbGwtb3BhY2l0eT0iMC4wMyI+PHBhdGggZD0iTTM2IDM0di00aC0ydjRoLTR2Mmg0djRoMnYtNGg0di0yaC00em0wLTMwVjBoLTJ2NGgtNHYyaDR2NGgyVjZoNFY0aC00ek02IDM0di00SDR2NGgwdjJoNHY0aDJ2LTRoNHYtMkg2ek02IDRWMEG0djRIMHYyaDR2NGgyVjZoNFY0SDZ6Ii8+PC9nPjwvZz48L3N2Zz4=')] opacity-20"></div>

      {/* Navbar */}
      <nav className="relative z-20 border-b border-[#7cb342]/20 bg-[#003f66]">
        <div className="max-w-7xl mx-auto px-6 py-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="relative">
                <div className="absolute inset-0 bg-white/20 rounded-xl blur-md opacity-50"></div>
                <div className="relative bg-white p-2 rounded-xl shadow-2xl">
                  <img 
                    src="/Aspyr-logo.svg" 
                    alt="Aspyr Labs" 
                    className="h-8 w-8"
                  />
                </div>
              </div>
              <div>
                <h2 className="text-xl font-bold">
                  <span className="text-[#9ccc65]">Aspyr</span>
                  <span className="text-white">Labs</span>
                </h2>
                <p className="text-sm text-[#7cb342]">Analytics Platform</p>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="relative z-10 p-6 max-w-7xl mx-auto">
        
        {/* Breadcrumbs */}
        <div className="flex items-center gap-2 text-sm mb-6">
          <a href="/" className="text-gray-400 hover:text-[#9ccc65] transition-colors">Home</a>
          <span className="text-gray-600">/</span>
          <span className="text-white font-medium">Sales Forecast</span>
        </div>

        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-[#7cb342]">
              <TrendingUp className="h-6 w-6 text-white" />
            </div>
            <h1 className="text-4xl font-bold text-white">Future Demand Forecast</h1>
          </div>
        </div>

        {/* Control Panel */}
        <div className="bg-[#004d7a] rounded-2xl border-2 border-[#005a8c] p-6 mb-6">
          <div className="flex items-center justify-between flex-wrap gap-4">
            
            {/* Left Section - Input & Button */}
            <div className="flex items-center gap-4">
              <div>
                <label className="block text-white text-sm font-medium mb-2">Product ID</label>
                <input
                  type="text"
                  value={productId}
                  onChange={(e) => setProductId(e.target.value.toUpperCase())}
                  className="bg-[#003f66] border border-[#7cb342]/30 rounded-xl px-4 py-2 text-white placeholder-gray-400 focus:outline-none focus:border-[#7cb342] w-32 uppercase"
                  placeholder="e.g. P12"
                />
              </div>
              <button
                onClick={fetchData}
                disabled={loading}
                className="mt-6 px-6 py-2 bg-[#7cb342] text-white font-semibold rounded-xl hover:bg-[#9ccc65] hover:shadow-lg hover:shadow-[#7cb342]/30 transition-all flex items-center gap-2 disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    Loading...
                  </>
                ) : (
                  <>
                    <BarChart3 className="h-5 w-5" />
                    Generate Forecast
                  </>
                )}
              </button>
            </div>

            {/* Right Section - View Toggle */}
            {rawData.length > 0 && (
              <div className="flex items-center gap-2 bg-[#003f66] p-1 rounded-xl">
                <button
                  onClick={() => setView("weekly")}
                  className={`px-4 py-2 rounded-lg font-medium transition-all ${
                    view === "weekly"
                      ? "bg-[#7cb342] text-white shadow-lg"
                      : "text-gray-300 hover:text-white"
                  }`}
                >
                  Weekly Detail
                </button>
                <button
                  onClick={() => setView("monthly")}
                  className={`px-4 py-2 rounded-lg font-medium transition-all ${
                    view === "monthly"
                      ? "bg-[#7cb342] text-white shadow-lg"
                      : "text-gray-300 hover:text-white"
                  }`}
                >
                  Monthly View
                </button>
                <button
                  onClick={() => setView("yearly")}
                  className={`px-4 py-2 rounded-lg font-medium transition-all ${
                    view === "yearly"
                      ? "bg-[#7cb342] text-white shadow-lg"
                      : "text-gray-300 hover:text-white"
                  }`}
                >
                  Yearly View
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 bg-red-500/10 border border-red-400/30 rounded-xl p-4">
            <p className="text-red-300">{error}</p>
          </div>
        )}

        {/* Chart Section */}
        {rawData.length > 0 && (
          <div className="bg-[#004d7a] rounded-2xl border-2 border-[#005a8c] p-6">
            
            {/* Chart Header */}
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 rounded-xl bg-[#7cb342]">
                <Calendar className="h-5 w-5 text-white" />
              </div>
              <h2 className="text-2xl font-bold text-white">
                Projected Sales ({view})
              </h2>
            </div>

            {/* Chart Area with Recharts */}
            <div className="bg-[#002640] rounded-xl p-4" style={{ height: '500px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                  <XAxis 
                    dataKey="date" 
                    stroke="#ffffff80"
                    style={{ fontSize: '12px' }}
                  />
                  <YAxis 
                    stroke="#ffffff80"
                    style={{ fontSize: '12px' }}
                  />
                  <Tooltip 
                    formatter={(value) => [`${value.toFixed(0)} units`, "Predicted Sales"]}
                    contentStyle={{ 
                      backgroundColor: '#003f66',
                      border: '1px solid #7cb342',
                      borderRadius: '10px',
                      color: '#ffffff'
                    }}
                  />
                  <Legend 
                    wrapperStyle={{ color: '#ffffff' }}
                  />
                  <Bar 
                    dataKey="sales" 
                    fill="#9ccc65" 
                    name="Predicted Sales" 
                    radius={[4, 4, 0, 0]} 
                  />
                  <ReferenceLine 
                    y={chartData.reduce((a, b) => a + b.sales, 0) / chartData.length} 
                    label={{ value: "Avg", fill: '#7cb342', fontSize: 12 }} 
                    stroke="#7cb342" 
                    strokeDasharray="3 3" 
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}