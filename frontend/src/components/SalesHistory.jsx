import React, { useState } from "react";
import { api, endpoints } from "../services/api";
import { getTotalRevenue, calculateGrowth } from "../utils/dataHelpers";
import { History, DollarSign, TrendingUp, ArrowLeft, AlertCircle } from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export default function SalesHistory() {
  const [productId, setProductId] = useState("P1");
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.get(endpoints.history(productId.toUpperCase()));
      const formatted = res.data.map((item) => ({
        date: item.ds.split("T")[0],
        sales: item.y,
      }));
      setData(formatted);
    } catch (err) {
      console.error("Failed to fetch history", err);
      if (err.response && err.response.status === 404) {
        setError("Product not found");
      } else {
        setError("Failed to fetch history");
      }
      setData([]);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e) => {
    setProductId(e.target.value.toUpperCase());
    setError(null);
  };

  return (
    <div className="min-h-screen bg-[#003f66] p-3">
      {/* Navbar */}
      <nav className="border-b border-[#7cb342]/20 bg-[#003f66] rounded-2xl mb-6 px-6 py-2">
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
          
          {/* <a 
            href="/" 
            className="flex items-center gap-2 px-4 py-2 bg-[#004d7a] hover:bg-[#005a8c] text-white rounded-xl border border-[#7cb342]/30 hover:border-[#7cb342] transition-all text-sm"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Home
          </a> */}
        </div>
      </nav>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto">
        
        {/* Breadcrumbs */}
        <div className="flex items-center gap-2 text-sm mb-4">
          <a href="/" className="text-gray-400 hover:text-[#9ccc65]">Home</a>
          <span className="text-gray-600">/</span>
          <span className="text-white font-medium">Sales History</span>
        </div>

        {/* Header */}
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 rounded-xl bg-[#7cb342]">
            <History className="h-6 w-6 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-white">Historical Performance</h1>
        </div>

        {/* Controls */}
        <div className="bg-[#004d7a] rounded-2xl border-2 border-[#005a8c] p-6 mb-6">
          <div className="mb-6">
            <label className="block text-white text-sm font-medium mb-2">Product ID</label>
            <div className="flex gap-3">
              <input
                type="text"
                value={productId}
                onChange={handleInputChange}
                disabled={loading}
                placeholder="Enter product ID..."
                className="flex-1 bg-[#003f66] border border-[#7cb342]/30 rounded-xl px-4 py-2 text-white placeholder-gray-400 focus:outline-none focus:border-[#7cb342] disabled:opacity-50 uppercase"
              />
              <button
                onClick={fetchData}
                disabled={loading}
                className="px-6 py-2 bg-[#7cb342] text-white font-semibold rounded-xl hover:bg-[#9ccc65] hover:shadow-lg hover:shadow-[#7cb342]/30 transition-all disabled:opacity-50 flex items-center gap-2"
              >
                {loading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    Loading...
                  </>
                ) : (
                  <>Analyze Product</>
                )}
              </button>
            </div>
          </div>

          {/* Error State */}
          {error && (
            <div className="mb-4 bg-red-500/10 border border-red-400/30 rounded-xl p-2">
              <p className="text-red-300 text-sm">{error}</p>
            </div>
          )}

          {data.length > 0 && (
            <>
              {/* KPI Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                
                {/* Total Sales Card */}
                <div className="bg-[#003f66] border border-[#7cb342]/20 rounded-xl p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <DollarSign className="h-5 w-5 text-[#9ccc65]" />
                    <h3 className="text-sm font-semibold text-white">Total Sales (1 Year)</h3>
                  </div>
                  <p className="text-3xl font-bold text-white">
                    {Number(getTotalRevenue(data)).toLocaleString()}
                  </p>
                  <p className="text-[#9ccc65] text-xs mt-1">Units sold in the past year</p>
                </div>

                {/* Growth Card */}
                <div className="bg-[#003f66] border border-[#7cb342]/20 rounded-xl p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <TrendingUp className="h-5 w-5 text-[#9ccc65]" />
                    <h3 className="text-sm font-semibold text-white">YoY Growth Trend</h3>
                  </div>
                  <p className="text-3xl font-bold text-[#9ccc65]">
                    {calculateGrowth(data)}%
                  </p>
                  <p className="text-[#9ccc65] text-xs mt-1">Year-over-year growth rate</p>
                </div>
              </div>

              {/* Chart Section */}
              <div className="bg-[#003f66] border border-[#7cb342]/20 rounded-xl p-4">
                <h3 className="text-lg font-bold text-white mb-3">Weekly Sales Volume</h3>
                <div className="bg-[#002640] rounded-lg p-2" style={{ height: '300px' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data}>
                      <defs>
                        <linearGradient id="colorSales" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#7cb342" stopOpacity={0.8} />
                          <stop offset="95%" stopColor="#7cb342" stopOpacity={0.1} />
                        </linearGradient>
                      </defs>
                      <XAxis 
                        dataKey="date" 
                        stroke="#ffffff80"
                        style={{ fontSize: '12px' }}
                      />
                      <YAxis 
                        stroke="#ffffff80"
                        style={{ fontSize: '12px' }}
                      />
                      <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#003f66',
                          border: '1px solid #7cb342',
                          borderRadius: '8px',
                          color: '#ffffff'
                        }}
                        formatter={(value) => [`${value} units`, 'Sales']}
                      />
                      <Area
                        type="monotone"
                        dataKey="sales"
                        stroke="#7cb342"
                        strokeWidth={2}
                        fillOpacity={1}
                        fill="url(#colorSales)"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                
                <div className="mt-3 pt-3 border-t border-[#7cb342]/20">
                  <p className="text-gray-400 text-xs">
                    Showing data for Product ID: <span className="text-[#9ccc65] font-semibold">{productId}</span>
                  </p>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Empty State */}
        {data.length === 0 && !loading && !error && (
          <div className="text-center py-12 bg-[#004d7a] rounded-2xl border-2 border-[#005a8c]">
            <div className="inline-flex p-4 bg-[#003f66] rounded-xl border border-[#7cb342]/20 mb-4">
              <History className="h-12 w-12 text-[#7cb342]" />
            </div>
            <h3 className="text-xl font-bold text-white mb-2">No Data Loaded</h3>
            <p className="text-gray-400 text-sm">Enter a Product ID and click "Analyze Product" to view historical data</p>
          </div>
        )}
      </div>
    </div>
  );
}