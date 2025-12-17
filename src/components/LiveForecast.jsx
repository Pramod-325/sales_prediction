import React, { useState } from "react";
import { api, endpoints } from "../services/api.js";
import { Sparkles, Zap } from "lucide-react";
import { LineChart, Line, XAxis, Tooltip, ResponsiveContainer } from "recharts";

export default function LiveForecast() {
  const [productId, setProductId] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleLiveUpdate = async () => {
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const res = await api.post(endpoints.live(productId.toUpperCase()));
      setResult(res.data);
    } catch (err) {
      if (err.response && err.response.status === 404) {
        setError("Product not found. Please check the Product ID.");
      } else {
        setError("Error triggering live forecast. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#003f66] p-3">
      <nav className="border-b border-[#7cb342]/20 bg-[#003f66] rounded-2xl mb-6 px-6 py-2">
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
      </nav>

      <div className="max-w-4xl mx-auto">
        <div className="flex items-center gap-2 text-sm mb-4">
          <a href="/" className="text-gray-400 hover:text-[#9ccc65]">Home</a>
          <span className="text-gray-600">/</span>
          <span className="text-white font-medium">Live Forecast</span>
        </div>

        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 rounded-xl bg-[#7cb342]">
            <Zap className="h-6 w-6 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-white">Live Model Retraining</h1>
        </div>

        <div className="mb-6 bg-[#7cb342]/10 border border-[#7cb342]/30 rounded-xl p-4">
          <p className="text-[#9ccc65] text-sm">
            <strong>Note:</strong> Future updates will allow uploading CSVs for new products. Currently, this triggers a re-train on existing database history.
          </p>
        </div>

        <div className="bg-[#004d7a] rounded-2xl border-2 border-[#005a8c] p-6">
          <div className="mb-6">
            <label className="block text-white text-sm font-medium mb-2">Product ID</label>
            <div className="flex gap-3">
              <input
                type="text"
                value={productId}
                onChange={(e) => setProductId(e.target.value.toUpperCase())}
                disabled={loading}
                placeholder="Enter product ID..."
                className="flex-1 bg-[#003f66] border border-[#7cb342]/30 rounded-xl px-4 py-2 text-white placeholder-gray-400 focus:outline-none focus:border-[#7cb342] disabled:opacity-50 uppercase"
              />
              <button
                onClick={handleLiveUpdate}
                disabled={loading || !productId}
                className="px-6 py-2 bg-[#7cb342] text-white font-semibold rounded-xl hover:bg-[#9ccc65] hover:shadow-lg hover:shadow-[#7cb342]/30 transition-all disabled:opacity-50 flex items-center gap-2"
              >
                {loading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    Retraining...
                  </>
                ) : (
                  <>Generate Forecast</>
                )}
              </button>
            </div>
          </div>

          {error && (
            <div className="mb-4 bg-red-500/10 border border-red-400/30 rounded-xl p-2">
              <p className="text-red-300 text-sm">{error}</p>
            </div>
          )}

          {result && (
            <div>
              <div className="mb-4 bg-[#7cb342]/10 border border-[#7cb342]/30 rounded-xl p-2">
                <p className="text-[#9ccc65] text-sm">{result.message}</p>
              </div>

              <div className="bg-[#003f66] border border-[#7cb342]/20 rounded-xl p-2">
                <h3 className="text-lg font-bold text-white mb-3">Forecast Preview (Next 2 Years)</h3>
                <div className="bg-[#002640] rounded-lg p-2" style={{ height: '200px' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={result.forecast}>
                      <XAxis dataKey="date" hide />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#003f66',
                          border: '1px solid #7cb342',
                          borderRadius: '8px',
                          color: '#ffffff'
                        }}
                      />
                      <Line type="monotone" dataKey="sales" stroke="#9ccc65" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}