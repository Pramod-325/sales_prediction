import React, { useEffect, useState } from "react";
import { api, endpoints } from "../services/api.js";
import { Activity, AlertCircle, Target, Cpu } from "lucide-react";

const MetricCard = ({ title, value, subtext, icon: Icon, gradient }) => (
  <div className="group relative bg-[#004d7a] rounded-2xl border-2 border-[#005a8c] p-6 overflow-hidden hover:border-[#7cb342] hover:shadow-2xl hover:shadow-[#7cb342]/30 transition-all">
    <div className={`absolute inset-0 bg-gradient-to-br ${gradient} opacity-0 group-hover:opacity-10 transition-opacity`}></div>
    <div className="relative z-10">
      <div className="flex items-center justify-between mb-4">
        <p className="text-gray-300 text-sm font-medium">{title}</p>
        <div className={`p-3 bg-gradient-to-br ${gradient} rounded-xl shadow-lg`}>
          <Icon className="h-6 w-6 text-white" />
        </div>
      </div>
      <p className="text-4xl font-bold text-white mb-2">{value}</p>
      <p className="text-gray-400 text-sm">{subtext}</p>
    </div>
  </div>
);

export default function ModelMetrics() {
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.get(endpoints.metrics)
      .then((res) => setMetrics(res.data))
      .catch((err) => {
        if (err.response && err.response.status === 404) {
          setError("Metrics not found. Please ensure the model has been trained.");
        } else {
          setError("Error loading metrics. Please try again later.");
        }
      });
  }, []);

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
          <span className="text-white font-medium">Model Metrics</span>
        </div>

        {/* Header */}
        <div className="flex items-center gap-3 mb-8">
          <div className="p-2 rounded-xl bg-[#7cb342]">
            <Activity className="h-6 w-6 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-white">Model Health & Performance</h1>
        </div>

        {/* Error State */}
        {error && (
          <div className="bg-red-500/10 border border-red-400/30 rounded-xl p-4 mb-6">
            <p className="text-red-300">{error}</p>
          </div>
        )}

        {/* Loading State */}
        {!metrics && !error ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <div className="w-16 h-16 border-4 border-[#7cb342]/30 border-t-[#7cb342] rounded-full animate-spin mx-auto mb-4"></div>
              <p className="text-gray-400">Loading metrics...</p>
            </div>
          </div>
        ) : metrics && (
          <>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              
              {/* Forecast Accuracy Card */}
              <MetricCard
                title="Forecast Accuracy"
                value={`${(metrics.accuracy).toFixed(1)}%`}
                subtext="1 - Accuracy (Higher is better)"
                icon={Target}
                gradient="from-[#7cb342] to-[#9ccc65]"
              />

              {/* Error Rate Card */}
              <MetricCard
                title="Error Rate (WMAPE)"
                value={`${(metrics.wmape).toFixed(1)}%`}
                subtext="Weighted Mean Abs Percentage Error"
                icon={AlertCircle}
                gradient="from-[#558b2f] to-[#7cb342]"
              />

              {/* Model Version Card */}
              <MetricCard
                title="Model Version"
                value={metrics.model_version}
                subtext={`Last Trained: ${new Date(metrics.last_trained).toLocaleDateString()}`}
                icon={Cpu}
                gradient="from-[#9ccc65] to-[#aed581]"
              />
            </div>

            {/* Model Status Section */}
            <div className="mt-8 bg-[#004d7a] rounded-2xl border-2 border-[#005a8c] p-6">
              <div className="flex items-center gap-3 mb-6">
                <div className="p-3 bg-[#7cb342] rounded-xl shadow-lg">
                  <Activity className="h-6 w-6 text-white" />
                </div>
                <h2 className="text-2xl font-bold text-white">Model Status</h2>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Status Indicator */}
                <div className="bg-[#003f66] rounded-xl p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="h-3 w-3 bg-[#7cb342] rounded-full animate-pulse"></div>
                    <p className="text-white font-semibold">Model Status: Active</p>
                  </div>
                  <p className="text-gray-400 text-sm mb-2">
                    The model is currently running and generating predictions with high accuracy.
                  </p>
                  <div className="flex items-center gap-2 text-[#9ccc65] text-sm mt-4">
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                    <span>All systems operational</span>
                  </div>
                </div>

                {/* Performance Summary */}
                <div className="bg-[#003f66] rounded-xl p-6">
                  <p className="text-white font-semibold mb-4">Performance Summary</p>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400 text-sm">Accuracy Level</span>
                      <span className="text-[#9ccc65] font-semibold">{metrics.accuracy >= 85 ? 'Excellent' : metrics.accuracy >= 70 ? 'Good' : 'Fair'}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400 text-sm">Error Rate</span>
                      <span className="text-[#7cb342] font-semibold">{metrics.wmape < 10 ? 'Low' : metrics.wmape < 20 ? 'Moderate' : 'High'}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400 text-sm">Model Age</span>
                      <span className="text-[#9ccc65] font-semibold">
                        {Math.floor((new Date() - new Date(metrics.last_trained)) / (1000 * 60 * 60 * 24))} days
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}