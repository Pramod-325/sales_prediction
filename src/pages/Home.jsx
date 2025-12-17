import { Link } from "react-router";
import { TrendingUp, History, Zap, BarChart3 } from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-screen bg-[#003f66] relative overflow-hidden">
      
      {/* Animated background effects */}
      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiNmZmIiIGZpbGwtb3BhY2l0eT0iMC4wMiI+PHBhdGggZD0iTTM2IDM0di00aC0ydjRoLTR2Mmg0djRoMnYtNGg0di0yaC00em0wLTMwVjBoLTJ2NGgtNHYyaDR2NGgyVjZoNFY0aC00ek02IDM0di00SDR2NGgwdjJoNHY0aDJ2LTRoNHYtMkg2ek02IDRWMEG0djRIMHYyaDR2NGgyVjZoNFY0SDZ6Ii8+PC9nPjwvZz48L3N2Zz4=')] opacity-30"></div>
      
      {/* Elegant decorative orbs */}
      <div className="absolute top-20 left-10 w-96 h-96 bg-[#7cb342]/10 rounded-full blur-3xl"></div>
      <div className="absolute bottom-20 right-10 w-80 h-80 bg-[#9ccc65]/10 rounded-full blur-3xl"></div>
      
      {/* Premium Navbar */}
      <nav className="relative z-20 border-b border-[#7cb342]/20 bg-[#003f66]">
        <div className="max-w-7xl mx-auto px-6 py-5">
          <div className="flex items-center justify-between">
            {/* Logo Section - Premium Highlight */}
            <div className="flex items-center gap-4">
              <div className="relative">
                {/* Premium glow effect */}
                <div className="absolute inset-0 bg-white/20 rounded-xl blur-md opacity-50"></div>
                <div className="relative bg-white p-3 rounded-xl shadow-2xl">
                  <img 
                    src="/Aspyr-logo.svg" 
                    alt="Aspyr Labs" 
                    className="h-10 w-10"
                  />
                </div>
              </div>
              <div>
                <h2 className="text-2xl font-bold">
                  <span className="text-[#9ccc65]">Aspyr</span>
                  <span className="text-white">Labs</span>
                </h2>
                <p className="text-sm text-[#7cb342] font-medium">Analytics Platform</p>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="relative z-10 flex flex-col items-center justify-center px-6 py-8">
        {/* Dashboard Header */}
        <div className="text-center mb-6">
          <h1 className="text-4xl font-bold mb-2">
            <span className="text-[#9ccc65]">Sales Prediction</span>
            <span className="text-white"> Dashboard</span>
          </h1>
          <p className="text-gray-300 text-base">
            AI-powered sales forecasting & business intelligence
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-5 w-full max-w-4xl">
          <Card 
            title="Sales Forecast" 
            path="/forecast" 
            icon={<TrendingUp className="w-7 h-7" />} 
            description="Advanced AI predictions for future sales"
          />
          <Card 
            title="Sales History" 
            path="/history" 
            icon={<History className="w-7 h-7" />} 
            description="Comprehensive historical data analysis"
          />
          
          <Card 
            title="Live Forecast" 
            path="/live" 
            icon={<Zap className="w-7 h-7" />} 
            description="Real-time market predictions & insights"
          />
          <Card 
            title="Model Metrics" 
            path="/metrics" 
            icon={<BarChart3 className="w-7 h-7" />} 
            description="Detailed performance analytics & KPIs"
          />
        </div>
      </div>
    </div>
  );
}

function Card({ title, path, icon, description }) {
  return (
    <Link to={path}>
      <div className="group relative bg-[#004d7a] p-6 rounded-2xl border-2 border-[#005a8c] hover:border-[#7cb342] shadow-xl hover:shadow-2xl hover:shadow-[#7cb342]/30 hover:scale-105 hover:-translate-y-1 transition-all duration-300 cursor-pointer overflow-hidden">
        
        {/* Subtle hover glow */}
        <div className="absolute inset-0 bg-[#7cb342] opacity-0 group-hover:opacity-5 transition-opacity duration-300"></div>
        
        {/* Content */}
        <div className="relative z-10">
          <div className="inline-flex p-3 rounded-xl bg-[#7cb342] mb-3 shadow-xl group-hover:shadow-2xl group-hover:shadow-[#7cb342]/50 group-hover:scale-110 transition-all duration-300">
            <div className="text-white">
              {icon}
            </div>
          </div>
          <h2 className="text-xl font-bold text-white mb-2 group-hover:text-[#9ccc65] transition-colors">{title}</h2>
          <p className="text-gray-300 text-sm leading-relaxed">
            {description}
          </p>
          <div className="mt-4 flex items-center font-semibold text-[#9ccc65] group-hover:gap-3 gap-2 transition-all duration-300">
            <span className="text-sm">Explore Now</span>
            <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M9 5l7 7-7 7" />
            </svg>
          </div>
        </div>
      </div>
    </Link>
  );
}