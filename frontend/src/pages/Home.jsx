import { Link } from "react-router-dom";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-100 to-purple-200 flex flex-col items-center justify-center p-6">
      
      <h1 className="text-4xl font-bold mb-2">
        Sales Prediction Dashboard
      </h1>
      <p className="text-gray-700 mb-10">
        ML-powered sales forecasting & analytics
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-4xl">
        
        <Card title="Sales History" path="/history" />
        <Card title="Sales Forecast" path="/forecast" />
        <Card title="Live Forecast" path="/live" />
        <Card title="Model Metrics" path="/metrics" />

      </div>
    </div>
  );
}

function Card({ title, path }) {
  return (
    <Link to={path}>
      <div className="bg-white p-6 rounded-xl shadow-lg hover:scale-105 transition transform cursor-pointer">
        <h2 className="text-xl font-semibold">{title}</h2>
        <p className="text-gray-600 mt-2">
          View {title.toLowerCase()}
        </p>
      </div>
    </Link>
  );
}
