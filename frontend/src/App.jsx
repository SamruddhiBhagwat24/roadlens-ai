import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import ImageUpload from './components/ImageUpload';
import DetectionCanvas from './components/DetectionCanvas';
import ImageQualityCard from './components/ImageQualityCard';
import RoadIntelligenceCard from './components/RoadIntelligenceCard';
import DetectionsList from './components/DetectionsList';
import PerformanceCard from './components/PerformanceCard';
import { checkHealth, analyzeRoadImage } from './services/api';

export default function App() {
  const [backendHealth, setBackendHealth] = useState(null);
  const [selectedCountry, setSelectedCountry] = useState('usa');
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [apiError, setApiError] = useState(null);

  useEffect(() => {
    // Initial health check
    checkHealth().then(setBackendHealth);
    const interval = setInterval(() => {
      checkHealth().then(setBackendHealth);
    }, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleImageSelected = (file) => {
    setSelectedFile(file);
    setAnalysisResult(null);
    setApiError(null);

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }

    if (file) {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setApiError(null);

    try {
      const result = await analyzeRoadImage(selectedFile, selectedCountry);
      setAnalysisResult(result);
    } catch (err) {
      setApiError(err.message || 'An error occurred during perception analysis');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      <Navbar backendHealth={backendHealth} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
        {/* API Error Toast / Banner */}
        {apiError && (
          <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-2xl text-rose-300 flex items-start gap-3 shadow-lg">
            <svg className="w-5 h-5 text-rose-400 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div className="flex-1 text-sm">
              <span className="font-bold block">Analysis Error</span>
              <span>{apiError}</span>
            </div>
            <button
              onClick={() => setApiError(null)}
              className="text-rose-400 hover:text-rose-200 text-xs px-2 py-1"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Top Section: Upload & Visual Detection Canvas */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-4 space-y-6">
            {/* Road Standard Selector */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl">
              <div className="flex items-center justify-between mb-2">
                <label
                  htmlFor="road-standard-selector"
                  className="text-xs font-semibold tracking-wide text-slate-300 uppercase flex items-center gap-2"
                >
                  <svg className="w-4 h-4 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Road Standard
                </label>
                <span className="text-[10px] px-2 py-0.5 rounded-full font-medium bg-slate-800 text-slate-400 border border-slate-700/50">
                  {selectedCountry === 'usa' ? 'MUTCD' : 'IRC / MoRTH'}
                </span>
              </div>
              <div className="relative">
                <select
                  id="road-standard-selector"
                  value={selectedCountry}
                  onChange={(e) => setSelectedCountry(e.target.value)}
                  className="w-full bg-slate-800/80 border border-slate-700 hover:border-slate-600 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 rounded-xl px-3.5 py-2 text-sm text-slate-200 font-medium appearance-none cursor-pointer transition shadow-inner outline-none pr-10"
                >
                  <option value="usa" className="bg-slate-900 text-slate-200">
                    🇺🇸 USA — MUTCD
                  </option>
                  <option value="india" className="bg-slate-900 text-slate-200">
                    🇮🇳 India — IRC / MoRTH
                  </option>
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3.5 text-slate-400">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
              </div>
            </div>

            <ImageUpload
              onImageSelected={handleImageSelected}
              onAnalyze={handleAnalyze}
              isAnalyzing={isAnalyzing}
              selectedFile={selectedFile}
            />

            {/* Quick Context Card */}
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4 text-xs text-slate-400 space-y-2">
              <p className="font-semibold text-slate-300 uppercase tracking-wider text-[10px]">
                Mini Challenge 2 — USA Benchmark Targets
              </p>
              <ul className="list-disc list-inside space-y-1 text-slate-400">
                <li>US License Plates (multi-state characters)</li>
                <li>Stop Signs & Speed Limit Signs</li>
                <li>Advisory Speed Plaques (number-only)</li>
                <li>Work-Zone & Multi-line Warning Signs</li>
              </ul>
            </div>
          </div>

          <div className="lg:col-span-8">
            <DetectionCanvas
              imageSrc={previewUrl}
              detections={analysisResult?.detections || []}
              imageDimensions={analysisResult?.image_dimensions}
            />
          </div>
        </div>

        {/* Bottom Section: Perception Results & Telemetry */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <RoadIntelligenceCard
            roadIntelligence={analysisResult?.road_intelligence}
          />

          <ImageQualityCard
            imageQuality={analysisResult?.image_quality}
          />

          <PerformanceCard
            performance={analysisResult?.performance}
          />
        </div>

        {/* Full Entities & Text Localized Table */}
        <DetectionsList
          detections={analysisResult?.detections || []}
          ocrResults={analysisResult?.ocr_results || []}
        />
      </main>

      <footer className="border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-600">
        <p>RoadLens AI • Automated Driving & ADAS Road Perception Research Prototype</p>
        <p className="mt-1">Research prototype only • No vehicle control • Zero fabricated metrics policy</p>
      </footer>
    </div>
  );
}
