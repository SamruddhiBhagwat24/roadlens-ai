import React from 'react';

export default function Navbar({ backendHealth }) {
  const isHealthy = backendHealth?.status === 'healthy';
  const isRocm = backendHealth?.rocm_available;

  return (
    <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/20">
            <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xl font-bold tracking-tight text-white">RoadLens<span className="text-blue-500">.AI</span></span>
              <span className="text-[10px] px-2 py-0.5 rounded-full font-semibold uppercase tracking-wider bg-blue-500/10 text-blue-400 border border-blue-500/20">
                Milestone 1
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">AI-Powered Road Perception & Road Intelligence</p>
          </div>
        </div>

        {/* Backend / Accelerator Status Badges */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2 px-3 py-1 rounded-lg bg-slate-800/80 border border-slate-700/60 text-xs">
            <span className={`w-2 h-2 rounded-full ${isHealthy ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`}></span>
            <span className="text-slate-300 font-medium">
              {isHealthy ? (backendHealth?.device || 'Backend Online') : 'Backend Standby'}
            </span>
          </div>

          <div className="flex items-center space-x-1.5 px-3 py-1 rounded-lg border text-xs font-semibold">
            {isRocm ? (
              <span className="text-rose-400 bg-rose-500/10 border-rose-500/20 px-2 py-0.5 rounded flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-rose-500"></span>
                AMD ROCm Active
              </span>
            ) : (
              <span className="text-slate-400 bg-slate-800/60 border-slate-700/40 px-2 py-0.5 rounded flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-slate-400"></span>
                CPU / Non-ROCm
              </span>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
