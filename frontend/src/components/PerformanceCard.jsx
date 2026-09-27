import React from 'react';

export default function PerformanceCard({ performance }) {
  if (!performance) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
        <h2 className="text-sm font-semibold tracking-wide text-slate-200 uppercase mb-3 flex items-center gap-2">
          <svg className="w-4 h-4 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          Hardware Acceleration & Latency Telemetry
        </h2>
        <p className="text-xs text-slate-500">Awaiting inference metrics...</p>
      </div>
    );
  }

  const { device_type, device_name, rocm_enabled, latency_ms = {}, gpu_utilization, gpu_memory_mb, status_note } = performance;

  const stages = [
    { name: 'Quality Assessment', key: 'quality_check' },
    { name: 'Adaptive Preprocessing', key: 'preprocessing' },
    { name: 'Region Detection', key: 'detection' },
    { name: 'Modular OCR', key: 'ocr' },
    { name: 'Semantic Reasoning', key: 'intelligence' },
  ];

  const totalMs = latency_ms.total || Object.values(latency_ms).reduce((a, b) => a + b, 0);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold tracking-wide text-slate-200 uppercase flex items-center gap-2">
          <svg className="w-4 h-4 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          Hardware Acceleration & Latency Telemetry
        </h2>
        <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded border tracking-wider bg-slate-800 text-slate-300 border-slate-700">
          Zero Fake Metrics Guaranteed
        </span>
      </div>

      {/* Device Overview Banner */}
      <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-3.5 mb-4 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div>
          <span className="text-[10px] uppercase tracking-wider font-semibold text-slate-500 block">
            Compute Backend
          </span>
          <p className="text-sm font-bold text-white flex items-center gap-2 mt-0.5">
            {device_name}
            {rocm_enabled && (
              <span className="text-[10px] px-2 py-0.5 bg-rose-500/20 text-rose-300 border border-rose-500/30 rounded font-semibold">
                AMD ROCm
              </span>
            )}
          </p>
          <p className="text-xs text-slate-400 mt-1">{status_note}</p>
        </div>

        <div className="text-left sm:text-right">
          <span className="text-[10px] uppercase tracking-wider font-semibold text-slate-500 block">
            End-to-End Latency
          </span>
          <p className="text-2xl font-black text-amber-400 mt-0.5">
            {totalMs.toFixed(1)} <span className="text-xs font-semibold text-slate-400">ms</span>
          </p>
        </div>
      </div>

      {/* Stage-by-Stage Latency Breakdown */}
      <div className="space-y-2">
        <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-1">
          Pipeline Latency Profile
        </span>
        {stages.map((st) => {
          const val = latency_ms[st.key] || 0.0;
          const pct = totalMs > 0 ? Math.min(100, Math.round((val / totalMs) * 100)) : 0;

          return (
            <div key={st.key} className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-slate-300 font-medium">{st.name}</span>
                <span className="text-slate-400 font-mono">{val.toFixed(1)} ms ({pct}%)</span>
              </div>
              <div className="w-full bg-slate-950 rounded-full h-1.5 overflow-hidden border border-slate-800">
                <div
                  className="bg-amber-500/80 h-full rounded-full transition-all duration-300"
                  style={{ width: `${pct}%` }}
                ></div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Hardware Telemetry Note */}
      <div className="mt-4 pt-3 border-t border-slate-800 text-[11px] text-slate-500 flex items-center justify-between">
        <span>GPU VRAM: {gpu_memory_mb !== null ? `${gpu_memory_mb} MB` : 'N/A (CPU Mode)'}</span>
        <span>GPU Util: {gpu_utilization !== null ? `${gpu_utilization}%` : 'N/A (CPU Mode)'}</span>
      </div>
    </div>
  );
}
