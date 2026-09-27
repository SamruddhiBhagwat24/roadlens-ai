import React from 'react';

const SEVERITY_COLORS = {
  low: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
  normal: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
  medium: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
  moderate_skew: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
  high: 'text-rose-400 bg-rose-500/10 border-rose-500/20',
  high_skew: 'text-rose-400 bg-rose-500/10 border-rose-500/20',
};

export default function ImageQualityCard({ imageQuality }) {
  if (!imageQuality) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
        <h2 className="text-sm font-semibold tracking-wide text-slate-200 uppercase mb-3 flex items-center gap-2">
          <svg className="w-4 h-4 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          Adverse Visual Conditions
        </h2>
        <p className="text-xs text-slate-500">Awaiting analysis metrics...</p>
      </div>
    );
  }

  const { blur, blur_score, brightness, brightness_score, noise, noise_score, glare, glare_score, perspective, perspective_score, adverse_conditions } = imageQuality;

  const metrics = [
    { label: 'Sharpness / Blur', level: blur, score: `${blur_score} var` },
    { label: 'Illumination', level: brightness, score: `${brightness_score} avg` },
    { label: 'Sensor Noise', level: noise, score: `σ ${noise_score}` },
    { label: 'Glare / Saturation', level: glare, score: `${glare_score}%` },
    { label: 'Perspective Skew', level: perspective, score: `${perspective_score}°` },
  ];

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold tracking-wide text-slate-200 uppercase flex items-center gap-2">
          <svg className="w-4 h-4 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          Adverse Condition Assessment
        </h2>
        <span className="text-[11px] text-slate-400 font-mono">Module 2 Quality Intelligence</span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        {metrics.map((m, idx) => (
          <div key={idx} className="bg-slate-950/70 border border-slate-800/80 rounded-xl p-3">
            <p className="text-[11px] font-medium text-slate-400 truncate">{m.label}</p>
            <div className="mt-2 flex items-center justify-between">
              <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border uppercase ${SEVERITY_COLORS[m.level] || 'text-slate-300'}`}>
                {m.level}
              </span>
              <span className="text-[11px] font-mono text-slate-500">{m.score}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Active Adverse Condition Warnings */}
      <div className="mt-4 pt-3 border-t border-slate-800/80">
        <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-2">Detected Adverse Conditions</p>
        {adverse_conditions && adverse_conditions.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {adverse_conditions.map((cond, i) => (
              <span key={i} className="text-xs px-2.5 py-1 rounded-lg bg-rose-500/10 text-rose-400 border border-rose-500/20 font-medium flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-rose-400 animate-ping"></span>
                {cond.replace('_', ' ')}
              </span>
            ))}
          </div>
        ) : (
          <div className="flex items-center gap-2 text-xs text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-3 py-1.5 rounded-lg">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            Optimal scene visibility — no severe degradation triggers detected
          </div>
        )}
      </div>
    </div>
  );
}
