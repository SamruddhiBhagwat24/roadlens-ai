import React from 'react';

export default function RoadIntelligenceCard({ roadIntelligence }) {
  if (!roadIntelligence) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
        <h2 className="text-sm font-semibold tracking-wide text-slate-200 uppercase mb-3 flex items-center gap-2">
          <svg className="w-4 h-4 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          Structured Road Intelligence
        </h2>
        <p className="text-xs text-slate-500">Awaiting visual perception synthesis...</p>
      </div>
    );
  }

  const { object_type, meaning, value, unit, detection_confidence, ocr_confidence, interpretation_confidence, advisory_notes, all_states } = roadIntelligence;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold tracking-wide text-slate-200 uppercase flex items-center gap-2">
          <svg className="w-4 h-4 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          Structured Road Intelligence
        </h2>
        <span className="text-[11px] text-blue-400 font-mono px-2 py-0.5 rounded bg-blue-500/10 border border-blue-500/20">
          Semantic Perception
        </span>
      </div>

      {/* Primary Interpretation Banner */}
      <div className="bg-gradient-to-r from-blue-950/60 to-slate-900 border border-blue-500/30 rounded-xl p-4 mb-4">
        <div className="flex items-start justify-between">
          <div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-blue-400">
              Primary Road Observation
            </span>
            <h3 className="text-lg font-bold text-white mt-1 capitalize">
              {meaning || object_type?.replace(/_/g, ' ') || 'Scene Processed'}
            </h3>
            {value && (
              <div className="mt-2 flex items-baseline gap-2">
                <span className="text-3xl font-extrabold text-blue-400 tracking-tight">{value}</span>
                {unit && <span className="text-sm font-semibold text-slate-400 uppercase">{unit}</span>}
              </div>
            )}
          </div>
          {object_type === 'stop_sign' && (
            <div className="w-12 h-12 rounded-xl bg-red-600 flex items-center justify-center text-white font-extrabold text-xs shadow-lg shadow-red-600/30">
              STOP
            </div>
          )}
          {object_type === 'speed_limit_sign' && (
            <div className="border-2 border-white rounded-lg px-2 py-1 bg-white text-black font-black text-center shadow">
              <span className="block text-[8px] leading-tight">SPEED</span>
              <span className="block text-[8px] leading-tight">LIMIT</span>
              <span className="block text-base leading-tight font-extrabold">{value || '--'}</span>
            </div>
          )}
        </div>
      </div>

      {/* Multi-Stage Confidence Reporting */}
      <div className="grid grid-cols-3 gap-2 mb-4">
        <div className="bg-slate-950/70 border border-slate-800 rounded-lg p-2.5 text-center">
          <p className="text-[10px] text-slate-400 font-medium">Detection Conf</p>
          <p className="text-sm font-bold text-slate-200 mt-1">
            {detection_confidence ? `${Math.round(detection_confidence * 100)}%` : 'N/A'}
          </p>
        </div>
        <div className="bg-slate-950/70 border border-slate-800 rounded-lg p-2.5 text-center">
          <p className="text-[10px] text-slate-400 font-medium">OCR Conf</p>
          <p className="text-sm font-bold text-slate-200 mt-1">
            {ocr_confidence ? `${Math.round(ocr_confidence * 100)}%` : 'N/A'}
          </p>
        </div>
        <div className="bg-slate-950/70 border border-slate-800 rounded-lg p-2.5 text-center">
          <p className="text-[10px] text-slate-400 font-medium">Semantic Conf</p>
          <p className="text-sm font-bold text-blue-400 mt-1">
            {interpretation_confidence ? `${Math.round(interpretation_confidence * 100)}%` : 'N/A'}
          </p>
        </div>
      </div>

      {/* Advisory Notes */}
      {advisory_notes && (
        <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-xl text-xs text-amber-300 flex items-start gap-2">
          <svg className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <div>
            <p className="font-semibold text-amber-200">Advisory Alert</p>
            <p className="mt-0.5 text-amber-300/90">{advisory_notes}</p>
          </div>
        </div>
      )}
    </div>
  );
}
