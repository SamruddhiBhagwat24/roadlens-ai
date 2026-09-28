import React from 'react';

export default function DetectionsList({ detections = [], ocrResults = [] }) {
  if (detections.length === 0 && ocrResults.length === 0) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
        <h2 className="text-sm font-semibold tracking-wide text-slate-200 uppercase mb-3 flex items-center gap-2">
          <svg className="w-4 h-4 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
          </svg>
          Localized Road Entities & OCR Extractions
        </h2>
        <p className="text-xs text-slate-500">No objects or text localized yet.</p>
      </div>
    );
  }

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold tracking-wide text-slate-200 uppercase flex items-center gap-2">
          <svg className="w-4 h-4 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
          </svg>
          Localized Road Entities & OCR Extractions
        </h2>
        <span className="text-xs text-slate-400 font-mono">
          {detections.length} objects localized • {ocrResults.length} text blocks
        </span>
      </div>

      <div className="space-y-3">
        {detections.map((det, idx) => (
          <div key={idx} className="bg-slate-950/70 border border-slate-800 rounded-xl p-3.5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold text-white tracking-wide">
                  {det.label || det.class_name}
                </span>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 font-semibold">
                  {Math.round(det.confidence * 100)}% match
                </span>
              </div>
              <p className="text-[11px] text-slate-500 font-mono mt-1">
                BBox: [{det.bounding_box.join(', ')}]
              </p>
            </div>

            {/* Matching OCR result if available */}
            {ocrResults[idx] && (
              <div className="bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-lg flex items-center gap-3">
                <div>
                  <span className="text-[9px] uppercase font-semibold text-slate-400 block">OCR Text</span>
                  <span className="text-xs font-mono font-bold text-emerald-400">
                    "{ocrResults[idx].text === 'OCR_UNREADABLE' ? 'OCR Unreadable' : ocrResults[idx].text}"
                  </span>
                </div>
                <div className="border-l border-slate-800 pl-3">
                  <span className="text-[9px] uppercase font-semibold text-slate-400 block">Engine</span>
                  <span className="text-[10px] font-mono text-slate-300">
                    {ocrResults[idx].engine}
                  </span>
                </div>
              </div>
            )}
          </div>
        ))}

        {/* Unmatched OCR Results */}
        {ocrResults.slice(detections.length).map((ocr, i) => (
          <div key={`ocr-${i}`} className="bg-slate-950/50 border border-slate-800/80 rounded-xl p-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono text-emerald-400 font-bold">"{ocr.text === 'OCR_UNREADABLE' ? 'OCR Unreadable' : ocr.text}"</span>
              <span className="text-[10px] text-slate-500 font-mono">[{ocr.bounding_box.join(', ')}]</span>
            </div>
            <span className="text-[10px] font-mono text-slate-400 bg-slate-900 px-2 py-0.5 rounded border border-slate-800">
              {ocr.engine} ({Math.round(ocr.confidence * 100)}%)
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
