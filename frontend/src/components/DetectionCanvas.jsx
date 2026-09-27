import React, { useRef, useState, useEffect } from 'react';

const CLASS_COLORS = {
  STOP_SIGN: { border: '#EF4444', bg: 'rgba(239, 68, 68, 0.2)', text: '#FCA5A5' },
  SPEED_LIMIT_SIGN: { border: '#3B82F6', bg: 'rgba(59, 130, 246, 0.2)', text: '#93C5FD' },
  ADVISORY_SPEED: { border: '#F59E0B', bg: 'rgba(245, 158, 11, 0.2)', text: '#FCD34D' },
  WARNING_SIGN: { border: '#F59E0B', bg: 'rgba(245, 158, 11, 0.2)', text: '#FCD34D' },
  WORK_ZONE_SIGN: { border: '#F97316', bg: 'rgba(249, 115, 22, 0.2)', text: '#FDBA74' },
  LICENSE_PLATE: { border: '#10B981', bg: 'rgba(16, 185, 129, 0.2)', text: '#6EE7B7' },
};

export default function DetectionCanvas({ imageSrc, detections = [], imageDimensions }) {
  const containerRef = useRef(null);
  const imgRef = useRef(null);
  const [scale, setScale] = useState({ scaleX: 1, scaleY: 1 });

  const updateScale = () => {
    if (imgRef.current && imageDimensions) {
      const renderedW = imgRef.current.clientWidth;
      const renderedH = imgRef.current.clientHeight;
      const originalW = imageDimensions.width || renderedW;
      const originalH = imageDimensions.height || renderedH;

      setScale({
        scaleX: renderedW / originalW,
        scaleY: renderedH / originalH,
      });
    }
  };

  useEffect(() => {
    updateScale();
    window.addEventListener('resize', updateScale);
    return () => window.removeEventListener('resize', updateScale);
  }, [imageDimensions, imageSrc]);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold tracking-wide text-slate-200 uppercase flex items-center gap-2">
          <svg className="w-4 h-4 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
          </svg>
          Visual Perception View
        </h2>
        {detections.length > 0 && (
          <span className="text-xs px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-medium">
            {detections.length} Target{detections.length > 1 ? 's' : ''} Localized
          </span>
        )}
      </div>

      <div
        ref={containerRef}
        className="relative overflow-hidden rounded-xl bg-slate-950/80 border border-slate-800/80 flex items-center justify-center min-h-[340px]"
      >
        {imageSrc ? (
          <div className="relative inline-block max-w-full">
            <img
              ref={imgRef}
              src={imageSrc}
              alt="Road Scene Preview"
              onLoad={updateScale}
              className="max-h-[500px] w-auto object-contain rounded-lg block"
            />

            {/* Bounding Box Overlays */}
            {detections.map((det, idx) => {
              const [x1, y1, x2, y2] = det.bounding_box;
              const styleColor = CLASS_COLORS[det.class_name] || {
                border: '#38BDF8',
                bg: 'rgba(56, 189, 248, 0.2)',
                text: '#BAE6FD',
              };

              const left = x1 * scale.scaleX;
              const top = y1 * scale.scaleY;
              const width = (x2 - x1) * scale.scaleX;
              const height = (y2 - y1) * scale.scaleY;

              return (
                <div
                  key={idx}
                  style={{
                    position: 'absolute',
                    left: `${left}px`,
                    top: `${top}px`,
                    width: `${width}px`,
                    height: `${height}px`,
                    borderColor: styleColor.border,
                    backgroundColor: styleColor.bg,
                  }}
                  className="border-2 rounded transition-all pointer-events-none group"
                >
                  <span
                    style={{
                      backgroundColor: styleColor.border,
                      color: '#FFFFFF',
                    }}
                    className="absolute -top-6 left-0 text-[10px] font-bold px-1.5 py-0.5 rounded shadow whitespace-nowrap"
                  >
                    {det.label || det.class_name} ({Math.round(det.confidence * 100)}%)
                  </span>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center text-slate-600 py-16">
            <svg className="w-12 h-12 mb-3 stroke-[1.2]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <p className="text-sm">No road scene image loaded</p>
            <p className="text-xs text-slate-500 mt-1">Upload a road image on the left to begin visual perception</p>
          </div>
        )}
      </div>
    </div>
  );
}
