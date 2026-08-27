import React, { useState, useRef } from 'react';
import { Columns, Layers, Sliders } from 'lucide-react';

interface ResultViewerProps {
  jobId: string;
}

export const ResultViewer: React.FC<ResultViewerProps> = ({ jobId }) => {
  const [viewMode, setViewMode] = useState<'slider' | 'blend'>('slider');
  const [sliderPos, setSliderPos] = useState<number>(50); // percentage (0-100)
  const [opacity, setOpacity] = useState<number>(0.5); // opacity (0-1)

  const containerRef = useRef<HTMLDivElement | null>(null);

  const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api/v1";
  const refUrl = `${API_BASE}/jobs/${jobId}/preview/reference`;
  const regUrl = `${API_BASE}/jobs/${jobId}/preview/registered`;

  const handleMouseMove = (e: React.MouseEvent) => {
    if (viewMode !== 'slider' || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPos(percentage);
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (viewMode !== 'slider' || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.touches[0].clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPos(percentage);
  };

  return (
    <div className="visualizer-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div className="visualizer-tabs">
          <button 
            type="button" 
            className={`tab-btn ${viewMode === 'slider' ? 'active' : ''}`}
            onClick={() => setViewMode('slider')}
          >
            <Columns size={14} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Split Slider
          </button>
          <button 
            type="button" 
            className={`tab-btn ${viewMode === 'blend' ? 'active' : ''}`}
            onClick={() => setViewMode('blend')}
          >
            <Layers size={14} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Blend Overlay
          </button>
        </div>

        {viewMode === 'blend' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px' }}>
            <Sliders size={14} style={{ color: 'var(--accent-blue)' }} />
            <span>Opacity:</span>
            <input 
              type="range" 
              min="0" 
              max="1" 
              step="0.01" 
              value={opacity} 
              onChange={e => setOpacity(parseFloat(e.target.value))}
              style={{ width: '100px', cursor: 'pointer' }}
            />
            <span style={{ width: '35px', textAlign: 'right', fontWeight: 600 }}>{Math.round(opacity * 100)}%</span>
          </div>
        )}
      </div>

      <div className="visualizer-body">
        {viewMode === 'slider' ? (
          <div 
            className="comparison-slider-container"
            ref={containerRef}
            onMouseMove={handleMouseMove}
            onTouchMove={handleTouchMove}
          >
            {/* Reference Image (bottom layer) */}
            <img 
              src={refUrl} 
              alt="Reference" 
              className="slider-img" 
            />
            
            {/* Warped Registered Image (top layer, dynamically clipped) */}
            <img 
              src={regUrl} 
              alt="Registered" 
              className="slider-overlay-img" 
              style={{ clipPath: `polygon(0 0, ${sliderPos}% 0, ${sliderPos}% 100%, 0 100%)` }}
            />

            {/* Sliding Divider Bar */}
            <div 
              className="slider-handle" 
              style={{ left: `${sliderPos}%` }}
            >
              <div className="slider-handle-button">
                <span>↔</span>
              </div>
            </div>

            <div className="slider-label src">Registered (Warped Source)</div>
            <div className="slider-label ref">LRO Reference</div>
          </div>
        ) : (
          <div className="comparison-slider-container" style={{ position: 'relative' }}>
            {/* Reference Image (bottom layer) */}
            <img 
              src={refUrl} 
              alt="Reference" 
              className="slider-img" 
            />
            {/* Warped Registered Image (top layer with opacity) */}
            <img 
              src={regUrl} 
              alt="Registered" 
              className="slider-img" 
              style={{ opacity: opacity, position: 'absolute', top: 0, left: 0 }}
            />
            <div className="slider-label src" style={{ bottom: '12px', left: '12px' }}>Blended Overlays</div>
          </div>
        )}
      </div>
    </div>
  );
};
