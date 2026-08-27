import React from 'react';
import { Award, Zap, Activity, CheckCircle2, Clock } from 'lucide-react';
import type { RegistrationResult } from '../api/client';

interface MetricsPanelProps {
  result: RegistrationResult;
}

export const MetricsPanel: React.FC<MetricsPanelProps> = ({ result }) => {
  const q_score = result.quality_metrics?.q_score ?? 0.0;
  const ssim = result.quality_metrics?.ssim ?? 0.0;
  const psnr = result.quality_metrics?.psnr ?? 0.0;
  const mutual_information = result.quality_metrics?.mutual_information ?? 0.0;
  const rmse = result.quality_metrics?.rmse ?? result.rmse ?? 0.0;
  const spatial_dist = result.quality_metrics?.spatial_distribution_score ?? 0.0;
  
  // Radial gauge parameters
  const radius = 45;
  const strokeDasharray = 2 * Math.PI * radius;
  const strokeDashoffset = strokeDasharray - (q_score * strokeDasharray);

  return (
    <div className="metrics-row">
      {/* Radial Q-Score Gauge Card */}
      <div className="glass-card qscore-gauge-card">
        <span className="qscore-label" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <Award size={14} style={{ color: 'var(--accent-blue)' }} /> Q-Score
        </span>
        <div className="circular-gauge">
          <svg width="110" height="110">
            <defs>
              <linearGradient id="cyan-purple-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#00f0ff" />
                <stop offset="100%" stopColor="#d000ff" />
              </linearGradient>
            </defs>
            <circle className="bg-circle" cx="55" cy="55" r={radius} />
            <circle 
              className="val-circle" 
              cx="55" 
              cy="55" 
              r={radius} 
              strokeDasharray={strokeDasharray}
              strokeDashoffset={strokeDashoffset}
            />
          </svg>
          <div className="gauge-val">{(q_score * 100).toFixed(0)}%</div>
        </div>
      </div>

      {/* Numeric Stats Grid */}
      <div className="glass-card" style={{ flexGrow: 1 }}>
        <h2 className="card-title"><Activity size={18} /> Performance Metrics</h2>
        <div className="metrics-panel-grid">
          <div className="metric-stat-card">
            <span className="metric-label" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Zap size={12} style={{ color: 'var(--accent-blue)' }} /> RMSE
            </span>
            <span className="metric-number">
              {rmse > 0 ? `${rmse.toFixed(3)} px` : "0.000 px"}
            </span>
          </div>

          <div className="metric-stat-card">
            <span className="metric-label" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <CheckCircle2 size={12} style={{ color: 'var(--accent-green)' }} /> Inliers
            </span>
            <span className="metric-number">
              {result.inlier_count ?? 0} pts
            </span>
          </div>

          <div className="metric-stat-card">
            <span className="metric-label">Spatial Dist.</span>
            <span className="metric-number">{(spatial_dist * 100).toFixed(1)}%</span>
          </div>

          <div className="metric-stat-card">
            <span className="metric-label">SSIM</span>
            <span className="metric-number">{ssim.toFixed(4)}</span>
          </div>

          <div className="metric-stat-card">
            <span className="metric-label">NMI (Mutual Info)</span>
            <span className="metric-number">{mutual_information.toFixed(4)}</span>
          </div>

          <div className="metric-stat-card">
            <span className="metric-label">PSNR</span>
            <span className="metric-number">
              {psnr === Infinity ? "∞ (Perfect)" : `${psnr.toFixed(1)} dB`}
            </span>
          </div>

          <div className="metric-stat-card">
            <span className="metric-label" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Clock size={12} style={{ color: 'var(--accent-purple)' }} /> Exec Time
            </span>
            <span className="metric-number">{result.execution_time_seconds.toFixed(2)} s</span>
          </div>
        </div>
      </div>
    </div>
  );
};
