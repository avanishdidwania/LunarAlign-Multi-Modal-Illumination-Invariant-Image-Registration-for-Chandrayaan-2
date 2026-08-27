import React, { useState } from 'react';
import { Upload, X, Settings } from 'lucide-react';
import type { RegistrationConfig } from '../api/client';

interface ImageUploaderProps {
  onRun: (src: File, ref: File, config: RegistrationConfig) => void;
  isLoading: boolean;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({ onRun, isLoading }) => {
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  
  const [config, setConfig] = useState<RegistrationConfig>({
    illumination_method: "phase_congruency",
    detection_method: "superpoint",
    matching_method: "lightglue",
    outlier_method: "magsac++",
    transform_type: null,
    refine_subpixel: true,
    device: "auto"
  });

  const handleSourceChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSourceFile(e.target.files[0]);
    }
  };

  const handleReferenceChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setReferenceFile(e.target.files[0]);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (sourceFile && referenceFile) {
      onRun(sourceFile, referenceFile, config);
    }
  };
  
  const setKey = (key: keyof RegistrationConfig, val: any) => {
    setConfig(prev => ({ ...prev, [key]: val }));
  };

  return (
    <form onSubmit={handleSubmit} className="config-form">
      <div className="upload-grid">
        {/* Source Image Upload Box */}
        <div className={`upload-box ${sourceFile ? 'has-file' : ''}`}>
          <input
            type="file"
            id="source-upload"
            accept=".tif,.tiff,.img,.pds,.jp2,.png"
            onChange={handleSourceChange}
            style={{ display: 'none' }}
          />
          <label htmlFor="source-upload" style={{ cursor: 'pointer', display: 'block' }}>
            <Upload className="upload-icon" size={32} style={{ margin: '0 auto 12px' }} />
            <h3>Source Image</h3>
            <p>OHRC (0.25m) or TMC-2 (5m)</p>
          </label>
          {sourceFile && (
            <div className="file-info">
              <span className="file-name">{sourceFile.name}</span>
              <button type="button" className="remove-file" onClick={() => setSourceFile(null)}>
                <X size={16} />
              </button>
            </div>
          )}
        </div>

        {/* Reference Image Upload Box */}
        <div className={`upload-box ${referenceFile ? 'has-file' : ''}`}>
          <input
            type="file"
            id="reference-upload"
            accept=".tif,.tiff,.img,.pds,.jp2,.png"
            onChange={handleReferenceChange}
            style={{ display: 'none' }}
          />
          <label htmlFor="reference-upload" style={{ cursor: 'pointer', display: 'block' }}>
            <Upload className="upload-icon" size={32} style={{ margin: '0 auto 12px' }} />
            <h3>Reference Image</h3>
            <p>LRO NAC reference (~0.5m)</p>
          </label>
          {referenceFile && (
            <div className="file-info">
              <span className="file-name">{referenceFile.name}</span>
              <button type="button" className="remove-file" onClick={() => setReferenceFile(null)}>
                <X size={16} />
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Configuration Panel */}
      <div className="glass-card">
        <h2 className="card-title"><Settings size={18} /> Parameters</h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div className="form-group">
            <label>Illumination Normalization</label>
            <select
              className="custom-select"
              value={config.illumination_method}
              onChange={e => setKey('illumination_method', e.target.value)}
            >
              <option value="phase_congruency">Phase Congruency (Invariant)</option>
              <option value="clahe">CLAHE Contrast Enhancement</option>
              <option value="gradient">Gradient Orientation Map</option>
              <option value="lnms">Local Normalized Mean Subtraction</option>
            </select>
          </div>

          <div className="form-group">
            <label>Pipeline Core Model</label>
            <select
              className="custom-select"
              value={`${config.detection_method}:${config.matching_method}`}
              onChange={e => {
                const [det, mat] = e.target.value.split(':');
                setConfig(prev => ({
                  ...prev,
                  detection_method: det,
                  matching_method: mat
                }));
              }}
            >
              <option value="superpoint:lightglue">SuperPoint + LightGlue (Recommended)</option>
              <option value="sift:bf">SIFT + Brute-Force L2 (Classical)</option>
              <option value="superpoint:loftr">LoFTR Dense Matching (Transformer)</option>
            </select>
          </div>

          <div className="form-group">
            <label>Outlier Rejection</label>
            <select
              className="custom-select"
              value={config.outlier_method}
              onChange={e => setKey('outlier_method', e.target.value)}
            >
              <option value="magsac++">MAGSAC++ (Robust Consensus)</option>
              <option value="ransac">RANSAC (Standard Consensus)</option>
              <option value="lmeds">LMEDS (Least Median)</option>
            </select>
          </div>

          <div className="form-group">
            <label>Transformation Model</label>
            <select
              className="custom-select"
              value={config.transform_type || ""}
              onChange={e => setKey('transform_type', e.target.value || null)}
            >
              <option value="">Auto Select (Spatial Span)</option>
              <option value="affine">Affine (6 DOF)</option>
              <option value="projective">Projective (8 DOF)</option>
            </select>
          </div>

          <div 
            className={`toggle-group ${config.refine_subpixel ? 'active' : ''}`}
            onClick={() => setKey('refine_subpixel', !config.refine_subpixel)}
          >
            <span>Sub-Pixel Refinement</span>
            <div className="toggle-switch"></div>
          </div>

          <div className="form-group">
            <label>Hardware Target</label>
            <select
              className="custom-select"
              value={config.device}
              onChange={e => setKey('device', e.target.value)}
            >
              <option value="auto">Auto Resolution (GPU preferred)</option>
              <option value="cuda">Force NVIDIA CUDA GPU</option>
              <option value="cpu">Force Intel/AMD CPU</option>
            </select>
          </div>
        </div>
      </div>

      <button
        type="submit"
        className="run-btn"
        disabled={!sourceFile || !referenceFile || isLoading}
      >
        {isLoading ? (
          <>
            <svg className="spinner" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
              <circle cx="12" cy="12" r="10" stroke="rgba(255, 255, 255, 0.2)" />
              <path d="M4 12a8 8 0 0 1 8-8" />
            </svg>
            Aligning...
          </>
        ) : (
          "Align Images"
        )}
      </button>
    </form>
  );
};
