import { useState, useEffect } from 'react';
import { Globe, RefreshCw, Layers, ShieldAlert, Download, Cpu } from 'lucide-react';
import { ImageUploader } from './components/ImageUploader';
import { MetricsPanel } from './components/MetricsPanel';
import { ResultViewer } from './components/ResultViewer';
import { MatchPointOverlay } from './components/MatchPointOverlay';
import { apiClient } from './api/client';
import type { JobResponse, RegistrationConfig } from './api/client';
import './App.css';

export default function App() {
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string>("idle"); // "idle" | "pending" | "running" | "completed" | "failed"
  const [jobResult, setJobResult] = useState<JobResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  
  const [visTab, setVisTab] = useState<'visualizer' | 'inspector'>('visualizer');

  // Poll job status when jobId changes
  useEffect(() => {
    if (!jobId || jobStatus === "completed" || jobStatus === "failed") return;

    let intervalId = setInterval(async () => {
      try {
        const resp = await apiClient.getJobStatus(jobId);
        setJobStatus(resp.status);
        setJobResult(resp);

        if (resp.status === "completed") {
          clearInterval(intervalId);
        } else if (resp.status === "failed") {
          clearInterval(intervalId);
          setErrorMsg(resp.result?.error_message || "Alignment job failed.");
        }
      } catch (err: any) {
        clearInterval(intervalId);
        setJobStatus("failed");
        setErrorMsg(err.message || "Failed to poll job status.");
      }
    }, 1000);

    return () => clearInterval(intervalId);
  }, [jobId, jobStatus]);

  const handleRunAlignment = async (src: File, ref: File, config: RegistrationConfig) => {
    setSourceFile(src);
    setReferenceFile(ref);
    setErrorMsg(null);
    setJobResult(null);
    setJobStatus("pending");

    try {
      const resp = await apiClient.register(src, ref, config);
      setJobId(resp.job_id);
      setJobStatus("pending");
    } catch (err: any) {
      setJobStatus("failed");
      setErrorMsg(err.message || "Failed to initiate alignment job.");
    }
  };

  const getStepStatusClass = (stepIdx: number) => {
    // Maps status to pipeline steps for the timeline visualization
    // 0: Loading, 1: Normalization, 2: Matching, 3: Outlier Rejection, 4: Refinement, 5: Warping/Export
    const statusMap: Record<string, number> = {
      "pending": 0,
      "running": 2, // Assume intermediate stage when running
      "completed": 6,
      "failed": -1
    };
    
    const currentIdx = statusMap[jobStatus] ?? 0;
    
    if (currentIdx > stepIdx) return "completed";
    if (currentIdx === stepIdx) return "active";
    return "";
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="logo-section">
          <Globe className="logo-icon" size={28} />
          <div className="logo-text">
            <h1>Lunar Registration Portal</h1>
            <p>ISRO Chandrayaan-2 Co-Registration Engine</p>
          </div>
        </div>
        <div className="system-status">
          <div className="status-dot"></div>
          <span>Local CPU/GPU Orchestrator Active</span>
        </div>
      </header>

      {/* Main Grid */}
      <div className="dashboard-grid">
        {/* Left Side: Uploads & Parameters OR Processing Timeline */}
        <div className="left-panel">
          {jobStatus === "pending" || jobStatus === "running" ? (
            <div className="glass-card polling-card">
              <div className="loader-anim"></div>
              <h2>Co-Aligning Imagery</h2>
              <p>Running high-precision registration pipeline stages in backend worker...</p>
              
              <div className="job-timeline">
                <div className={`timeline-step ${getStepStatusClass(0)}`}>
                  <div className="timeline-circle">1</div>
                  <span>Image Formats Verification</span>
                </div>
                <div className={`timeline-step ${getStepStatusClass(1)}`}>
                  <div className="timeline-circle">2</div>
                  <span>Illumination Preprocessing</span>
                </div>
                <div className={`timeline-step ${getStepStatusClass(2)}`}>
                  <div className="timeline-circle">3</div>
                  <span>Gaussian Resolution Pyramids</span>
                </div>
                <div className={`timeline-step ${getStepStatusClass(3)}`}>
                  <div className="timeline-circle">4</div>
                  <span>Feature Extraction & Deep Matching</span>
                </div>
                <div className={`timeline-step ${getStepStatusClass(4)}`}>
                  <div className="timeline-circle">5</div>
                  <span>Robust Outlier Rejector (MAGSAC++)</span>
                </div>
                <div className={`timeline-step ${getStepStatusClass(5)}`}>
                  <div className="timeline-circle">6</div>
                  <span>Quadratic Sub-Pixel Fitting</span>
                </div>
                <div className={`timeline-step ${getStepStatusClass(6)}`}>
                  <div className="timeline-circle">7</div>
                  <span>GeoTIFF Metadata Re-projection</span>
                </div>
              </div>
            </div>
          ) : (
            <ImageUploader onRun={handleRunAlignment} isLoading={jobStatus === "pending" || jobStatus === "running"} />
          )}
        </div>

        {/* Right Side: Workspace Results / Welcome Panel */}
        <div className="right-panel">
          {jobStatus === "completed" && jobResult?.result && sourceFile && referenceFile ? (
            <div className="results-workspace">
              {/* Metrics Header Cards */}
              <MetricsPanel result={jobResult.result} />

              {/* Visualization Container */}
              <div className="glass-card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                  <div className="visualizer-tabs">
                    <button 
                      type="button" 
                      className={`tab-btn ${visTab === 'visualizer' ? 'active' : ''}`}
                      onClick={() => setVisTab('visualizer')}
                    >
                      <Layers size={14} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Overlay Alignment Slider
                    </button>
                    <button 
                      type="button" 
                      className={`tab-btn ${visTab === 'inspector' ? 'active' : ''}`}
                      onClick={() => setVisTab('inspector')}
                    >
                      <RefreshCw size={14} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Interactive Match Inspector
                    </button>
                  </div>

                  <a 
                    href={apiClient.getRegisteredImageUrl(jobResult.job_id)}
                    className="download-link"
                    download
                  >
                    <Download size={14} /> Download GeoTIFF
                  </a>
                </div>

                {visTab === 'visualizer' ? (
                  <ResultViewer jobId={jobResult.job_id} />
                ) : (
                  <MatchPointOverlay 
                    sourceFile={sourceFile} 
                    referenceFile={referenceFile} 
                    matches={jobResult.result.match_points} 
                  />
                )}
              </div>
            </div>
          ) : jobStatus === "failed" ? (
            <div className="glass-card welcome-card" style={{ border: '1px solid rgba(255, 56, 56, 0.2)' }}>
              <ShieldAlert className="welcome-icon" size={64} style={{ color: 'var(--accent-red)' }} />
              <h2 style={{ color: 'var(--accent-red)' }}>Registration Execution Failed</h2>
              <p style={{ maxWidth: '400px' }}>{errorMsg || "The pipeline run failed due to degenerate inputs, insufficient points, or scale ratios."}</p>
              <button 
                type="button" 
                className="run-btn" 
                onClick={() => setJobStatus("idle")}
                style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--border-color)', boxShadow: 'none' }}
              >
                Reset Dashboard
              </button>
            </div>
          ) : (
            <div className="glass-card welcome-card">
              <Globe className="welcome-icon" size={64} />
              <h2>Co-Registration Control Center</h2>
              <p>Drag and drop your Chandrayaan-2 payload rasters on the left, select your matching configuration parameters, and run the co-alignment algorithm.</p>
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <footer className="footer">
        <span>SIH 26166 — Space Applications Centre (SAC), ISRO</span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Cpu size={12} /> GPU Acceleration (LightGlue/LoFTR Models) Active
        </span>
      </footer>
    </div>
  );
}
