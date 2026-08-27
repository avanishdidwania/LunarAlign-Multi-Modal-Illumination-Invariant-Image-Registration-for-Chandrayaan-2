export interface RegistrationConfig {
  illumination_method: string;
  detection_method: string;
  matching_method: string;
  outlier_method: string;
  transform_type: string | null;
  refine_subpixel: boolean;
  device: string;
}

export interface QualityMetrics {
  ssim: number;
  psnr: number;
  mutual_information: number;
  inlier_ratio: number;
  q_score: number;
  rmse: number;
  spatial_distribution_score: number;
}

export interface MatchPoint {
  source_pt: [number, number];
  reference_pt: [number, number];
  confidence: number;
}

export interface RegistrationResult {
  success: boolean;
  quality_metrics?: QualityMetrics;
  rmse?: number;
  inlier_count?: number;
  inlier_ratio?: number;
  execution_time_seconds: number;
  error_message?: string;
  match_points: MatchPoint[];
}

export interface JobResponse {
  job_id: string;
  status: string; // "pending" | "running" | "completed" | "failed"
  result?: RegistrationResult;
  registered_image_url?: string;
}

// Support fallback to current host if base URL is not specified
const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api/v1";

export const apiClient = {
  async getMethods() {
    const res = await fetch(`${API_BASE}/config/methods`);
    if (!res.ok) throw new Error("Failed to load methods config");
    return res.json();
  },

  async register(
    sourceFile: File,
    referenceFile: File,
    config: RegistrationConfig
  ): Promise<JobResponse> {
    const formData = new FormData();
    formData.append("source_image", sourceFile);
    formData.append("reference_image", referenceFile);
    formData.append("illumination_method", config.illumination_method);
    formData.append("detection_method", config.detection_method);
    formData.append("matching_method", config.matching_method);
    formData.append("outlier_method", config.outlier_method);
    if (config.transform_type) {
      formData.append("transform_type", config.transform_type);
    }
    formData.append("refine_subpixel", String(config.refine_subpixel));
    formData.append("device", config.device);

    const res = await fetch(`${API_BASE}/register`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const errText = await res.text().catch(() => "Unknown error");
      console.error(`Registration failed: HTTP ${res.status}`, errText);
      let detail = "Failed to submit job";
      try {
        const errJson = JSON.parse(errText);
        detail = errJson.detail || detail;
      } catch { /* not JSON */ 
        detail = errText || detail;
      }
      throw new Error(detail);
    }
    return res.json();
  },

  async getJobStatus(job_id: string): Promise<JobResponse> {
    const res = await fetch(`${API_BASE}/jobs/${job_id}`);
    if (!res.ok) {
      throw new Error(`Job ${job_id} not found`);
    }
    return res.json();
  },

  async getMatchesGeoJSON(job_id: string) {
    const res = await fetch(`${API_BASE}/jobs/${job_id}/matches`);
    if (!res.ok) {
      throw new Error("Failed to retrieve matches GeoJSON");
    }
    return res.json();
  },

  getRegisteredImageUrl(job_id: string): string {
    return `${API_BASE}/jobs/${job_id}/image`;
  },
};
