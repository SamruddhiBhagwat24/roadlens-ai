/**
 * RoadLens AI API Client Service
 */

const API_BASE = 'https://roadlens-ai-backend.onrender.com/api';
export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) {
      throw new Error(`Health check failed (${res.status})`);
    }
    return await res.json();
  } catch (err) {
    return { status: 'offline', error: err.message, rocm_available: false };
  }
}

export async function fetchConfig() {
  try {
    const res = await fetch(`${API_BASE}/config`);
    if (!res.ok) throw new Error('Config fetch failed');
    return await res.json();
  } catch (err) {
    return { max_upload_size_mb: 15, allowed_extensions: ['jpg', 'jpeg', 'png', 'webp'] };
  }
}

export async function fetchProfiles() {
  try {
    const res = await fetch(`${API_BASE}/profiles`);
    if (!res.ok) throw new Error('Profiles fetch failed');
    return await res.json();
  } catch (err) {
    return {
      active_default: 'usa',
      profiles: [
        { code: 'usa', name: 'United States (MUTCD)', speed_unit: 'mph', status: 'production_active', standards: ['FHWA MUTCD'] },
        { code: 'india', name: 'India (IRC / MoRTH)', speed_unit: 'km/h', status: 'staged_specification', standards: ['IRC:67-2022'] }
      ]
    };
  }
}

export async function analyzeRoadImage(file, countryCode = 'usa') {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/analyze?country_code=${encodeURIComponent(countryCode)}`, {
    method: 'POST',
    body: formData,
  });

  let data;
  try {
    data = await response.json();
  } catch (err) {
    if (!response.ok) {
      throw new Error(`Server returned status ${response.status} (${response.statusText || 'Request failed'})`);
    }
    throw new Error('Invalid JSON response received from perception server');
  }

  if (!response.ok) {
    const errorMessage = data?.detail || data?.error || `Analysis failed (${response.status})`;
    throw new Error(errorMessage);
  }

  return data;
}
