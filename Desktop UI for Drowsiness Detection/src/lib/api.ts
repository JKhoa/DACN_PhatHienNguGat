/**
 * API utility — localhost only.
 * Plain fetch tới BACKEND_URL (mặc định http://127.0.0.1:5000).
 * Override bằng VITE_BACKEND_URL khi cần.
 */

const BACKEND_URL =
  (import.meta as any)?.env?.VITE_BACKEND_URL || 'http://127.0.0.1:5000';

class API {
  private async request(method: string, endpoint: string, data?: unknown): Promise<Response> {
    const url = `${BACKEND_URL}${endpoint.startsWith('/') ? '' : '/'}${endpoint}`;
    const init: RequestInit = { method, headers: {} };
    if (data !== undefined && data !== null) {
      (init.headers as Record<string, string>)['Content-Type'] = 'application/json';
      init.body = JSON.stringify(data);
    }
    return fetch(url, init);
  }

  async get(endpoint: string): Promise<Response> {
    return this.request('GET', endpoint);
  }

  async post(endpoint: string, data?: unknown): Promise<Response> {
    return this.request('POST', endpoint, data);
  }

  async put(endpoint: string, data?: unknown): Promise<Response> {
    return this.request('PUT', endpoint, data);
  }

  async delete(endpoint: string): Promise<Response> {
    return this.request('DELETE', endpoint);
  }
}

const api = new API();
export { api };

export const apiGet    = (endpoint: string)                  => api.get(endpoint);
export const apiPost   = (endpoint: string, data?: unknown)  => api.post(endpoint, data);
export const apiPut    = (endpoint: string, data?: unknown)  => api.put(endpoint, data);
export const apiDelete = (endpoint: string)                  => api.delete(endpoint);

/**
 * Download a PDF/Excel report — direct POST to backend, response is binary file.
 */
export async function apiExport(
  format: 'pdf' | 'excel',
  period: string,
  cameraIds?: string[],
): Promise<boolean> {
  const ext = format === 'pdf' ? 'pdf' : 'xlsx';
  const filename = `drowsiness_report_${period}.${ext}`;

  const res = await fetch(`${BACKEND_URL}/api/logs/export/${format}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cameraIds ? { period, camera_ids: cameraIds } : { period }),
  });
  if (!res.ok) return false;

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement('a'), { href: url, download: filename });
  document.body.appendChild(a);
  a.click();
  URL.revokeObjectURL(url);
  a.remove();
  return true;
}
