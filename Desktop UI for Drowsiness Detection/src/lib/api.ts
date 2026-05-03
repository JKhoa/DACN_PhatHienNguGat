/**
 * API utility — two transports:
 *   - Electron mode: routes through window.appApi (IPC bridge to main process)
 *   - Web mode (localhost): plain fetch to BACKEND_URL
 * Detection is runtime (presence of window.appApi). Public surface
 * (apiGet / apiPost / apiPut / apiDelete) is unchanged.
 */

declare global {
  interface Window {
    appApi?: {
      invoke: (channel: string, ...args: unknown[]) => Promise<unknown>;
      on: (channel: string, cb: (...args: unknown[]) => void) => () => void;
      removeAllListeners: (channel: string) => void;
    };
  }
}

interface IpcApiResult {
  status: number;
  data: unknown;
}

const BACKEND_URL =
  (import.meta as any)?.env?.VITE_BACKEND_URL || 'http://127.0.0.1:5000';

function hasIpc(): boolean {
  return typeof window !== 'undefined' && !!window.appApi;
}

class API {
  private async request(method: string, endpoint: string, data?: unknown): Promise<Response> {
    if (hasIpc()) {
      const result = (await window.appApi!.invoke('api:request', {
        method,
        endpoint,
        data,
      })) as IpcApiResult;
      return new Response(JSON.stringify(result.data), {
        status: result.status,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // Web (localhost) fallback — go straight to the backend.
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
 * Download a PDF/Excel report. Triggers a browser download.
 * Works in both Electron (IPC → base64) and Web (direct POST → blob).
 */
export async function apiExport(
  format: 'pdf' | 'excel',
  period: string,
  cameraIds?: string[],
): Promise<boolean> {
  const ext = format === 'pdf' ? 'pdf' : 'xlsx';
  const filename = `drowsiness_report_${period}.${ext}`;

  const triggerDownload = (blob: Blob) => {
    const url = URL.createObjectURL(blob);
    const a = Object.assign(document.createElement('a'), { href: url, download: filename });
    document.body.appendChild(a);
    a.click();
    URL.revokeObjectURL(url);
    a.remove();
  };

  if (hasIpc()) {
    const result = (await window.appApi!.invoke('api:export', {
      format,
      period,
      camera_ids: cameraIds,
    })) as { status: number; base64: string | null; contentType: string | null };
    if (result.status >= 200 && result.status < 300 && result.base64) {
      const bytes = Uint8Array.from(atob(result.base64), (c) => c.charCodeAt(0));
      triggerDownload(new Blob([bytes], { type: result.contentType ?? 'application/octet-stream' }));
      return true;
    }
    return false;
  }

  // Web mode — direct POST to backend, response is the binary file.
  const res = await fetch(`${BACKEND_URL}/api/logs/export/${format}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cameraIds ? { period, camera_ids: cameraIds } : { period }),
  });
  if (!res.ok) return false;
  triggerDownload(await res.blob());
  return true;
}
