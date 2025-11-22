import { ENV } from '../config/env';

/**
 * API utility helper for making HTTP requests to backend
 */
class API {
  private baseURL: string;

  constructor() {
    this.baseURL = ENV.API_BASE_URL;
  }

  /**
   * Build full API URL
   */
  private buildURL(endpoint: string): string {
    // Remove leading slash if present to avoid double slashes
    const cleanEndpoint = endpoint.startsWith('/') ? endpoint.slice(1) : endpoint;
    return `${this.baseURL}/${cleanEndpoint}`;
  }

  /**
   * GET request
   */
  async get(endpoint: string): Promise<Response> {
    return fetch(this.buildURL(endpoint));
  }

  /**
   * POST request
   */
  async post(endpoint: string, data?: any): Promise<Response> {
    return fetch(this.buildURL(endpoint), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  /**
   * PUT request
   */
  async put(endpoint: string, data?: any): Promise<Response> {
    return fetch(this.buildURL(endpoint), {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  /**
   * DELETE request
   */
  async delete(endpoint: string): Promise<Response> {
    return fetch(this.buildURL(endpoint), {
      method: 'DELETE',
    });
  }

  /**
   * Get full URL for external use (e.g., download links)
   */
  getURL(endpoint: string): string {
    return this.buildURL(endpoint);
  }
}

// Export singleton instance
export const api = new API();

// Export convenience methods
export const apiGet = (endpoint: string) => api.get(endpoint);
export const apiPost = (endpoint: string, data?: any) => api.post(endpoint, data);
export const apiPut = (endpoint: string, data?: any) => api.put(endpoint, data);
export const apiDelete = (endpoint: string) => api.delete(endpoint);
export const apiURL = (endpoint: string) => api.getURL(endpoint);
