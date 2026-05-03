/**
 * Auto-fix utility to resolve common errors in the application
 * Monitors console errors and automatically fixes them
 */

import { apiGet, apiPost } from '../lib/api';

interface ErrorPattern {
  pattern: RegExp;
  fix: () => Promise<void>;
  description: string;
}

class AutoFixManager {
  private errorPatterns: ErrorPattern[] = [];
  // Track last fix timestamp per pattern description → used for rate-limit + dedup.
  private lastFixAt: Map<string, number> = new Map();
  // Reentrance guard: which patterns are currently being fixed (prevents parallel duplicates).
  private inFlight: Set<string> = new Set();
  private isRunning = false;
  // Captured native console fns — used for internal logging to avoid recursive override.
  private nativeError: (...args: any[]) => void = console.error.bind(console);
  private nativeWarn: (...args: any[]) => void = console.warn.bind(console);
  private nativeLog: (...args: any[]) => void = console.log.bind(console);
  // Minimum gap between fixes of the same pattern (ms).
  private static readonly FIX_COOLDOWN_MS = 10_000;

  constructor() {
    this.setupErrorPatterns();
    this.startMonitoring();
  }

  private setupErrorPatterns() {
    // Pattern 1: 404 for /api/camera/<id>/detection - Camera not in backend
    this.errorPatterns.push({
      pattern: /404.*\/api\/camera\/([^/]+)\/detection/,
      fix: async () => {
        await this.fixCameraNotInBackend();
      },
      description: 'Camera not found in backend - attempting to add/start',
    });

    // Pattern 2: Failed to fetch detection
    this.errorPatterns.push({
      pattern: /Failed to fetch detection for ([^:]+)/,
      fix: async () => {
        await this.fixCameraDetectionFailure();
      },
      description: 'Detection fetch failure - checking camera status',
    });

    // Pattern 3: Network error
    this.errorPatterns.push({
      pattern: /Network error|Failed to fetch|ECONNREFUSED/,
      fix: async () => {
        await this.fixNetworkError();
      },
      description: 'Network error - checking backend connection',
    });

    // Pattern 4: No detection data
    this.errorPatterns.push({
      pattern: /No detection data|no detection data/,
      fix: async () => {
        await this.fixNoDetectionData();
      },
      description: 'No detection data - ensuring detection is enabled',
    });
  }

  private async fixCameraNotInBackend() {
    try {
      // Get all cameras from backend
      const camerasResponse = await apiGet('api/cameras');
      if (!camerasResponse.ok) return;

      const camerasData = await camerasResponse.json();
      const backendCameras = camerasData.cameras || [];
      const backendCamIds = new Set(backendCameras.map((c: any) => c.id));

      // Signal to App.tsx to sync cameras - it will handle adding missing cameras
      window.dispatchEvent(new CustomEvent('autofix-sync-cameras', { 
        detail: { backendCamIds: Array.from(backendCamIds) } 
      }));
      
      console.log('[AutoFix] Triggered camera sync');
    } catch (error) {
      console.warn('[AutoFix] Could not fix camera sync:', error);
    }
  }

  private async fixCameraDetectionFailure() {
    try {
      // Check backend health
      const healthResponse = await apiGet('api/health');
      if (!healthResponse.ok) {
        console.warn('[AutoFix] Backend not responding');
        return;
      }

      // Ensure YOLO detector is initialized
      try {
        await apiPost('api/detection/initialize', { model_path: 'yolo11n-pose.pt' });
        console.log('[AutoFix] Attempted to initialize YOLO detector');
      } catch {
        // Ignore if already initialized
      }
    } catch (error) {
      console.warn('[AutoFix] Could not fix detection failure:', error);
    }
  }

  private async fixNetworkError() {
    try {
      // Test backend connection
      const response = await apiGet('api/health');
      if (response.ok) {
        console.log('[AutoFix] Backend is accessible');
      } else {
        console.warn('[AutoFix] Backend returned error status:', response.status);
        // Trigger backend restart notification
        window.dispatchEvent(new CustomEvent('autofix-backend-restart-needed'));
      }
    } catch (error) {
      console.error('[AutoFix] Backend connection failed:', error);
      window.dispatchEvent(new CustomEvent('autofix-backend-restart-needed'));
    }
  }

  private async fixNoDetectionData() {
    try {
      // Get all cameras and ensure detection is enabled
      const camerasResponse = await apiGet('api/cameras');
      if (!camerasResponse.ok) return;

      const camerasData = await camerasResponse.json();
      const backendCameras = camerasData.cameras || [];

      // For each running camera, ensure detection is enabled
      for (const cam of backendCameras) {
        if (cam.status === 'running') {
          try {
            await apiPost(`api/camera/${cam.id}/detection/toggle`, { enabled: true });
            console.log(`[AutoFix] Enabled detection for camera ${cam.id}`);
          } catch {
            // Ignore individual failures
          }
        }
      }
    } catch (error) {
      console.warn('[AutoFix] Could not fix no detection data:', error);
    }
  }

  private startMonitoring() {
    if (this.isRunning) return;
    this.isRunning = true;

    // Capture the original native functions BEFORE overriding so internal logs
    // (from checkAndFix itself) cannot recurse back through the overridden fns.
    this.nativeError = console.error.bind(console);
    this.nativeWarn = console.warn.bind(console);
    this.nativeLog = console.log.bind(console);

    console.error = (...args: any[]) => {
      this.nativeError(...args);
      // Do not analyze internal AutoFix logs — they are our own diagnostics.
      const message = args.join(' ');
      if (message.includes('[AutoFix]')) return;
      this.checkAndFix(message);
    };

    console.warn = (...args: any[]) => {
      this.nativeWarn(...args);
      const message = args.join(' ');
      if (message.includes('[AutoFix]')) return;
      if (message.includes('Failed to fetch') || message.includes('404') || message.includes('detection')) {
        this.checkAndFix(message);
      }
    };

    // Also listen to unhandled promise rejections
    window.addEventListener('unhandledrejection', (event) => {
      const error = event.reason?.message || String(event.reason);
      this.checkAndFix(error);
    });

    this.nativeLog('[AutoFix] Error monitoring started');
  }

  private async checkAndFix(errorMessage: string) {
    if (!this.isRunning) return;

    // Find the first matching pattern.
    for (const errorPattern of this.errorPatterns) {
      if (!errorPattern.pattern.test(errorMessage)) continue;

      const key = errorPattern.description;

      // Reentrance guard — skip if a fix for this pattern is already running.
      if (this.inFlight.has(key)) return;

      // Rate-limit — skip if we fixed this pattern within the cooldown window.
      const last = this.lastFixAt.get(key) ?? 0;
      if (Date.now() - last < AutoFixManager.FIX_COOLDOWN_MS) return;

      this.inFlight.add(key);
      this.lastFixAt.set(key, Date.now());
      this.nativeLog(`[AutoFix] Applying fix: ${key}`);

      try {
        await errorPattern.fix();
        this.nativeLog(`[AutoFix] Fix applied: ${key}`);
      } catch (err) {
        this.nativeWarn(`[AutoFix] Fix failed for ${key}:`, err);
      } finally {
        this.inFlight.delete(key);
      }

      break;
    }
  }

  public getFixHistory(): string[] {
    // Backwards-compatible shape: list of "description-timestamp" entries.
    return Array.from(this.lastFixAt.entries()).map(([d, t]) => `${d}-${t}`);
  }

  public stopMonitoring() {
    this.isRunning = false;
  }
}

// Initialize global auto-fix manager
let autoFixManager: AutoFixManager | null = null;

export function initAutoFix() {
  if (!autoFixManager) {
    autoFixManager = new AutoFixManager();
  }
  return autoFixManager;
}

export function getAutoFixManager() {
  return autoFixManager;
}

