/**
 * Auto-fix utility to resolve common errors in the application
 * Monitors console errors and automatically fixes them
 */

interface ErrorPattern {
  pattern: RegExp;
  fix: () => Promise<void>;
  description: string;
}

class AutoFixManager {
  private errorPatterns: ErrorPattern[] = [];
  private fixHistory: string[] = [];
  private isRunning = false;

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
      const camerasResponse = await fetch('http://127.0.0.1:5000/api/cameras');
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
      const healthResponse = await fetch('http://127.0.0.1:5000/api/health');
      if (!healthResponse.ok) {
        console.warn('[AutoFix] Backend not responding');
        return;
      }

      // Ensure YOLO detector is initialized
      try {
        await fetch('http://127.0.0.1:5000/api/detection/initialize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model_path: 'yolo11n-pose.pt' }),
        });
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
      const response = await fetch('http://127.0.0.1:5000/api/health');
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
      const camerasResponse = await fetch('http://127.0.0.1:5000/api/cameras');
      if (!camerasResponse.ok) return;

      const camerasData = await camerasResponse.json();
      const backendCameras = camerasData.cameras || [];

      // For each running camera, ensure detection is enabled
      for (const cam of backendCameras) {
        if (cam.status === 'running') {
          try {
            await fetch(`http://127.0.0.1:5000/api/camera/${cam.id}/detection/toggle`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ enabled: true }),
            });
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

    // Override console.error to catch errors
    const originalError = console.error;
    const originalWarn = console.warn;

    console.error = (...args: any[]) => {
      originalError.apply(console, args);
      this.checkAndFix(args.join(' '));
    };

    console.warn = (...args: any[]) => {
      originalWarn.apply(console, args);
      const message = args.join(' ');
      // Only check for specific warnings that indicate errors
      if (message.includes('Failed to fetch') || message.includes('404') || message.includes('detection')) {
        this.checkAndFix(message);
      }
    };

    // Also listen to unhandled promise rejections
    window.addEventListener('unhandledrejection', (event) => {
      const error = event.reason?.message || String(event.reason);
      this.checkAndFix(error);
    });

    console.log('[AutoFix] Error monitoring started');
  }

  private async checkAndFix(errorMessage: string) {
    // Avoid duplicate fixes
    if (this.fixHistory.includes(errorMessage)) {
      return;
    }

    // Check each pattern
    for (const errorPattern of this.errorPatterns) {
      const match = errorMessage.match(errorPattern.pattern);
      if (match) {
        const errorId = `${errorPattern.description}-${Date.now()}`;
        
        // Avoid fixing the same error multiple times in quick succession
        if (this.fixHistory.some(h => h.startsWith(errorPattern.description))) {
          const recentFix = this.fixHistory
            .filter(h => h.startsWith(errorPattern.description))
            .pop();
          if (recentFix) {
            const timestamp = parseInt(recentFix.split('-').pop() || '0');
            if (Date.now() - timestamp < 5000) {
              // Fixed recently, skip
              return;
            }
          }
        }

        console.log(`[AutoFix] Detected error: ${errorPattern.description}`);
        console.log(`[AutoFix] Attempting to fix: ${errorPattern.description}`);

        try {
          await errorPattern.fix();
          this.fixHistory.push(errorId);
          console.log(`[AutoFix] Fix applied: ${errorPattern.description}`);
          
          // Clear history after 30 seconds to allow retry if error persists
          setTimeout(() => {
            this.fixHistory = this.fixHistory.filter(h => !h.startsWith(errorPattern.description));
          }, 30000);
        } catch (error) {
          console.warn(`[AutoFix] Fix failed for ${errorPattern.description}:`, error);
        }
        
        break;
      }
    }
  }

  public getFixHistory(): string[] {
    return [...this.fixHistory];
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

