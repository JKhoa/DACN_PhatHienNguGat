import React, { useEffect, useState } from 'react';
import { Badge } from './ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Switch } from './ui/switch';
import { Label } from './ui/label';
import { 
  Brain, 
  Users, 
  Eye, 
  AlertTriangle, 
  Activity,
  Zap,
  Clock,
  Target
} from 'lucide-react';

interface DetectionResult {
  frame_id: number;
  timestamp: number;
  persons: PersonDetection[];
  fps: number;
  processing_time: number;
}

interface PersonDetection {
  id: number;
  bbox: [number, number, number, number];
  confidence: number;
  keypoints: Keypoint[];
  drowsiness_score: number;
  drowsiness_state: 'awake' | 'drowsy' | 'sleeping';
  last_update: number;
}

interface Keypoint {
  x: number;
  y: number;
  confidence: number;
  visible: boolean;
}

interface YOLODetectionPanelProps {
  cameraId: string;
  isEnabled: boolean;
  onToggleDetection: (enabled: boolean) => void;
}

export function YOLODetectionPanel({ 
  cameraId, 
  isEnabled, 
  onToggleDetection 
}: YOLODetectionPanelProps) {
  const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch detection results
  const fetchDetectionResults = async () => {
    if (!isEnabled) return;
    
    try {
      setIsLoading(true);
      const response = await fetch(`http://127.0.0.1:5000/api/camera/${cameraId}/detection`);
      
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setDetectionResult(data.detection_result);
          setError(null);
        } else {
          setError(data.error || 'Failed to get detection results');
        }
      } else {
        setError('Failed to fetch detection results');
      }
    } catch (err) {
      setError('Network error while fetching detection results');
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch results periodically when enabled
  useEffect(() => {
    if (!isEnabled) {
      setDetectionResult(null);
      return;
    }

    const interval = setInterval(fetchDetectionResults, 1000); // Update every second
    fetchDetectionResults(); // Initial fetch

    return () => clearInterval(interval);
  }, [cameraId, isEnabled]);

  const handleToggleDetection = async (enabled: boolean) => {
    try {
      const response = await fetch(`http://127.0.0.1:5000/api/camera/${cameraId}/detection/toggle`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ enabled }),
      });

      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          onToggleDetection(enabled);
        }
      }
    } catch (err) {
      console.error('Failed to toggle detection:', err);
    }
  };

  const getDrowsinessColor = (state: string) => {
    switch (state) {
      case 'sleeping':
        return 'bg-red-500';
      case 'drowsy':
        return 'bg-orange-500';
      case 'awake':
        return 'bg-green-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getDrowsinessIcon = (state: string) => {
    switch (state) {
      case 'sleeping':
        return '😴';
      case 'drowsy':
        return '😪';
      case 'awake':
        return '😊';
      default:
        return '❓';
    }
  };

  return (
    <Card className="w-full">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          <Brain className="h-5 w-5" />
          YOLO Detection Panel
        </CardTitle>
        <div className="flex items-center space-x-2">
          <Switch
            id="detection-toggle"
            checked={isEnabled}
            onCheckedChange={handleToggleDetection}
          />
          <Label htmlFor="detection-toggle">
            {isEnabled ? 'Detection Enabled' : 'Detection Disabled'}
          </Label>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {error && (
          <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-md">
            <AlertTriangle className="h-4 w-4 text-red-500" />
            <span className="text-sm text-red-700">{error}</span>
          </div>
        )}

        {isLoading && (
          <div className="flex items-center gap-2 p-3 bg-blue-50 border border-blue-200 rounded-md">
            <Activity className="h-4 w-4 text-blue-500 animate-pulse" />
            <span className="text-sm text-blue-700">Loading detection results...</span>
          </div>
        )}

        {detectionResult && (
          <div className="space-y-4">
            {/* Performance Stats */}
            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-center gap-2 p-2 bg-gray-50 rounded-md">
                <Zap className="h-4 w-4 text-blue-500" />
                <span className="text-sm font-medium">FPS: {detectionResult.fps.toFixed(1)}</span>
              </div>
              <div className="flex items-center gap-2 p-2 bg-gray-50 rounded-md">
                <Clock className="h-4 w-4 text-green-500" />
                <span className="text-sm font-medium">
                  Process: {(detectionResult.processing_time * 1000).toFixed(1)}ms
                </span>
              </div>
            </div>

            {/* Detection Summary */}
            <div className="flex items-center gap-2 p-3 bg-blue-50 border border-blue-200 rounded-md">
              <Users className="h-4 w-4 text-blue-500" />
              <span className="text-sm font-medium">
                Detected {detectionResult.persons.length} person(s)
              </span>
            </div>

            {/* Person Detections */}
            {detectionResult.persons.length > 0 && (
              <div className="space-y-3">
                <h4 className="text-sm font-medium text-gray-700">Detected Persons:</h4>
                {detectionResult.persons.map((person) => (
                  <div key={person.id} className="p-3 border rounded-md bg-white">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <Target className="h-4 w-4 text-gray-500" />
                        <span className="text-sm font-medium">Person ID: {person.id}</span>
                      </div>
                      <Badge 
                        className={`${getDrowsinessColor(person.drowsiness_state)} text-white`}
                      >
                        {getDrowsinessIcon(person.drowsiness_state)} {person.drowsiness_state.toUpperCase()}
                      </Badge>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-2 text-xs text-gray-600">
                      <div>Confidence: {(person.confidence * 100).toFixed(1)}%</div>
                      <div>Drowsiness: {(person.drowsiness_score * 100).toFixed(1)}%</div>
                      <div>Keypoints: {person.keypoints.filter(k => k.visible).length}/17</div>
                      <div>BBox: [{person.bbox.map(b => Math.round(b)).join(', ')}]</div>
                    </div>

                    {/* Keypoints Info */}
                    <div className="mt-2 text-xs text-gray-500">
                      <div className="flex items-center gap-1">
                        <Eye className="h-3 w-3" />
                        <span>
                          Eyes: {person.keypoints.slice(1, 3).filter(k => k.visible).length}/2 visible
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* No detections */}
            {detectionResult.persons.length === 0 && (
              <div className="text-center py-4 text-gray-500">
                <Users className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">No persons detected in current frame</p>
              </div>
            )}
          </div>
        )}

        {!detectionResult && !isLoading && !error && isEnabled && (
          <div className="text-center py-4 text-gray-500">
            <Brain className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">Waiting for detection results...</p>
          </div>
        )}

        {!isEnabled && (
          <div className="text-center py-4 text-gray-500">
            <Brain className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">Detection is disabled</p>
            <p className="text-xs mt-1">Enable detection to see YOLO results</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
