import { SystemStats } from '../types';
import { Badge } from './ui/badge';
import { Activity, Camera, Cpu, AlertCircle, Users, AlertTriangle } from 'lucide-react';

interface StatusBarProps {
  stats: SystemStats;
}

export function StatusBar({ stats }: StatusBarProps) {
  return (
    <div className="border-t px-4 py-2 flex items-center gap-4 bg-muted/50">
      <div className="flex items-center gap-2">
        <Activity className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm">
          FPS: <span className="font-mono">{stats.totalFPS.toFixed(1)}</span>
        </span>
      </div>

      <div className="flex items-center gap-2">
        <Camera className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm">
          Camera: <span className="font-mono">{stats.runningCameras}/{stats.totalCameras}</span>
        </span>
      </div>

      <div className="flex items-center gap-2">
        <Users className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm">
          Học sinh: <span className="font-mono">{stats.totalStudents}</span>
        </span>
      </div>

      {stats.sleepyStudents > 0 && (
        <div className="flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-red-500" />
          <Badge variant="destructive">
            {stats.sleepyStudents} buồn ngủ
          </Badge>
        </div>
      )}

      <div className="flex items-center gap-2">
        <Cpu className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm">
          CPU: <span className="font-mono">{stats.cpuUsage}%</span>
        </span>
        {stats.gpuUsage !== undefined && (
          <span className="text-sm ml-2">
            GPU: <span className="font-mono">{stats.gpuUsage}%</span>
          </span>
        )}
      </div>

      {stats.reconnectCount > 0 && (
        <div className="flex items-center gap-2">
          <AlertCircle className="h-4 w-4 text-orange-500" />
          <Badge variant="outline" className="text-orange-500">
            {stats.reconnectCount} đang kết nối lại
          </Badge>
        </div>
      )}

      <div className="ml-auto text-sm text-muted-foreground">
        {new Date().toLocaleTimeString('vi-VN')}
      </div>
    </div>
  );
}
