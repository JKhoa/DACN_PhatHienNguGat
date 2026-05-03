import { useEffect, useRef, useState } from 'react';
import { Camera } from '../types';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from './ui/dropdown-menu';
import {
  Play,
  Square,
  MoreVertical,
  Maximize2,
  Tag,
  Eye,
  FileText,
  Camera as CameraIcon,
  Video,
  Circle,
  AlertCircle,
  Loader2,
} from 'lucide-react';
import { cn } from './ui/utils';

interface CameraCardProps {
  camera: Camera;
  onToggle: (cameraId: string) => void;
  onPopOut: (cameraId: string) => void;
  onConfigure: (cameraId: string) => void;
  showOverlay: boolean;
  showPerformance: boolean;
}

export function CameraCard({
  camera,
  onToggle,
  onPopOut,
  onConfigure,
  showOverlay,
  showPerformance,
}: CameraCardProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [sleepDuration, setSleepDuration] = useState(0);

  // Mock classroom video feed with multiple students
  useEffect(() => {
    if (!canvasRef.current || camera.status === 'offline') return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let frameId: number;
    let time = 0;

    const drawFrame = () => {
      time += 0.016;
      
      // Classroom background
      ctx.fillStyle = '#1a1a1a';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Draw desks grid (simulated classroom)
      ctx.strokeStyle = '#333';
      ctx.lineWidth = 1;
      for (let row = 0; row < 5; row++) {
        for (let col = 0; col < 8; col++) {
          const x = 40 + col * 75;
          const y = 60 + row * 60;
          ctx.strokeRect(x, y, 60, 45);
        }
      }

      // Draw students
      camera.students.forEach((student, index) => {
        const { x, y } = student.position;
        
        // Student color based on state
        let color = '#22c55e'; // green - normal
        if (student.state === 'sleepy') color = '#ef4444'; // red
        if (student.state === 'head_down') color = '#a855f7'; // purple
        
        // Draw student circle (head)
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.fill();
        
        if (showOverlay) {
          // Draw bounding box for detected students
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.5;
          ctx.strokeRect(x - 15, y - 15, 30, 40);
          
          // Draw keypoints
          ctx.fillStyle = color;
          // Shoulders
          ctx.beginPath();
          ctx.arc(x - 8, y + 10, 3, 0, Math.PI * 2);
          ctx.fill();
          ctx.beginPath();
          ctx.arc(x + 8, y + 10, 3, 0, Math.PI * 2);
          ctx.fill();
          
          // Confidence label
          if (student.confidence > 0.7) {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.fillRect(x - 15, y - 25, 30, 8);
            ctx.fillStyle = '#fff';
            ctx.font = '7px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(`${(student.confidence * 100).toFixed(0)}%`, x, y - 19);
            ctx.textAlign = 'left';
          }
        }
      });

      // Performance HUD
      if (showPerformance) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(5, 5, 120, 80);
        ctx.fillStyle = '#fff';
        ctx.font = '11px monospace';
        ctx.fillText(`FPS: ${camera.fps}`, 10, 20);
        ctx.fillText(`Students: ${camera.students.length}`, 10, 35);
        ctx.fillText(`Sleepy: ${camera.sleepyStudents}`, 10, 50);
        ctx.fillText(`Latency: ${Math.floor(Math.random() * 50 + 20)}ms`, 10, 65);
        ctx.fillText(`Conf: ${(camera.students.reduce((sum, s) => sum + s.confidence, 0) / camera.students.length || 0).toFixed(2)}`, 10, 80);
      }
      
      // Camera info overlay
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(0, 0, canvas.width, 25);
      ctx.fillStyle = '#fff';
      ctx.font = '12px sans-serif';
      ctx.fillText(camera.name, 10, 16);
      
      if (camera.sleepyStudents > 0) {
        ctx.fillStyle = 'rgba(239, 68, 68, 0.9)';
        ctx.fillRect(canvas.width - 80, 5, 75, 15);
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 11px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(`⚠ ${camera.sleepyStudents} buồn ngủ`, canvas.width - 5, 16);
        ctx.textAlign = 'left';
      }

      if (camera.isRunning) {
        frameId = requestAnimationFrame(drawFrame);
      }
    };

    drawFrame();

    return () => {
      cancelAnimationFrame(frameId);
    };
  }, [camera.isRunning, camera.status, camera.students, camera.fps, camera.sleepyStudents, showOverlay, showPerformance, camera.name]);

  // Update sleep duration from longest sleeping student
  useEffect(() => {
    const maxSleepDuration = Math.max(
      ...camera.students.filter(s => s.state === 'sleepy').map(s => s.sleepDuration),
      0
    );
    setSleepDuration(maxSleepDuration);
  }, [camera.students]);

  const getStatusIcon = () => {
    switch (camera.status) {
      case 'online':
        return <Circle className="h-3 w-3 fill-green-500 text-green-500" />;
      case 'offline':
        return <Circle className="h-3 w-3 fill-red-500 text-red-500" />;
      case 'reconnecting':
        return <Loader2 className="h-3 w-3 text-orange-500 animate-spin" />;
    }
  };

  const getStatusBadge = () => {
    if (camera.sleepyStudents > 0) {
      return {
        text: `${camera.sleepyStudents} Buồn ngủ`,
        class: 'bg-red-500/10 text-red-500 border-red-500/20',
      };
    }
    if (camera.status === 'online' && camera.students.length > 0) {
      return {
        text: `${camera.students.length}/${camera.totalStudents} HS`,
        class: 'bg-green-500/10 text-green-500 border-green-500/20',
      };
    }
    return {
      text: 'Chưa phát hiện',
      class: 'bg-gray-500/10 text-gray-500 border-gray-500/20',
    };
  };

  return (
    <div className="border rounded-lg overflow-hidden bg-card hover:shadow-lg transition-shadow">
      {/* Header */}
      <div className="p-2 border-b flex items-center justify-between bg-muted/50">
        <div className="flex items-center gap-2 flex-1 min-w-0">
          {getStatusIcon()}
          <span className="text-sm truncate">{camera.name}</span>
        </div>

        <div className="flex items-center gap-1">
          <Badge variant="outline" className={cn("text-xs", getStatusBadge().class)}>
            {getStatusBadge().text}
          </Badge>
          
          {camera.status === 'online' && (
            <span className="text-xs text-muted-foreground font-mono ml-2">
              {camera.fps} FPS
            </span>
          )}

          <Button
            size="sm"
            variant="ghost"
            className="h-6 w-6 p-0"
            onClick={() => onToggle(camera.id)}
          >
            {camera.isRunning ? (
              <Square className="h-3 w-3" />
            ) : (
              <Play className="h-3 w-3" />
            )}
          </Button>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button size="sm" variant="ghost" className="h-6 w-6 p-0">
                <MoreVertical className="h-3 w-3" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => onPopOut(camera.id)}>
                <Maximize2 className="h-4 w-4 mr-2" />
                Pop Out
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => onConfigure(camera.id)}>
                <Tag className="h-4 w-4 mr-2" />
                Cấu hình
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem>
                <Eye className="h-4 w-4 mr-2" />
                Toggle Overlay
              </DropdownMenuItem>
              <DropdownMenuItem>
                <FileText className="h-4 w-4 mr-2" />
                Toggle Logging
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem>
                <CameraIcon className="h-4 w-4 mr-2" />
                Chụp ảnh
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Video className="h-4 w-4 mr-2" />
                Ghi video
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* Video Feed */}
      <div className="relative aspect-video bg-black">
        {camera.status === 'offline' ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-2" />
              <p className="text-sm text-muted-foreground">Camera offline</p>
            </div>
          </div>
        ) : camera.status === 'reconnecting' ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <Loader2 className="h-12 w-12 text-orange-500 mx-auto mb-2 animate-spin" />
              <p className="text-sm text-muted-foreground">Đang kết nối lại...</p>
            </div>
          </div>
        ) : (
          <canvas
            ref={canvasRef}
            width={640}
            height={360}
            className="w-full h-full"
          />
        )}

        {/* Sleepy Students Count Badge */}
        {camera.sleepyStudents > 0 && (
          <div className="absolute top-2 right-2 flex gap-2">
            <Badge variant="destructive" className="animate-pulse">
              ⚠ {camera.sleepyStudents} học sinh
            </Badge>
            {sleepDuration > 0 && (
              <Badge variant="destructive">
                {Math.floor(sleepDuration / 60)}:{(sleepDuration % 60).toString().padStart(2, '0')}
              </Badge>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
