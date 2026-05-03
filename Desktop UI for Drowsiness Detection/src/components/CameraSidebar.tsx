import { useState } from 'react';
import { Camera } from '../types';
import { Input } from './ui/input';
import { ScrollArea } from './ui/scroll-area';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Search, Circle, AlertCircle, Loader2, Settings, Plus, Trash2 } from 'lucide-react';
import { cn } from './ui/utils';

interface CameraSidebarProps {
  cameras: Camera[];
  selectedCameraId?: string;
  onSelectCamera: (cameraId: string) => void;
  onAddCamera: () => void;
  onDeleteCamera: () => void;
  onConfigureCamera: (cameraId: string) => void;
}

export function CameraSidebar({ 
  cameras, 
  selectedCameraId, 
  onSelectCamera, 
  onAddCamera, 
  onDeleteCamera, 
  onConfigureCamera 
}: CameraSidebarProps) {
  const [searchQuery, setSearchQuery] = useState('');

  const filteredCameras = cameras.filter(camera =>
    camera.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getStatusIcon = (status: Camera['status']) => {
    switch (status) {
      case 'online':
        return <Circle className="h-3 w-3 fill-green-500 text-green-500" />;
      case 'offline':
        return <Circle className="h-3 w-3 fill-red-500 text-red-500" />;
      case 'reconnecting':
        return <Loader2 className="h-3 w-3 text-orange-500 animate-spin" />;
    }
  };

  const getStudentStats = (camera: Camera) => {
    if (camera.sleepyStudents > 0) {
      return {
        text: `${camera.sleepyStudents}/${camera.students.length} Buồn ngủ`,
        color: 'bg-red-500/10 text-red-500 border-red-500/20',
      };
    }
    if (camera.students.length > 0) {
      return {
        text: `${camera.students.length}/${camera.totalStudents} Học sinh`,
        color: 'bg-green-500/10 text-green-500 border-green-500/20',
      };
    }
    return {
      text: 'Chưa phát hiện',
      color: 'bg-gray-500/10 text-gray-500 border-gray-500/20',
    };
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 border-b">
        <div className="flex items-center justify-between mb-3">
          <h3>Danh sách Camera</h3>
          <Button
            size="sm"
            variant="outline"
            onClick={onAddCamera}
            className="h-8 px-2"
          >
            <Plus className="h-4 w-4 mr-1" />
            Thêm
          </Button>
        </div>
        <div className="relative">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Tìm kiếm camera..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-8"
          />
        </div>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-2 space-y-1">
          {filteredCameras.map((camera) => (
            <button
              key={camera.id}
              onClick={() => onSelectCamera(camera.id)}
              className={cn(
                "w-full text-left p-3 rounded-lg border transition-all hover:bg-accent",
                selectedCameraId === camera.id && "bg-accent border-primary"
              )}
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  {getStatusIcon(camera.status)}
                  <span className="text-sm truncate">{camera.name}</span>
                </div>
                <div className="flex items-center gap-1">
                  {camera.sleepyStudents > 0 && (
                    <Badge variant="destructive" className="h-5 px-1.5 text-xs shrink-0">
                      {camera.sleepyStudents}
                    </Badge>
                  )}
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-6 w-6 p-0"
                    onClick={(e) => {
                      e.stopPropagation();
                      onConfigureCamera(camera.id);
                    }}
                  >
                    <Settings className="h-3 w-3" />
                  </Button>
                </div>
              </div>

              <div className="flex items-center gap-2 flex-wrap">
                <Badge variant="outline" className={cn("text-xs", getStudentStats(camera).color)}>
                  {getStudentStats(camera).text}
                </Badge>
                {camera.status === 'online' && (
                  <span className="text-xs text-muted-foreground font-mono">
                    {camera.fps} FPS
                  </span>
                )}
              </div>

              {camera.status === 'reconnecting' && (
                <div className="flex items-center gap-1 mt-2 text-xs text-orange-500">
                  <AlertCircle className="h-3 w-3" />
                  Đang kết nối lại...
                </div>
              )}
            </button>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}
