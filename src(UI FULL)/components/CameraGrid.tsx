import { Camera } from '../types';
import { CameraCard } from './CameraCard';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from './ui/select';
import { Grid2X2, Grid3X3, LayoutGrid } from 'lucide-react';

interface CameraGridProps {
  cameras: Camera[];
  gridSize: '1x1' | '2x2' | '3x3' | '4x4';
  onGridSizeChange: (size: '1x1' | '2x2' | '3x3' | '4x4') => void;
  onToggleCamera: (cameraId: string) => void;
  onPopOut: (cameraId: string) => void;
  onConfigure: (cameraId: string) => void;
  showOverlay: boolean;
  showPerformance: boolean;
}

export function CameraGrid({
  cameras,
  gridSize,
  onGridSizeChange,
  onToggleCamera,
  onPopOut,
  onConfigure,
  showOverlay,
  showPerformance,
}: CameraGridProps) {
  const getGridCols = () => {
    switch (gridSize) {
      case '1x1':
        return 'grid-cols-1';
      case '2x2':
        return 'grid-cols-2';
      case '3x3':
        return 'grid-cols-3';
      case '4x4':
        return 'grid-cols-4';
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-3 border-b flex items-center justify-between bg-muted/50">
        <div className="flex items-center gap-2">
          <LayoutGrid className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm">Camera Grid</span>
        </div>
        <Select value={gridSize} onValueChange={onGridSizeChange as any}>
          <SelectTrigger className="w-32">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="1x1">
              <div className="flex items-center gap-2">
                <Grid2X2 className="h-4 w-4" />
                1×1
              </div>
            </SelectItem>
            <SelectItem value="2x2">
              <div className="flex items-center gap-2">
                <Grid2X2 className="h-4 w-4" />
                2×2
              </div>
            </SelectItem>
            <SelectItem value="3x3">
              <div className="flex items-center gap-2">
                <Grid3X3 className="h-4 w-4" />
                3×3
              </div>
            </SelectItem>
            <SelectItem value="4x4">
              <div className="flex items-center gap-2">
                <Grid3X3 className="h-4 w-4" />
                4×4
              </div>
            </SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="flex-1 p-4 overflow-auto">
        <div className={`grid ${getGridCols()} gap-4 auto-rows-fr`}>
          {cameras.map((camera) => (
            <CameraCard
              key={camera.id}
              camera={camera}
              onToggle={onToggleCamera}
              onPopOut={onPopOut}
              onConfigure={onConfigure}
              showOverlay={showOverlay}
              showPerformance={showPerformance}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
