import { Button } from './ui/button';
import { 
  Play, 
  Square, 
  Plus, 
  Trash2, 
  Upload, 
  Download, 
  Save, 
  RotateCcw,
  Eye,
  Activity,
  FileText,
  Moon,
  Sun,
  Settings,
  BarChart3
} from 'lucide-react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from './ui/tooltip';
import { Separator } from './ui/separator';

interface ToolbarProps {
  isAllRunning: boolean;
  onStartAll: () => void;
  onStopAll: () => void;
  onAddCamera: () => void;
  onDeleteCamera: () => void;
  onClearAllCameras: () => void;
  onImportConfig: () => void;
  onExportConfig: () => void;
  onSaveLayout: () => void;
  onRestoreLayout: () => void;
  onToggleOverlay: () => void;
  onTogglePerformance: () => void;
  onToggleLogging: () => void;
  onToggleTheme: () => void;
  onOpenSettings: () => void;
  onToggleStatistics?: () => void;
  isDarkMode: boolean;
  overlayEnabled: boolean;
  performanceEnabled: boolean;
  loggingEnabled: boolean;
  showStatistics?: boolean;
}

export function Toolbar({
  isAllRunning,
  onStartAll,
  onStopAll,
  onAddCamera,
  onDeleteCamera,
  onClearAllCameras,
  onImportConfig,
  onExportConfig,
  onSaveLayout,
  onRestoreLayout,
  onToggleOverlay,
  onTogglePerformance,
  onToggleLogging,
  onToggleTheme,
  onOpenSettings,
  onToggleStatistics,
  isDarkMode,
  overlayEnabled,
  performanceEnabled,
  loggingEnabled,
  showStatistics,
}: ToolbarProps) {
  return (
    <TooltipProvider>
      <div className="border-b px-3 py-2 flex items-center gap-2 bg-background">
        {/* App brand */}
        <div className="flex items-center gap-2 mr-2 select-none">
          <div className="h-8 w-8 rounded-md bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-sm">
            <Eye className="h-4 w-4 text-white" />
          </div>
          <div className="leading-tight">
            <div className="text-sm font-semibold">Drowsiness Detection</div>
            <div className="text-[11px] text-muted-foreground">YOLO Pose · Realtime</div>
          </div>
        </div>

        <Separator orientation="vertical" className="h-8" />

        {/* Trạng thái tổng */}
        <div className="hidden md:flex items-center gap-1.5 mr-1">
          <span className={`inline-block h-2 w-2 rounded-full ${isAllRunning ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />
          <span className="text-xs text-muted-foreground">
            {isAllRunning ? 'Đang chạy' : 'Đang tắt'}
          </span>
        </div>

        <Separator orientation="vertical" className="h-8" />

        <div className="flex items-center gap-2">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                size="sm"
                variant={isAllRunning ? 'secondary' : 'default'}
                onClick={onStartAll}
                disabled={isAllRunning}
              >
                <Play className="h-4 w-4 mr-1" />
                Start All
              </Button>
            </TooltipTrigger>
            <TooltipContent>Khởi động tất cả camera</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                size="sm"
                variant={isAllRunning ? 'destructive' : 'outline'}
                onClick={onStopAll}
                disabled={!isAllRunning}
              >
                <Square className="h-4 w-4 mr-1" />
                Stop All
              </Button>
            </TooltipTrigger>
            <TooltipContent>Dừng tất cả camera</TooltipContent>
          </Tooltip>
        </div>

        <Separator orientation="vertical" className="h-8" />

        <div className="flex items-center gap-2">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button size="sm" variant="outline" onClick={onAddCamera}>
                <Plus className="h-4 w-4 mr-1" />
                Thêm
              </Button>
            </TooltipTrigger>
            <TooltipContent>Thêm camera mới</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button size="sm" variant="outline" onClick={onDeleteCamera}>
                <Trash2 className="h-4 w-4 mr-1" />
                Xóa
              </Button>
            </TooltipTrigger>
            <TooltipContent>Xóa camera đã chọn</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button size="sm" variant="destructive" onClick={onClearAllCameras}>
                <Trash2 className="h-4 w-4 mr-1" />
                Clear All
              </Button>
            </TooltipTrigger>
            <TooltipContent>Xóa tất cả camera</TooltipContent>
          </Tooltip>
        </div>

        <Separator orientation="vertical" className="h-8" />

        <div className="flex items-center gap-2">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button size="sm" variant="outline" onClick={onImportConfig}>
                <Upload className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Import YAML</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button size="sm" variant="outline" onClick={onExportConfig}>
                <Download className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Export YAML</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button size="sm" variant="outline" onClick={onSaveLayout}>
                <Save className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Lưu bố cục</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button size="sm" variant="outline" onClick={onRestoreLayout}>
                <RotateCcw className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Khôi phục bố cục</TooltipContent>
          </Tooltip>
        </div>

        <Separator orientation="vertical" className="h-8" />

        <div className="flex items-center gap-2">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button 
                size="sm" 
                variant={overlayEnabled ? 'default' : 'outline'}
                onClick={onToggleOverlay}
              >
                <Eye className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Toggle Overlay</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button 
                size="sm" 
                variant={performanceEnabled ? 'default' : 'outline'}
                onClick={onTogglePerformance}
              >
                <Activity className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Toggle Performance HUD</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button 
                size="sm" 
                variant={loggingEnabled ? 'default' : 'outline'}
                onClick={onToggleLogging}
              >
                <FileText className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Toggle Logging</TooltipContent>
          </Tooltip>

          {onToggleStatistics && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button 
                  size="sm" 
                  variant={showStatistics ? 'default' : 'outline'}
                  onClick={onToggleStatistics}
                >
                  <BarChart3 className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Thống kê ngủ gật</TooltipContent>
            </Tooltip>
          )}
        </div>

        <div className="ml-auto flex items-center gap-2">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button size="sm" variant="outline" onClick={onOpenSettings}>
                <Settings className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Cài đặt</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button size="sm" variant="outline" onClick={onToggleTheme}>
                {isDarkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
              </Button>
            </TooltipTrigger>
            <TooltipContent>{isDarkMode ? 'Light Mode' : 'Dark Mode'}</TooltipContent>
          </Tooltip>
        </div>
      </div>
    </TooltipProvider>
  );
}
