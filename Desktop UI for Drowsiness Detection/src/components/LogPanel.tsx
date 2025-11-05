import { useState } from 'react';
import { LogEvent, Camera } from '../types';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { ScrollArea } from './ui/scroll-area';
import { Badge } from './ui/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from './ui/select';
import { Download, Search, Filter } from 'lucide-react';
import { cn } from './ui/utils';

interface LogPanelProps {
  logs: LogEvent[];
  cameras: Camera[];
  onExport: () => void;
}

export function LogPanel({ logs, cameras, onExport }: LogPanelProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [filterCamera, setFilterCamera] = useState<string>('all');
  const [filterType, setFilterType] = useState<string>('all');

  const filteredLogs = logs.filter(log => {
    const matchesSearch = log.message.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         log.cameraName.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCamera = filterCamera === 'all' || log.cameraId === filterCamera;
    const matchesType = filterType === 'all' || log.type === filterType;
    return matchesSearch && matchesCamera && matchesType;
  });

  const getLogColor = (type: LogEvent['type']) => {
    switch (type) {
      case 'sleepy':
        return 'bg-red-500/10 text-red-500 border-red-500/20';
      case 'wake_up':
        return 'bg-green-500/10 text-green-500 border-green-500/20';
      case 'head_down':
      case 'sleeping':
        return 'bg-purple-500/10 text-purple-500 border-purple-500/20';
      case 'connection':
        return 'bg-blue-500/10 text-blue-500 border-blue-500/20';
      case 'error':
        return 'bg-orange-500/10 text-orange-500 border-orange-500/20';
    }
  };

  const getLogTypeText = (type: LogEvent['type']) => {
    switch (type) {
      case 'sleepy':
        return 'Buồn ngủ';
      case 'wake_up':
        return 'Tỉnh táo';
      case 'head_down':
      case 'sleeping':
        return 'Gục xuống';
      case 'connection':
        return 'Kết nối';
      case 'error':
        return 'Lỗi';
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 border-b space-y-3">
        <div className="flex items-center justify-between">
          <h3>Log Sự kiện</h3>
          <Button size="sm" variant="outline" onClick={onExport}>
            <Download className="h-4 w-4 mr-1" />
            Export CSV
          </Button>
        </div>

        <div className="relative">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Tìm kiếm..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-8"
          />
        </div>

        <div className="flex gap-2">
          <Select value={filterCamera} onValueChange={setFilterCamera}>
            <SelectTrigger className="flex-1">
              <SelectValue placeholder="Lọc theo camera" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">Tất cả camera</SelectItem>
              {cameras.map(camera => (
                <SelectItem key={camera.id} value={camera.id}>
                  {camera.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Select value={filterType} onValueChange={setFilterType}>
            <SelectTrigger className="flex-1">
              <SelectValue placeholder="Loại sự kiện" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">Tất cả</SelectItem>
              <SelectItem value="sleepy">Buồn ngủ</SelectItem>
              <SelectItem value="sleeping">Ngủ</SelectItem>
              <SelectItem value="wake_up">Tỉnh táo</SelectItem>
              <SelectItem value="head_down">Gục xuống</SelectItem>
              <SelectItem value="connection">Kết nối</SelectItem>
              <SelectItem value="error">Lỗi</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-4 space-y-2">
          {filteredLogs.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <Filter className="h-12 w-12 mx-auto mb-2 opacity-20" />
              <p>Không có log nào</p>
            </div>
          ) : (
            filteredLogs.map((log) => (
              <div
                key={log.id}
                className="p-3 border rounded-lg hover:bg-accent transition-colors"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2 flex-wrap">
                    <Badge variant="outline" className={cn("text-xs", getLogColor(log.type))}>
                      {getLogTypeText(log.type)}
                    </Badge>
                    <span className="text-sm truncate">{log.cameraName}</span>
                  </div>
                  <span className="text-xs text-muted-foreground shrink-0">
                    {log.timestamp.toLocaleTimeString('vi-VN')}
                  </span>
                </div>
                <p className="text-sm text-muted-foreground">{log.message}</p>
                {log.studentPosition && (
                  <p className="text-xs text-muted-foreground mt-1">
                    Vị trí: {log.studentPosition}
                  </p>
                )}
                {log.duration !== undefined && log.duration > 0 && (
                  <p className="text-xs text-muted-foreground mt-1">
                    Thời lượng: {Math.floor(log.duration / 60)}:{(log.duration % 60).toString().padStart(2, '0')}
                  </p>
                )}
                {log.studentCount !== undefined && (
                  <p className="text-xs text-muted-foreground mt-1">
                    Số học sinh: {log.studentCount}
                  </p>
                )}
              </div>
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
