import { useState, useEffect } from 'react';
import { LogEvent, Camera } from '../types';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { ScrollArea } from './ui/scroll-area';
import { Badge } from './ui/badge';
import { Card } from './ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from './ui/select';
import { Download, Search, Filter, Calendar, TrendingUp, Users } from 'lucide-react';
import { cn } from './ui/utils';
import { format } from 'date-fns';
import { vi } from 'date-fns/locale';
import { apiGet } from '../lib/api';

interface LogPanelProps {
  logs: LogEvent[];
  cameras: Camera[];
  onExport: () => void;
}

export function LogPanel({ logs, cameras, onExport }: LogPanelProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [filterCamera, setFilterCamera] = useState<string>('all');
  const [filterType, setFilterType] = useState<string>('all');
  const [period, setPeriod] = useState<string>('today');
  const [startHour, setStartHour] = useState<string>('00');
  const [endHour, setEndHour] = useState<string>('23');
  
  // Statistics from backend
  const [stats, setStats] = useState<any>(null);
  const [weeklyStats, setWeeklyStats] = useState<any>(null);
  const [monthlyStats, setMonthlyStats] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [backendLogs, setBackendLogs] = useState<LogEvent[]>([]);

  // Fetch statistics and logs from backend
  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      try {
        // Fetch today stats
        const todayRes = await apiGet('api/logs/summary?period=today');
        if (todayRes.ok) {
          const data = await todayRes.json();
          setStats(data.summary);
        }

        // Fetch weekly stats
        const weekRes = await apiGet('api/logs/summary?period=week');
        if (weekRes.ok) {
          const data = await weekRes.json();
          setWeeklyStats(data.summary);
        }

        // Fetch monthly stats
        const monthRes = await apiGet('api/logs/summary?period=month');
        if (monthRes.ok) {
          const data = await monthRes.json();
          setMonthlyStats(data.summary);
        }

        // Fetch all logs from backend - get from all cameras
        const allCameraLogs: LogEvent[] = [];

        // Get list of cameras first
        const camerasRes = await apiGet('api/logs/cameras');
        if (camerasRes.ok) {
          const camerasData = await camerasRes.json();
          const cameraList = camerasData.cameras || [];

          // Fetch logs for each camera
          for (const cam of cameraList) {
            try {
              const logsRes = await apiGet(`api/logs/events/${cam.id}?period=today`);
              if (logsRes.ok) {
                const data = await logsRes.json();
                if (data.success && data.events) {
                  // Convert backend events to LogEvent format
                  const convertedLogs: LogEvent[] = data.events.map((event: any, index: number) => ({
                    id: `log-${cam.id}-${Date.now()}-${index}`,
                    timestamp: new Date(event.timestamp),
                    type: event.event_type === 'start_drowsy' ? 'sleepy' : 
                          event.event_type === 'end_drowsy' ? 'wake_up' : 
                          event.event_type as LogEvent['type'],
                    message: event.event_type === 'start_drowsy' 
                      ? `Học sinh #${event.student_id} BẮT ĐẦU ngủ gật`
                      : event.event_type === 'end_drowsy'
                      ? `Học sinh #${event.student_id} TỈNH LẠI (Ngủ gật: ${event.duration_display || ''})`
                      : event.message || '',
                    cameraId: event.camera_id,
                    cameraName: event.camera_name || cameras.find(c => c.id === event.camera_id)?.name || event.camera_id,
                    studentPosition: event.student_id ? `#${event.student_id}` : undefined,
                    duration: event.duration_seconds,
                  }));
                  allCameraLogs.push(...convertedLogs);
                }
              }
            } catch (err) {
              console.error(`Error fetching logs for camera ${cam.id}:`, err);
            }
          }
        }
        
        setBackendLogs(allCameraLogs);
      } catch (error) {
        console.error('Error fetching data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
    // Refresh every 5 seconds to catch new drowsy events
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, [cameras]);

  // Merge props logs and backend logs
  const allLogs = [...logs, ...backendLogs].sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());

  const filteredLogs = allLogs.filter(log => {
    const matchesSearch = log.message.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         log.cameraName.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCamera = filterCamera === 'all' || log.cameraId === filterCamera;
    const matchesType = filterType === 'all' || log.type === filterType;
    
    // Filter by time range (hour)
    const logHour = log.timestamp.getHours();
    const matchesTimeRange = logHour >= parseInt(startHour) && logHour <= parseInt(endHour);
    
    return matchesSearch && matchesCamera && matchesType && matchesTimeRange;
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
      {/* Statistics Cards */}
      <div className="p-4 border-b bg-muted/30">
        <div className="grid grid-cols-3 gap-3 mb-3">
          {/* Today Stats */}
          <Card className="p-3">
            <div className="flex items-center gap-2 mb-2">
              <Calendar className="h-4 w-4 text-blue-500" />
              <span className="text-xs font-medium">Hôm nay</span>
            </div>
            {stats ? (
              <>
                <div className="text-2xl font-bold text-blue-600">
                  {stats.total_drowsy_students_unique || 0}
                </div>
                <div className="text-xs text-muted-foreground">
                  Học sinh ngủ gật
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  {stats.total_events || 0} sự kiện • {stats.total_duration_display || '0s'}
                </div>
              </>
            ) : (
              <div className="text-sm text-muted-foreground">Đang tải...</div>
            )}
          </Card>

          {/* Weekly Stats */}
          <Card className="p-3">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="h-4 w-4 text-green-500" />
              <span className="text-xs font-medium">Tuần này</span>
            </div>
            {weeklyStats ? (
              <>
                <div className="text-2xl font-bold text-green-600">
                  {weeklyStats.total_drowsy_students_unique || 0}
                </div>
                <div className="text-xs text-muted-foreground">
                  Tổng học sinh ngủ gật
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  {weeklyStats.total_events || 0} sự kiện • {weeklyStats.total_duration_display || '0s'}
                </div>
              </>
            ) : (
              <div className="text-sm text-muted-foreground">Đang tải...</div>
            )}
          </Card>

          {/* Monthly Stats */}
          <Card className="p-3">
            <div className="flex items-center gap-2 mb-2">
              <Users className="h-4 w-4 text-purple-500" />
              <span className="text-xs font-medium">Tháng này</span>
            </div>
            {monthlyStats ? (
              <>
                <div className="text-2xl font-bold text-purple-600">
                  {monthlyStats.total_drowsy_students_unique || 0}
                </div>
                <div className="text-xs text-muted-foreground">
                  Tổng học sinh ngủ gật
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  {monthlyStats.total_events || 0} sự kiện • {monthlyStats.total_duration_display || '0s'}
                </div>
              </>
            ) : (
              <div className="text-sm text-muted-foreground">Đang tải...</div>
            )}
          </Card>
        </div>
      </div>

      <div className="p-4 border-b space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold">Log Sự kiện</h3>
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

        <div className="grid grid-cols-2 gap-2">
          <Select value={filterCamera} onValueChange={setFilterCamera}>
            <SelectTrigger>
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
            <SelectTrigger>
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

        {/* Time Range Filter */}
        <div className="flex items-center gap-2">
          <Calendar className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm text-muted-foreground">Khoảng giờ:</span>
          <Select value={startHour} onValueChange={setStartHour}>
            <SelectTrigger className="w-20">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {Array.from({ length: 24 }, (_, i) => (
                <SelectItem key={i} value={i.toString().padStart(2, '0')}>
                  {i.toString().padStart(2, '0')}:00
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <span className="text-sm text-muted-foreground">đến</span>
          <Select value={endHour} onValueChange={setEndHour}>
            <SelectTrigger className="w-20">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {Array.from({ length: 24 }, (_, i) => (
                <SelectItem key={i} value={i.toString().padStart(2, '0')}>
                  {i.toString().padStart(2, '0')}:59
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {(startHour !== '00' || endHour !== '23') && (
            <Button 
              size="sm" 
              variant="ghost" 
              onClick={() => {
                setStartHour('00');
                setEndHour('23');
              }}
            >
              Reset
            </Button>
          )}
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
                <div className="flex items-center gap-2">
                  <p className="text-sm text-muted-foreground">{log.message}</p>
                  {log.studentPosition && (
                    <Badge variant="secondary" className="text-xs">
                      {log.studentPosition}
                    </Badge>
                  )}
                </div>
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
