import { useState, useEffect } from 'react';
import { Camera } from '../types';
import { Card } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { ScrollArea } from './ui/scroll-area';
import { Calendar } from './ui/calendar';
import { Popover, PopoverContent, PopoverTrigger } from './ui/popover';
import { CalendarIcon, BarChart3, TrendingUp, Users, Clock, Download, AlertTriangle, Sun, Moon, Sunset } from 'lucide-react';
import { cn } from './ui/utils';

// Date utility functions
const formatDate = (date: Date): string => {
  const day = date.getDate().toString().padStart(2, '0');
  const month = (date.getMonth() + 1).toString().padStart(2, '0');
  const year = date.getFullYear();
  return `${day}/${month}/${year}`;
};

const startOfMonth = (date: Date): Date => {
  return new Date(date.getFullYear(), date.getMonth(), 1);
};

const endOfMonth = (date: Date): Date => {
  return new Date(date.getFullYear(), date.getMonth() + 1, 0);
};

const startOfWeek = (date: Date): Date => {
  const d = new Date(date);
  const day = d.getDay();
  const diff = d.getDate() - day + (day === 0 ? -6 : 1); // adjust when day is Sunday
  return new Date(d.setDate(diff));
};

const endOfWeek = (date: Date): Date => {
  const d = new Date(startOfWeek(date));
  return new Date(d.setDate(d.getDate() + 6));
};

const format = (date: Date, formatStr: string): string => {
  if (formatStr === 'dd/MM/yyyy') {
    return formatDate(date);
  }
  // Add more formats as needed
  return date.toLocaleDateString('vi-VN');
};

interface StatisticsPanelProps {
  cameras: Camera[];
}

interface Statistics {
  totalDrowsy: number;
  totalSleeping: number;
  totalWakeUps: number;
  byCamera: Array<{
    cameraId: string;
    cameraName: string;
    drowsy: number;
    sleeping: number;
    wakeUps: number;
    currentDrowsy?: number; // Số học sinh đang ngủ gật hiện tại trong phòng
    totalStudents?: number; // Tổng số học sinh trong phòng (ước tính)
    drowsyRate?: number; // Tỷ lệ ngủ gật (%)
  }>;
  byDate: Array<{
    date: string;
    drowsy: number;
    sleeping: number;
  }>;
  byTimeSlot: Array<{
    timeSlot: string; // 'morning', 'afternoon', 'evening'
    drowsy: number;
    sleeping: number;
    totalStudents: number;
    drowsyRate: number;
  }>;
  currentDrowsyCount: number; // Số học sinh đang ngủ gật hiện tại
  alerts?: Array<{
    cameraId: string;
    cameraName: string;
    message: string;
    severity: 'warning' | 'critical';
    drowsyRate: number;
  }>;
}

export function StatisticsPanel({ cameras }: StatisticsPanelProps) {
  const [selectedCamera, setSelectedCamera] = useState<string>('all');
  const [timeRange, setTimeRange] = useState<'today' | 'week' | 'month' | 'custom'>('today');
  const [startDate, setStartDate] = useState<Date>(new Date());
  const [endDate, setEndDate] = useState<Date>(new Date());
  const [statistics, setStatistics] = useState<Statistics | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchStatistics = async (showLoading = true) => {
    if (showLoading) {
      setLoading(true);
    }
    try {
      let start: Date;
      let end: Date = new Date();

      switch (timeRange) {
        case 'today':
          start = new Date();
          start.setHours(0, 0, 0, 0);
          break;
        case 'week':
          start = startOfWeek(new Date());
          end = endOfWeek(new Date());
          break;
        case 'month':
          start = startOfMonth(new Date());
          end = endOfMonth(new Date());
          break;
        case 'custom':
          start = startDate;
          end = endDate;
          break;
        default:
          start = new Date();
          start.setHours(0, 0, 0, 0);
      }

      const params = new URLSearchParams({
        start_time: start.toISOString(),
        end_time: end.toISOString(),
        ...(selectedCamera !== 'all' && { camera_id: selectedCamera }),
      });

      const response = await fetch(`http://127.0.0.1:5000/api/statistics?${params}`);
      const data = await response.json();

      if (data.success) {
        setStatistics(data.statistics);
      }
    } catch (error) {
      console.error('Error fetching statistics:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Initial load with loading indicator
    fetchStatistics(true);
    
    // Auto-refresh statistics every 3 seconds to update current drowsy count (without loading indicator)
    const interval = setInterval(() => fetchStatistics(false), 3000);
    
    return () => clearInterval(interval);
  }, [selectedCamera, timeRange, startDate, endDate]);

  const exportStatistics = () => {
    if (!statistics) return;

    const csv = [
      'Thống kê ngủ gật',
      `Khoảng thời gian: ${timeRange === 'custom' ? format(startDate, 'dd/MM/yyyy') : timeRange} - ${timeRange === 'custom' ? format(endDate, 'dd/MM/yyyy') : format(new Date(), 'dd/MM/yyyy')}`,
      '',
      'Tổng quát',
      `Tổng buồn ngủ: ${statistics.totalDrowsy}`,
      `Tổng ngủ gật: ${statistics.totalSleeping}`,
      `Tổng tỉnh lại: ${statistics.totalWakeUps}`,
      '',
      'Theo phòng',
      'Phòng,Đang ngủ gật,Buồn ngủ,Ngủ gật,Tỉnh lại',
      ...statistics.byCamera.map(cam => 
        `${cam.cameraName},${cam.currentDrowsy || 0},${cam.drowsy},${cam.sleeping},${cam.wakeUps}`
      ),
      '',
      'Số học sinh đang ngủ gật',
      `Hiện tại: ${statistics.currentDrowsyCount}`,
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `thong-ke-ngu-gat-${format(new Date(), 'yyyy-MM-dd')}.csv`;
    a.click();
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 border-b space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">Thống kê ngủ gật</h3>
          <Button size="sm" variant="outline" onClick={exportStatistics} disabled={!statistics}>
            <Download className="h-4 w-4 mr-1" />
            Export CSV
          </Button>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label>Phòng camera</Label>
            <Select value={selectedCamera} onValueChange={setSelectedCamera}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">Tất cả phòng</SelectItem>
                {cameras.map(cam => (
                  <SelectItem key={cam.id} value={cam.id}>
                    {cam.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label>Khoảng thời gian</Label>
            <Select value={timeRange} onValueChange={(v: any) => setTimeRange(v)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="today">Hôm nay</SelectItem>
                <SelectItem value="week">Tuần này</SelectItem>
                <SelectItem value="month">Tháng này</SelectItem>
                <SelectItem value="custom">Tùy chỉnh</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        {timeRange === 'custom' && (
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>Từ ngày</Label>
              <Popover>
                <PopoverTrigger asChild>
                  <Button
                    variant="outline"
                    className={cn(
                      "w-full justify-start text-left font-normal",
                      !startDate && "text-muted-foreground"
                    )}
                  >
                    <CalendarIcon className="mr-2 h-4 w-4" />
                    {startDate ? format(startDate, "dd/MM/yyyy") : <span>Chọn ngày</span>}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-auto p-0">
                  <Calendar
                    selected={startDate}
                    onSelect={(date: Date | undefined) => {
                      if (date) {
                        setStartDate(date);
                      }
                    }}
                  />
                </PopoverContent>
              </Popover>
            </div>

            <div className="space-y-2">
              <Label>Đến ngày</Label>
              <Popover>
                <PopoverTrigger asChild>
                  <Button
                    variant="outline"
                    className={cn(
                      "w-full justify-start text-left font-normal",
                      !endDate && "text-muted-foreground"
                    )}
                  >
                    <CalendarIcon className="mr-2 h-4 w-4" />
                    {endDate ? format(endDate, "dd/MM/yyyy") : <span>Chọn ngày</span>}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-auto p-0">
                  <Calendar
                    selected={endDate}
                    onSelect={(date: Date | undefined) => {
                      if (date) {
                        setEndDate(date);
                      }
                    }}
                  />
                </PopoverContent>
              </Popover>
            </div>
          </div>
        )}
      </div>

      <ScrollArea className="flex-1 p-4">
        {loading ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <Clock className="h-8 w-8 animate-spin mx-auto mb-2 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">Đang tải thống kê...</p>
            </div>
          </div>
        ) : statistics ? (
          <div className="space-y-6">
            {/* Tổng quát */}
            <div className="grid grid-cols-3 gap-4">
              <Card className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Buồn ngủ</p>
                    <p className="text-2xl font-bold text-orange-500">{statistics.totalDrowsy}</p>
                  </div>
                  <TrendingUp className="h-8 w-8 text-orange-500/50" />
                </div>
              </Card>

              <Card className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Ngủ gật</p>
                    <p className="text-2xl font-bold text-red-500">{statistics.totalSleeping}</p>
                  </div>
                  <Users className="h-8 w-8 text-red-500/50" />
                </div>
              </Card>

              <Card className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Tỉnh lại</p>
                    <p className="text-2xl font-bold text-green-500">{statistics.totalWakeUps}</p>
                  </div>
                  <BarChart3 className="h-8 w-8 text-green-500/50" />
                </div>
              </Card>
            </div>

            {/* Theo phòng */}
            <Card className="p-4">
              <h4 className="font-semibold mb-4">Thống kê theo phòng học</h4>
              <div className="space-y-3">
                {statistics.byCamera.length > 0 ? (
                  statistics.byCamera.map((cam) => {
                    const drowsyRate = cam.drowsyRate !== undefined ? cam.drowsyRate : 0;
                    const hasHighRate = drowsyRate > 30;
                    return (
                      <div 
                        key={cam.cameraId} 
                        className={`flex items-center justify-between p-3 border rounded-lg ${
                          hasHighRate ? 'bg-red-50 border-red-300' : ''
                        }`}
                      >
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <p className="font-medium">{cam.cameraName}</p>
                            {cam.currentDrowsy !== undefined && cam.currentDrowsy > 0 && (
                              <Badge variant="destructive" className="animate-pulse">
                                Đang ngủ gật: {cam.currentDrowsy}
                              </Badge>
                            )}
                            {cam.totalStudents !== undefined && (
                              <Badge variant="secondary" className="text-xs">
                                ~{cam.totalStudents} học sinh
                              </Badge>
                            )}
                            {drowsyRate > 30 && (
                              <Badge variant="destructive" className="text-xs">
                                <AlertTriangle className="h-3 w-3 mr-1" />
                                Cảnh báo
                              </Badge>
                            )}
                          </div>
                          <div className="flex gap-4 flex-wrap items-center">
                            <Badge variant="outline" className="text-orange-500">
                              Buồn ngủ: {cam.drowsy}
                            </Badge>
                            <Badge variant="outline" className="text-red-500">
                              Gục xuống: {cam.sleeping}
                            </Badge>
                            <Badge variant="outline" className="text-green-500">
                              Tỉnh lại: {cam.wakeUps}
                            </Badge>
                            {drowsyRate > 0 && (
                              <Badge 
                                variant={drowsyRate > 30 ? 'destructive' : drowsyRate > 15 ? 'outline' : 'secondary'}
                                className="text-xs"
                              >
                                Tỷ lệ: {drowsyRate.toFixed(1)}%
                              </Badge>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <p className="text-sm text-muted-foreground text-center py-4">
                    Không có dữ liệu trong khoảng thời gian này
                  </p>
                )}
              </div>
            </Card>

            {/* Cảnh báo tỷ lệ ngủ gật cao */}
            {statistics.alerts && statistics.alerts.length > 0 && (
              <Card className="p-4 border-red-300 bg-red-50">
                <div className="flex items-center gap-2 mb-4">
                  <AlertTriangle className="h-5 w-5 text-red-500" />
                  <h4 className="font-semibold text-red-700">Cảnh báo</h4>
                </div>
                <div className="space-y-2">
                  {statistics.alerts.map((alert, idx) => (
                    <div key={idx} className={`p-3 rounded-lg border ${
                      alert.severity === 'critical' 
                        ? 'bg-red-100 border-red-300' 
                        : 'bg-orange-100 border-orange-300'
                    }`}>
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium text-sm">{alert.cameraName}</p>
                          <p className="text-xs text-muted-foreground mt-1">{alert.message}</p>
                        </div>
                        <Badge variant={alert.severity === 'critical' ? 'destructive' : 'outline'} className="text-xs">
                          {alert.drowsyRate.toFixed(1)}%
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </Card>
            )}

            {/* Số học sinh đang ngủ gật hiện tại */}
            <Card className="p-4">
              <h4 className="font-semibold mb-4">Học sinh đang ngủ gật</h4>
              <div className="flex items-center justify-center py-6">
                <div className="text-center">
                  <p className="text-4xl font-bold text-red-500 mb-2">
                    {statistics.currentDrowsyCount}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    học sinh đang ngủ gật
                  </p>
                </div>
              </div>
            </Card>

            {/* Thống kê theo khung giờ */}
            {statistics.byTimeSlot && statistics.byTimeSlot.length > 0 && (
              <Card className="p-4">
                <h4 className="font-semibold mb-4">Thống kê theo khung giờ học</h4>
                <div className="space-y-3">
                  {statistics.byTimeSlot.map((slot) => {
                    const getTimeSlotIcon = () => {
                      switch(slot.timeSlot) {
                        case 'morning': return <Sun className="h-5 w-5 text-yellow-500" />;
                        case 'afternoon': return <Sunset className="h-5 w-5 text-orange-500" />;
                        case 'evening': return <Moon className="h-5 w-5 text-blue-500" />;
                        default: return <Clock className="h-5 w-5" />;
                      }
                    };
                    const getTimeSlotName = () => {
                      switch(slot.timeSlot) {
                        case 'morning': return 'Ca sáng (6h-12h)';
                        case 'afternoon': return 'Ca chiều (12h-18h)';
                        case 'evening': return 'Ca tối (18h-22h)';
                        default: return slot.timeSlot;
                      }
                    };
                    return (
                      <div key={slot.timeSlot} className="flex items-center justify-between p-3 border rounded-lg">
                        <div className="flex items-center gap-3 flex-1">
                          {getTimeSlotIcon()}
                          <div className="flex-1">
                            <p className="font-medium text-sm">{getTimeSlotName()}</p>
                            <p className="text-xs text-muted-foreground mt-1">
                              Ước tính {slot.totalStudents} học sinh
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-4">
                          <div className="text-right">
                            <p className="text-sm font-medium text-orange-500">{slot.drowsy}</p>
                            <p className="text-xs text-muted-foreground">Buồn ngủ</p>
                          </div>
                          <div className="text-right">
                            <p className="text-sm font-medium text-red-500">{slot.sleeping}</p>
                            <p className="text-xs text-muted-foreground">Gục xuống</p>
                          </div>
                          <div className="text-right">
                            <Badge 
                              variant={slot.drowsyRate > 30 ? 'destructive' : slot.drowsyRate > 15 ? 'outline' : 'secondary'}
                              className="min-w-[60px]"
                            >
                              {slot.drowsyRate.toFixed(1)}%
                            </Badge>
                            <p className="text-xs text-muted-foreground mt-1">Tỷ lệ</p>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </Card>
            )}

            {/* Theo ngày */}
            <Card className="p-4">
              <h4 className="font-semibold mb-4">Thống kê theo ngày</h4>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {statistics.byDate.length > 0 ? (
                  statistics.byDate.map((date) => {
                    try {
                      const dateObj = new Date(date.date + 'T00:00:00');
                      return (
                        <div key={date.date} className="flex items-center justify-between p-2 border rounded">
                          <span className="text-sm font-medium">{format(dateObj, 'dd/MM/yyyy')}</span>
                          <div className="flex gap-3">
                            <Badge variant="outline" className="text-orange-500">
                              {date.drowsy}
                            </Badge>
                            <Badge variant="outline" className="text-red-500">
                              {date.sleeping}
                            </Badge>
                          </div>
                        </div>
                      );
                    } catch (e) {
                      return null;
                    }
                  })
                ) : (
                  <p className="text-sm text-muted-foreground text-center py-4">
                    Không có dữ liệu
                  </p>
                )}
              </div>
            </Card>
          </div>
        ) : (
          <div className="flex items-center justify-center h-full">
            <p className="text-sm text-muted-foreground">Không có dữ liệu thống kê</p>
          </div>
        )}
      </ScrollArea>
    </div>
  );
}

