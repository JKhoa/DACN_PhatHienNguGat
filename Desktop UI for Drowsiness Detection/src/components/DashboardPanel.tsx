/**
 * Dashboard Panel - Real-time monitoring cho tất cả camera
 * Hiển thị tổng quan, grid view, và số liệu real-time
 */

import React, { useState, useEffect } from 'react';
import { apiGet, apiExport } from '../lib/api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { 
  Activity, 
  Users, 
  AlertTriangle, 
  Eye, 
  TrendingUp,
  Calendar,
  Download,
  Clock,
  BarChart3,
  AlertCircle,
  CheckCircle,
  X,
  ChevronRight
} from 'lucide-react';

interface CameraInfo {
  camera_id: string;
  camera_name: string;
  active_drowsy_count: number;
  total_events?: number;
  unique_students?: number;
  total_duration?: string;
  last_event_time?: string;
}

interface SummaryStats {
  period: string;
  total_cameras: number;
  total_drowsy_students_unique: number;
  total_events: number;
  total_duration_display: string;
  currently_drowsy_all_cameras: number;
}

interface ActiveStudent {
  camera_id: string;
  camera_name: string;
  student_id: number;
  current_duration_seconds: number;
  current_duration_display: string;
}

interface CameraDetailStats {
  camera_id: string;
  camera_name: string;
  total_events: number;
  unique_students: number;
  total_duration: string;
  avg_duration: string;
  longest_duration: string;
  most_frequent_student?: number;
  events_by_hour: { [hour: string]: number };
}

export function DashboardPanel() {
  const [cameras, setCameras] = useState<CameraInfo[]>([]);
  const [summary, setSummary] = useState<SummaryStats | null>(null);
  const [activeStudents, setActiveStudents] = useState<ActiveStudent[]>([]);
  const [period, setPeriod] = useState('today');
  const [isLoading, setIsLoading] = useState(true);
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);
  const [cameraDetail, setCameraDetail] = useState<CameraDetailStats | null>(null);

  // Fetch camera detail stats
  const fetchCameraDetail = async (cameraId: string) => {
    try {
      const response = await apiGet(`api/logs/events/${encodeURIComponent(cameraId)}?period=${period}`);
      const data = await response.json();
      
      if (data.success && data.events) {
        // Calculate detailed statistics
        const events = data.events;
        const uniqueStudents = new Set(events.map((e: any) => e.student_id)).size;
        const totalDuration = events.reduce((sum: number, e: any) => sum + (e.duration_seconds || 0), 0);
        const avgDuration = events.length > 0 ? totalDuration / events.length : 0;
        const longestEvent = events.reduce((max: any, e: any) => 
          (e.duration_seconds || 0) > (max?.duration_seconds || 0) ? e : max, events[0]);
        
        // Count events by hour
        const eventsByHour: { [hour: string]: number } = {};
        events.forEach((e: any) => {
          const hour = new Date(e.start_time).getHours();
          const hourKey = `${hour.toString().padStart(2, '0')}:00`;
          eventsByHour[hourKey] = (eventsByHour[hourKey] || 0) + 1;
        });

        // Find most frequent student
        const studentCounts: { [id: number]: number } = {};
        events.forEach((e: any) => {
          studentCounts[e.student_id] = (studentCounts[e.student_id] || 0) + 1;
        });
        const mostFrequent = Object.entries(studentCounts).reduce((max, [id, count]) => 
          count > (max[1] as number) ? [id, count] : max, ['0', 0]);

        const camera = cameras.find(c => c.camera_id === cameraId);
        
        setCameraDetail({
          camera_id: cameraId,
          camera_name: camera?.camera_name || cameraId,
          total_events: events.length,
          unique_students: uniqueStudents,
          total_duration: formatDuration(totalDuration),
          avg_duration: formatDuration(Math.round(avgDuration)),
          longest_duration: formatDuration(longestEvent?.duration_seconds || 0),
          most_frequent_student: parseInt(mostFrequent[0]),
          events_by_hour: eventsByHour
        });
      }
    } catch (error) {
      console.error('Error fetching camera detail:', error);
    }
  };

  const formatDuration = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) return `${hours}h ${minutes}m ${secs}s`;
    if (minutes > 0) return `${minutes}m ${secs}s`;
    return `${secs}s`;
  };

  // Fetch data from backend
  const fetchDashboardData = async () => {
    try {
      // Fetch cameras list
      const camerasRes = await apiGet('api/logs/cameras');
      const camerasData = await camerasRes.json();
      if (camerasData.success) {
        setCameras(camerasData.cameras || []);
      }

      // Fetch summary stats
      const summaryRes = await apiGet(`api/logs/summary?period=${period}`);
      const summaryData = await summaryRes.json();
      if (summaryData.success) {
        setSummary(summaryData.summary);
      }

      // Fetch active drowsy students
      const activeRes = await apiGet('api/logs/active');
      const activeData = await activeRes.json();
      if (activeData.success) {
        setActiveStudents(activeData.active_drowsy_students || []);
      }

      setIsLoading(false);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    // Refresh every 5 seconds
    const interval = setInterval(fetchDashboardData, 5000);
    return () => clearInterval(interval);
  }, [period]);

  // Fetch camera detail when selected
  useEffect(() => {
    if (selectedCamera) {
      fetchCameraDetail(selectedCamera);
    }
  }, [selectedCamera, period]);

  const exportReport = async (format: 'pdf' | 'excel') => {
    try {
      await apiExport(format, period);
    } catch (error) {
      console.error(`Error exporting ${format}:`, error);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-lg">Đang tải dữ liệu...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">📊 Dashboard Giám Sát</h1>
          <p className="text-muted-foreground">Tổng quan hệ thống phát hiện ngủ gật</p>
        </div>
        
        <div className="flex gap-2">
          <select 
            value={period} 
            onChange={(e) => setPeriod(e.target.value)}
            className="border rounded px-3 py-2"
            title="Chọn khoảng thời gian"
          >
            <option value="today">Hôm nay</option>
            <option value="week">Tuần này</option>
            <option value="month">Tháng này</option>
          </select>
          
          <Button onClick={() => exportReport('pdf')} variant="outline">
            <Download className="w-4 h-4 mr-2" />
            PDF
          </Button>
          
          <Button onClick={() => exportReport('excel')} variant="outline">
            <Download className="w-4 h-4 mr-2" />
            Excel
          </Button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Tổng số phòng</CardTitle>
            <Eye className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{summary?.total_cameras || 0}</div>
            <p className="text-xs text-muted-foreground">Đang giám sát</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Học sinh ngủ gật</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{summary?.total_drowsy_students_unique || 0}</div>
            <p className="text-xs text-muted-foreground">Tổng số người</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Đang ngủ gật</CardTitle>
            <AlertTriangle className="h-4 w-4 text-orange-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-orange-500">
              {summary?.currently_drowsy_all_cameras || 0}
            </div>
            <p className="text-xs text-muted-foreground">Real-time</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Tổng sự kiện</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{summary?.total_events || 0}</div>
            <p className="text-xs text-muted-foreground">{summary?.total_duration_display || '0s'}</p>
          </CardContent>
        </Card>
      </div>

      {/* Camera Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>Tình trạng các phòng học</CardTitle>
              <CardDescription>Click vào phòng để xem chi tiết thống kê</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {cameras.map((camera) => {
                  const drowsyCount = camera.active_drowsy_count;
                  const severity = drowsyCount === 0 ? 'normal' : drowsyCount <= 2 ? 'warning' : 'critical';
                  const isSelected = selectedCamera === camera.camera_id;
                  
                  return (
                    <Card 
                      key={camera.camera_id}
                      onClick={() => setSelectedCamera(isSelected ? null : camera.camera_id)}
                      className={`cursor-pointer transition-all hover:shadow-lg ${
                        isSelected ? 'ring-2 ring-blue-500 bg-blue-50' : ''
                      } ${
                        severity === 'critical' ? 'border-red-500 bg-red-50' :
                        severity === 'warning' ? 'border-yellow-500 bg-yellow-50' :
                        'border-green-500 bg-green-50'
                      }`}
                    >
                      <CardHeader>
                        <div className="flex items-center justify-between">
                          <CardTitle className="text-base flex items-center gap-2">
                            📹 {camera.camera_name}
                            {isSelected && <ChevronRight className="w-4 h-4 text-blue-500" />}
                          </CardTitle>
                          <Badge 
                            variant={severity === 'critical' ? 'destructive' : severity === 'warning' ? 'default' : 'secondary'}
                          >
                            {drowsyCount} ngủ gật
                          </Badge>
                        </div>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-2">
                          <div className="flex items-center gap-2">
                            <Activity className={`w-4 h-4 ${
                              severity === 'critical' ? 'text-red-500' :
                              severity === 'warning' ? 'text-yellow-500' :
                              'text-green-500'
                            }`} />
                            <span className="text-sm font-medium">
                              {severity === 'critical' ? '🔴 Cần chú ý ngay!' :
                               severity === 'warning' ? '🟡 Cảnh báo' :
                               '🟢 Bình thường'}
                            </span>
                          </div>
                          
                          {camera.total_events !== undefined && (
                            <div className="grid grid-cols-3 gap-2 text-xs text-muted-foreground">
                              <div className="flex flex-col gap-1">
                                <span className="font-semibold text-gray-700">{camera.total_events}</span>
                                <span>Sự kiện</span>
                              </div>
                              <div className="flex flex-col gap-1">
                                <span className="font-semibold text-gray-700">{camera.unique_students || 0}</span>
                                <span>Học sinh</span>
                              </div>
                              <div className="flex flex-col gap-1">
                                <span className="font-semibold text-gray-700">{camera.total_duration || '0s'}</span>
                                <span>Tổng thời gian</span>
                              </div>
                            </div>
                          )}

                          {camera.last_event_time && (
                            <div className="flex items-center gap-1 text-xs text-muted-foreground mt-2 pt-2 border-t">
                              <Clock className="w-3 h-3" />
                              <span>Lần cuối: {new Date(camera.last_event_time).toLocaleTimeString('vi-VN')}</span>
                            </div>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Camera Detail Panel */}
        <div className="lg:col-span-1">
          {selectedCamera && cameraDetail ? (
            <Card className="sticky top-6 border-blue-500">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base">📊 Chi tiết {cameraDetail.camera_name}</CardTitle>
                  <Button 
                    variant="ghost" 
                    size="sm" 
                    onClick={() => setSelectedCamera(null)}
                    className="h-6 w-6 p-0"
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </div>
                <CardDescription>Thống kê chi tiết theo {period === 'today' ? 'hôm nay' : period === 'week' ? 'tuần' : 'tháng'}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Key Metrics */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 bg-blue-50 rounded-lg">
                    <div className="text-2xl font-bold text-blue-700">{cameraDetail.total_events}</div>
                    <div className="text-xs text-muted-foreground">Tổng sự kiện</div>
                  </div>
                  <div className="p-3 bg-green-50 rounded-lg">
                    <div className="text-2xl font-bold text-green-700">{cameraDetail.unique_students}</div>
                    <div className="text-xs text-muted-foreground">Học sinh</div>
                  </div>
                </div>

                {/* Duration Stats */}
                <div className="space-y-2">
                  <h4 className="font-semibold text-sm">⏱️ Thời gian ngủ gật</h4>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Tổng cộng:</span>
                      <span className="font-semibold">{cameraDetail.total_duration}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Trung bình:</span>
                      <span className="font-semibold">{cameraDetail.avg_duration}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Lâu nhất:</span>
                      <span className="font-semibold text-red-600">{cameraDetail.longest_duration}</span>
                    </div>
                  </div>
                </div>

                {/* Most Frequent Student */}
                {cameraDetail.most_frequent_student && (
                  <div className="p-3 bg-orange-50 rounded-lg border border-orange-200">
                    <div className="text-xs text-muted-foreground mb-1">👤 Hay ngủ gật nhất</div>
                    <div className="text-lg font-bold text-orange-700">
                      Học sinh #{cameraDetail.most_frequent_student}
                    </div>
                  </div>
                )}

                {/* Events by Hour Chart */}
                {Object.keys(cameraDetail.events_by_hour).length > 0 && (
                  <div className="space-y-2">
                    <h4 className="font-semibold text-sm">📈 Phân bố theo giờ</h4>
                    <div className="space-y-1">
                      {Object.entries(cameraDetail.events_by_hour)
                        .sort(([a], [b]) => a.localeCompare(b))
                        .map(([hour, count]) => {
                          const maxCount = Math.max(...Object.values(cameraDetail.events_by_hour));
                          const percentage = (count / maxCount) * 100;
                          
                          return (
                            <div key={hour} className="flex items-center gap-2">
                              <span className="text-xs w-12 text-muted-foreground">{hour}</span>
                              <div className="flex-1 bg-gray-200 rounded-full h-4 overflow-hidden">
                                <div 
                                  className="bg-blue-500 h-full rounded-full transition-all flex items-center justify-end pr-1"
                                  style={{ ['--bar-width' as any]: `${percentage}%`, width: 'var(--bar-width)' } as React.CSSProperties}
                                >
                                  {count > 0 && (
                                    <span className="text-xs text-white font-semibold">{count}</span>
                                  )}
                                </div>
                              </div>
                            </div>
                          );
                        })}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          ) : (
            <Card className="sticky top-6">
              <CardContent className="flex flex-col items-center justify-center h-64 text-center">
                <Eye className="w-12 h-12 text-muted-foreground mb-3" />
                <p className="text-sm text-muted-foreground">
                  Chọn một phòng học bên trái để xem chi tiết thống kê
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* Active Students List */}
      {activeStudents.length > 0 && (
        <Card className="border-orange-300 bg-orange-50">
          <CardHeader>
            <CardTitle className="text-orange-700">
              🔴 Học sinh đang ngủ gật ({activeStudents.length})
            </CardTitle>
            <CardDescription>Real-time monitoring</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {activeStudents.map((student, idx) => (
                <div 
                  key={`${student.camera_id}-${student.student_id}`}
                  className="flex items-center justify-between p-3 bg-white rounded border border-orange-200"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                    <div>
                      <div className="font-semibold">{student.camera_name}</div>
                      <div className="text-sm text-muted-foreground">
                        Học sinh #{student.student_id}
                      </div>
                    </div>
                  </div>
                  <Badge variant="destructive">
                    {student.current_duration_display}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Camera Detail Panel */}
      {selectedCamera && cameraDetail && (
        <Card className="border-blue-300 bg-blue-50">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-blue-700">
                  📊 Chi tiết: {cameraDetail.camera_name}
                </CardTitle>
                <CardDescription>Thống kê chi tiết {period === 'today' ? 'hôm nay' : period === 'week' ? 'tuần này' : 'tháng này'}</CardDescription>
              </div>
              <Button 
                variant="ghost" 
                size="sm"
                onClick={() => setSelectedCamera(null)}
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              {/* Total Events */}
              <Card className="bg-white">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground">
                    Tổng sự kiện
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between">
                    <div className="text-2xl font-bold">{cameraDetail.total_events}</div>
                    <BarChart3 className="w-5 h-5 text-blue-500" />
                  </div>
                </CardContent>
              </Card>

              {/* Unique Students */}
              <Card className="bg-white">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground">
                    Số học sinh
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between">
                    <div className="text-2xl font-bold">{cameraDetail.unique_students}</div>
                    <Users className="w-5 h-5 text-green-500" />
                  </div>
                </CardContent>
              </Card>

              {/* Total Duration */}
              <Card className="bg-white">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground">
                    Tổng thời gian
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between">
                    <div className="text-2xl font-bold">{cameraDetail.total_duration}</div>
                    <Clock className="w-5 h-5 text-orange-500" />
                  </div>
                </CardContent>
              </Card>

              {/* Average Duration */}
              <Card className="bg-white">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground">
                    TB/sự kiện
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between">
                    <div className="text-2xl font-bold">{cameraDetail.avg_duration}</div>
                    <TrendingUp className="w-5 h-5 text-purple-500" />
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Additional Details */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Key Metrics */}
              <Card className="bg-white">
                <CardHeader>
                  <CardTitle className="text-sm">Các chỉ số quan trọng</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <div className="flex items-center gap-2">
                      <AlertCircle className="w-4 h-4 text-red-500" />
                      <span className="text-sm">Ngủ gật lâu nhất</span>
                    </div>
                    <span className="font-semibold">{cameraDetail.longest_duration}</span>
                  </div>

                  {cameraDetail.most_frequent_student && (
                    <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                      <div className="flex items-center gap-2">
                        <Users className="w-4 h-4 text-blue-500" />
                        <span className="text-sm">HS ngủ gật nhiều nhất</span>
                      </div>
                      <span className="font-semibold">#{cameraDetail.most_frequent_student}</span>
                    </div>
                  )}

                  <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <div className="flex items-center gap-2">
                      <CheckCircle className="w-4 h-4 text-green-500" />
                      <span className="text-sm">TB mỗi sự kiện</span>
                    </div>
                    <span className="font-semibold">{cameraDetail.avg_duration}</span>
                  </div>
                </CardContent>
              </Card>

              {/* Events by Hour Chart */}
              <Card className="bg-white">
                <CardHeader>
                  <CardTitle className="text-sm">Phân bố theo giờ</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {Object.entries(cameraDetail.events_by_hour || {})
                      .sort(([a], [b]) => a.localeCompare(b))
                      .slice(0, 10)
                      .map(([hour, count]) => {
                        const maxCount = Math.max(...Object.values(cameraDetail.events_by_hour || {}));
                        const percentage = maxCount > 0 ? (count / maxCount) * 100 : 0;
                        
                        return (
                          <div key={hour} className="space-y-1">
                            <div className="flex items-center justify-between text-xs">
                              <span className="font-medium">{hour}</span>
                              <span className="text-muted-foreground">{count} sự kiện</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div 
                                className="bg-blue-500 h-2 rounded-full transition-all"
                                style={{ ['--bar-width' as any]: `${Math.min(100, Math.max(0, percentage))}%`, width: 'var(--bar-width)' } as React.CSSProperties}
                              />
                            </div>
                          </div>
                        );
                      })}
                    
                    {Object.keys(cameraDetail.events_by_hour || {}).length === 0 && (
                      <div className="text-center text-sm text-muted-foreground py-4">
                        Chưa có dữ liệu
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
