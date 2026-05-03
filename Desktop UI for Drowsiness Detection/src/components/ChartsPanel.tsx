/**
 * Charts Panel - Biểu đồ thống kê drowsiness
 * Line chart (xu hướng giờ), Bar chart (so sánh phòng), Pie chart (phân bố)
 */

import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { apiGet } from '../lib/api';

interface CameraStats {
  camera_id: string;
  camera_name: string;
  total_drowsy_students: number;
  total_events: number;
  total_duration_seconds: number;
  total_duration_display: string;
}

interface HourlyData {
  hour: string;
  count: number;
  duration: number;
}

interface EventData {
  camera_id: string;
  camera_name: string;
  student_id: number;
  start_time: string;
  duration_seconds: number;
}

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

export function ChartsPanel() {
  const [period, setPeriod] = useState('today');
  const [selectedCameraId, setSelectedCameraId] = useState<string>('all'); // NEW: Camera filter
  const [cameraStats, setCameraStats] = useState<CameraStats[]>([]);
  const [hourlyData, setHourlyData] = useState<HourlyData[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchChartsData();
  }, [period, selectedCameraId]); // Re-fetch when camera changes

  const fetchChartsData = async () => {
    try {
      setIsLoading(true);

      // Fetch camera statistics
      const statsRes = await apiGet(`api/logs/stats?period=${period}`);
      const statsData = await statsRes.json();
      
      if (statsData.success && statsData.camera_stats) {
        setCameraStats(statsData.camera_stats);
        
        // Process hourly data
        const hourlyMap: { [key: string]: { count: number; duration: number } } = {};
        
        // Determine which cameras to process
        const camerasToProcess = selectedCameraId === 'all' 
          ? statsData.camera_stats 
          : statsData.camera_stats.filter((c: CameraStats) => c.camera_id === selectedCameraId);
        
        // Get events for selected camera(s)
        for (const camera of camerasToProcess) {
          const eventsRes = await apiGet(
            `api/logs/events/${camera.camera_id}?period=${period}`
          );
          const eventsData = await eventsRes.json();
          
          if (eventsData.success && eventsData.events) {
            eventsData.events.forEach((event: EventData) => {
              // Extract hour from start_time (format: "2025-11-10 09:15:30")
              const hour = event.start_time.split(' ')[1]?.split(':')[0] || '00';
              const hourKey = `${hour}:00`;
              
              if (!hourlyMap[hourKey]) {
                hourlyMap[hourKey] = { count: 0, duration: 0 };
              }
              
              hourlyMap[hourKey].count += 1;
              hourlyMap[hourKey].duration += event.duration_seconds;
            });
          }
        }
        
        // Convert to array and sort
        const hourlyArray = Object.entries(hourlyMap)
          .map(([hour, data]) => ({
            hour,
            count: data.count,
            duration: Math.round(data.duration / 60) // Convert to minutes
          }))
          .sort((a, b) => a.hour.localeCompare(b.hour));
        
        setHourlyData(hourlyArray);
      }
      
      setIsLoading(false);
    } catch (error) {
      console.error('Error fetching charts data:', error);
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-lg">Đang tải biểu đồ...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6">
      {/* Header with Camera Selector */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">📈 Biểu Đồ Thống Kê</h1>
          <p className="text-muted-foreground">Phân tích dữ liệu ngủ gật</p>
        </div>
        
        <div className="flex gap-3">
          {/* Camera Selector */}
          <select 
            value={selectedCameraId} 
            onChange={(e) => setSelectedCameraId(e.target.value)}
            className="border rounded px-3 py-2 min-w-[200px]"
            aria-label="Chọn phòng học"
          >
            <option value="all">📹 Tất cả phòng học</option>
            {cameraStats.map((camera) => (
              <option key={camera.camera_id} value={camera.camera_id}>
                📹 {camera.camera_name}
              </option>
            ))}
          </select>
          
          {/* Period Selector */}
          <select 
            value={period} 
            onChange={(e) => setPeriod(e.target.value)}
            className="border rounded px-3 py-2"
            aria-label="Chọn khoảng thời gian"
          >
            <option value="today">Hôm nay</option>
            <option value="week">Tuần này</option>
            <option value="month">Tháng này</option>
          </select>
        </div>
      </div>

      {/* Filter Info */}
      {selectedCameraId !== 'all' && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 flex items-center gap-2">
          <span className="text-blue-600 font-medium">
            🔍 Đang xem dữ liệu của: {cameraStats.find(c => c.camera_id === selectedCameraId)?.camera_name}
          </span>
          <button 
            onClick={() => setSelectedCameraId('all')}
            className="ml-auto text-blue-600 hover:text-blue-800 underline text-sm"
          >
            Xem tất cả
          </button>
        </div>
      )}

      <Tabs defaultValue="hourly" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="hourly">Xu hướng theo giờ</TabsTrigger>
          <TabsTrigger value="comparison">So sánh phòng</TabsTrigger>
          <TabsTrigger value="distribution">Phân bố</TabsTrigger>
        </TabsList>

        {/* Line Chart - Hourly Trend */}
        <TabsContent value="hourly">
          <Card>
            <CardHeader>
              <CardTitle>📊 Xu hướng ngủ gật theo giờ</CardTitle>
              <CardDescription>
                Số lượt ngủ gật và thời gian trung bình trong ngày
              </CardDescription>
            </CardHeader>
            <CardContent>
              {hourlyData.length > 0 ? (
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={hourlyData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="hour" 
                      label={{ value: 'Giờ trong ngày', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      yAxisId="left"
                      label={{ value: 'Số lượt', angle: -90, position: 'insideLeft' }}
                    />
                    <YAxis 
                      yAxisId="right" 
                      orientation="right"
                      label={{ value: 'Thời gian (phút)', angle: 90, position: 'insideRight' }}
                    />
                    <Tooltip />
                    <Legend />
                    <Line 
                      yAxisId="left"
                      type="monotone" 
                      dataKey="count" 
                      stroke="#3b82f6" 
                      strokeWidth={2}
                      name="Số lượt ngủ gật"
                      dot={{ r: 5 }}
                    />
                    <Line 
                      yAxisId="right"
                      type="monotone" 
                      dataKey="duration" 
                      stroke="#10b981" 
                      strokeWidth={2}
                      name="Tổng thời gian (phút)"
                      dot={{ r: 5 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-[400px] flex items-center justify-center text-muted-foreground">
                  Chưa có dữ liệu trong khoảng thời gian này
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Bar Chart - Camera Comparison */}
        <TabsContent value="comparison">
          <Card>
            <CardHeader>
              <CardTitle>🏫 So sánh giữa các phòng học</CardTitle>
              <CardDescription>
                {selectedCameraId === 'all' 
                  ? 'Số lượng học sinh ngủ gật và tổng số sự kiện ở tất cả phòng'
                  : `Chi tiết thống kê của ${cameraStats.find(c => c.camera_id === selectedCameraId)?.camera_name}`
                }
              </CardDescription>
            </CardHeader>
            <CardContent>
              {cameraStats.length > 0 ? (
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={selectedCameraId === 'all' ? cameraStats : cameraStats.filter(c => c.camera_id === selectedCameraId)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="camera_name" 
                      angle={-45}
                      textAnchor="end"
                      height={100}
                    />
                    <YAxis label={{ value: 'Số lượng', angle: -90, position: 'insideLeft' }} />
                    <Tooltip />
                    <Legend />
                    <Bar 
                      dataKey="total_drowsy_students" 
                      fill="#3b82f6" 
                      name="Số học sinh"
                      radius={[8, 8, 0, 0]}
                    />
                    <Bar 
                      dataKey="total_events" 
                      fill="#10b981" 
                      name="Số sự kiện"
                      radius={[8, 8, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-[400px] flex items-center justify-center text-muted-foreground">
                  Chưa có dữ liệu
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Pie Chart - Distribution */}
        <TabsContent value="distribution">
          <Card>
            <CardHeader>
              <CardTitle>🥧 Phân bố số lượng ngủ gật theo phòng</CardTitle>
              <CardDescription>
                {selectedCameraId === 'all' 
                  ? 'Tỷ lệ phần trăm học sinh ngủ gật ở mỗi phòng'
                  : `Chi tiết phân bố của ${cameraStats.find(c => c.camera_id === selectedCameraId)?.camera_name}`
                }
              </CardDescription>
            </CardHeader>
            <CardContent>
              {cameraStats.length > 0 ? (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <ResponsiveContainer width="100%" height={400}>
                    <PieChart>
                      <Pie
                        data={selectedCameraId === 'all' ? cameraStats : cameraStats.filter(c => c.camera_id === selectedCameraId)}
                        dataKey="total_drowsy_students"
                        nameKey="camera_name"
                        cx="50%"
                        cy="50%"
                        outerRadius={120}
                        label={(entry) => `${entry.camera_name}: ${entry.total_drowsy_students}`}
                      >
                        {(selectedCameraId === 'all' ? cameraStats : cameraStats.filter(c => c.camera_id === selectedCameraId)).map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>

                  {/* Legend with details */}
                  <div className="space-y-3">
                    <h3 className="font-semibold text-lg">Chi tiết phòng học</h3>
                    {(selectedCameraId === 'all' ? cameraStats : cameraStats.filter(c => c.camera_id === selectedCameraId)).map((camera, index) => (
                      <div 
                        key={camera.camera_id}
                        className="flex items-center justify-between p-3 bg-muted rounded"
                      >
                        <div className="flex items-center gap-3">
                          <div 
                            className="w-4 h-4 rounded"
                            style={{ ['--color' as any]: COLORS[index % COLORS.length], backgroundColor: 'var(--color)' } as React.CSSProperties}
                          />
                          <span className="font-medium">{camera.camera_name}</span>
                        </div>
                        <div className="text-right">
                          <div className="font-bold">{camera.total_drowsy_students} học sinh</div>
                          <div className="text-sm text-muted-foreground">
                            {camera.total_events} sự kiện • {camera.total_duration_display}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="h-[400px] flex items-center justify-center text-muted-foreground">
                  Chưa có dữ liệu
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
