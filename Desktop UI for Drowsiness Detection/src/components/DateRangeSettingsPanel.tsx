/**
 * Date Range Settings Panel - Cho phép chọn khoảng thời gian tùy chỉnh
 * Hiển thị khi nhấn nút trên camera
 */

import React, { useState } from 'react';
import { apiGet, apiExport } from '../lib/api';
import { format } from 'date-fns';
import { Calendar as CalendarIcon, Download } from 'lucide-react';
import { Button } from './ui/button';
import { Calendar } from './ui/calendar';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from './ui/dialog';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from './ui/popover';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';

interface DateRangeSettingsPanelProps {
  cameraId?: string;
  cameraName?: string;
}

interface CameraStats {
  camera_id: string;
  camera_name: string;
  total_drowsy_students: number;
  currently_drowsy: number;
  total_events: number;
  total_duration_display: string;
}

export function DateRangeSettingsPanel({ cameraId, cameraName }: DateRangeSettingsPanelProps) {
  const [startDate, setStartDate] = useState<Date>();
  const [endDate, setEndDate] = useState<Date>();
  const [stats, setStats] = useState<CameraStats | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);

  const fetchStatsForRange = async () => {
    if (!startDate || !endDate) {
      alert('Vui lòng chọn cả ngày bắt đầu và ngày kết thúc');
      return;
    }

    setIsLoading(true);
    
    try {
      const startStr = format(startDate, 'yyyy-MM-dd');
      const endStr = format(endDate, 'yyyy-MM-dd');
      const period = `${startStr}_${endStr}`;
      
      const endpoint = cameraId
        ? `api/logs/stats/${cameraId}?period=${period}`
        : `api/logs/summary?period=${period}`;

      const response = await apiGet(endpoint);
      const data = await response.json();
      
      if (data.success) {
        if (cameraId) {
          setStats(data.stats);
        } else {
          // Convert summary to stats format
          setStats({
            camera_id: 'all',
            camera_name: 'Tất cả phòng',
            total_drowsy_students: data.summary.total_drowsy_students_unique,
            currently_drowsy: data.summary.currently_drowsy_all_cameras,
            total_events: data.summary.total_events,
            total_duration_display: data.summary.total_duration_display
          });
        }
      }
    } catch (error) {
      console.error('Error fetching stats:', error);
      alert('Lỗi khi tải dữ liệu');
    } finally {
      setIsLoading(false);
    }
  };

  const exportReport = async (exportFormat: 'pdf' | 'excel') => {
    if (!startDate || !endDate) {
      alert('Vui lòng chọn khoảng thời gian trước khi xuất báo cáo');
      return;
    }

    try {
      const startStr = format(startDate, 'yyyy-MM-dd');
      const endStr = format(endDate, 'yyyy-MM-dd');
      const period = `${startStr}_${endStr}`;
      const camera_ids = cameraId ? [cameraId] : undefined;
      const ok = await apiExport(exportFormat, period, camera_ids);
      if (!ok) alert(`Lỗi khi xuất ${exportFormat.toUpperCase()}`);
    } catch (error) {
      console.error(`Error exporting ${exportFormat}:`, error);
      alert(`Lỗi khi xuất ${exportFormat.toUpperCase()}`);
    }
  };

  return (
    <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm">
          <CalendarIcon className="w-4 h-4 mr-2" />
          Chọn khoảng thời gian
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>
            ⚙️ Tùy chỉnh khoảng thời gian
            {cameraName && ` - ${cameraName}`}
          </DialogTitle>
          <DialogDescription>
            Chọn ngày bắt đầu và ngày kết thúc để xem thống kê và xuất báo cáo
          </DialogDescription>
        </DialogHeader>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 py-4">
          {/* Start Date Picker */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Ngày bắt đầu</label>
            <Popover>
              <PopoverTrigger asChild>
                <Button variant="outline" className="w-full justify-start text-left font-normal">
                  <CalendarIcon className="mr-2 h-4 w-4" />
                  {startDate ? format(startDate, 'dd/MM/yyyy') : 'Chọn ngày'}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0">
                <Calendar
                  mode="single"
                  selected={startDate}
                  onSelect={setStartDate}
                  initialFocus
                />
              </PopoverContent>
            </Popover>
          </div>

          {/* End Date Picker */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Ngày kết thúc</label>
            <Popover>
              <PopoverTrigger asChild>
                <Button variant="outline" className="w-full justify-start text-left font-normal">
                  <CalendarIcon className="mr-2 h-4 w-4" />
                  {endDate ? format(endDate, 'dd/MM/yyyy') : 'Chọn ngày'}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0">
                <Calendar
                  mode="single"
                  selected={endDate}
                  onSelect={setEndDate}
                  initialFocus
                  disabled={(date: Date) => startDate ? date < startDate : false}
                />
              </PopoverContent>
            </Popover>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-2">
          <Button 
            onClick={fetchStatsForRange} 
            disabled={!startDate || !endDate || isLoading}
            className="flex-1"
          >
            {isLoading ? 'Đang tải...' : 'Xem thống kê'}
          </Button>
          
          <Button 
            onClick={() => exportReport('pdf')} 
            disabled={!startDate || !endDate}
            variant="outline"
          >
            <Download className="w-4 h-4 mr-2" />
            PDF
          </Button>
          
          <Button 
            onClick={() => exportReport('excel')} 
            disabled={!startDate || !endDate}
            variant="outline"
          >
            <Download className="w-4 h-4 mr-2" />
            Excel
          </Button>
        </div>

        {/* Statistics Display */}
        {stats && (
          <Card className="mt-4">
            <CardHeader>
              <CardTitle>📊 Thống kê khoảng thời gian đã chọn</CardTitle>
              <CardDescription>
                {startDate && endDate && (
                  <>
                    Từ {format(startDate, 'dd/MM/yyyy')} đến {format(endDate, 'dd/MM/yyyy')}
                  </>
                )}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="space-y-1">
                  <div className="text-sm text-muted-foreground">Phòng học</div>
                  <div className="text-2xl font-bold">{stats.camera_name}</div>
                </div>
                
                <div className="space-y-1">
                  <div className="text-sm text-muted-foreground">Học sinh ngủ gật</div>
                  <div className="text-2xl font-bold text-orange-600">
                    {stats.total_drowsy_students}
                  </div>
                </div>
                
                <div className="space-y-1">
                  <div className="text-sm text-muted-foreground">Số sự kiện</div>
                  <div className="text-2xl font-bold">{stats.total_events}</div>
                </div>
                
                <div className="space-y-1">
                  <div className="text-sm text-muted-foreground">Tổng thời gian</div>
                  <div className="text-2xl font-bold">{stats.total_duration_display}</div>
                </div>
              </div>
              
              {stats.currently_drowsy > 0 && (
                <div className="mt-4 p-3 bg-orange-100 rounded border border-orange-300">
                  <div className="flex items-center gap-2">
                    <Badge variant="destructive">{stats.currently_drowsy}</Badge>
                    <span className="text-sm font-medium">học sinh đang ngủ gật hiện tại</span>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Quick Presets */}
        <div className="border-t pt-4">
          <div className="text-sm font-medium mb-2">Chọn nhanh:</div>
          <div className="flex flex-wrap gap-2">
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => {
                const today = new Date();
                setStartDate(today);
                setEndDate(today);
              }}
            >
              Hôm nay
            </Button>
            
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => {
                const today = new Date();
                const weekAgo = new Date(today);
                weekAgo.setDate(today.getDate() - 7);
                setStartDate(weekAgo);
                setEndDate(today);
              }}
            >
              7 ngày qua
            </Button>
            
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => {
                const today = new Date();
                const monthAgo = new Date(today);
                monthAgo.setMonth(today.getMonth() - 1);
                setStartDate(monthAgo);
                setEndDate(today);
              }}
            >
              30 ngày qua
            </Button>
            
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => {
                const today = new Date();
                const firstDay = new Date(today.getFullYear(), today.getMonth(), 1);
                setStartDate(firstDay);
                setEndDate(today);
              }}
            >
              Tháng này
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
