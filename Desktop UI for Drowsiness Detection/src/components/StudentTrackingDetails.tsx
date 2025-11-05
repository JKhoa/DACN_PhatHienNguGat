import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { ScrollArea } from './ui/scroll-area';
import { 
  Users, 
  Eye, 
  EyeOff, 
  Clock, 
  MapPin, 
  Activity,
  AlertTriangle,
  CheckCircle,
  XCircle
} from 'lucide-react';

interface Student {
  id: string;
  position: { x: number; y: number };
  state: 'normal' | 'sleepy' | 'head_down';
  confidence: number;
  sleepDuration: number;
  lastUpdate: string | Date;
  bbox: [number, number, number, number];
  headBbox?: [number, number, number, number];
}

interface StudentTrackingDetailsProps {
  students: Student[];
  cameraName: string;
  isActive: boolean;
}

export const StudentTrackingDetails: React.FC<StudentTrackingDetailsProps> = ({
  students,
  cameraName,
  isActive
}) => {
  const getStateIcon = (state: Student['state']) => {
    switch (state) {
      case 'normal':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'sleepy':
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      case 'head_down':
        return <XCircle className="h-4 w-4 text-red-500" />;
    }
  };

  const getStateColor = (state: Student['state']) => {
    switch (state) {
      case 'normal':
        return 'bg-green-500/10 text-green-500 border-green-500/20';
      case 'sleepy':
        return 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20';
      case 'head_down':
        return 'bg-red-500/10 text-red-500 border-red-500/20';
    }
  };

  const getStateText = (state: Student['state']) => {
    switch (state) {
      case 'normal':
        return 'Tỉnh táo';
      case 'sleepy':
        return 'Buồn ngủ';
      case 'head_down':
        return 'Gục xuống';
    }
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const formatTime = (input: string | Date) => {
    const d = input instanceof Date ? input : new Date(input);
    return d.toLocaleTimeString('vi-VN');
  };

  const sleepyStudents = students.filter(s => s.state !== 'normal');
  const normalStudents = students.filter(s => s.state === 'normal');

  return (
    <Card className="w-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg font-semibold flex items-center gap-2">
            <Users className="h-5 w-5" />
            Chi tiết Tracking - {cameraName}
          </CardTitle>
          <div className="flex items-center gap-2">
            {isActive ? (
              <Badge variant="default" className="bg-green-500">
                <Activity className="h-3 w-3 mr-1" />
                Đang hoạt động
              </Badge>
            ) : (
              <Badge variant="secondary">
                <EyeOff className="h-3 w-3 mr-1" />
                Tạm dừng
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Summary Stats */}
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <div className="text-2xl font-bold text-blue-900 dark:text-blue-100">
              {students.length}
            </div>
            <div className="text-sm text-blue-600 dark:text-blue-300">Tổng học sinh</div>
          </div>
          <div className="text-center p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
            <div className="text-2xl font-bold text-green-900 dark:text-green-100">
              {normalStudents.length}
            </div>
            <div className="text-sm text-green-600 dark:text-green-300">Tỉnh táo</div>
          </div>
          <div className="text-center p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
            <div className="text-2xl font-bold text-red-900 dark:text-red-100">
              {sleepyStudents.length}
            </div>
            <div className="text-sm text-red-600 dark:text-red-300">Cần chú ý</div>
          </div>
        </div>

        {/* Student List */}
        <div>
          <h4 className="font-semibold mb-3 flex items-center gap-2">
            <Eye className="h-4 w-4" />
            Danh sách học sinh ({students.length})
          </h4>
          
          <ScrollArea className="h-64">
            <div className="space-y-2">
              {students.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <Users className="h-12 w-12 mx-auto mb-2 opacity-20" />
                  <p>Chưa phát hiện học sinh nào</p>
                  <p className="text-sm">Hãy đảm bảo camera đang hoạt động</p>
                </div>
              ) : (
                students.map((student) => (
                  <div
                    key={student.id}
                    className="p-3 border rounded-lg hover:bg-accent transition-colors"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        {getStateIcon(student.state)}
                        <span className="font-medium">{student.id}</span>
                        <Badge 
                          variant="outline" 
                          className={`text-xs ${getStateColor(student.state)}`}
                        >
                          {getStateText(student.state)}
                        </Badge>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {formatTime(student.lastUpdate)}
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div className="flex items-center gap-2">
                        <MapPin className="h-3 w-3 text-muted-foreground" />
                        <span>Vị trí: ({student.position.x}, {student.position.y})</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Activity className="h-3 w-3 text-muted-foreground" />
                        <span>Độ tin cậy: {(student.confidence * 100).toFixed(1)}%</span>
                      </div>
                    </div>

                    {student.state !== 'normal' && (
                      <div className="mt-2 flex items-center gap-2 text-sm">
                        <Clock className="h-3 w-3 text-muted-foreground" />
                        <span>
                          Thời gian: {formatDuration(student.sleepDuration)}
                        </span>
                      </div>
                    )}

                    {/* Bounding Box Info */}
                    <div className="mt-2 text-xs text-muted-foreground">
                      <div>Full Body: [{student.bbox.join(', ')}]</div>
                      {student.headBbox && (
                        <div>Head Only: [{student.headBbox.join(', ')}]</div>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          </ScrollArea>
        </div>

        {/* Alert Summary */}
        {sleepyStudents.length > 0 && (
          <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="h-4 w-4 text-red-600" />
              <span className="font-semibold text-red-900 dark:text-red-100">
                Cảnh báo: {sleepyStudents.length} học sinh cần chú ý
              </span>
            </div>
            <div className="text-sm text-red-700 dark:text-red-200">
              {sleepyStudents.map(s => s.id).join(', ')}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};


