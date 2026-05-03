import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from './ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Label } from './ui/label';
import { Input } from './ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from './ui/select';
import { Switch } from './ui/switch';
import { Button } from './ui/button';
import { Upload, Download } from 'lucide-react';

interface SettingsDialogProps {
  open: boolean;
  onClose: () => void;
}

export function SettingsDialog({ open, onClose }: SettingsDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Cài đặt Hệ thống</DialogTitle>
          <DialogDescription>
            Cấu hình model, hiệu năng và giao diện
          </DialogDescription>
        </DialogHeader>

        <Tabs defaultValue="model" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="model">Model & Detection</TabsTrigger>
            <TabsTrigger value="performance">Hiệu năng</TabsTrigger>
            <TabsTrigger value="ui">Giao diện</TabsTrigger>
            <TabsTrigger value="config">Cấu hình</TabsTrigger>
          </TabsList>

          <TabsContent value="model" className="space-y-4">
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Model Pose</Label>
                <Select defaultValue="yolo11n">
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="yolo11n">yolo11n-pose.pt (Nhanh)</SelectItem>
                    <SelectItem value="yolo11s">yolo11s-pose.pt (Cân bằng)</SelectItem>
                    <SelectItem value="yolo11m">yolo11m-pose.pt (Chính xác)</SelectItem>
                    <SelectItem value="yolo11l">yolo11l-pose.pt (Rất chính xác)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Detection Strategy</Label>
                <Select defaultValue="yolo">
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="yolo">YOLO Pose</SelectItem>
                    <SelectItem value="mediapipe">Mediapipe</SelectItem>
                    <SelectItem value="ear">EAR (Eye Aspect Ratio)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Confidence Threshold: 0.5</Label>
                <Input type="range" min="0" max="1" step="0.05" defaultValue="0.5" />
              </div>

              <div className="space-y-2">
                <Label>Image Size</Label>
                <Select defaultValue="640">
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="320">320 (Nhanh nhất)</SelectItem>
                    <SelectItem value="640">640 (Khuyến nghị)</SelectItem>
                    <SelectItem value="1280">1280 (Chi tiết)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Auto Download Model</Label>
                  <p className="text-sm text-muted-foreground">
                    Tự động tải model nếu chưa có
                  </p>
                </div>
                <Switch defaultChecked />
              </div>
            </div>
          </TabsContent>

          <TabsContent value="performance" className="space-y-4">
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Target FPS</Label>
                <Input type="number" defaultValue="30" min="10" max="60" />
              </div>

              <div className="space-y-2">
                <Label>Frame Queue Size</Label>
                <Input type="number" defaultValue="2" min="1" max="10" />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Drop Oldest Frame</Label>
                  <p className="text-sm text-muted-foreground">
                    Bỏ khung hình cũ khi queue đầy
                  </p>
                </div>
                <Switch defaultChecked />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>GPU Acceleration</Label>
                  <p className="text-sm text-muted-foreground">
                    Sử dụng GPU nếu có (CUDA)
                  </p>
                </div>
                <Switch defaultChecked />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Multi-threading</Label>
                  <p className="text-sm text-muted-foreground">
                    Xử lý đa luồng cho nhiều camera
                  </p>
                </div>
                <Switch defaultChecked />
              </div>

              <div className="space-y-2">
                <Label>Max Latency (ms)</Label>
                <Input type="number" defaultValue="2000" min="500" max="5000" step="100" />
              </div>
            </div>
          </TabsContent>

          <TabsContent value="ui" className="space-y-4">
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Theme</Label>
                <Select defaultValue="system">
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="light">Light</SelectItem>
                    <SelectItem value="dark">Dark</SelectItem>
                    <SelectItem value="system">System</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Font Size</Label>
                <Select defaultValue="medium">
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="small">Nhỏ</SelectItem>
                    <SelectItem value="medium">Vừa</SelectItem>
                    <SelectItem value="large">Lớn</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Language</Label>
                <Select defaultValue="vi">
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="vi">Tiếng Việt</SelectItem>
                    <SelectItem value="en">English</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>High DPI Support</Label>
                  <p className="text-sm text-muted-foreground">
                    Tối ưu cho màn hình độ phân giải cao
                  </p>
                </div>
                <Switch defaultChecked />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Show Notifications</Label>
                  <p className="text-sm text-muted-foreground">
                    Hiện thông báo khi phát hiện ngủ gật
                  </p>
                </div>
                <Switch defaultChecked />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Play Alert Sound</Label>
                  <p className="text-sm text-muted-foreground">
                    Phát âm thanh cảnh báo
                  </p>
                </div>
                <Switch />
              </div>
            </div>
          </TabsContent>

          <TabsContent value="config" className="space-y-4">
            <div className="space-y-4">
              <div className="p-4 border rounded-lg space-y-4">
                <div>
                  <h4 className="mb-2">Import/Export Configuration</h4>
                  <p className="text-sm text-muted-foreground mb-4">
                    Lưu và khôi phục cấu hình camera với decorators
                  </p>
                  <div className="flex gap-2">
                    <Button variant="outline" className="flex-1">
                      <Upload className="h-4 w-4 mr-2" />
                      Import YAML
                    </Button>
                    <Button variant="outline" className="flex-1">
                      <Download className="h-4 w-4 mr-2" />
                      Export YAML
                    </Button>
                  </div>
                </div>
              </div>

              <div className="p-4 border rounded-lg space-y-4">
                <div>
                  <h4 className="mb-2">Layout Management</h4>
                  <p className="text-sm text-muted-foreground mb-4">
                    Quản lý bố cục giao diện
                  </p>
                  <div className="flex gap-2">
                    <Button variant="outline" className="flex-1">
                      Lưu bố cục hiện tại
                    </Button>
                    <Button variant="outline" className="flex-1">
                      Khôi phục bố cục mặc định
                    </Button>
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Auto-save Layout</Label>
                  <p className="text-sm text-muted-foreground">
                    Tự động lưu bố cục khi thay đổi
                  </p>
                </div>
                <Switch defaultChecked />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Restore on Startup</Label>
                  <p className="text-sm text-muted-foreground">
                    Khôi phục cấu hình khi khởi động
                  </p>
                </div>
                <Switch defaultChecked />
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}
