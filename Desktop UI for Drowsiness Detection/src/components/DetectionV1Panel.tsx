/**
 * Panel phát hiện ngủ gật (chính) + bấm điện thoại (phụ) — pipeline `/api/v1/detect`.
 *
 * Layout 3 cột (mirror tab Camera):
 *   trái  — chọn nguồn (Realtime / Ảnh / Video) + cấu hình
 *   giữa — canvas hiển thị stream/preview với bbox overlay
 *   phải  — 3 stat cards + kết quả hiện tại + log alert gần đây
 *
 * Realtime: WebSocket qua IPC bridge → namespace /api/v1/detect/realtime.
 */
import React, { useEffect, useRef, useState } from 'react';
import { apiGet, apiPost } from '../lib/api';
import { wsDetectV1 } from '../lib/wsDetectV1';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { ScrollArea } from './ui/scroll-area';
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from './ui/resizable';
import { Camera, Image as ImageIcon, Film, AlertTriangle, Activity } from 'lucide-react';

interface DetectionObject {
  class_name: string;
  display_name: string;
  confidence: number;
  bbox: [number, number, number, number];
  severity: 'danger' | 'warn' | 'info';
  source: string;
}

interface TopKItem {
  class_name: string;
  display_name: string;
  confidence: number;
  source: string;
}

interface DetectionResponse {
  objects: DetectionObject[];
  top_k: TopKItem[];
  inference_time_ms: number;
  image_size: [number, number];
}

interface AlertEntry {
  id: string;
  timestamp: string;
  display_name: string;
  severity: 'danger' | 'warn';
  confidence: number;
  source: 'realtime' | 'image' | 'video';
}

type Mode = 'realtime' | 'image' | 'video';

const BEEP_SRC = 'data:audio/wav;base64,UklGRhQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQ==';
const MAX_ALERTS = 50;

async function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const s = String(reader.result || '');
      resolve(s.includes(',') ? s.split(',')[1] : s);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function drawOverlay(
  canvas: HTMLCanvasElement,
  src: HTMLImageElement | HTMLVideoElement,
  objs: DetectionObject[]
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const w = (src as HTMLImageElement).naturalWidth || (src as HTMLVideoElement).videoWidth;
  const h = (src as HTMLImageElement).naturalHeight || (src as HTMLVideoElement).videoHeight;
  if (!w || !h) return;
  canvas.width = w;
  canvas.height = h;
  ctx.drawImage(src, 0, 0, w, h);
  for (const o of objs) {
    const color =
      o.severity === 'danger' ? '#ef4444' : o.severity === 'warn' ? '#eab308' : '#22c55e';
    const [x1, y1, x2, y2] = o.bbox;
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2, Math.round(w / 320));
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = `${o.display_name} ${o.confidence.toFixed(2)}`;
    ctx.font = `${Math.max(14, Math.round(w / 60))}px system-ui`;
    const pad = 4;
    const tw = ctx.measureText(label).width;
    const th = Math.max(14, Math.round(w / 60));
    ctx.fillStyle = color;
    ctx.fillRect(x1, Math.max(0, y1 - th - pad * 2), tw + pad * 2, th + pad * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, x1 + pad, Math.max(th, y1 - pad));
  }
}

function severityChipClass(sev: 'danger' | 'warn' | 'info'): string {
  if (sev === 'danger') return 'bg-red-600 text-white';
  if (sev === 'warn') return 'bg-yellow-500 text-black';
  return 'bg-green-600 text-white';
}

// ─── Sidebar (cột trái) ────────────────────────────────────────────────────

interface SidebarProps {
  mode: Mode;
  setMode: (m: Mode) => void;
  conf: number;
  setConf: (v: number) => void;
  streaming: boolean;
  wsConnected: boolean;
  onStartRealtime: () => void;
  onStopRealtime: () => void;
  onImageSelected: (file: File) => void;
  onVideoSelected: (file: File) => void;
  health: { primary?: string; secondary?: string | null } | null;
  imgBusy: boolean;
  videoBusy: boolean;
}

function Sidebar(props: SidebarProps) {
  const MODES: { key: Mode; label: string; icon: React.ReactNode }[] = [
    { key: 'realtime', label: 'Camera Realtime', icon: <Camera size={16} /> },
    { key: 'image', label: 'Upload Ảnh', icon: <ImageIcon size={16} /> },
    { key: 'video', label: 'Upload Video', icon: <Film size={16} /> },
  ];

  return (
    <div className="flex flex-col h-full p-4 gap-4">
      <div>
        <div className="text-base font-semibold mb-1">Nguồn phát hiện</div>
        <div className="text-xs text-muted-foreground">
          Chọn nguồn đầu vào cho pipeline V1
        </div>
      </div>

      <div className="flex flex-col gap-1.5">
        {MODES.map((m) => {
          const active = props.mode === m.key;
          return (
            <button
              key={m.key}
              onClick={() => props.setMode(m.key)}
              className={`flex items-center gap-2.5 px-3 py-2.5 rounded-lg border text-sm text-left transition-colors ${
                active
                  ? 'bg-primary/10 border-primary text-primary font-medium'
                  : 'bg-card hover:bg-accent border-border'
              }`}
            >
              {m.icon}
              <span>{m.label}</span>
            </button>
          );
        })}
      </div>

      <div className="border-t pt-4 flex flex-col gap-3">
        <div>
          <Label className="text-xs">Confidence ({props.conf.toFixed(2)})</Label>
          <Input
            type="range"
            min={0.1}
            max={0.95}
            step={0.05}
            value={props.conf}
            onChange={(e) => props.setConf(Number(e.target.value))}
            className="mt-1"
          />
        </div>

        {props.mode === 'realtime' && (
          <>
            {!props.streaming ? (
              <Button onClick={props.onStartRealtime} className="w-full">
                Bật camera
              </Button>
            ) : (
              <Button variant="destructive" onClick={props.onStopRealtime} className="w-full">
                Dừng
              </Button>
            )}
            <div className="flex items-center gap-2 text-xs">
              <span
                className={`w-2 h-2 rounded-full ${
                  props.wsConnected ? 'bg-green-500' : 'bg-gray-400'
                }`}
              />
              <span className="text-muted-foreground">
                WS: {props.wsConnected ? 'connected' : 'disconnected'}
              </span>
            </div>
          </>
        )}

        {props.mode === 'image' && (
          <div>
            <Label className="text-xs">Chọn ảnh</Label>
            <Input
              type="file"
              accept="image/*"
              onChange={(e) => e.target.files && props.onImageSelected(e.target.files[0])}
              className="mt-1 text-xs"
            />
            {props.imgBusy && (
              <div className="text-xs text-muted-foreground mt-1">Đang phân tích…</div>
            )}
          </div>
        )}

        {props.mode === 'video' && (
          <div>
            <Label className="text-xs">Chọn video</Label>
            <Input
              type="file"
              accept="video/*"
              onChange={(e) => e.target.files && props.onVideoSelected(e.target.files[0])}
              className="mt-1 text-xs"
            />
            {props.videoBusy && (
              <div className="text-xs text-muted-foreground mt-1">Đang xử lý video…</div>
            )}
          </div>
        )}
      </div>

      <div className="mt-auto border-t pt-3">
        <div className="text-xs font-medium mb-1.5">Model</div>
        {props.health ? (
          <div className="space-y-1">
            <Badge variant="outline" className="bg-green-500/10 text-green-700 border-green-500/30">
              Online
            </Badge>
            <div className="text-xs text-muted-foreground">
              Primary: <b className="text-foreground">{props.health.primary || '—'}</b>
            </div>
            {props.health.secondary && (
              <div className="text-xs text-muted-foreground">
                Secondary: <b className="text-foreground">{props.health.secondary}</b>
              </div>
            )}
          </div>
        ) : (
          <Badge variant="destructive">Offline</Badge>
        )}
      </div>
    </div>
  );
}

// ─── Right panel (cột phải) ─────────────────────────────────────────────────

interface RightProps {
  result: DetectionResponse | null;
  alerts: AlertEntry[];
  stats: { drowsy: number; phone: number; avgInfMs: number };
}

function RightPanel({ result, alerts, stats }: RightProps) {
  return (
    <div className="flex flex-col h-full p-4 gap-3 overflow-hidden">
      {/* 3 stat cards */}
      <div className="grid grid-cols-3 gap-2 shrink-0">
        <div className="rounded-lg border p-2.5 bg-red-50">
          <div className="flex items-center gap-1 text-xs text-red-700">
            <AlertTriangle size={12} /> Ngủ gật
          </div>
          <div className="text-xl font-bold text-red-700">{stats.drowsy}</div>
        </div>
        <div className="rounded-lg border p-2.5 bg-yellow-50">
          <div className="flex items-center gap-1 text-xs text-yellow-700">📱 Điện thoại</div>
          <div className="text-xl font-bold text-yellow-700">{stats.phone}</div>
        </div>
        <div className="rounded-lg border p-2.5 bg-blue-50">
          <div className="flex items-center gap-1 text-xs text-blue-700">
            <Activity size={12} /> Avg ms
          </div>
          <div className="text-xl font-bold text-blue-700">
            {stats.avgInfMs ? stats.avgInfMs.toFixed(0) : '—'}
          </div>
        </div>
      </div>

      {/* Current result */}
      <Card className="shrink-0">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Kết quả hiện tại</CardTitle>
          {result && (
            <CardDescription className="text-xs">
              {result.inference_time_ms.toFixed(0)}ms • {result.objects.length} object(s)
            </CardDescription>
          )}
        </CardHeader>
        <CardContent className="pt-0">
          {!result ? (
            <div className="text-xs text-muted-foreground">Chưa có kết quả.</div>
          ) : result.objects.length === 0 && result.top_k.length === 0 ? (
            <div className="text-xs text-green-700">✓ Bình thường</div>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {result.objects.map((o, i) => (
                <Badge key={i} className={`text-xs ${severityChipClass(o.severity)}`}>
                  {o.display_name} {o.confidence.toFixed(2)}
                </Badge>
              ))}
              {result.objects.length === 0 &&
                result.top_k.slice(0, 3).map((t, i) => (
                  <Badge key={`tk-${i}`} variant="outline" className="text-xs">
                    {t.display_name} ({t.confidence.toFixed(2)})
                  </Badge>
                ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Alerts log */}
      <Card className="flex-1 overflow-hidden flex flex-col">
        <CardHeader className="pb-2 shrink-0">
          <CardTitle className="text-sm">Log cảnh báo ({alerts.length})</CardTitle>
          <CardDescription className="text-xs">50 alert gần nhất</CardDescription>
        </CardHeader>
        <CardContent className="pt-0 flex-1 overflow-hidden">
          <ScrollArea className="h-full pr-2">
            {alerts.length === 0 ? (
              <div className="text-xs text-muted-foreground">Chưa có cảnh báo.</div>
            ) : (
              <div className="space-y-1.5">
                {alerts.map((a) => (
                  <div
                    key={a.id}
                    className={`text-xs rounded border p-2 ${
                      a.severity === 'danger'
                        ? 'border-red-200 bg-red-50 text-red-800'
                        : 'border-yellow-200 bg-yellow-50 text-yellow-800'
                    }`}
                  >
                    <div className="flex justify-between">
                      <b>{a.display_name}</b>
                      <span className="opacity-70">{a.timestamp}</span>
                    </div>
                    <div className="opacity-80">
                      conf {a.confidence.toFixed(2)} • {a.source}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
}

// ─── Main panel ─────────────────────────────────────────────────────────────

export default function DetectionV1Panel() {
  const [mode, setMode] = useState<Mode>('realtime');
  const [conf, setConf] = useState(0.35);
  const confRef = useRef(conf);
  useEffect(() => {
    confRef.current = conf;
  }, [conf]);

  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [alerts, setAlerts] = useState<AlertEntry[]>([]);
  const [health, setHealth] = useState<{ primary?: string; secondary?: string | null } | null>(null);

  const [streaming, setStreaming] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [imgBusy, setImgBusy] = useState(false);
  const [videoBusy, setVideoBusy] = useState(false);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string>('');

  const videoRef = useRef<HTMLVideoElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const beepRef = useRef<HTMLAudioElement | null>(null);
  const intervalRef = useRef<number | null>(null);
  const pendingRef = useRef(false);
  const infHistoryRef = useRef<number[]>([]);

  const stats = React.useMemo(() => {
    const drowsy = alerts.filter((a) => a.display_name.toLowerCase().includes('ngủ')).length;
    const phone = alerts.filter((a) => a.display_name.toLowerCase().includes('điện thoại')).length;
    const avg =
      infHistoryRef.current.length === 0
        ? 0
        : infHistoryRef.current.reduce((a, b) => a + b, 0) / infHistoryRef.current.length;
    return { drowsy, phone, avgInfMs: avg };
  }, [alerts, result]);

  function recordAlerts(objs: DetectionObject[], source: AlertEntry['source']) {
    const now = new Date().toLocaleTimeString('vi-VN', { hour12: false });
    const fresh: AlertEntry[] = objs
      .filter((o) => o.severity === 'danger' || o.severity === 'warn')
      .map((o, i) => ({
        id: `${Date.now()}-${i}-${o.class_name}`,
        timestamp: now,
        display_name: o.display_name,
        severity: o.severity as 'danger' | 'warn',
        confidence: o.confidence,
        source,
      }));
    if (fresh.length === 0) return;
    setAlerts((prev) => [...fresh, ...prev].slice(0, MAX_ALERTS));
  }

  function handleResult(json: DetectionResponse, source: AlertEntry['source']) {
    setResult(json);
    infHistoryRef.current = [...infHistoryRef.current, json.inference_time_ms].slice(-30);
    if (json.objects.some((o) => o.severity === 'danger')) {
      beepRef.current?.play().catch(() => {
        /* autoplay blocked */
      });
    }
    recordAlerts(json.objects, source);
  }

  // Health check
  useEffect(() => {
    apiGet('/api/v1/detect/health')
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  // WS setup (luôn ready, chỉ emit frame khi đang streaming)
  useEffect(() => {
    beepRef.current = new Audio(BEEP_SRC);

    wsDetectV1.onStatus((c) => setWsConnected(c));
    wsDetectV1.onResult((msg) => {
      pendingRef.current = false;
      const json = msg as DetectionResponse;
      handleResult(json, 'realtime');
      if (canvasRef.current && videoRef.current && videoRef.current.videoWidth > 0) {
        drawOverlay(canvasRef.current, videoRef.current, json.objects);
      }
    });
    wsDetectV1.onError(() => {
      pendingRef.current = false;
    });
    wsDetectV1.connect();

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      const stream = videoRef.current?.srcObject as MediaStream | null;
      stream?.getTracks().forEach((t) => t.stop());
      wsDetectV1.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function startRealtime() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setStreaming(true);
      intervalRef.current = window.setInterval(() => {
        if (pendingRef.current || !videoRef.current) return;
        const v = videoRef.current;
        if (v.videoWidth === 0) return;
        const off = document.createElement('canvas');
        off.width = v.videoWidth;
        off.height = v.videoHeight;
        const ctx = off.getContext('2d');
        if (!ctx) return;
        ctx.drawImage(v, 0, 0);
        const b64 = off.toDataURL('image/jpeg', 0.7).split(',')[1];
        pendingRef.current = true;
        wsDetectV1.sendFrame(b64, confRef.current);
        window.setTimeout(() => {
          pendingRef.current = false;
        }, 3000);
      }, 500);
    } catch (e) {
      console.error('[DetectV1] getUserMedia', e);
    }
  }

  function stopRealtime() {
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = null;
    const stream = videoRef.current?.srcObject as MediaStream | null;
    stream?.getTracks().forEach((t) => t.stop());
    setStreaming(false);
  }

  async function handleImageFile(file: File) {
    setImgBusy(true);
    setResult(null);
    setImagePreviewUrl(URL.createObjectURL(file));
    try {
      const b64 = await fileToBase64(file);
      const resp = await apiPost(`/api/v1/detect/image?conf=${conf}`, { image_base64: b64 });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const json = (await resp.json()) as DetectionResponse;
      handleResult(json, 'image');
      if (imgRef.current && canvasRef.current) {
        imgRef.current.onload = () =>
          drawOverlay(canvasRef.current!, imgRef.current!, json.objects);
      }
    } catch (e) {
      console.error('[DetectV1/image]', e);
    } finally {
      setImgBusy(false);
    }
  }

  async function handleVideoFile(file: File) {
    setVideoBusy(true);
    setResult(null);
    try {
      const b64 = await fileToBase64(file);
      const resp = await apiPost(`/api/v1/detect/video?conf=${conf}`, {
        video_base64: b64,
        filename: file.name,
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = (await resp.json()) as {
        alerts: Array<{ display_name: string; since_seconds: number; duration_seconds: number }>;
      };
      const now = new Date().toLocaleTimeString('vi-VN', { hour12: false });
      const fresh: AlertEntry[] = data.alerts.map((a, i) => ({
        id: `vid-${Date.now()}-${i}`,
        timestamp: now,
        display_name: `${a.display_name} @${a.since_seconds}s (${a.duration_seconds}s)`,
        severity: 'danger',
        confidence: 1,
        source: 'video',
      }));
      setAlerts((prev) => [...fresh, ...prev].slice(0, MAX_ALERTS));
    } catch (e) {
      console.error('[DetectV1/video]', e);
    } finally {
      setVideoBusy(false);
    }
  }

  return (
    <div className="flex-1 flex flex-col gap-3 p-4 bg-muted/30 overflow-hidden">
      <Card className="rounded-xl border shadow-sm shrink-0">
        <CardHeader className="py-3">
          <CardTitle className="text-lg flex items-center gap-2">
            ⚠️ Phát hiện ngủ gật & điện thoại (pipeline V1)
          </CardTitle>
        </CardHeader>
      </Card>

      <ResizablePanelGroup direction="horizontal" className="flex-1 gap-3">
        <ResizablePanel
          defaultSize={20}
          minSize={15}
          maxSize={30}
          className="bg-card rounded-xl border shadow-sm overflow-hidden"
        >
          <Sidebar
            mode={mode}
            setMode={setMode}
            conf={conf}
            setConf={setConf}
            streaming={streaming}
            wsConnected={wsConnected}
            onStartRealtime={startRealtime}
            onStopRealtime={stopRealtime}
            onImageSelected={handleImageFile}
            onVideoSelected={handleVideoFile}
            health={health}
            imgBusy={imgBusy}
            videoBusy={videoBusy}
          />
        </ResizablePanel>

        <ResizableHandle className="bg-transparent w-1" />

        <ResizablePanel
          defaultSize={55}
          minSize={30}
          className="bg-card rounded-xl border shadow-sm overflow-hidden"
        >
          <div className="flex flex-col h-full p-4 overflow-hidden">
            <div className="text-sm font-semibold mb-2 shrink-0">
              {mode === 'realtime'
                ? '📹 Camera Realtime'
                : mode === 'image'
                ? '🖼️ Xem ảnh'
                : '🎞️ Xem video'}
            </div>
            <div className="flex-1 overflow-auto flex items-center justify-center bg-black/5 rounded-lg">
              {/* Hidden source elements (read-only) */}
              <video ref={videoRef} className="hidden" playsInline muted />
              {imagePreviewUrl && (
                <img ref={imgRef} src={imagePreviewUrl} alt="preview" className="hidden" />
              )}
              <canvas
                ref={canvasRef}
                className="max-w-full max-h-full rounded"
                style={{ display: result ? 'block' : 'none' }}
              />
              {!result && (
                <div className="text-sm text-muted-foreground">
                  {mode === 'realtime'
                    ? 'Bấm "Bật camera" ở cột trái để bắt đầu.'
                    : mode === 'image'
                    ? 'Chọn ảnh ở cột trái để phân tích.'
                    : 'Chọn video ở cột trái để phân tích.'}
                </div>
              )}
            </div>
          </div>
        </ResizablePanel>

        <ResizableHandle className="bg-transparent w-1" />

        <ResizablePanel
          defaultSize={25}
          minSize={20}
          maxSize={35}
          className="bg-card rounded-xl border shadow-sm overflow-hidden"
        >
          <RightPanel result={result} alerts={alerts} stats={stats} />
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  );
}
