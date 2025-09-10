import streamlit as st, streamlit.components.v1 as components
import cv2, tempfile, time
from ultralytics import YOLO
from pathlib import Path
import math

st.set_page_config(layout="wide", page_title="Sleepy YOLO Pose — HUD", page_icon="🎯")

# ---------- Load HUD header + CSS ----------
css = ""
if Path("frontend/style.css").exists():
    css = Path("frontend/style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
if Path("frontend/index.html").exists():
    header = Path("frontend/index.html").read_text(encoding="utf-8")
    components.html(f"<style>{css}</style>{header}", height=90)

st.markdown('<div class="panel">', unsafe_allow_html=True)

# ---------- Controls ----------
c1, c2, c3 = st.columns(3)
with c1:
    model_path = st.text_input("Model path", "yolo11n-pose.pt")
with c2:
    src = st.selectbox("Nguồn", ["Webcam", "Video"], index=0)
with c3:
    cam_index = st.number_input("Camera index", min_value=0, step=1, value=0)

c4, c5, c6 = st.columns(3)
with c4:
    flip_mode = st.selectbox("Lật khung hình", ["Không lật", "Lật ngang", "Lật dọc", "Quay 180°"], index=0)
with c5:
    fps_limit = st.slider("Giới hạn FPS hiển thị", 5, 60, 30, 1)
with c6:
    imgsz = st.selectbox("Kích thước suy luận (imgsz)", [640, 960, 1280], index=1)

c7, c8, c9 = st.columns(3)
with c7:
    capture_res = st.selectbox("Độ phân giải camera", ["640x480","960x540","1280x720","1920x1080"], index=2)
with c8:
    mjpg = st.checkbox("Ưu tiên MJPG (tránh nén mạnh)", value=True)
with c9:
    line_width = st.slider("Độ dày khung vẽ", 1, 4, 2, 1)

c10, c11, c12 = st.columns(3)
with c10:
    render_w = st.selectbox("Chiều rộng hiển thị", [640, 960, 1280, 1600], index=2)
with c11:
    sharpen = st.checkbox("Tăng nét (unsharp mask)", value=False)
with c12:
    run = st.toggle("▶️ Run", value=False)

uploaded = None
if src == "Video":
    uploaded = st.file_uploader("Chọn video", type=["mp4","mov","avi"])

# ---------- Placeholders ----------
col1, col2 = st.columns([2, 1])
with col1:
    frame_placeholder = st.empty()
with col2:
    log_placeholder = st.empty()
status_placeholder = st.empty()

@st.cache_resource(show_spinner=False)
def load_model(p):
    return YOLO(p)

def parse_res(txt: str):
    w, h = txt.split("x")
    return int(w), int(h)

def open_capture(idx: int, res_txt: str, use_mjpg: bool):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(int(idx), backend)
        if cap.isOpened():
            w, h = parse_res(res_txt)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, 30)
            # if use_mjpg:
            #     cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'M','J','P','G'))
            return cap
    return None

if 'sleep_log' not in st.session_state:
    st.session_state.sleep_log = []

# ---------- RUN LOOP ----------
if run:
    model = load_model(model_path)

    cap = None
    SLEEP_ANGLE = 45
    SLEEP_FRAMES = 45
    sleep_states = {}
    ema_fps = None
    if src == "Webcam":
        cap = open_capture(cam_index, capture_res, mjpg)
        if cap is None:
            st.error("❌ Không mở được webcam. Thử index khác hoặc tắt MJPG / giảm độ phân giải.")
            st.stop()
    else:
        if not uploaded:
            st.warning("Hãy upload video trước khi Run."); st.stop()
        else:
            t = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            t.write(uploaded.read()); t.flush()
            cap = cv2.VideoCapture(t.name)
    try:
        while cap is not None and cap.isOpened():
            t0 = time.time()
            ok, frame = cap.read()
            if not ok or frame is None:
                st.warning("Hết video hoặc lỗi camera."); break

            # Flip (nếu cần)
            if flip_mode == "Lật ngang":
                frame = cv2.flip(frame, 1)
            elif flip_mode == "Lật dọc":
                frame = cv2.flip(frame, 0)
            elif flip_mode == "Quay 180°":
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            # Inference
            res = model(frame, imgsz=int(imgsz), conf=0.5)
            r0 = res[0] if res and len(res) > 0 else None
            vis = r0.plot(line_width=int(line_width), conf=True) if r0 is not None else frame.copy()

            # Phát hiện ngủ gật dựa vào keypoints
            sleepy_count = 0
            if r0 is not None and hasattr(r0, "keypoints"):
                for i, kp in enumerate(r0.keypoints):
                    k = kp.xy[0].cpu().numpy()
                    if len(k) >= 7:
                        nose, l_sh, r_sh = k[0], k[5], k[6]
                        neck = ((l_sh[0]+r_sh[0])/2, (l_sh[1]+r_sh[1])/2)
                        dx, dy = nose[0]-neck[0], nose[1]-neck[1]
                        ang = abs(math.degrees(math.atan2(dx, dy)))
                        prev = sleep_states.get(i, 0)
                        sleep_states[i] = prev + 1 if ang > SLEEP_ANGLE else 0
                        state = "NGỦ GẬT" if sleep_states[i] >= SLEEP_FRAMES else "Bình thường"
                        if state == "NGỦ GẬT": 
                            sleepy_count += 1
                            if sleep_states[i] == SLEEP_FRAMES: # Log khi mới phát hiện
                                st.session_state.sleep_log.insert(0, f"[{time.strftime('%H:%M:%S')}] Người {i+1} có dấu hiệu ngủ gật!\n")
                        cv2.putText(vis, f"{state} ({ang:.1f}°)", (int(nose[0]), int(nose[1])-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255) if state=="NGỦ GẬT" else (0,255,0), 2)

            # BGR -> RGB
            try:
                rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            except Exception:
                rgb = vis

            # Sharpen (optional)
            if sharpen:
                blur = cv2.GaussianBlur(rgb, (0,0), 1.0)
                rgb  = cv2.addWeighted(rgb, 1.25, blur, -0.25, 0)

            # Render exact width to avoid blur
            h, w = rgb.shape[:2]
            if render_w != w:
                scale = render_w / max(1, w)
                rgb = cv2.resize(rgb, (render_w, int(h*scale)), interpolation=cv2.INTER_CUBIC)

            frame_placeholder.image(
                rgb, channels="RGB", output_format="PNG", width=render_w, caption="frame"
            )

            log_placeholder.text_area(
                "Log phát hiện",
                "".join(st.session_state.sleep_log),
                height=400,
                key="sleep_log_textarea",
            )

            # Hiển thị trạng thái tổng
            status_text = f"<div class='status-line'><b>FPS:</b> {{fps:.1f}} &nbsp;|&nbsp; <b>Latency:</b> {{latency}} ms &nbsp;|&nbsp; <b>Số người ngủ gật:</b> {sleepy_count}</div>"
            t1 = time.time()
            iter_time = t1 - t0
            fps_now = 1.0 / max(iter_time, 1e-6)
            ema_fps = fps_now if ema_fps is None else (0.8*ema_fps + 0.2*fps_now)
            status_placeholder.markdown(
                status_text.format(fps=ema_fps, latency=int(iter_time*1000)), unsafe_allow_html=True
            )

            # FPS limiter
            delay = max(0, (1.0 / max(1, fps_limit)) - (time.time() - t1))
            if delay:
                time.sleep(delay)
    finally:
        if cap is not None:
            cap.release()

st.markdown('</div>', unsafe_allow_html=True)
