from ultralytics import YOLO
import cv2, math, ctypes, numpy as np

MODEL_PATH = "yolo11n-pose.pt"
WINDOW_NAME = "Sleepy Pose Demo"

def head_angle(nose, neck):
    dx, dy = nose[0]-neck[0], nose[1]-neck[1]
    return abs(math.degrees(math.atan2(dx, dy)))

def get_screen_size():
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

def letterbox_to_screen(img):
    sw, sh = get_screen_size()
    h, w = img.shape[:2]
    scale = min(sw/w, sh/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
    x, y = (sw-nw)//2, (sh-nh)//2
    canvas[y:y+nh, x:x+nw] = resized
    return canvas

def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    cap.set(cv2.CAP_PROP_FPS,30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    SLEEP_ANGLE = 45
    SLEEP_FRAMES = 45
    consec = 0
    mirror = True

    while True:
        ok, frame = cap.read()
        if not ok: break
        if mirror: frame = cv2.flip(frame,1)

        res = model(frame, imgsz=960)
        plot = res[0].plot(line_width=2)

        for kp in res[0].keypoints:
            k = kp.xy[0].cpu().numpy()
            nose, l_sh, r_sh = k[0], k[5], k[6]
            neck = ((l_sh[0]+r_sh[0])/2, (l_sh[1]+r_sh[1])/2)
            ang = head_angle(nose, neck)
            consec = consec + 1 if ang > SLEEP_ANGLE else 0
            state = "SLEEPY" if consec >= SLEEP_FRAMES else "OK"
            cv2.putText(plot, f"{state} ({ang:.1f}°)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        show = letterbox_to_screen(plot)
        cv2.imshow(WINDOW_NAME, show)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')): break
        elif key in (ord('m'), ord('M')): mirror = not mirror

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
