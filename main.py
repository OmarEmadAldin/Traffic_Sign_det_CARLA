import time
import cv2
import numpy as np
import carla
from ultralytics import YOLO
from CarlaCodes.spawn_vec_cam import SpawnManager

# ── Hardcoded config ──────────────────────────────────────────────────────────
WEIGHTS  = "Weights/best.pt"
HOST     = "localhost"
PORT     = 2000
TOWN     = "Town02"
VEHICLE  = "vehicle.tesla.model3"
CONF     = 0.45
IOU      = 0.45
IMGSZ    = 640
SAVE     = "output.avi"          # set to "" to disable saving
# ─────────────────────────────────────────────────────────────────────────────

TARGET_CLASSES = {
    "Pedestrian",
    "Stop",
    "Speed Limit 20",
    "Speed Limit 30",
    "Speed Limit 50",
    "Speed Limit 60",
    "Speed Limit 70",
    "Speed Limit 80",
    "Speed Limit 100",
    "Speed Limit 120",
}

print("[main] Loading model...")
model = YOLO(WEIGHTS)


print(f"[main] Connecting to CARLA {HOST}:{PORT} ...")
client = carla.Client(HOST, PORT)
client.set_timeout(20.0)
print(f"[main] Server: {client.get_server_version()}")

manager = SpawnManager(client, town=TOWN)
writer  = None
fps     = 0.0
t_prev  = time.time()
snap_i  = 0


def get_tl_color_bgr(color_name):
    if color_name == "Red":
        return (0, 0, 255)
    elif color_name == "Yellow":
        return (0, 255, 255)
    else:
        return (0, 255, 0)



try:
    manager.spawn(vehicle_filter=VEHICLE)
    print("[main] Running — q=quit  s=snapshot")

    while True:

        manager.world.tick()
        if not manager.frames_ready():
            continue

        rgb, _ = manager.pop_frames()
        results = model(rgb, conf=CONF, iou=IOU, imgsz=IMGSZ, verbose=False)
        result  = results[0]

        names = model.names
        frame = rgb.copy()
        tl_color_name = manager.get_traffic_light_color()
        tl_color_bgr = get_tl_color_bgr(tl_color_name)

        for box in result.boxes:

            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            name   = names[cls_id]
            print(f'name of the label{name}')

            if name not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if name == "Pedestrian": 
                color = (255, 0, 0) 
                label = "Pedestrian Sign"
     
            elif name == "Stop":
                color = (0, 0, 255)
                label = "STOP Sign"      
            else: 
                color = (0, 255, 0)
                label = "Speed Limit Sign"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            # ── Label background ──
            (tw, th), baseline = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
                
            y_label = max(y1, th + 10)
            cv2.rectangle(frame,(x1, y_label - th - baseline - 6),(x1 + tw + 10, y_label),color,-1)
            cv2.putText(frame,label,(x1 + 5, y_label - 5),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 255, 255),2,cv2.LINE_AA)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Detections: {len(result.boxes)}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 1)
        cv2.putText(frame,f"Traffic Light: {tl_color_name}",(10, 90),cv2.FONT_HERSHEY_SIMPLEX,0.8,tl_color_bgr,2,cv2.LINE_AA)
        
        if writer is None and SAVE:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(SAVE, fourcc, 20, (w, h))
            print(f"[main] Saving to: {SAVE}")

        if writer:
            writer.write(frame)

        # ── Display ──
        cv2.imshow("CARLA Traffic Sign Detection | q=quit s=snapshot", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("s"):
            path = f"snap_{snap_i:04d}.jpg"
            cv2.imwrite(path, frame)
            print(f"[main] Snapshot: {path}")
            snap_i += 1

except KeyboardInterrupt:
    print("\n[main] Interrupted.")

finally:
    manager.destroy()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[main] Done.")