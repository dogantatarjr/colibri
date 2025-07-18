import cv2
import numpy as np
from pymavlink import mavutil
import math
import time

# --- MAVLink bağlantısını güvenli başlatma ---
def init_mavlink(connection_str, timeout=30):
    try:
        master = mavutil.mavlink_connection(connection_str)
        print("Heartbeat bekleniyor...")
        master.wait_heartbeat(timeout=timeout)
        print("Heartbeat alındı, MAVLink bağlantısı kuruldu.")
        return master
    except Exception as e:
        print(f"MAVLink bağlantı hatası: {e}")
        return None

# MAVLink bağlantısını başlat
master = init_mavlink('udp:127.0.0.1:14552')
if master is None:
    raise SystemExit("MAVLink bağlantısı kurulamadı. Program sonlandırılıyor.")

# Video yakalama (webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Kamera açılamadı. Program sonlandırılıyor.")

# Parametreler
correction_threshold = 30  # Görsel hizalama için piksel eşiği
alignment_start_time = None  # Hizalamanın başlatıldığı zaman
water_dropped = False  # Suyun bırakılıp bırakılmadığı bayrağı

enhancement_mode = "Mission Planner Route"

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera akışı alınamıyor, döngü sonlandırılıyor.")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0,100,100]), np.array([10,255,255])
    lower_red2, upper_red2 = np.array([160,100,100]), np.array([180,255,255])
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                          cv2.inRange(hsv, lower_red2, upper_red2))
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    control_roll, control_pitch = 0, 0
    mode = enhancement_mode

    if contours:
        c = max(contours, key=cv2.contourArea)
        ((x,y), radius) = cv2.minEnclosingCircle(c)
        if radius >= 10:
            # VTOL mod geçişi
            if radius < 50:
                mode = "VTOL Vertical Mode"
                master.mav.command_long_send(
                    master.target_system, master.target_component,
                    mavutil.mavlink.MAV_CMD_DO_VTOL_TRANSITION,
                    0, 0,0,0,0,0,0,0)
            else:
                mode = "VTOL Fixed Wing Mode"
                master.mav.command_long_send(
                    master.target_system, master.target_component,
                    mavutil.mavlink.MAV_CMD_DO_VTOL_TRANSITION,
                    0, 1,0,0,0,0,0,0)

            M = cv2.moments(c)
            cx = int(M['m10']/M['m00']) if M['m00'] else int(x)
            cy = int(M['m01']/M['m00']) if M['m00'] else int(y)
            frame_cx, frame_cy = frame.shape[1]//2, frame.shape[0]//2
            error_x, error_y = cx - frame_cx, cy - frame_cy
            error_mag = math.hypot(error_x, error_y)

            if error_mag > correction_threshold:
                # Hala hizalanmadı, düzeltme
                mode += " | Correcting Orientation"
                control_roll = int(np.clip(error_x, -500, 500))
                control_pitch = int(np.clip(error_y, -500, 500))
                alignment_start_time = None
                water_dropped = False
            else:
                # Hizalama tamamlandı
                if alignment_start_time is None:
                    alignment_start_time = time.time()
                elif not water_dropped and time.time() - alignment_start_time >= 1:
                    # 1 saniye geçtikten sonra suyu otomatik bırak
                    print("Hedef üzerine gelindi, su bırakılıyor...")
                    master.mav.command_long_send(
                        master.target_system, master.target_component,
                        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                        0, 9, 1900, 0,0,0,0,0)
                    water_dropped = True

            # Görsel geri bildirim
            cv2.circle(frame, (cx,cy), int(radius), (0,255,0), 2)
            cv2.circle(frame, (cx,cy), 5, (255,0,0), -1)
            cv2.line(frame, (frame_cx,frame_cy), (cx,cy), (255,255,0), 2)

    # RC override
    try:
        master.mav.rc_channels_override_send(
            master.target_system, master.target_component,
            1500 + control_roll, 1500 + control_pitch,
            1500, 1500, 0,0,0,0)
    except Exception as e:
        print(f"RC override hatası: {e}")

    cv2.putText(frame, f"Mode: {mode}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == 27:
        print("Çıkış tuşuna basıldı. Döngü sonlandırılıyor.")
        break

cap.release()
cv2.destroyAllWindows()