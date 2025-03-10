import cv2
import numpy as np
from pymavlink import mavutil
import math
import time

# MAVLink bağlantısını başlat (SITL simülasyonu veya gerçek uçuş kartı için uygun port)
master = mavutil.mavlink_connection('udp:127.0.0.1:14552')
print("Heartbeat bekleniyor...")
master.wait_heartbeat()
print("Heartbeat alındı, MAVLink bağlantısı kuruldu.")

# Video yakalama (webcam)
cap = cv2.VideoCapture(0)

# Print işlemleri için zaman kontrolü
last_print_time = time.time()

# Döngü
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü HSV renk uzayına çevir
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Kırmızı renk için iki farklı HSV aralığını tanımla
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Maskeleri birleştir
    mask = cv2.bitwise_or(mask1, mask2)

    # Gürültüyü azaltmak için morfolojik işlemler
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # Kontur tespiti
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Varsayılan kontrol değerleri
    control_roll = 0
    control_pitch = 0
    mode = "Mission Planner Route"  # Varsayılan mod: override yapılmıyor

    if contours:
        # En büyük konturu seç
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius < 10:
            mode = "No Override"
            # Eğer yangın alanı çok küçükse override yapılmıyor
        else:
            # Konturun merkezini hesapla
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = int(x), int(y)
            
            # Tespit edilen alanı çizdir
            cv2.circle(frame, (cx, cy), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            
            # Görüntü merkezini hesapla
            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2
            
            # Görsel hata: hedef ile görüntü merkezi arasındaki fark
            error_x = cx - frame_center_x
            error_y = cy - frame_center_y
            
            # Hata değerlerini kullanarak RC override ofsetlerini hesapla
            visual_roll = int(np.clip(error_x, -500, 500))
            visual_pitch = int(np.clip(error_y, -500, 500))
            
            cv2.line(frame, (frame_center_x, frame_center_y), (cx, cy), (255, 255, 0), 2)
            cv2.putText(frame, f"Visual Err: ({error_x}, {error_y})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # VTOL modunun belirlenmesi: yarıçap aralıklarına göre
            if radius < 50:
                mode = "VTOL Vertical Mode"
                master.mav.command_long_send(
                    master.target_system, master.target_component,
                    mavutil.mavlink.MAV_CMD_DO_VTOL_TRANSITION,
                    0, 0, 0, 0, 0, 0, 0, 0)
            else:
                mode = "VTOL Fixed Wing Mode"
                master.mav.command_long_send(
                    master.target_system, master.target_component,
                    mavutil.mavlink.MAV_CMD_DO_VTOL_TRANSITION,
                    0, 1, 0, 0, 0, 0, 0, 0)
            
            control_roll = visual_roll
            control_pitch = visual_pitch

    # 3 saniye aralığında print yapmak için zaman kontrolü
    current_time = time.time()
    if current_time - last_print_time >= 3:
        print(f"Control: Roll offset: {control_roll}, Pitch offset: {control_pitch}, Mode: {mode}")
        last_print_time = current_time

    # MAVLink üzerinden RC override komutlarının gönderilmesi
    master.mav.rc_channels_override_send(
        master.target_system, master.target_component,
        1500 + control_roll,   # Roll kanalı
        1500 + control_pitch,  # Pitch kanalı
        1500,                  # Throttle
        1500,                  # Yaw veya diğer kanal
        0, 0, 0, 0
    )

    # Görüntü ve maske pencerelerini göster
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
