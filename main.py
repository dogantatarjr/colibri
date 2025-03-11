import cv2
import numpy as np
from pymavlink import mavutil
import math
import time

# MAVLink bağlantısını başlat (SITL veya gerçek uçuş kartı için uygun port)
master = mavutil.mavlink_connection('udp:127.0.0.1:14552')
print("Heartbeat bekleniyor...")
master.wait_heartbeat()
print("Heartbeat alındı, MAVLink bağlantısı kuruldu.")

# Video yakalama (webcam)
cap = cv2.VideoCapture(0)

# Düz hizalama için hata eşik değeri (pixel cinsinden)
correction_threshold = 30

# Su bırakma onayının sadece bir kez sorulması için bayrak
water_prompt_done = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Kırmızı renk tespiti için iki farklı HSV aralığı kullan
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Varsayılan kontrol değerleri
    control_roll = 0
    control_pitch = 0
    mode = "Mission Planner Route"  # Varsayılan: mevcut rota korunuyor

    if contours:
        # En büyük konturu seç
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        # Hedef alan çok küçükse override yapılmaz
        if radius < 10:
            mode = "No Override"
            print("Hedef alan çok küçük (radius < 10): Override yapılmıyor.")
            water_prompt_done = False
        else:
            # İlk adım: VTOL modunu belirleme (harekete geçmeden önce)
            if radius < 50:
                mode = "VTOL Vertical Mode"
                print("Hedef tespit edildi: VTOL dikey moda geçiliyor (10 <= radius < 50).")
                master.mav.command_long_send(
                    master.target_system, master.target_component,
                    mavutil.mavlink.MAV_CMD_DO_VTOL_TRANSITION,
                    0, 0, 0, 0, 0, 0, 0, 0)
            else:
                mode = "VTOL Fixed Wing Mode"
                print("Hedef tespit edildi: VTOL sabit kanat moduna geçiliyor (radius >= 50).")
                master.mav.command_long_send(
                    master.target_system, master.target_component,
                    mavutil.mavlink.MAV_CMD_DO_VTOL_TRANSITION,
                    0, 1, 0, 0, 0, 0, 0, 0)

            # Hedefin merkezi hesaplanıyor
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = int(x), int(y)
            
            # Tespit edilen alan ekrana çizdiriliyor
            cv2.circle(frame, (cx, cy), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            
            # Görüntü merkezinin hesaplanması
            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2

            # Görsel hata hesaplanıyor: Tespit edilen alan ile görüntü merkezi arasındaki fark
            error_x = cx - frame_center_x
            error_y = cy - frame_center_y
            visual_roll = int(np.clip(error_x, -500, 500))
            visual_pitch = int(np.clip(error_y, -500, 500))
            
            cv2.line(frame, (frame_center_x, frame_center_y), (cx, cy), (255, 255, 0), 2)
            cv2.putText(frame, "Visual Err: ({}, {})".format(error_x, error_y), (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            error_magnitude = math.sqrt(error_x**2 + error_y**2)

            # İkinci adım: Uçağın yönünü düzeltme
            if error_magnitude > correction_threshold:
                mode += " | Correcting Orientation"
                control_roll = visual_roll
                control_pitch = visual_pitch
                print("Orientasyon düzeltiliyor, hata büyüklüğü: {:.1f} > {}".format(error_magnitude, correction_threshold))
                water_prompt_done = False
            else:
                # Uçak hedefe hizalıysa, yani görsel hata düşükse
                # Üçüncü adım: Hedef alanda (yangın bölgesinde) su bırakma onayını sor
                if not water_prompt_done:
                    response = input("Uçak hedefe ulaştı. Su bırakılacak mı? (y/n): ")
                    print("Kullanıcı cevabı:", response)
                    water_prompt_done = True
                control_roll = 0
                control_pitch = 0

    cv2.putText(frame, "Mode: {}".format(mode), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    master.mav.rc_channels_override_send(
        master.target_system, master.target_component,
        1500 + control_roll,   # Roll kanalı
        1500 + control_pitch,  # Pitch kanalı
        1500,                  # Throttle
        1500,                  # Yaw veya diğer kanal
        0, 0, 0, 0
    )

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
