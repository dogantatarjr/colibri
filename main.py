import cv2
import numpy as np
from pymavlink import mavutil
import math

# MAVLink bağlantısını başlat (SITL simülasyonu için Mission Planner UDP portunu kullanın)
master = mavutil.mavlink_connection('udp:127.0.0.1:14552')

# Video yakalama (webcam)
cap = cv2.VideoCapture(0)

# Stabilite kontrolü için hata büyüklüğü eşik değeri
mode_switch_stability_threshold = 100  # Bu değeri ihtiyacınıza göre ayarlayın

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü HSV renk uzayına çevir
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Kırmızı renk için iki farklı aralık (HSV) tanımlama
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # İki maskeyi birleştir
    mask = cv2.bitwise_or(mask1, mask2)

    # Gürültüyü azaltmak için morfolojik işlemler
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # Kontur tespiti
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fire_detected = False
    visual_roll = 0
    visual_pitch = 0

    if contours:
        # En büyük konturu seç
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius > 10:  # Gürültü engelleme için minimum radius
            # Konturun merkezini hesaplama
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = int(x), int(y)

            # Görüntü üzerinde tespit edilen alanı çizdirme
            cv2.circle(frame, (cx, cy), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            # Görüntü merkezini hesaplama
            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2

            # Hata (error) değerini hesaplama: kırmızı alan ile görüntü merkezi arasındaki fark
            error_x_visual = cx - frame_center_x
            error_y_visual = cy - frame_center_y

            # Görsel hata değerlerini RC komutlarına dönüştür (ölçeklendirme doğrudan aktarılıyor; ayarlanabilir)
            visual_roll = int(np.clip(error_x_visual, -500, 500))
            visual_pitch = int(np.clip(error_y_visual, -500, 500))

            cv2.putText(frame, f"Fire Detected! Visual Err: ({error_x_visual}, {error_y_visual})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            fire_detected = True

            # Ek olarak, görsel hata büyüklüğünü hesaplayıp stabilite kontrolü yapalım
            error_magnitude = math.sqrt(error_x_visual**2 + error_y_visual**2)
            if error_magnitude > mode_switch_stability_threshold:
                cv2.putText(frame, "Stability Low - No Mode Change", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                print("Düşük stabilite: Mod değişikliği uygulanmadı.")
                fire_detected = False  # Stabil değilse görsel mod değişikliği yapılmaz

    # Eğer görsel mod değişikliği uygulanamazsa, o zaman global (GPS) veriler veya nötr komutlar kullanılır.
    # (Burada GPS verileri kullanılmıyor; yalnızca görüntü işleme bazlı kontrol mevcut)

    # --- MAVLink üzerinden RC override komutlarının gönderilmesi ---
    # Her kanal için nötr referans değeri 1500; üzerine eklenen ofsetler kontrol komutlarını belirler.
    if fire_detected:
        mode = "Visual Override"
        control_roll = visual_roll
        control_pitch = visual_pitch
    else:
        mode = "No Mode Change"
        control_roll = 0
        control_pitch = 0

    cv2.putText(frame, f"Mode: {mode}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    print(f"Control: Roll: {control_roll}, Pitch: {control_pitch}, Mode: {mode}")

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
