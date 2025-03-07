import cv2
import numpy as np
from pymavlink import mavutil

# MAVLink bağlantısını başlat (SITL simülasyonu için Mission Planner UDP portunu kullanın)
master = mavutil.mavlink_connection('udp:127.0.0.1:14552')

# Video yakalama (webcam)
cap = cv2.VideoCapture(0)

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
            error_x = cx - frame_center_x
            error_y = cy - frame_center_y

            # Hata değerlerini kullanarak roll ve pitch hesaplama (ölçeklendirme ayarlanabilir)
            roll = int(np.clip(error_x, -500, 500))
            pitch = int(np.clip(error_y, -500, 500))

            # Görüntü üzerinde hata bilgilerini çizdirme
            cv2.line(frame, (frame_center_x, frame_center_y), (cx, cy), (255, 255, 0), 2)
            cv2.putText(frame, f"Error: ({error_x}, {error_y})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            print(f"Error: ({error_x}, {error_y}), Roll: {roll}, Pitch: {pitch}")

            # Yangın alanının genişliğine göre uçuş modunu belirleme
            # Örneğin, radius < 50 ise dar alan yangını (dikey mod), >= 50 ise geniş alan yangını (sabit kanat modu)
            if radius < 50:
                cv2.putText(frame, "VTOL: Vertical Mode", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print("Dar alan yangını tespit edildi: Dikey moda geçiliyor.")
                master.mav.command_long_send(
                    master.target_system, master.target_component,
                    mavutil.mavlink.MAV_CMD_DO_VTOL_TRANSITION,
                    0,    # Confirmation
                    0,    # 0: Dikey (hover) mod
                    0, 0, 0, 0, 0, 0, 0)
            else:
                cv2.putText(frame, "VTOL: Fixed Wing Mode", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print("Geniş alan yangını tespit edildi: Sabit kanat moduna geçiliyor.")
                master.mav.command_long_send(
                    master.target_system, master.target_component,
                    mavutil.mavlink.MAV_CMD_DO_VTOL_TRANSITION,
                    0,    # Confirmation
                    1,    # 1: Sabit kanat modu
                    0, 0, 0, 0, 0, 0, 0)

            # RC override komutları gönderme (roll ve pitch hatalarına göre)
            master.mav.rc_channels_override_send(
                master.target_system, master.target_component,
                1500 + roll,   # Roll kanalı
                1500 + pitch,  # Pitch kanalı
                1500,          # Throttle
                1500,          # Yaw veya diğer kanal
                0, 0, 0, 0)

    # Görüntü ve maske pencerelerini göster
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    # ESC tuşuna basılırsa çıkış yap
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
