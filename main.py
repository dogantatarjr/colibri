import collections
import collections.abc
# Python ≥3.10 compatibility patch
collections.MutableMapping = collections.abc.MutableMapping

from dronekit import connect, VehicleMode, LocationGlobalRelative
import cv2
import numpy as np
import time
import math
import sys

# --- Fonksiyon: Küçük alandaki yangın konumuna yönlenme algoritması ---
def handle_small_area(vehicle, fire_lat, fire_lon, target_alt):
    """
    vehicle: DroneKit arayüzü
    fire_lat, fire_lon: yangın konum koordinatları
    target_alt: yükseklik (m)
    """
    # 1) 5 saniye mevcut rotada devam et
    print("Continuing current heading for 5 seconds before diverting to fire center...")
    time.sleep(5.0)

    # 2) GUIDED moda geç ve goto komutunu gönder
    print("Switching to GUIDED mode...")
    vehicle.mode = VehicleMode("GUIDED")
    while vehicle.mode.name != "GUIDED":
        time.sleep(0.1)
    print(f"Sending simple_goto to fire center at ({fire_lat:.6f}, {fire_lon:.6f}), alt {target_alt}m")
    vehicle.simple_goto(LocationGlobalRelative(fire_lat, fire_lon, target_alt))

    # 3) Varışı bekle (2m içi veya 30s max), sonra QLOITER
    arrival_start = time.time()
    while True:
        curr = vehicle.location.global_relative_frame
        dNorth = (fire_lat - curr.lat) * 111111
        dEast  = (fire_lon - curr.lon) * 111111 * math.cos(math.radians(curr.lat))
        dist   = math.hypot(dNorth, dEast)
        if dist < 2.0 or time.time() - arrival_start > 30.0:
            print(f"Arrived within {dist:.1f}m of fire center → switching to QLOITER")
            vehicle.mode = VehicleMode("QLOITER")
            break
        time.sleep(0.5)


# --- Fonksiyon: Büyük alan için (henüz çağrılmıyor) ---
def handle_large_area(vehicle, north, east, target_alt, far_scale=5.0):
    curr = vehicle.location.global_relative_frame
    # Yakın hedef
    target = LocationGlobalRelative(
        curr.lat + north/111111,
        curr.lon + east/(111111 * math.cos(math.radians(curr.lat))),
        target_alt
    )
    print("GUIDED → moving to close target (large-area)")
    vehicle.mode = VehicleMode("GUIDED"); time.sleep(0.5)
    vehicle.simple_goto(target)
    time.sleep(1.0)

    # Uzak hedef
    far_target = LocationGlobalRelative(
        curr.lat + (north * far_scale)/111111,
        curr.lon + (east  * far_scale)/(111111 * math.cos(math.radians(curr.lat))),
        target_alt
    )
    print("GUIDED → moving to far target (large-area)")
    vehicle.simple_goto(far_target)


# --- Bağlantı & Ayarlar ---
connection_string = 'tcp:127.0.0.1:5762'
print(f"Connecting to vehicle on: {connection_string}")
vehicle = connect(connection_string, wait_ready=True)

target_alt      = 70.0      # m
area_thresh     = 500       # px eşiği
scale_m_per_px  = 0.5       # m/piksel
pixel_thresh    = 20        # jitter dead‐zone

# Kamera ayarları
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Camera open failed; exiting.")
    vehicle.close()
    sys.exit(1)
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.namedWindow('Mask',  cv2.WINDOW_NORMAL)

lower_hsv = np.array([35,100,100])
upper_hsv = np.array([85,255,255])

# Veri tutucular
area_list         = []
pos_list          = []
detection_started = False
detection_start   = 0.0
last_record_time  = 0.0

RECORD_INTERVAL   = 2.0   # saniye
COLLECT_DURATION  = 10.0  # saniye

print("Entering in-flight detection loop. ESC to abort.")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Maskeleme
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5,5),np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5,5),np.uint8))

        cv2.imshow('Frame', frame)
        cv2.imshow('Mask', mask)
        if cv2.waitKey(1) & 0xFF == 27:
            print("User aborted; exiting loop.")
            break

        # Kontur tespiti
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c    = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        now = time.time()
        # İlk tespit
        if not detection_started and area >= area_thresh:
            detection_started = True
            detection_start   = now
            last_record_time  = now - RECORD_INTERVAL
            print(f"First detection at t={detection_start:.1f}s, area={area:.1f}px")

        # 10s boyunca her 2s’de bir kaydet
        if detection_started and now - detection_start <= COLLECT_DURATION:
            if now - last_record_time >= RECORD_INTERVAL:
                (x, y), _ = cv2.minEnclosingCircle(c)
                h, w      = frame.shape[:2]
                err_x = x - w/2
                err_y = y - h/2
                north = -err_y * scale_m_per_px
                east  =  err_x * scale_m_per_px
                curr   = vehicle.location.global_relative_frame
                lat    = curr.lat + north/111111
                lon    = curr.lon + east/(111111 * math.cos(math.radians(curr.lat)))
                area_list.append(area)
                pos_list.append((lat, lon))
                last_record_time = now
                print(f"Recorded #{len(area_list)}: area={area:.1f}px at ({lat:.6f}, {lon:.6f})")

        # Süre dolduysa yangın konumunu belirle ve ilgili fonksiyonu çağır
        if detection_started and now - detection_start > COLLECT_DURATION:
            max_idx    = int(np.argmax(area_list))
            fire_area  = area_list[max_idx]
            fire_lat, fire_lon = pos_list[max_idx]
            print("\n-- Fire estimation complete --")
            print(f"Max area: {fire_area:.1f}px at idx {max_idx}")
            print(f"Estimated fire location: ({fire_lat:.6f}, {fire_lon:.6f})")

            if fire_area > 20000:
                # Büyük alan: offset hesapla ve handle_large_area çağır
                curr = vehicle.location.global_relative_frame
                north = (fire_lat - curr.lat) * 111111
                east  = (fire_lon - curr.lon) * 111111 * math.cos(math.radians(curr.lat))
                handle_large_area(vehicle, north, east, target_alt)
            else:
                # Küçük alan
                handle_small_area(vehicle, fire_lat, fire_lon, target_alt)

            break

        time.sleep(0.1)

finally:
    cap.release()
    cv2.destroyAllWindows()
    vehicle.close()
    print("Detection loop ended. Appropriate handler executed.")
