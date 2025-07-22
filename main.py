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

# --- Fonksiyon: Küçük alan için VTOL→GUIDED→QLOITER ---
def handle_small_area(vehicle, fire_lat, fire_lon, target_alt):
    print("Continuing current heading for 5 seconds before diverting to fire center...")
    time.sleep(5.0)

    print("Switching to GUIDED mode...")
    vehicle.mode = VehicleMode("GUIDED")
    while vehicle.mode.name != "GUIDED":
        time.sleep(0.1)

    print(f"Sending simple_goto to fire center at ({fire_lat:.6f}, {fire_lon:.6f}), alt {target_alt}m")
    vehicle.simple_goto(LocationGlobalRelative(fire_lat, fire_lon, target_alt))

    # Varışı bekle
    arrival_start = time.time()
    while True:
        curr = vehicle.location.global_relative_frame
        dNorth = (fire_lat - curr.lat) * 111111
        dEast  = (fire_lon - curr.lon) * 111111 * math.cos(math.radians(curr.lat))
        dist   = math.hypot(dNorth, dEast)
        if dist < 2.0 or time.time() - arrival_start > 30.0:
            print(f"Arrived within {dist:.1f}m → switching to QLOITER")
            vehicle.mode = VehicleMode("QLOITER")
            break
        time.sleep(0.5)


# --- Fonksiyon: Büyük alan için direkt merkeze→RTL ---
def handle_large_area(vehicle, north, east, target_alt):
    curr = vehicle.location.global_relative_frame
    target = LocationGlobalRelative(
        curr.lat + north/111111,
        curr.lon + east/(111111 * math.cos(math.radians(curr.lat))),
        target_alt
    )
    print("GUIDED → moving to fire center (large-area)")
    vehicle.mode = VehicleMode("GUIDED")
    while vehicle.mode.name != "GUIDED":
        time.sleep(0.1)
    vehicle.simple_goto(target)

    # Varışı bekle → RTL
    arrival_start = time.time()
    while True:
        curr = vehicle.location.global_relative_frame
        dNorth = (target.lat - curr.lat) * 111111
        dEast  = (target.lon - curr.lon) * 111111 * math.cos(math.radians(curr.lat))
        dist   = math.hypot(dNorth, dEast)
        if dist < 2.0 or time.time() - arrival_start > 30.0:
            print(f"Arrived within {dist:.1f}m → switching to RTL")
            vehicle.mode = VehicleMode("RTL")
            break
        time.sleep(0.5)


def main():
    # --- Bağlantı & Ayarlar ---
    connection_string = 'tcp:127.0.0.1:5762'
    print(f"Connecting to vehicle on: {connection_string}")
    vehicle = connect(connection_string, wait_ready=True)

    target_alt     = 70.0      # m
    area_thresh    = 500       # px
    scale_m_per_px = 0.5
    pixel_thresh   = 20

    # HSV aralıkları (kırmızı)
    lower1, upper1 = np.array([0, 100, 100]), np.array([10, 255, 255])
    lower2, upper2 = np.array([160, 100, 100]), np.array([180, 255, 255])

    # Ölçüm ayarları
    RECORD_INTERVAL  = 2.0   # saniye
    COLLECT_DURATION = 10.0  # saniye

    # Veri tutucular
    area_list         = []
    pos_list          = []
    detection_started = False
    detection_start   = 0.0
    last_record_time  = 0.0

    # Kamera aç
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Camera open failed; exiting.")
        vehicle.close()
        sys.exit(1)

    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Mask',  cv2.WINDOW_NORMAL)

    try:
        print("Entering in-flight detection loop. ESC to abort.")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask  = cv2.bitwise_or(mask1, mask2)
            mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5,5),np.uint8))
            mask  = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5,5),np.uint8))

            cv2.imshow('Frame', frame)
            cv2.imshow('Mask', mask)
            if cv2.waitKey(1) & 0xFF == 27:
                print("User aborted; exiting loop.")
                break

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            c    = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)

            now = time.time()
            # İlk tespit → ölçüme başla
            if not detection_started and area >= area_thresh:
                detection_started  = True
                detection_start    = now
                last_record_time   = now - RECORD_INTERVAL
                print(f"First detection at t={detection_start:.1f}s, area={area:.1f}px")

            # Ölçüm süresi içinde her RECORD_INTERVAL’da kaydet
            if detection_started and now - detection_start <= COLLECT_DURATION:
                if now - last_record_time >= RECORD_INTERVAL:
                    (x, y), _ = cv2.minEnclosingCircle(c)
                    h, w      = frame.shape[:2]
                    err_x = x - w/2
                    err_y = y - h/2
                    north = -err_y * scale_m_per_px
                    east  =  err_x * scale_m_per_px
                    curr  = vehicle.location.global_relative_frame
                    lat   = curr.lat + north/111111
                    lon   = curr.lon + east/(111111 * math.cos(math.radians(curr.lat)))
                    area_list.append(area)
                    pos_list.append((lat, lon))
                    last_record_time = now
                    print(f"Recorded #{len(area_list)}: area={area:.1f}px at ({lat:.6f}, {lon:.6f})")

            # Süre bitince döngüyü kır
            if detection_started and now - detection_start > COLLECT_DURATION:
                break

            time.sleep(0.1)

        # Ölçüm tamamlandı: en büyük alan ve konumunu seç
        max_idx    = int(np.argmax(area_list))
        fire_area  = area_list[max_idx]
        fire_lat, fire_lon = pos_list[max_idx]
        print(f"\n-- Fire estimation --")
        print(f"Max area: {fire_area:.1f}px at idx {max_idx}")
        print(f"Estimated fire location: ({fire_lat:.6f}, {fire_lon:.6f})")

        # Alan büyüklüğüne göre handler çağır
        if fire_area > 20000:
            curr = vehicle.location.global_relative_frame
            north = (fire_lat - curr.lat) * 111111
            east  = (fire_lon - curr.lon) * 111111 * math.cos(math.radians(curr.lat))
            handle_large_area(vehicle, north, east, target_alt)
        else:
            handle_small_area(vehicle, fire_lat, fire_lon, target_alt)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        vehicle.close()
        print("Cleanup done, exiting.")

if __name__ == "__main__":
    main()
