from ultralytics import YOLO
import cv2
import numpy as np
from paddleocr import PaddleOCR
from take_pic import take_pic
from img_visualize import visualize_results
from util import get_car, read_license_plate, write_csv
from sklearn.cluster import KMeans
import webcolors

from server_req import post_block
from car import Car
from plate import Plate
from tracker import Tracker

# Helper: get dominant CSS3 color of a vehicle crop
def get_vehicle_color(vehicle_crop):
    crop_rgb = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB)
    pixels = crop_rgb.reshape(-1, 3)

    # KMeans to find dominant color
    kmeans = KMeans(n_clusters=1, random_state=0).fit(pixels)
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    rgb_triplet = webcolors.IntegerRGB(*dominant_color)

    # Try exact match first
    try:
        color_name = webcolors.rgb_to_name(rgb_triplet, spec="css3")
    except ValueError:
        # Find closest CSS3 color
        min_dist = float('inf')
        color_name = "unknown"
        for name in webcolors.names(spec="css3"):
            css_rgb = webcolors.name_to_rgb(name)
            dist = sum((a - b) ** 2 for a, b in zip(dominant_color, (css_rgb.red, css_rgb.green, css_rgb.blue)))
            if dist < min_dist:
                min_dist = dist
                color_name = name
    return color_name

print("=== Starting Script ===")

# Initialize models
coco_model = YOLO("yolov8n.pt")
license_plate_detector = YOLO("license_plate_detector.pt")
ocr_reader = PaddleOCR(use_angle_cls=True, lang='en')

results = {}
frame_nmr = 0
mot_tracker = Tracker()  # dummy tracker
vehicles = [2, 3, 5, 7]  # COCO vehicle classes
vehicle_class_map = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

while True:
    frame_path = "./input/latest.jpg"
    take_pic(frame_path) # capture live
    frame = cv2.imread(frame_path)
    if frame is None:
        raise FileNotFoundError(frame_path)
    print(f"[OK] Image loaded: {frame.shape}")

    results[frame_nmr] = {}

    # --- Vehicle Detection ---
    detections = coco_model(frame)[0]
    filtered_dets = [
        det for det in detections.boxes.data.tolist() if int(det[5]) in vehicles and det[4] >= 0.8
    ]
    print("Filtered vehicle detections:", filtered_dets)

    # --- Update tracker ---
    track_ids = mot_tracker.update(np.asarray(filtered_dets))
    print(f"[INFO] Found {len(track_ids)} vehicles after tracking")

    # --- Initialize Car objects immediately ---
    tracker_id_to_det = {}
    for i, det in enumerate(filtered_dets):
        x1, y1, x2, y2, score, cls = det
        car_id = int(track_ids[i][-1]) if track_ids.ndim > 1 else int(track_ids[i])

        car_crop = frame[int(y1):int(y2), int(x1):int(x2)]
        color = get_vehicle_color(car_crop) if car_crop.size > 0 else "unknown"
        style = vehicle_class_map.get(int(cls), "unknown")

        results[frame_nmr][car_id] = {
            "car_obj": Car( prob=det[4], color=color, style=style, plates=[]),
            "bbox": [x1, y1, x2, y2]
        }

        tracker_id_to_det[car_id] = det

    # --- Detect license plates ---
    plates_detected = license_plate_detector(frame)[0]

    for plate in plates_detected.boxes.data.tolist():
        x1, y1, x2, y2, score, _ = plate

        # Skip very low-confidence plates
        if score < 0.5:
            print( f"[DEBUG] Skipping low-confidence plate: { score }")
            continue

        xcar1, ycar1, xcar2, ycar2, car_id = get_car(plate, track_ids)
        if car_id == -1:
            print("[WARN] No matching car for license plate")
            continue

        # Crop license plate
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            print("[DEBUG] Skipping due 0 crop size")
            continue

        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # OCR for license plate
        text, conf = read_license_plate(ocr_reader, crop_gray)
        if text is None or conf < 0.8:
            print(f"[DEBUG] Skipped plate '{text}' (score={conf})")
            continue
        print(f"[INFO] OCR plate: '{text}' (score={conf}) for car_id={car_id}")

        # Append plate info
        if car_id in results[frame_nmr]:
            results[frame_nmr][car_id]["car_obj"].plates.append(Plate(text=text, prob=conf))

    # --- Send updates to server ---
    cars_to_send = [data["car_obj"] for data in results[frame_nmr].values() if "car_obj" in data]

    if not cars_to_send:
        # Send empty request
        cars_to_send = []

    try:
        response = post_block(block=2, cars=cars_to_send)
        print(f"[OK] Sent {len(cars_to_send)} car(s) to server, status: {response.status_code}")
    except Exception as e:
        print("[FAIL] Error sending data to server:", e)

    frame_nmr += 1