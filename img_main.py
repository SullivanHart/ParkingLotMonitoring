from ultralytics import YOLO
import cv2
import numpy as np
from img_visualize import visualize_results
from tqdm import tqdm

print("=== Starting Script ===")

# Local imports
try:
    import util
    from sort.sort import Sort
    from util import get_car, read_license_plate, write_csv
    print("[OK] Utility modules imported")
except Exception as e:
    print("[FAIL] Error importing util modules:", e)
    raise

results = {}

# Initialize tracker
class Tracker:
    def __init__(self):
        self.next_id = 0

    def update(self, detections):
        """
        Assign a unique ID to each detection.
        detections: numpy array of [x1, y1, x2, y2, score] or similar
        Returns: numpy array of [x1, y1, x2, y2, id] for each detection
        """
        tracked = []
        for det in detections:
            x1, y1, x2, y2 = det[:4]  # ignore score
            tracked.append([x1, y1, x2, y2, self.next_id])
            self.next_id += 1
        return np.array(tracked)


# Initialize dummy tracker
mot_tracker = Tracker()
print("[OK] Dummy tracker initialized")

# Load models
try:
    coco_model = YOLO("yolov8n.pt")
    license_plate_detector = YOLO("license_plate_detector.pt")
    print("[OK] Models loaded")
except Exception as e:
    print("[FAIL] Model loading error:", e)
    raise

# Load still image
frame_path = "./input/latest.jpg"
frame = cv2.imread(frame_path)
if frame is None:
    print(f"[FAIL] Could not load image at {frame_path}")
    raise FileNotFoundError(frame_path)
else:
    print(f"[OK] Image loaded: {frame.shape}")

frame_nmr = 0
results[frame_nmr] = {}

# Vehicle classes in COCO
vehicles = [2, 3, 5, 7]

# Detect vehicles
print("[INFO] Running vehicle detection...")
detections = coco_model(frame)[0]
print(f"[OK] Vehicle detection complete. Found {len(detections.boxes)} objects.")

detections_ = []
for detection in detections.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = detection
    if int(class_id) in vehicles:
        detections_.append([x1, y1, x2, y2, score])

print(f"[OK] Filtered {len(detections_)} vehicles out of {len(detections.boxes)} total detections")

# Track vehicles
track_ids = mot_tracker.update(np.asarray(detections_))
print(f"[OK] Tracker assigned {len(track_ids)} IDs")

# Detect license plates
print("[INFO] Running license plate detection...")
license_plates = license_plate_detector(frame)[0]
print(f"[OK] License plate detection complete. Found {len(license_plates.boxes)} candidates")

for license_plate in license_plates.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = license_plate

    # Assign license plate to car
    xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
    if car_id == -1:
        print("[WARN] No matching car found for license plate")
        continue

    # Crop license plate
    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
    if license_plate_crop.size == 0:
        print("[WARN] Empty license plate crop")
        continue

    # Process license plate (grayscale + threshold)
    # license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    # _, license_plate_crop_thresh = cv2.threshold(
    #     license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV
    # )

    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    license_plate_crop_thresh = cv2.adaptiveThreshold(
        license_plate_crop_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  # block size (odd number)
        2    # C value, subtracts from mean
    )


    # Save crops for inspection
    cv2.imwrite(f"./output/crop/gray_{frame_nmr}_{car_id}.jpg", license_plate_crop_gray)
    cv2.imwrite(f"./output/crop/thresh_{frame_nmr}_{car_id}.jpg", license_plate_crop_thresh)
    print(f"[DEBUG] Saved cropped plate images for car_id={car_id}")

    # Read license plate number
    # license_plate_text, license_plate_text_score = read_license_plate(
    #     license_plate_crop_thresh
    # )    
    
    license_plate_text, license_plate_text_score = read_license_plate(
        license_plate_crop_gray
    )

    print(f"[DEBUG] OCR result: {license_plate_text} (score={license_plate_text_score})")

    if license_plate_text is not None:
        results[frame_nmr][car_id] = {
            "car": {"bbox": [xcar1, ycar1, xcar2, ycar2]},
            "license_plate": {
                "bbox": [x1, y1, x2, y2],
                "text": license_plate_text,
                "bbox_score": score,
                "text_score": license_plate_text_score,
            },
        }

# Write results
try:
    write_csv(results, "./output/test.csv")
    print("[OK] Results written to ./output/test.csv")
except Exception as e:
    print("[FAIL] Error writing CSV:", e)

# Visualize results
try:
    visualize_results(frame_path, results, "./output/out.jpg")
    print("[OK] Visualization written to ./output/out.jpg")
except Exception as e:
    print("[FAIL] Visualization error:", e)

print("=== Script Finished ===")
