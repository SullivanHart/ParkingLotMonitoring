import string
from paddleocr import PaddleOCR
import cv2
import numpy as np

# Initialize the PaddleOCR reader
ocr_reader = PaddleOCR(use_angle_cls=True, lang='en')  # lang='en' for English

# Map prohibitted characters in US
ocr_corrections = {
    'O': '0',  # OCR often confuses zero and letter O
    'S': '5',  # optional
}

def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format(
            'frame_nmr', 'car_id', 'car_bbox',
            'license_plate_bbox', 'license_plate_bbox_score',
            'license_number', 'license_number_score'
        ))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id] and \
                   'license_plate' in results[frame_nmr][car_id] and \
                   'text' in results[frame_nmr][car_id]['license_plate']:
                    f.write('{},{},{},{},{},{},{}\n'.format(
                        frame_nmr,
                        car_id,
                        '[{} {} {} {}]'.format(*results[frame_nmr][car_id]['car']['bbox']),
                        '[{} {} {} {}]'.format(*results[frame_nmr][car_id]['license_plate']['bbox']),
                        results[frame_nmr][car_id]['license_plate']['bbox_score'],
                        results[frame_nmr][car_id]['license_plate']['text'],
                        results[frame_nmr][car_id]['license_plate']['text_score']
                    ))

def license_complies_format(text):
    """
    Accept 2–8 characters, uppercase letters and digits.
    Allow almost any combination to handle all U.S. plates, including custom ones.
    """
    clean_text = text.replace(' ', '').replace('-', '').upper()
    if len(clean_text) < 2 or len(clean_text) > 8:
        return False

    allowed_chars = set(string.ascii_uppercase + '0123456789')
    for c in clean_text:
        if c not in allowed_chars:
            return False
    return True

def format_license(text):
    text = text.upper().replace(" ", "")
    formatted = ""
    for c in text:
        formatted += ocr_corrections.get(c, c)
    return formatted[:8]

def read_license_plate(license_plate_crop):
    if license_plate_crop is None or license_plate_crop.size == 0:
        print("[WARN] Empty license plate crop")
        return None, None

    if len(license_plate_crop.shape) == 2 or license_plate_crop.shape[2] == 1:
        license_plate_crop = cv2.cvtColor(license_plate_crop, cv2.COLOR_GRAY2BGR)

    ocr_result = ocr_reader.predict(license_plate_crop)
    if not ocr_result:
        print("[DEBUG] OCR found nothing")
        return None, None

    rec_texts = ocr_result[0].get('rec_texts', [])
    rec_scores = ocr_result[0].get('rec_scores', [])

    for text, score in zip(rec_texts, rec_scores):
        text_clean = text.upper().replace(' ', '')
        print(f"[DEBUG] OCR raw text: {text} -> cleaned: {text_clean}, score: {score}")

        if license_complies_format(text_clean):
            formatted = format_license(text_clean)
            print(f"[DEBUG] License plate formatted: {formatted}")
            return formatted, score

    return None, None

def get_car(license_plate, vehicle_track_ids):
    """Find the car bbox and ID that contains the license plate bbox."""
    x1, y1, x2, y2, score, class_id = license_plate
    for j, (xcar1, ycar1, xcar2, ycar2, car_id) in enumerate(vehicle_track_ids):
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id
    return -1, -1, -1, -1, -1
