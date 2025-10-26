import string
import cv2
import numpy as np

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
                # print(results[frame_nmr][car_id])
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

def read_license_plate(ocr_reader, license_plate_crop):
    try:
        ocr_result = ocr_reader.ocr(license_plate_crop, cls=True)

        if not ocr_result or len(ocr_result[0]) == 0:
            return None, 0.0

        # Each element in ocr_result[0] is like: [box, (text, confidence)]
        text_conf_pairs = [item[1] for item in ocr_result[0] if len(item) > 1]
        texts = [t[0] for t in text_conf_pairs]
        scores = [t[1] for t in text_conf_pairs]

        # Combine recognized parts
        license_plate_text = "".join(texts)
        license_plate_text_score = sum(scores) / len(scores) if scores else 0.0

        return license_plate_text, license_plate_text_score

    except Exception as e:
        print(f"[ERROR] OCR failed: {e}")
        return None, 0.0

def get_car(license_plate, vehicle_track_ids):
    #Find the car bbox and ID that contains the license plate bbox
    x1, y1, x2, y2, score, class_id = license_plate
    for j, (xcar1, ycar1, xcar2, ycar2, car_id) in enumerate(vehicle_track_ids):
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id
    return -1, -1, -1, -1, -1
