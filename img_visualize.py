import cv2
import numpy as np

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=50, line_length_y=50):
    """Draws corner-style borders instead of a full rectangle."""
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


def visualize_results(image_path, results, output_path="./output/out.jpg"):
    """
    Visualize detection + OCR results on a single still image.

    Args:
        image_path (str): Path to input image.
        results (dict): Results dictionary from detection script.
        output_path (str): Path to save annotated image.
    """
    frame = cv2.imread(image_path)
    frame_nmr = list(results.keys())[0]  # only one frame
    frame_results = results[frame_nmr]

    for car_id, data in frame_results.items():
        # Draw car bounding box
        x1, y1, x2, y2 = map(int, data['car']['bbox'])
        draw_border(frame, (x1, y1), (x2, y2), (0, 255, 0), 15, 60, 60)

        # Draw license plate bbox
        lp_data = data['license_plate']
        lx1, ly1, lx2, ly2 = map(int, lp_data['bbox'])
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 0, 255), 5)

        # Crop license plate for display
        license_crop = frame[ly1:ly2, lx1:lx2]
        if license_crop.size > 0:
            H, W = license_crop.shape[:2]
            try:
                # place crop above car
                y_offset = max(0, y1 - H - 50)
                x_offset = int((x1 + x2 - W) / 2)
                frame[y_offset:y_offset+H, x_offset:x_offset+W] = license_crop
            except:
                pass

        # Write license plate text above car
        lp_text = lp_data.get('text', '')
        if lp_text:
            (text_w, text_h), _ = cv2.getTextSize(lp_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 5)
            cv2.putText(frame,
                        lp_text,
                        (int((x1 + x2 - text_w) / 2), max(40, y1 - 60)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.0,
                        (0, 0, 0),
                        5)

    cv2.imwrite(output_path, frame)
    print(f"Visualization saved to {output_path}")
