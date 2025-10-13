import os
import logging
from collections import Counter

import cv2

from pipeline.process.ocr import perform_ocr

logger = logging.getLogger(__name__)

from collections import Counter

def analyze_detections(detection_results, capture_dir):
    images_dir = os.path.join(capture_dir, "images")

    logger.info("Starting analysis of detected objects")

    for obj in detection_results:
        track_id = obj["id"]
        
        # Initialize lists to store eartag_number and is_muzzle_clean for later analysis
        eartag_numbers = []
        muzzle_clean_status = []

        for image_entry in obj["images"]:
            image_name = image_entry["name"]
            bbox_track = image_entry.get("track_bbox", None)
            detections = image_entry.get("detections", [])

            image_path = os.path.join(images_dir, image_name)
            original_img = cv2.imread(image_path)
            if original_img is None:
                logger.warning(f"Could not load original image: {image_path}")
                continue

            h, w = original_img.shape[:2]

            if bbox_track:
                x1 = max(0, bbox_track["x1"])
                y1 = max(0, bbox_track["y1"])
                x2 = min(w, bbox_track["x2"])
                y2 = min(h, bbox_track["y2"])
                track_crop = original_img[y1:y2, x1:x2]
            else:
                track_crop = original_img

            for det_idx, detection in enumerate(detections):
                cls_name = detection["class"].lower()
                bbox = detection["bbox"]

                dx1 = max(0, bbox["x1"])
                dy1 = max(0, bbox["y1"])
                dx2 = max(dx1, bbox["x2"])
                dy2 = max(dy1, bbox["y2"])

                det_img = track_crop[dy1:dy2, dx1:dx2]

                if det_img.size == 0:
                    logger.warning(f"Empty detection crop for track {track_id}, image {image_name}, detection {det_idx}")
                    continue

                if cls_name == "muzzle":
                    # TODO: Add muzzle cleanliness classification logic
                    is_muzzle_clean = True
                    detection["is_muzzle_clean"] = is_muzzle_clean
                    logger.info(f"Track {track_id} muzzle classification: {'clean' if is_muzzle_clean else 'dirty'}")

                    # Track muzzle cleanliness for later analysis
                    muzzle_clean_status.append(detection["is_muzzle_clean"])

                elif cls_name == "eartag" or cls_name == "tag":
                    text = perform_ocr(det_img)
                    detection["eartag_number"] = text
                    logger.info(f"Track {track_id} eartag number OCR: {text}")

                    # Track eartag numbers for later analysis
                    if text != "":
                        eartag_numbers.append(text)

        # After processing all detections for the track, calculate the most common values
        if eartag_numbers:
            most_common_eartag = Counter(eartag_numbers).most_common(1)[0][0]
        else:
            most_common_eartag = None

        if muzzle_clean_status:
            most_common_muzzle_clean = Counter(muzzle_clean_status).most_common(1)[0][0]
        else:
            most_common_muzzle_clean = None

        obj["result"] = {
            "eartag_number": most_common_eartag,
            "is_muzzle_clean": most_common_muzzle_clean
        }

    logger.info("Completed analysis of detected objects")
    return detection_results