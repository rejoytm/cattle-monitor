import os
import logging

import cv2
from ultralytics import YOLO

from pipeline.utils.config import load_yaml_config

logger = logging.getLogger(__name__)
config = load_yaml_config("pipeline/config.yaml")
models_dir = os.path.abspath(config.get("models_dir"))

def _save_detection_result(track_image, bbox, save_dir, file_name):
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    crop = track_image[y1:y2, x1:x2]
    if crop.size == 0:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, file_name), crop)

def detect_objects(tracking_results, capture_dir, save_intermediate_results=False):
    model = YOLO(os.path.join(models_dir, 'cow_muzzle_eartag_yolo11n_v1.pt'))
    images_dir = os.path.join(capture_dir, "images")
    tracks_dir = os.path.join(capture_dir, "tracks")

    logger.info("Starting object detection")

    for obj in tracking_results:
        track_id = obj["id"]
        logger.info(f"Running object detection on Track ID {track_id}")

        for image_entry in obj["images"]:
            image_name = image_entry["name"]
            bbox = image_entry["track_bbox"]

            image_path = os.path.join(images_dir, image_name)
            original_img = cv2.imread(image_path)
            if original_img is None:
                logger.warning(f"Could not load original image: {image_path}")
                continue

            h, w = original_img.shape[:2]
            x1, y1 = max(0, bbox["x1"]), max(0, bbox["y1"])
            x2, y2 = min(w, bbox["x2"]), min(h, bbox["y2"])
            track_img = original_img[y1:y2, x1:x2]

            image_entry["detections"] = []

            results_yolo = model(track_img, verbose=False)[0]

            for det_idx, box in enumerate(results_yolo.boxes):
                x1_det, y1_det, x2_det, y2_det = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, str(cls_id))

                detection = {
                    "class": cls_name,
                    "bbox": {
                        "x1": x1_det,
                        "y1": y1_det,
                        "x2": x2_det,
                        "y2": y2_det
                    }
                }

                image_entry["detections"].append(detection)

                if save_intermediate_results:
                    save_dir = os.path.join(tracks_dir, str(track_id), cls_name)
                    file_name = f"{os.path.splitext(image_name)[0]}_{det_idx}.png"
                    _save_detection_result(track_img, detection["bbox"], save_dir, file_name)

    logger.info("Completed object detection")
    return tracking_results