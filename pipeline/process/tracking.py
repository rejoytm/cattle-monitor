import os
import shutil
import glob
import time
import logging

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from pipeline.utils.config import load_yaml_config

logger = logging.getLogger(__name__)
config = load_yaml_config("pipeline/config.yaml")
models_dir = os.path.abspath(config.get("models_dir"))

def _save_tracking_results(object_map, capture_dir):
    image_dir = os.path.join(capture_dir, "images")
    tracks_dir = os.path.join(capture_dir, "tracks")

    if os.path.exists(tracks_dir):
        shutil.rmtree(tracks_dir)
    os.makedirs(tracks_dir, exist_ok=True)

    for obj in object_map.values():
        track_id = obj["id"]
        track_dir = os.path.join(tracks_dir, str(track_id))
        os.makedirs(track_dir, exist_ok=True)

        for image_entry in obj["images"]:
            image_name = image_entry["name"]
            bbox = image_entry["track_bbox"]

            image_path = os.path.join(image_dir, image_name)
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"Could not load image: {image_path}")
                continue

            h, w = img.shape[:2]
            x1, y1 = max(0, bbox["x1"]), max(0, bbox["y1"])
            x2, y2 = min(w, bbox["x2"]), min(h, bbox["y2"])

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                logger.warning(f"Empty crop for image {image_path}, track {track_id}")
                continue

            save_path = os.path.join(track_dir, image_name)
            cv2.imwrite(save_path, crop)

def deepsort(capture_dir, max_age=5, target_classes=[0], save_intermediate_results=False):
    model = YOLO(os.path.join(models_dir, 'cow_face_yolo11n_v1.pt'))

    images_dir = os.path.join(capture_dir, "images")
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))

    if not image_paths:
        logger.warning(f"No images found in directory: {images_dir}")
        return []

    logger.info("Starting tracking")
    tracker = DeepSort(max_age=max_age, n_init=2, half=True)
    object_map = {}

    inference_times = []
    tracking_times = []
    total_images = len(image_paths)

    try:
        for idx, image_path in enumerate(image_paths, 1):
            frame = cv2.imread(image_path)
            if frame is None:
                logger.warning(f"Failed to load image: {image_path}")
                continue

            # Inference
            inference_start_time = time.time()
            results = model(frame, verbose=False)[0]
            inference_times.append(time.time() - inference_start_time)

            # Prepare detections
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                if cls_id in target_classes:
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))

            # Tracking
            tracking_start_time = time.time()
            tracks = tracker.update_tracks(detections, frame=frame)
            tracking_times.append(time.time() - tracking_start_time)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                base_name = os.path.basename(image_path)

                if track_id not in object_map:
                    object_map[track_id] = {
                        "id": track_id,
                        "images": []
                    }

                object_map[track_id]["images"].append({
                    "name": base_name,
                    "track_bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    },
                })

            logger.info(f"Processed {idx}/{total_images} images")

    except Exception as e:
        logger.exception(f"An error occurred during tracking: {e}")
    finally:
        logger.info("Completed tracking")

        avg_inference = sum(inference_times) / len(inference_times) if inference_times else 0
        avg_tracking = sum(tracking_times) / len(tracking_times) if tracking_times else 0

        logger.info(f"Number of objects tracked: {len(object_map)}")
        logger.info(f"Average inference time per frame: {avg_inference:.4f} seconds")
        logger.info(f"Average tracking update time per frame: {avg_tracking:.4f} seconds")

        if save_intermediate_results:
            _save_tracking_results(object_map, capture_dir)

        return list(object_map.values())