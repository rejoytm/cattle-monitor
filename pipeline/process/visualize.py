import os
import logging

import numpy as np
import cv2

logger = logging.getLogger(__name__)

def visualize_analysis_results(analysis_results, capture_dir):
    images_dir = os.path.join(capture_dir, "images")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    font_color = (0, 0, 0)
    font_thickness = 2
    figure_size = (150, 150)
    figure_spacing = 10
    label_margin_top = 10
    max_row_images = 10

    for obj in analysis_results:
        track_id = obj.get("id", "N/A")
        result = obj.get("result", {})
        eartag_number = result.get("eartag_number", "N/A")
        is_muzzle_clean = result.get("is_muzzle_clean", "N/A")

        total_images = len(obj.get("images", []))
        total_eartags = 0
        total_muzzles = 0        

        track_crops = []
        eartag_crops = []
        muzzle_crops = []

        track_names = []
        eartag_texts = []
        muzzle_flags = []

        for image_entry in obj.get("images", []):
            image_name = image_entry.get("name", "unknown")
            image_path = os.path.join(images_dir, image_name)
            original_img = cv2.imread(image_path)
            if original_img is None:
                continue

            bbox_track = image_entry.get("track_bbox", None)
            if bbox_track:
                x1 = max(0, bbox_track["x1"])
                y1 = max(0, bbox_track["y1"])
                x2 = bbox_track["x2"]
                y2 = bbox_track["y2"]
                track_crop = original_img[y1:y2, x1:x2]
            else:
                track_crop = original_img

            track_crops.append(track_crop)
            track_names.append(image_name)

            for det in image_entry.get("detections", []):
                cls_name = det.get("class", "").lower()
                bbox = det.get("bbox", {})
                dx1 = max(0, bbox.get("x1", 0))
                dy1 = max(0, bbox.get("y1", 0))
                dx2 = bbox.get("x2", dx1)
                dy2 = bbox.get("y2", dy1)
                det_crop = track_crop[dy1:dy2, dx1:dx2]

                if det_crop.size == 0:
                    continue

                if cls_name == "tag":   
                        total_eartags += 1
                        eartag_crops.append(det_crop)
                        label = det.get("eartag_number", "").strip()
                        eartag_texts.append(label if label else "N/A")

                elif cls_name == "muzzle":
                        total_muzzles += 1
                        muzzle_crops.append(det_crop)
                        muzzle_flags.append("Clean" if det.get("is_muzzle_clean", False) else "Dirty")

        def resize_images(images):
            return [cv2.resize(img, figure_size) for img in images]

        def label_images(images, labels):
            labeled = []
            for img, label in zip(images, labels):
                img_copy = img.copy()
                label_height = 30 + label_margin_top
                margin_img = cv2.copyMakeBorder(
                    img_copy, 0, label_height, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
                )
                text_position = (5, img_copy.shape[0] + label_margin_top + 20)
                cv2.putText(margin_img, str(label), text_position, font, font_scale, font_color, font_thickness)
                labeled.append(margin_img)
            return labeled
        
        def stack_horizontally(images):
            if not images:
                # Create blank images
                blank_img = 0 * np.ones((figure_size[1], figure_size[0], 3), dtype=np.uint8)
                images = [blank_img.copy() for _ in range(max_row_images)]

            spaced_images = []
            for img in images:
                img_with_margin = cv2.copyMakeBorder(
                    img,
                    0, 0, 0, figure_spacing,
                    cv2.BORDER_CONSTANT,
                    value=(255, 255, 255)
                )
                spaced_images.append(img_with_margin)

            # Remove extra spacing from the last image
            spaced_images[-1] = spaced_images[-1][:, :-figure_spacing]

            return cv2.hconcat(spaced_images)
        
        def pad_to_width(image, target_width):
            _, w = image.shape[:2]
            if w == target_width:
                return image
            padding = target_width - w
            return cv2.copyMakeBorder(image, 0, 0, 0, padding, cv2.BORDER_CONSTANT, value=(255, 255, 255))       

        track_labeled = label_images(resize_images(track_crops[:max_row_images]), track_names[:max_row_images])
        eartag_labeled = label_images(resize_images(eartag_crops[:max_row_images]), eartag_texts[:max_row_images])
        muzzle_labeled = label_images(resize_images(muzzle_crops[:max_row_images]), muzzle_flags[:max_row_images])
       
        track_row = stack_horizontally(track_labeled)
        eartag_row = stack_horizontally(eartag_labeled)
        muzzle_row = stack_horizontally(muzzle_labeled)

        min_row_width = (figure_size[0] + figure_spacing) * max_row_images - figure_spacing
        max_row_width = max(
            max(img.shape[1] if img is not None else 0 for img in [track_row, eartag_row, muzzle_row]),
            min_row_width
        )
        track_row = pad_to_width(track_row, max_row_width)
        eartag_row = pad_to_width(eartag_row, max_row_width)
        muzzle_row = pad_to_width(muzzle_row, max_row_width)

        banner_height = 60
        banner = np.ones((banner_height, max_row_width, 3), dtype=np.uint8) * 255

        banner_left_text = f"Track ID: {track_id} | Eartag: {eartag_number} | Muzzle Clean: {is_muzzle_clean}"
        cv2.putText(banner, banner_left_text, (10, 40), font, font_scale, font_color, font_thickness)

        banner_right_text = f"Total Images: {total_images} | Eartags: {total_eartags} | Muzzles: {total_muzzles}"
        (w, _), _ = cv2.getTextSize(banner_right_text, font, font_scale, font_thickness)
        cv2.putText(banner, banner_right_text, (max_row_width - w - 10, 40), font, font_scale, font_color, font_thickness)

        gallery = cv2.vconcat([banner, track_row, eartag_row, muzzle_row])

        window_title = f"Track {track_id} Summary"
        cv2.imshow(window_title, gallery)
        cv2.waitKey(0)
        cv2.destroyWindow(window_title)

def log_analysis_results(analysis_results):
    for obj in analysis_results:
        track_id = obj.get("id", "N/A")
        logger.info(f"Track ID: {track_id}")
        
        result = obj.get("result", {})
        eartag_number = result.get("eartag_number", "N/A")
        is_muzzle_clean = result.get("is_muzzle_clean", "N/A")
        logger.info(f"  Result: eartag_number={eartag_number}, is_muzzle_clean={is_muzzle_clean}")
        
        for image_entry in obj.get("images", []):
            image_name = image_entry.get("name", "unknown")
            detections = image_entry.get("detections", [])
            
            for det in detections:
                cls = det.get("class", "unknown").lower()
                info = []
                
                if cls == "tag":
                    info.append(f"eartag_number='{det.get('eartag_number', '')}'")
                elif cls == "muzzle":
                    info.append(f"is_muzzle_clean={det.get('is_muzzle_clean', 'N/A')}")                    
                
                logger.info(f"  Image: {image_name}, Class: {cls}, Info: {', '.join(info)}")    