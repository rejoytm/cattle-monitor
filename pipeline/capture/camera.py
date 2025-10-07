import logging

import numpy as np
import cv2

from pipeline.utils.config import load_yaml_config

logger = logging.getLogger(__name__)
config = load_yaml_config("pipeline/config.yaml")
use_mock_camera = config.get("use_mock_camera", False)

if not use_mock_camera:
    from picamera2 import Picamera2

_mock_camera_frame_number = 0

def initialize_camera():
    if use_mock_camera:
        logger.info("Using mock camera (use_mock_camera=true)")
        logger.info("Mock camera ready")
        return "mock_camera"
    
    logger.info("Initializing Picamera2 (use_mock_camera=false)")
    picam2 = Picamera2()
    picam2_config = picam2.create_video_configuration(
        raw={"size": (1640, 1232)}, main={"size": (640, 480), "format": "XRGB8888"}
    )
    picam2.align_configuration(picam2_config)
    picam2.configure(picam2_config)
    picam2.start()

    logger.info("Picamera2 ready")
    return picam2

def capture_image(picam2):
    global _mock_camera_frame_number
    
    if use_mock_camera:
        # Return a mock image
        _mock_camera_frame_number += 1
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        text = f"Mock Image {_mock_camera_frame_number}"
        cv2.putText(image, text, (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return image    
    
    return picam2.capture_array()