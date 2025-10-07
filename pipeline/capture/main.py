import os
import logging
import time
import json

import cv2

from pipeline.utils.log import setup_logging
from pipeline.utils.config import load_yaml_config
from pipeline.capture.camera import initialize_camera, capture_image
from pipeline.capture.rfid_reader import initialize_rfid_reader, get_rfid_readings

setup_logging()
logger = logging.getLogger(__name__)
config = load_yaml_config("pipeline/config.yaml")
capture_dir = os.path.abspath(config.get("capture_dir", "captured_data"))
capture_image_format = config.get("capture_image_format", "png")
capture_fps = config.get("capture_fps", 1)

def main():
    camera = initialize_camera()
    rfid_reader = initialize_rfid_reader()

    # Create timestamped session directory
    timestamp_ms = str(int(time.time() * 1000))
    session_dir = os.path.abspath(os.path.join(capture_dir, timestamp_ms))
    images_dir = os.path.join(session_dir, "images")
    rfid_readings_path = os.path.join(session_dir, "rfid_readings.json")

    os.makedirs(images_dir, exist_ok=True)
    logger.info(f"Captured data will be saved to {session_dir}")
    
    counter = 0
    rfid_data = {}

    # Calculate delay based on capture_fps
    delay = int(1000 / capture_fps)
    
    while True:
        # Define a counter based id to uniquely identify the current capture
        counter += 1
        capture_id = f"{counter:04d}"

        image = capture_image(camera)
        rfid_readings = get_rfid_readings(rfid_reader)

        # Save image
        image_filename = f"{capture_id}.{capture_image_format}"
        image_path = os.path.join(images_dir, image_filename)
        cv2.imwrite(image_path, image)

        # Save RFID readings
        rfid_data[capture_id] = rfid_readings
        with open(rfid_readings_path, "w") as f:
            json.dump(rfid_data, f, indent=4)

        # Display image
        cv2.imshow("Image", image)
        
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()      

if __name__ == '__main__':
    main()