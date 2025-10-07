import logging

from pipeline.utils.config import load_yaml_config

logger = logging.getLogger(__name__)
config = load_yaml_config("pipeline/config.yaml")
use_mock_rfid_reader = config.get("use_mock_rfid_reader", False)

def initialize_rfid_reader():
    if use_mock_rfid_reader:
        logger.info("Using mock RFID reader (use_mock_rfid_reader=true)")
        logger.info("Mock RFID reader ready")
        return "mock_rfid_reader"
    
    logger.info("Initializing RFID Reader (use_mock_rfid_reader=false)")
    logger.warning("RFID reader initialization logic is not implemented yet")
    # TODO: Add RFID reader initialization logic

    return "unimplemented_rfid_reader"

def get_rfid_readings(rfid_reader):
    if use_mock_rfid_reader:
        # Return mock RFID readings
        return [str(i).zfill(3) for i in range(1, 51)] 
    
    logger.warning("RFID reading logic is not implemented yet")
    # TODO: Add RFID reading logic

    return []
