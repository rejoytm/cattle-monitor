import logging
import time

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

def log_time_taken(process_name, start_time):
    logger = logging.getLogger(__name__) 
    elapsed_time = time.time() - start_time
    logger.info(f"{process_name} took {elapsed_time:.2f} seconds")