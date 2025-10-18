import argparse
import logging
import time
import os

from pipeline.utils.config import load_yaml_config
from pipeline.utils.log import setup_logging, log_time_taken
from pipeline.process.tracking import deepsort
from pipeline.process.detection import detect_objects
from pipeline.process.analysis import analyze_detections
from pipeline.process.visualization import visualize_analysis_results

setup_logging()
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run the processing pipeline on captured data")
    parser.add_argument("capture_name", type=str, help="name of the subfolder under capture_dir containing the captured data")
    parser.add_argument("--save_intermediate_results", action="store_true", help="flag to save intermediate results")
    
    args = parser.parse_args()
    config = load_yaml_config("pipeline/config.yaml")
    capture_dir = os.path.abspath(os.path.join(config.get("capture_dir"), args.capture_name))

    logger.info(f"Starting processing pipeline for capture: {args.capture_name}")
    if args.save_intermediate_results:
        logger.info(f"Intermediate results will be saved to: {capture_dir}")

    pipeline_start_time = time.time()

    tracking_start_time = time.time()
    tracking_results = deepsort(capture_dir, save_intermediate_results=args.save_intermediate_results)
    log_time_taken("Tracking", tracking_start_time)

    detection_start_time = time.time()
    detection_results = detect_objects(tracking_results, capture_dir, save_intermediate_results=args.save_intermediate_results)
    log_time_taken("Object detection", detection_start_time)

    analysis_start_time = time.time()
    analysis_results = analyze_detections(detection_results, capture_dir)
    log_time_taken("Analysis", analysis_start_time)

    log_time_taken("Processing Pipeline", pipeline_start_time)
    visualize_analysis_results(analysis_results, capture_dir)

if __name__ == "__main__":
    main()