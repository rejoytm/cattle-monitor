import os

import cv2

def extract_frames_from_video(video_path, output_path, target_fps, image_format='png'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    os.makedirs(output_path, exist_ok=True)

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    # Determine how many frames to skip based on FPS ratio
    frame_interval = int(round(original_fps / target_fps)) if target_fps < original_fps else 1

    frame_index = 0
    saved_frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            filename = os.path.join(output_path, f"{saved_frame_index:04d}.{image_format}")
            cv2.imwrite(filename, frame)
            saved_frame_index += 1

        frame_index += 1

    cap.release()