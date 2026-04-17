from pre_processing_ben import *
from pathlib import Path
import cv2
import random

class_folder = Path(r"C:\Users\HV\Desktop\bruno_work\save_electrospray\DMF\SPLIT CLIPS")
output_base = Path(r"C:\Users\HV\Desktop\bruno_work\save_electrospray\DMF\PROCESSED CLIPS")

video_files = sorted(list(class_folder.glob('*.mp4')), reverse=True)

random.shuffle(video_files)
for video_file in video_files:
    out_img_path = output_base / (video_file.stem + '.png')
    if out_img_path.exists():
        print(f"Skipping {video_file}, output already exists.")
        continue
    print(f"Processing {video_file}")
    cap = cv2.VideoCapture(str(video_file))
    frames = read_gray_frames(cap)
    if not frames:
        print(f"No frames found in {video_file}")
        continue
    # Use merged RGB image of three methods, then crop/resize to 256x256
    merged_image = cv2.merge(
        (temporal_median_background(frames), tiny_particle_detector(frames), original_optical_flow(frames)))
    processed_img = max_pool_to_size(merged_image, (256, 256))
    # Save processed image with same name as video, but .png
    cv2.imwrite(str(out_img_path), processed_img)
    print(f"Saved {out_img_path}")
