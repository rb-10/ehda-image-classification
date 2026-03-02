#Use Optical Flow Image to turn a video into an image

import cv2
import numpy as np
from pathlib import Path

def clip_to_combined_image(video_path, output_size=(512, 512)):
    """
    Returns a 3-channel image where:
      - Channel 0 (R): first frame (grayscale)
      - Channel 1 (G): last frame (grayscale)
      - Channel 2 (B): averaged optical flow magnitude
    """
    cap = cv2.VideoCapture(str(video_path))

    # --- Read first frame ---
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError(f"Could not read video: {video_path}")
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # --- Read through all frames, computing flow ---
    flow_magnitude_sum = None
    frame_count = 0
    prev_gray = first_gray.copy()
    last_gray = first_gray.copy()  # fallback if only 1 frame

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Only keep magnitude (how much movement), not direction
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        if flow_magnitude_sum is None:
            flow_magnitude_sum = np.zeros_like(magnitude)
        flow_magnitude_sum += magnitude

        prev_gray = curr_gray
        last_gray = curr_gray
        frame_count += 1

    cap.release()

    if frame_count == 0:
        raise ValueError("Video has fewer than 2 frames")

    # --- Average and normalize flow to 0-255 ---
    flow_avg = flow_magnitude_sum / frame_count
    flow_normalized = cv2.normalize(flow_avg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --- Resize all three channels to output size ---
    first_resized = cv2.resize(first_gray, output_size)
    last_resized  = cv2.resize(last_gray,  output_size)
    flow_resized  = cv2.resize(flow_normalized, output_size)

    # --- Stack into a single 3-channel image ---
    combined = cv2.merge([first_resized, last_resized, flow_resized])  # shape: (224, 224, 3)

    return combined


def process_dataset(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for video_path in input_dir.rglob("*.mp4"):
        relative = video_path.relative_to(input_dir)
        out_path = output_dir / relative.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            print(f"Skipping {video_path.name} (already processed)")
            continue

        try:
            img = clip_to_combined_image(video_path)
            cv2.imwrite(str(out_path), img)
            print(f"Saved: {out_path}")
        except Exception as e:
            print(f"Error processing {video_path}: {e}")


if __name__ == "__main__":
    process_dataset("datasets\\open_setup\\videos\\undefined", "datasets\\open_setup\\optical images\\undefined")
    
    
    
    """
    Test case:
    img = clip_to_combined_image("video.mp4")
    cv2.imwrite("combined_output.png", img)
    print(f"Output shape: {img.shape}")  # should be (224, 224, 3)

    # View each channel separately to verify
    cv2.imshow("First frame",  img[..., 0])
    cv2.imshow("Last frame",   img[..., 1])
    cv2.imshow("Flow magnitude", img[..., 2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """