import cv2
import os

# -------- SETTINGS --------
input_folder = "datasets\\open_setup\\optical images\\multi_jet"
output_folder = "datasets\\open_setup\\optical images_cropped\\multi_jet"

crop_width = 200
crop_height = 512# --------------------------

os.makedirs(output_folder, exist_ok=True)

images = [f for f in os.listdir(input_folder) if f.endswith(".png")]
images.sort()

for img_name in images:

    input_path = os.path.join(input_folder, img_name)
    output_path = os.path.join(output_folder, img_name)

    img = cv2.imread(input_path)

    if img is None:
        print("Error reading:", img_name)
        continue

    height, width = img.shape[:2]

    # Check if crop size is larger than image
    if crop_width > width or crop_height > height:
        print(f"Skipping {img_name} (crop bigger than image)")
        continue

    # Center coordinates
    cx = width // 2-20 # to set a bit to the left
    cy = height // 2

    x1 = cx - crop_width // 2
    y1 = cy - crop_height // 2
    x2 = x1 + crop_width
    y2 = y1 + crop_height

    cropped = img[y1:y2, x1:x2]

    cv2.imwrite(output_path, cropped)

    print("Saved:", img_name)

print("Done.")