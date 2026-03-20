import cv2
import os
import shutil
import json
import re

liquid = "EW82"
json_folder = f"C:\\Users\\HV\\Desktop\\bruno_work\\save_electrospray\\dataset\\current\\{liquid}\\unclassified\\"
input_folder = f"C:\\Users\\HV\\Desktop\\bruno_work\\save_electrospray\\dataset\\images\\{liquid}\\unclassified"
output_base = f"C:\\Users\\HV\\Desktop\\bruno_work\\save_electrospray\\dataset\\images\\{liquid}"

classes = ["cone_jet", "dripping", "intermitent", "multi_jet", "unconclusive", "undefined"]

for cls in classes:
    os.makedirs(os.path.join(output_base, cls), exist_ok=True)

videos = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]
videos.sort()

print("Controls:")
print("1–6 → assign class")
print("q   → quit")
print("n   → skip video")

# Cache loaded JSON files to avoid reloading repeatedly
json_cache = {}

def load_experiment_json(experiment_idx):
    """Load and cache the JSON for a given experiment index."""
    if experiment_idx in json_cache:
        return json_cache[experiment_idx]
    
    json_filename = f"experiment_{experiment_idx}.json"
    json_path = os.path.join(json_folder, json_filename)
    
    if not os.path.exists(json_path):
        print(f"  [WARNING] JSON not found: {json_path}")
        return None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    json_cache[experiment_idx] = (json_path, data)
    return json_cache[experiment_idx]

def save_experiment_json(experiment_idx):
    """Save the cached JSON back to disk."""
    if experiment_idx not in json_cache:
        return
    json_path, data = json_cache[experiment_idx]
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"  Saved JSON: {os.path.basename(json_path)}")

for video_name in videos:
    # Parse experiment and sample index from filename: clip_yyy_xxx.mp4
    match = re.match(r"clip_(\d+)_(\d+)\.mp4", video_name)
    if not match:
        print(f"Skipping (unexpected filename format): {video_name}")
        continue

    experiment_idx = int(match.group(1))
    sample_idx = int(match.group(2))
    sample_key = f"sample {sample_idx}"

    # Load the corresponding JSON
    result = load_experiment_json(experiment_idx)
    if result is None:
        print(f"  Skipping {video_name} — no matching JSON for experiment {experiment_idx}")
        continue

    json_path, data = result

    if sample_key not in data:
        print(f"  [WARNING] '{sample_key}' not found in {os.path.basename(json_path)}, skipping.")
        continue

    current_label = data[sample_key].get("spray_mode", "N/A")

    video_path = os.path.join(input_folder, video_name)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening:", video_name)
        continue

    print(f"\nLabeling: {video_name}  |  Experiment: {experiment_idx}, Sample: {sample_idx}  |  Current label: {current_label}")

    labeled = False
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        cv2.imshow("Video", frame)
        key = cv2.waitKey(int(1000 / cap.get(cv2.CAP_PROP_FPS))) & 0xFF

        if key == ord('q'):
            # Save all modified JSONs before quitting
            for exp_idx in list(json_cache.keys()):
                save_experiment_json(exp_idx)
            cap.release()
            cv2.destroyAllWindows()
            print("Quit — all changes saved.")
            exit()

        elif key == ord('n'):
            print("  Skipped")
            break

        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')]:
            class_index = int(chr(key)) - 1
            chosen_class = classes[class_index]

            # Update spray_mode in the correct sample
            data[sample_key]["spray_mode"] = chosen_class
            save_experiment_json(experiment_idx)

            # Move video to labeled folder
            destination = os.path.join(output_base, chosen_class, video_name)
            cap.release()
            cv2.destroyAllWindows()
            shutil.move(video_path, destination)
            print(f"  Labeled as '{chosen_class}' → moved to {chosen_class}/")
            labeled = True
            break

    if not labeled:
        cap.release()
        cv2.destroyAllWindows()

print("\nDone labeling.")
# Final save of any remaining cached JSONs
for exp_idx in list(json_cache.keys()):
    save_experiment_json(exp_idx)