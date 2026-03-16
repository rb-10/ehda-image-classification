import cv2
import os
import shutil

input_folder = "datasets\\open_setup\\unclassified"
output_base = "datasets\\open_setup\\videos"

classes = ["cone_jet", "dripping", "intermitent", "multi_jet", "no_flow", "undefined"]

for cls in classes:
    os.makedirs(os.path.join(output_base, cls), exist_ok=True)

videos = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]
videos.sort()

print("Controls:")
print("1–6 → assign class")
print("q   → quit")
print("n   → skip video")

for video_name in videos:

    video_path = os.path.join(input_folder, video_name)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening:", video_name)
        continue

    print(f"\nLabeling: {video_name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        cv2.imshow("Video", frame)

        key = cv2.waitKey(int(1000 / cap.get(cv2.CAP_PROP_FPS))) & 0xFF

        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

        elif key == ord('n'):
            print("Skipped")
            break

        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')]:
            class_index = int(chr(key)) - 1
            destination = os.path.join(output_base, classes[class_index], video_name)

            cap.release()
            cv2.destroyAllWindows()

            shutil.move(video_path, destination)
            print(f"Moved to {classes[class_index]}")
            break

    cap.release()
    cv2.destroyAllWindows()

print("Done labeling.")
