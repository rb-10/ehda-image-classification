from pathlib import Path

class_folder = Path(r"C:\Users\HV\Desktop\bruno_work\save_electrospray\Ethanol\CLASSIFIED")
i = 0

lst = []
for x in class_folder.iterdir():
    if x.is_dir():
        lst.append(x)

for folder in lst:
    video_files = list(folder.glob('*.png'))
    i += len(video_files)

print(i)