from pathlib import Path
from split_video import split_video

solution = "ethanol_hv_nozzle"
folder = Path(rf"C:\Users\HV\Desktop\bruno_work\save_electrospray\{solution}")

for file_name in Path(folder / 'raw').glob('*.mp4'):
    split_video(folder, file_name)
