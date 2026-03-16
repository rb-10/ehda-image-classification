"""
predict.py

Opens all images in a folder sequentially and prints the prediction for each, showing all class confidences.

Usage:
    python predict.py <images_folder> <model_path>

Example:
    python predict.py "captures/experiment_1" "final_model/export.pkl"
"""

import sys
from pathlib import Path
from fastai.vision.all import load_learner, PILImage

def predict_folder(images_folder: str, model_path: str):

    folder = Path(images_folder)
    exts   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = sorted([f for f in folder.iterdir() if f.suffix.lower() in exts])

    if not images:
        print(f"[PREDICT] No images found in {folder}")
        return

    print(f"[PREDICT] Loading model: {model_path}")
    learn = load_learner(model_path)

    print(f"[PREDICT] Found {len(images)} images in {folder}")
    print(f"[PREDICT] Classes: {learn.dls.vocab}\n")
    # Print header
    header = f"{'Image':<40} " + " ".join([f"{cls:<15}" for cls in learn.dls.vocab])
    print(header)
    print("─" * (40 + 16 * len(learn.dls.vocab)))

    for img_path in images:
        img = PILImage.create(img_path)
        label, _, probs = learn.predict(img)
        # Print image name and confidence for each class
        confidences = " ".join([f"{(probs[i].item()*100):>14.1f}%" for i in range(len(learn.dls.vocab))])
        print(f"{img_path.name:<40} {confidences}")

if __name__ == "__main__":

    predict_folder("datasets/open_setup/test", "final_model/export.pkl")