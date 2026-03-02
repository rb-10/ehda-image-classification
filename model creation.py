from video_to_img import process_dataset
from fastai.vision.all import *
import cv2
import numpy as np
from pathlib import Path

path = Path('datasets/open_setup/optical images_cropped')

if __name__ == "__main__":

    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock), 
        get_items=get_image_files, 
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')]
    ).dataloaders(path, bs=32)


    learn = vision_learner(dls, resnet34, metrics=accuracy)
    learn.fine_tune(5)