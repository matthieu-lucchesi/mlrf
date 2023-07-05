import os
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from src import get_project_root

IMG_FOLDER = get_project_root() / "data"


def extract(folder=IMG_FOLDER / "data_batch_1") -> None:
    """Loops over images and extract features from image and write in a file next to it"""
    for file in os.scandir(folder):
        if ".png" in file.name:
            image = cv2.imread(file.path)
            # HSV extraction
            image_reduced = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # reduce colors
            hist = cv2.calcHist(
                [image_reduced],
                channels=[0, 1, 2],
                mask=None,
                histSize=[4, 4, 4],
                ranges=[0, 180, 0, 256, 0, 256],
            )
            with open(
                Path(folder) / (file.name.split(".")[0] + "_hsv.list"), "wb"
            ) as file:
                pickle.dump(hist.flatten(), file)

            # HOG extraction
            winSize = (32, 32)
            blockSize = (16, 16)  # Defaults values from here
            blockStride = (8, 8)
            cellSize = (8, 8)
            nbins = 9
            hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
            hist = hog.compute(image)
            with open(
                Path(folder) / (file.name.split(".")[0] + "_hog.list"), "wb"
            ) as file:
                pickle.dump(hist.flatten(), file)


def extract_all(folder=IMG_FOLDER) -> None:
    """Loops over the batches to get all features for all datas"""
    for dir in os.scandir(folder):
        if "_batch" in dir.name and dir.is_dir():
            print(f"Extracting features for images in {dir.name}")
            extract(dir)
    print("Features extraction DONE")


if __name__ == "__main__":
    extract_all(IMG_FOLDER)
