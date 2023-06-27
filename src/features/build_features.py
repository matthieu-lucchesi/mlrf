from src import get_project_root
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.linear_model import SGDClassifier


IMG_FOLDER = get_project_root() / "data/batch1"


def extract(folder):
    for file in os.scandir(folder):
        if "_" not in file.name:
            image = cv2.imread(file.path)
            #TRY cv2.COLOR_RGB2HLS
            image_reduced = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            plt.imsave(fname=folder / (file.name.split('.')[0] + "_reduced.png"), arr=image_reduced)



def get_X_train(folder):
    res= []
    for file in os.scandir(folder):
        if "_" in file.name:
            sample=[]
            image = cv2.imread(file.path) 
            for line in image:
                for pix in line:
                    for val in pix:
                        sample.append(val)
            res.append(sample)
    return res


if __name__ == "__main__":
    extract(IMG_FOLDER)
    print("Features DONE")