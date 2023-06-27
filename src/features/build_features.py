from src import get_project_root
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier


IMG_FOLDER = get_project_root() / "data/"


def extract(folder=IMG_FOLDER / "data_batch_1"):
    for file in os.scandir(folder):
        if "_" not in file.name and ".png" in file.name:
            image = cv2.imread(file.path)
            #TRY cv2.COLOR_RGB2HLS
            image_reduced = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            plt.imsave(fname=folder / (file.name.split('.')[0] + "_reduced.png"), arr=image_reduced)



def get_X(folder):
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

def get_y(folder):
    for file in os.scandir(folder):
        if ".txt" in file.name:
            return file.path

if __name__ == "__main__":
    extract(IMG_FOLDER / "data_batch_1")
    extract(IMG_FOLDER / "data_batch_2")
    print("Features DONE")