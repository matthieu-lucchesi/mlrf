from src import get_project_root
import os
import cv2
import matplotlib.pyplot as plt
import pickle

IMG_FOLDER = get_project_root() / "data/"


def extract(folder=IMG_FOLDER / "data_batch_1"):
    for file in os.scandir(folder):
        if "_" not in file.name and ".png" in file.name:
            image = cv2.imread(file.path)
            #TRY cv2.COLOR_RGB2HLS
            image_reduced = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            plt.imsave(fname=folder / (file.name.split('.')[0] + "_reduced.png"), arr=image_reduced)
  
            # reduce colors
            hist = cv2.calcHist([image_reduced],channels=[0,1,2],mask=None, histSize=[4,4,4], ranges=[0,180,0,256,0,256])
            with open(folder /(file.name.split('.')[0] + "_reduced.list"),"wb") as file:
                pickle.dump(hist.flatten(), file)





def get_X(folder):
    res= []
    for file_numb in range(10_000):
        with open(folder / f"{file_numb}_reduced.png", "rb")as file:
            img = plt.imread(fname=file)
            sample = list(img.flatten())

        with open(folder / f"{file_numb}_reduced.list", "rb")as file:
            features = pickle.load(file)
            for feat in features:
                sample.append(feat)

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