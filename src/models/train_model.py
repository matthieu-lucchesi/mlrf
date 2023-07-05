from typing import List
from src import get_project_root

import os
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestCentroid
from pathlib import Path

IMG_FOLDER = get_project_root() / "data"
METHODS = ["flatten", "HSV", "HOG"]
MODELS = [
    RandomForestClassifier(random_state=42),
    NearestCentroid(),
    LinearSVC(random_state=42),
]
MODELS = [LinearSVC(random_state=42)]


def get_X(folder, extract_method="flatten") -> List[List[float]]:
    """Get the data from a batch of images with the features asked"""
    res = []
    for file_numb in range(10_000):
        sample = []

        if extract_method == "flatten":
            with open(folder / f"{file_numb}.png", "rb") as file:
                img = plt.imread(fname=file)
                sample = list(img.flatten())

        elif extract_method == "HSV":
            with open(folder / f"{file_numb}_hsv.list", "rb") as file:
                features = pickle.load(file)
                for feat in features:
                    sample.append(feat)

        elif extract_method == "HOG":
            with open(folder / f"{file_numb}_hog.list", "rb") as file:
                features = pickle.load(file)
                for feat in features:
                    sample.append(feat)
        else:
            print("Method of extraction not available")
            return
        res.append(sample)

    return res


def get_y(folder) -> Path:
    """get the labels of the data wanted"""
    for file in os.scandir(folder):
        if ".txt" in file.name:
            return file.path


def train_model(
    batches=["data_batch_1"],
    model_choice=MODELS[0],
    extract_method="HSV",
):
    """Loops over batches to retrieve data asked and train a model with the data"""
    X_train = []
    y_train = []
    for batch in batches:
        folder = IMG_FOLDER / batch

        X_train += get_X(folder, extract_method)
        with open(folder / "labels.txt", "rb") as file:
            y_train = pickle.load(file)

    print(
        f"Trainning... {type(model_choice).__name__} with features from {extract_method}"
    )

    model_choice.fit(X_train, y_train)
    print("Trainning DONE")

    with open(
        Path(__file__).parent
        / f"{len(batches)}_{type(model_choice).__name__}_{extract_method}.txt",
        "wb",
    ) as file:
        pickle.dump(model_choice, file)
        print("Model saved in " + file.name+"\n")
    return file.name


def train_all(batches, models=MODELS, extraction_methods=METHODS) -> None:
    """Loops over models and extract methods to train data"""
    for model in models:
        for method in extraction_methods:
            train_model(batches, model, method)


if __name__ == "__main__":
    train_all(["data_batch_1"])
    # train_model(["data_batch_2"], model_choice=MODEL[1], extract_method="HOG")
    # train_model(["data_batch_3"], model_choice=MODEL[2], extract_method="HOG")

    print("DONE")
