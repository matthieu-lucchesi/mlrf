import os
import pickle
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC

from src import get_project_root

IMG_FOLDER = get_project_root() / "data"
METHODS = ["flatten", "HSV", "HOG"]
MODELS = {
    "RandomForestClassifier": RandomForestClassifier(random_state=42),
    "NearestCentroid": NearestCentroid(),
    "LinearSVC": LinearSVC(random_state=42, dual="auto"),
    # "RandomForestClassifierGrid": GridSearchCV(
    #     RandomForestClassifier(random_state=42),
    #     param_grid={"max_depth": [10, 20, 30], "bootstrap": [True, False]},
    #     verbose=3,
    #     cv=3,
    # ),
    # "NearestCentroidGrid": GridSearchCV(
    #     NearestCentroid(),
    #     param_grid={
    #         "metric": ["euclidean", "manhattan"],
    #         "shrink_threshold": [None, 0.1, 0.5, 1],
    #     },
    #     verbose=3,
    #     cv=3,
    # ),
    # "LinearSVCGrid": GridSearchCV(
    #     LinearSVC(random_state=42, dual="auto"),
    #     param_grid={
    #         "C": [1, 10, 100, 1000],
    #     },
    #     verbose=3,
    #     cv=3,
    # ),
    # Models obtained after doing the grid search
    "RandomForestClassifierBest": RandomForestClassifier(
        random_state=42, max_depth=30, bootstrap=False
    ),
    "NearestCentroidBest": NearestCentroid(metric="manhattan", shrink_threshold=None),
}


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
    key_choice="RandomForestClassifier",
    model_choice=MODELS["RandomForestClassifier"],
    extract_method="HSV",
):
    """Loops over batches to retrieve data asked and train a model with the data"""
    X_train = []
    y_train = []
    for batch in batches:
        folder = IMG_FOLDER / batch

        X_train += get_X(folder, extract_method)
        with open(folder / "labels.txt", "rb") as file:
            y_train += pickle.load(file)

    print(f"Trainning... {key_choice} with features from {extract_method}")

    if "Grid" in key_choice:
        model_tuned = model_choice.fit(X_train, y_train).best_estimator_
        print(model_choice.best_params_)
    else:
        model_tuned = model_choice.fit(X_train, y_train)

    print("DONE")

    with open(
        Path(__file__).parent / f"{key_choice}_{extract_method}.txt",
        "wb",
    ) as file:
        pickle.dump(model_tuned, file)
        print("Model saved in " + file.name + "\n")
    return file.name


def train_all(batches, models=MODELS, extraction_methods=METHODS) -> None:
    """Loops over models and extract methods to train data"""
    for key, model in models.items():
        for method in extraction_methods:
            train_model(batches, key, model, method)


if __name__ == "__main__":
    train_all(
        ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    )
    # train_model(["data_batch_2"], model_choice=MODEL[1], extract_method="HOG")
    # train_model(["data_batch_3"], model_choice=MODEL[2], extract_method="HOG")

    print("DONE")
