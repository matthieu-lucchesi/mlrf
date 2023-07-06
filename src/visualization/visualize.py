import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
)
from sklearn.preprocessing import LabelBinarizer

from src import get_project_root
from src.models.train_model import get_X, get_y

IMG_FOLDER = get_project_root() / "data"


def evaluation(model_file, test_folder, extract_method) -> None:
    """Evaluates a model saved in ../models with a test batch and an extract method"""
    with open(model_file, "rb") as model_f, open(get_y(test_folder), "rb") as y_test_f:
        model_name = model_f.name.split("/")[-1].split(".")[0]
        model = pickle.load(model_f)
        y_test = pickle.load(y_test_f)

    # Creates the name of the evaluation folder
    eval_name = test_folder.name.split("/")[-1]
    print(f"evaluation of {model_name} with the datas from {eval_name}")

    # Retrieves datas for the test
    X_test = get_X(test_folder, extract_method)
    y_pred = model.predict(X_test)

    # Creates the folder of the evaluation
    folder = get_project_root() / "reports" / (model_name + "_" + eval_name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Creates the classif_report
    with open(folder / "report.txt", "w") as file:
        file.write(classification_report(y_test, y_pred))

    # Creates the conf matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, values_format=".1%", normalize="all"
    )
    plt.savefig(folder / "confmatrix.png")

    # Try to plot the roc curve
    label_binarizer = LabelBinarizer().fit(y_test)
    y_onehot_test = label_binarizer.transform(y_test)
    try:
        RocCurveDisplay.from_predictions(
            y_onehot_test.ravel(),
            model.predict_proba(X_test).ravel(),
            name="micro-average OvR",
            color="darkorange",
        )
        plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Micro-averaged One-vs-Rest\nReceiver Operating Characteristic")
        plt.legend()
        plt.savefig(folder / "RocCurve.png")
    except:
        try:
            RocCurveDisplay.from_predictions(
                y_onehot_test.ravel(),
                model.decision_function(X_test).ravel(),
                name="micro-average OvR",
                color="darkorange",
            )
            plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
            plt.axis("square")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Micro-averaged One-vs-Rest\nReceiver Operating Characteristic")
            plt.legend()
            plt.savefig(folder / "RocCurve.png")
        except:
            print("No ROC curve for this model")


def visu() -> None:
    """evaluate all models form folder ../models"""
    for model in os.scandir(Path(__file__).parent.parent / "models"):
        if ".txt" in model.name:
            evaluation(
                model,
                IMG_FOLDER / "test_batch",
                model.name.split("_")[-1].split(".")[0],
            )


if __name__ == "__main__":
    visu()
