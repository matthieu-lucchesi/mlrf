from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
import pickle
from src.features.build_features import get_X, get_y
from src import get_project_root
from pathlib import Path
import os
import matplotlib.pyplot as plt

IMG_FOLDER = get_project_root() / "data"


def evaluation(model_file, X_test_folder, y_test_folder):
    with open(model_file, "rb") as model_f, open(
        get_y(y_test_folder), "rb"
    ) as y_test_f:
        model_name = model_f.name.split("/")[-1].split(".")[0]
        eval_name = X_test_folder.name.split("/")[-1]
        print(f"evaluation of {model_name} with the datas from {eval_name}")
        model = pickle.load(model_f)
        y_test = pickle.load(y_test_f)
        X_test = get_X(X_test_folder)
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred)

        folder = get_project_root() / "reports" / (model_name + "_" + eval_name)
        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(folder / "report.txt", "w") as file:
            file.write(report)
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, values_format=".1%", normalize="all"
        )
        plt.savefig(folder / "confmatrix.png")

        label_binarizer = LabelBinarizer().fit(y_test)
        y_onehot_test = label_binarizer.transform(y_test)
        y_onehot_test.shape  # (n_samples, n_classes)
        try:
            RocCurveDisplay.from_predictions(
                y_onehot_test.ravel(),
                model.predict_proba(X_test).ravel(),
                name="micro-average OvR",
                color="darkorange",
            )
        except:
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

def visu():
    for model in os.scandir(Path(__file__).parent.parent / "models"):
        if ".txt" in model.name:
            if "1" in model.name:
                evaluation(
                    model, IMG_FOLDER / "data_batch_2", IMG_FOLDER / "data_batch_2"
                )
            if "2" in model.name:
                evaluation(
                    model, IMG_FOLDER / "data_batch_1", IMG_FOLDER / "data_batch_1"
                )

if __name__ == "__main__":
    visu()