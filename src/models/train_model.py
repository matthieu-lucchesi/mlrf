from src import get_project_root
from src.features.build_features import get_X_train
import pickle 
from sklearn.linear_model import SGDClassifier
from pathlib import Path

IMG_FOLDER = get_project_root() / "data/batch1"


if __name__ == "__main__":
    X_train = get_X_train(IMG_FOLDER)
    with open(get_project_root() / "data/labels.txt", "rb") as file:
        y_train = pickle.load(file)

    clf = SGDClassifier(random_state=42)
    print("Trainning...")
    clf.fit(X_train, y_train)
    with open(Path(__file__).parent / "model.txt", "wb") as file:
        pickle.dump(clf, file)
        print("Model saved in " + file.path)
    print("DONE")