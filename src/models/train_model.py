from src import get_project_root
from src.features.build_features import get_X
import pickle
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

IMG_FOLDER = get_project_root() / "data"


def train_model(file_name="data_batch_1", model_choice=SGDClassifier(random_state=42)):
    folder = IMG_FOLDER / file_name
    X_train = get_X(folder)
    with open(folder / "labels.txt", "rb") as file:
        # print(type(file.name))
        y_train = pickle.load(file)

    print(f"Trainning... {type(model_choice).__name__}")
    model_choice.fit(X_train, y_train)
    print("Trainning DONE")
    with open(
        Path(__file__).parent / f"{file_name}_{type(model_choice).__name__}.txt", "wb"
    ) as file:
        pickle.dump(model_choice, file)
        print("Model saved in " + file.name)
    return file.name



if __name__ == "__main__":
    # train_model()
    # train_model("data_batch_2")
    train_model(model_choice=RandomForestClassifier())
    train_model("data_batch_2", model_choice=RandomForestClassifier())
    
    print("DONE")
