import tarfile
from typing import List, Dict
from src import get_project_root
import os
import matplotlib.pyplot as plt
import pickle




root = get_project_root()
FOLDER_DATA = root / "data"


### extracting file
def extract_data(folder=FOLDER_DATA, dest=FOLDER_DATA) -> None:
    try:
        file = tarfile.open(folder / "cifar-10-python.tar.gz")
        file.extractall(dest)
        file.close()
        return dest
    except:
        print("extract_data ERROR")
        return folder




def get_batches(folder=FOLDER_DATA) -> List[os.PathLike]:
    for dir in os.scandir(folder):
        if dir.is_dir() and dir.name == "cifar-10-batches-py":
            return [file.path for file in os.scandir(dir.path) if "batch" in file.name]


### FROM THE DOC
def unpickle(file) -> Dict:
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def extract_files(folder) -> Dict[str, Dict]:
    """Apply func to files
    returns a dict with file_name: data"""
    return {file.split("/")[-1]: unpickle(file) for file in get_batches(folder)}


def data2img(array) -> List[List[int]]:
    """Converts the data to an array of img readable"""

    ### Split by colors 
    array = array.reshape(10000, 3, 32, 32)
    return [
        # Imshow wants arrays of tuple instead of three arrays
        image.transpose(1, 2, 0) for image in array
        ]

def save_images(images, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i, image in enumerate(images):
        print(folder / (str(i) + ".png"))
        plt.imsave(fname=folder / f"{i}.png", arr=image)

def load_labels(folder, file_name):
    with open(folder / "labels.txt", "wb") as file:
        pickle.dump(extract_files(folder)[file_name][b"labels"], file=file)


def load_batch(file_name="data_batch_1"):
    folder_data_extracted = extract_data()
    save_images(data2img(extract_files(folder_data_extracted)[file_name][b"data"]), folder_data_extracted / "batch1")
    load_labels(folder_data_extracted, file_name)
    print("Etraction of" + file_name + " done !")

if __name__ == "__main__":
    load_batch()
    load_batch("data_batch_2")
    