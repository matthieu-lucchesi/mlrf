import tarfile
from typing import List, Dict
from src import get_project_root
import os
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

root = get_project_root()
FOLDER_DATA = root / "data"


### extracting file
def extract_data(folder=FOLDER_DATA, dest=FOLDER_DATA) -> Path  :
    """extract the tar file in the folder project/data"""
    try:
        file = tarfile.open(folder / "cifar-10-python.tar.gz")
        file.extractall(dest)
        file.close()
        return dest
    except:
        print("extract_data ERROR")
        return folder


def get_batches(folder=FOLDER_DATA) -> List[Path]:
    """Return the list of the batches"""
    for dir in os.scandir(folder):
        if dir.is_dir() and dir.name == "cifar-10-batches-py":
            return [file.path for file in os.scandir(dir.path) if "_batch" in file.name]


### FROM THE DOC
def unpickle(file) -> Dict:
    """Doc's method to load images"""
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def extract_files(folder) -> Dict[str, Dict]:
    """Returns a dict with batch_number: data loaded form batch"""
    return {file.split("/")[-1]: unpickle(file) for file in get_batches(folder)}


def data2img(array) -> List[List[int]]:
    """Converts the data to an array of img readable RGB"""

    # Split by colors 
    array = array.reshape(10000, 3, 32, 32)
    return [
        # Imshow wants arrays of tuple instead of three arrays
        image.transpose(1, 2, 0) for image in array
        ]

def save_images(images, folder) -> None:
    """save all image in the list images to the folder given as argument"""
    
    #Creates the folder if not existing
    if not os.path.exists(folder):
        os.makedirs(folder)
    print("Extracting images from folder ", folder.name, "...")
    for i, image in enumerate(images):
        plt.imsave(fname=folder / f"{i}.png", arr=image)

def load_labels(folder, file_name) -> None:
    """Extract the labels of the images from a batch and write labels in the file **/labels.txt"""
    with open(folder / file_name /"labels.txt", "wb") as file:
        pickle.dump(extract_files(folder)[file_name][b"labels"], file=file)


def load_batch(file_name="data_batch_1") -> None:
    """Save images and labels from a batch"""
    folder_data_extracted = extract_data()
    save_images(data2img(extract_files(folder_data_extracted)[file_name][b"data"]), folder_data_extracted / file_name)
    load_labels(folder_data_extracted, file_name)
    print("Etraction of " + file_name + " done !\n")

def load_data(list_=["data_batch_1"]) -> None:
    """Loops over all batches and call the previous function to load a batch of images."""
    for file in list_:
        load_batch(file.split('/')[-1])


if __name__ == "__main__":
    load_data(get_batches())
    # load_batch()
    