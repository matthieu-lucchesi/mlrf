import tarfile
from typing import List, Dict
from src import get_project_root
import os

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


FOLDER_DATA_EXTRACTED = extract_data()


def get_batches(folder=FOLDER_DATA_EXTRACTED) -> List[os.PathLike]:
    for dir in os.scandir(folder):
        if dir.is_dir():
            return [file.path for file in os.scandir(dir.path) if "batch" in file.name]


### FROM THE DOC
def unpickle(file) -> Dict:
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def extract_files(folder=FOLDER_DATA_EXTRACTED) -> Dict[str, Dict]:
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




# plt.imshow(data2img(extract_files()["data_batch_1"][b"data"])[12])
# plt.show()
