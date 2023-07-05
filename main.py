"""I'm the main file, it's easy to do whatever from here"""

import os

from src import get_project_root
from src.data.make_dataset import get_batches, load_data
from src.features.build_features import extract_all
from src.models.train_model import train_all
from src.visualization.visualize import visu

if __name__ == "__main__.py":
    # Installs the dependencies
    os.system("pip install -r requirements.txt")

    # You may do this command if you have src Module Not Found
    os.system('export PYTHONPATH="${PYTHONPATH}:/mlrf-project"')

    # Extracts the data in .data/
    load_data(get_batches())

    # Extracts the data's features in .data/
    extract_all(get_project_root() / "data")

    # Trains the models and store them in ./src/models/
    train_all(["data_batch_1"])
    # train_all(["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"])

    # Evaluates the models from .src/models/ and stores the outputs in .reports
    visu()
