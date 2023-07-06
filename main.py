"""I'm the main file, it's easy to do whatever from here"""

import os

from src import get_project_root
from src.data.make_dataset import get_batches, load_data
from src.features.build_features import extract_all
from src.models.train_model import train_all
from src.visualization.visualize import visu

if __name__ == "__main__":
    ### 1.Installs the dependencies

    os.system("pip install -r requirements.txt")

    ### You may do this command if you have src Module Not Found

    # os.system('export PYTHONPATH="${PYTHONPATH}:/mlrf-project"')

    ### 2.Extracts the data in .data/ YOU MUST HAVE THE CIFAR.tar FILE IN .data/

    load_data(get_batches())

    ### 3.Extracts the data's features in .data/*_batch*/

    extract_all(get_project_root() / "data")

    ### 4.Trains the models and store them in ./src/models/
    ### models are yet given but you can delete all .txt files in src/models/ and uncomment :
    ### If you don't want to train you need to extract all .txt files in the folder ./src/models/
    # train_all(["data_batch_1"])      # For fastest results
    # train_all(["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"])   #If you want to train on all the data

    ### 5.Evaluates the models from .src/models/ and stores the outputs in .reports/
    ### visualisations are given in .reports/reports.zip but you can delete them and uncomment:

    # visu()
