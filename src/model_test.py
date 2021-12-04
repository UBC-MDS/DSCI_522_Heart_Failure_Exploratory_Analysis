"""Test the model on test dataset.
Usage: test.py [--data_file=<data_file>] [--out_dir=<out_dir>]

Options:
[--data_file=<data_file>]        Data set file test data are saved as csv.
[--out_dir=<out_dir>]            Output path to save results, tables and images.
"""

import os
import sys
from pathlib import Path

project_root = str(Path(__file__).parents[2])
sys.path.append(project_root)

from docopt import docopt
import matplotlib.pyplot as plt
import mglearn
import numpy as np
import pandas as pd
import dataframe_image as dfi
import pickle
from sklearn.metrics import get_scorer

# Customer imports
from utils.util import get_config, get_logger

# Define logger
logger = get_logger()

def main(data_file, out_dir):
    """load the best model and fit on test data
    Parameters
    ----------
    data_file : string
        Path to test data
    out_dir : string
        Path to directory where the test result should be saved
    """
    
    test_df = pd.read_csv(data_file)
    best_model = pickle.load(open(out_dir + "/best_model.sav", "rb"))

    # show the score of best model on test data in a table
    result = test_model(best_model, test_df)
    


def test_model(best_model, test_df):
    """test the model on test dataset
    Parameters
    ----------
    best_model : 
        the trained model
    data_file : string
        Path to test data
    Returns
    -------
    float
        The score
    """
    logger.info("Testing on test set...")
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
    X_test = test_df.drop("DEATH_EVENT", axis=1)
    y_test = test_df['DEATH_EVENT']
    
    df = pd.DataFrame(scoring_metrics, columns = ["Metrics"])

    r = []
    for m in scoring_metrics:
        r.append(get_scorer(m)(best_model, X_test, y_test))
    df["Test Result"] = r
    df.set_index('Metrics')
    path = os.path.join(out_dir, "test_result_table.png")
    dfi.export(df, path)
    logger.info("Test set results saved as a table")



if __name__ == "__main__":

    # Parse command line parameters
    opt = docopt(__doc__)

    data_file = opt["--data_file"]
    out_dir = opt["--out_dir"]

    # Read it from config file
    # if command line arguments are missing
    if not data_file:
        data_file = os.path.join(project_root, get_config("model.test.data_file"))

    if not out_dir:
        out_dir = os.path.join(project_root, get_config("model.test.out_dir"))

    # Run the main function
    logger.info("Running testing...")
    main(data_file, out_dir)
    logger.info("Test script successfully completed. Exiting!")
