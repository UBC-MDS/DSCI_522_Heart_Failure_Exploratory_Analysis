"""Fit a classifier based on input train data.
Save the models and coefficients in a table as png.
Usage: train.py [--data_file=<data_file>] [--out_dir=<out_dir>]

Options:
[--data_file=<data_file>]        Data set file train are saved as csv.
[--out_dir=<out_dir>]            Output path to save model, tables and images.
"""

# Import all the modules from project root directory
from pathlib import Path
import sys

project_root = str(Path(__file__).parents[2])
sys.path.append(project_root)

import os
from docopt import docopt
import IPython
import ipywidgets as widgets
import matplotlib.pyplot as plt
import mglearn
import numpy as np
import pandas as pd
import dataframe_image as dfi
import pickle
from IPython.display import HTML, display
from ipywidgets import interact, interactive
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import (
    ColumnTransformer,
    make_column_transformer
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Customer imports
from utils.util import get_config, get_logger

# Define logger
logger = get_logger()

def main(data_file, out_dir):
    """run all helper functions to find the best model and get the 
    hyperparameter tuning result
    Parameters
    ----------
    input_file : string
        the path to the training dataset
    out_dir : string
        the path to store the results
    """
    # If a directory path doesn't exist, create one
    os.makedirs(out_dir, exist_ok=True)
    
    train_df = pd.read_csv(data_file)
    pipe = build_pipe()
    best_model, train_results = fit_model(train_df, pipe)

    # save the best model
    pickle.dump(best_model, open(out_dir + "/best_model.sav", "wb"))

    
    # save train results as a table
    train_df_table(train_results, out_dir)
    

def build_pipe():

    """build a randomforest classifier pipeline with column transformer
    to preprocess every column

    Returns
    -------
    sklearn.pipeline.Pipeline
        ML pipeline
    """

    logger.info("Building the pipeline...")

    # build column transformer
    numeric_features = ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine', 'serum_sodium','time']
    binary_feats = ['sex', 'diabetes', 'high_blood_pressure','anaemia']
    target = 'DEATH_EVENT'

    preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(), binary_feats),)

    # build pipe line
    pipe_rf = make_pipeline(
    col_trans,
    RandomForestClassifier(random_state=123))
    
    logger.info("Successfully built the pipeline...")
    
    return pipe


def fit_model(train_df, pipe):
    """Train the logistic model by using random search
    with cross validation
    
    Parameters
    ----------
    data_file : string
        Train data set file path, including filename

    Returns
    -------
    train_results: dataframe
        A data frame with train score results from each model
    """
    
    logger.info("Fitting the model...")
    
    # split train data for cross validation
    X_train = train_df.drop("DEATH_EVENT", axis=1)
    y_train = train_df['DEATH_EVENT']

    
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    baseline_results['RandomForest_default'] = pd.DataFrame(cross_validate(pipe, X_train, y_train, scoring=scoring)).mean()

    # Export baseline_results
    baseline_results = pd.DataFrame(baseline_results)
    baseline_results_path = os.path.join(opt['--out_dir'], "baseline_result.csv")
    baseline_results.to_csv(baseline_results_path)
    print(f"Baseline Result saved to {baseline_results_path}")

    # Hyperparameter Tuning
    param_dist = {
        "randomforestclassifier__class_weight": [None, "balanced"],
        "randomforestclassifier__n_estimators": [10, 20, 50, 100, 200, 500],
        "randomforestclassifier__max_depth": np.arange(10, 20, 2)
    }
    rand_search_rf = RandomizedSearchCV(pipe_forest, param_dist, n_iter=20, 
                                        random_state=123, scoring=scoring, refit="precision")

    print("Model Training In Progess...")
    rand_search_rf.fit(X_train, y_train)
    print("Model Training Done!")

    hyperparam_result = pd.DataFrame(
        rand_search_rf.cv_results_
    ).sort_values("rank_test_f1")[['param_randomforestclassifier__n_estimators',
                                        'param_randomforestclassifier__max_depth',
                                        'param_randomforestclassifier__class_weight',
                                        'mean_test_accuracy',
                                        'mean_test_precision',
                                        'mean_test_recall',
                                        'mean_test_f1'
                                        ]]
    # Export hyperparam_result
    hyperparam_result_path = os.path.join(opt['--out_dir'], "hyperparam_result.csv")
    hyperparam_result.to_csv(hyperparam_result_path)
    print(f"Hyperparameter Tuning Result saved to {hyperparam_result_path}")
    
    # find the best model
    best_model = rand_search_rf.best_estimator_
    
    logger.info("Model fitted...")
    
    return best_model, hyperparam_result



def train_df_table(train_results, out_dir):

    logger.info("Making train results table...")
    path = os.path.join(out_dir, "train_result_table.png")
    dfi.export(train_results, path)
    logger.info(f"Train results table saved to {out_dir}")


if __name__ == "__main__":

    # Parse command line parameters
    opt = docopt(__doc__)

    data_file = opt["--data_file"]
    out_dir = opt["--out_dir"]

    # Read it from config file
    # if command line arguments are missing
    if not data_file:
        data_file = os.path.join(project_root, get_config("model.train.data_file"))

    if not out_dir:
        out_dir = os.path.join(project_root, get_config("model.train.out_dir"))

    # Run the main function
    logger.info("Running training...")
    main(data_file, out_dir)
    logger.info("Training script successfully completed. Exiting!")

