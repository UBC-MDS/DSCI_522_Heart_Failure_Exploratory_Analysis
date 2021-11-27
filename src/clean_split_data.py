"""Cleans and preprocesses the data:

Usage:
clean_data.py --raw_data=<raw> --out_dir=<out_dir>
    
Options:
--raw_data=<raw>       File path to raw data including file name
--out_dir=<out_dir>    File path to output processed data
"""

from docopt import docopt
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

opt = docopt(__doc__)


def main(raw_data, out_dir):
    # Read Input Data
    df = pd.read_csv(raw_data)
    
    # Clean up column names
    df.columns = df.columns.str.strip()
    
        
    # Write cleaned data file
    train_file = os.path.join(out_dir, 'Heart_Failure_Data_train.csv')
    test_file = os.path.join(out_dir, 'Heart_Failure_Data_test.csv')
    
    # Split data into train/test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)
    
        
    # Write cleaned data file
    out_file = os.path.join(out_dir, 'Heart_Failure_Data_clean.csv')
    try:
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
    except:
        # Create output directory
        os.mkdir(out_dir)
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)

    print("File created in:", out_dir)
    
    
    
        
if __name__ == "__main__":
    main(opt["--raw_data"], opt["--out_dir"])