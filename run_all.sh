# Script: run_all.sh
#
# run_all.sh script executes all the dependent scripts 
# 
# Usage: bash run_all.sh

# Download data
echo "Downloading data"
python src/download_data.py --url="https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv" --out_file="data/raw/heart_failure_clinical_records_dataset.csv"

# Data preprocessing
echo "Running data preprocessing script"
python src/clean_split_data.py --raw_data="data/raw/heart_failure_clinical_records_dataset.csv" --out_dir="data/processed/"

# Training data
echo "Running training script"
python src/model_train.py --data_file="data/processed/Heart_Failure_Data_train.csv" --out_dir="results/model"

# Test data
echo "Running test script"
python src/models/model_test.py --data_file="data/processed/Heart_Failure_Data_test.csv" --out_dir="results/model"

echo "Exiting! Script successfully completed."