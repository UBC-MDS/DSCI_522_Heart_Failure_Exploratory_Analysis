#This script builds the model to predict 
# the death event of the patients from their clinical features. 

PROJECT_NAME = Exploratory Data Analysis of the Heart Failure Clinical Records

# make all
ifeq ($(OS),Windows_NT) 
    py_exe := python
else
    py_exe := python3
endif

all: data/raw/heart_failure_clinical_records_dataset.csv data/processed/Heart_Failure_Data_train.csv data/processed/Heart_Failure_Data_test.csv \
	 results/model 

# Download Data
data/raw/heart_failure_clinical_records_dataset.csv : src/download_data.py
	$(py_exe) src/download_data.py --url="https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv" --out_file="data/raw/heart_failure_clinical_records_dataset.csv"

# Data Clean and split
data/processed/Heart_Failure_Data_train.csv data/processed/Heart_Failure_Data_test.csv: src/clean_split_data.py data/raw/heart_failure_clinical_records_dataset.csv 
	$(py_exe) src/clean_split_data.py --raw_data="data/raw/heart_failure_clinical_records_dataset.csv" --out_dir="data/processed"

##  Model building
results/model: data/processed/Heart_Failure_Data_train.csv data/processed/Heart_Failure_Data_test.csv
	$(py_exe) src/model_train.py --data_file="data/processed/Heart_Failure_Data_train.csv" --out_dir="results/model"
	$(py_exe) src/model_test.py --data_file="data/processed/Heart_Failure_Data_test.csv" --out_dir="results/model"


#Format
format:
	black src

#Clean
clean:
	rm -rf data/raw/*
	rm -rf data/processed/*
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


help: 
	@echo "usage: make [target] ..."
	@echo ""
