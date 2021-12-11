# Exploratory Data Analysis of the Heart Failure Clinical Records

- author: Riddhi Sansare

A data analysis project for DSCI 522 (Data Science workflows); a
course in the Master of Data Science program at the University of
British Columbia

## Usage

To set up the project locally please follow the steps :

```bash
#Clone the repository
git clone https://github.com/UBC-MDS/DSCI_522_Heart_Failure_Exploratory_Analysis.git
#set up the environment 
conda env create -f Heart_failure_environment.yml
conda activate Heart-failure
```

### To download the data, run the script below

```bash
python src/download_data.py --url="https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clini
cal_records_dataset.csv" --out_file="data/raw/heart_failure_clinical_records_dataset.csv"
```

### Data Cleaning

```bash
 python src/clean_split_data.py --raw_data="data/raw/heart_failure_clinical_records_dataset.csv" --out_dir="data/processed/"
```

### Model Building

The model building process can be accessed [here](https://github.com/UBC-MDS/DSCI_522_Heart_Failure_Exploratory_Analysis/blob/main/src/Heart_Failure_Model_Building.ipynb)

## Project Proposal

Last updated: Nov 20th, 2021

- **Introduction**

   Cardiovascular diseases kill millions of people globally every year, and they mainly exhibit as myocardial infarctions and heart failures.When the heart cannot pump enough blood to meet the needs of the body , chances of  Heart failure are extremely high.We can use the available medical data of patients quantify symptoms, body features, and clinical laboratory test values to perform statistical analysis aimed at highlighting patterns and correlations which could be undetectable by medical doctors. Machine learning can predict patients’ survival from their data and can characterize the most important features among those included in their medical records.
   The heart_failure_clinical_records data set used in this project was sourced from the UC Irvine Machine Learning Repository published in 2020. It can be found [here]("https://archive-beta.ics.uci.edu/ml/datasets/heart+failure+clinical+records"). This dataset contains the medical records of 299 patients who had heart failure, collected during their follow-up period. Each patient profile has 13 clinical features and our target feature is "death event".

- **Predictive Research Questions**

    In this project we are aiming to explore the available data of the patients and analyze the features of importance.Carry out a binary classification to predict the survival of each patient having heart failure symptoms and to detect the important clinical features that may lead to the heart failure.

- **Preliminary Analysis Plan**

  - Data download

  - Exploratory Data Analysis on the features

  - Data preprocessing

  - Classification- Using the best suitable classification model to predict the target.

  - Final report on the results

- ## Report

The report can be accessed [here](https://github.com/UBC-MDS/DSCI_522_Heart_Failure_Exploratory_Analysis/blob/main/doc/Report.pdf)

- ## License
  
This dataset is licensed under a[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode) (CC BY 4.0) license.

This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.

## Source

[Link to the dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00519/)

[The original dataset version was collected by Tanvir Ahmad, Assia Munir, Sajjad Haider Bhatti, Muhammad Aftab, and Muhammad Ali Raza (Government College University, Faisalabad, Pakistan) and made available by them on FigShare under the Attribution 4.0 International (CC BY 4.0: freedom to share and adapt the material) copyright in July 2017.](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records)

The current version of the dataset was elaborated by Davide Chicco (Krembil Research Institute, Toronto, Canada) and donated to the University of California Irvine Machine Learning Repository under the same Attribution 4.0 International (CC BY 4.0) copyright in January 2020.

## Dependencies

  The dependencies for this project are mentioned in the `Heart-failure.yml` environment file in the directory of this project

- ipykernel
- matplotlib>=3.2.2
- scikit-learn>=1.0
- pandas>=1.3.*
- python-graphviz
- pip
- altair>=4.1.0
- altair_data_server
- altair_saver
- docopt==0.6.2
- pandoc>=1.12.3
- seaborn
- R version 4.1.1 and R packages:
  - knitr==1.26
  - tidyverse==1.2.1

## Reference

Davide Chicco, Giuseppe Jurman: "Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone". BMC Medical Informatics and Decision Making 20, 16 (2020). [Web Link](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5#citeas)
