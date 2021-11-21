# Exploratory Data Analysis for the Heart Failure Clinical Records

- author: Riddhi Sansare

A data analysis project for DSCI 522 (Data Science workflows); a
course in the Master of Data Science program at the University of
British Columbia

## Usage

To download the data, run the script below.

```bash
python src/download_data.py --url="https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv" --outputfile="data/raw/heart_failure_clinical_records_dataset.csv"
```

## Project Proposal

Last updated: Nov 20th, 2021

- **About**

   Cardiovascular diseases kill millions of people globally every year, and they mainly exhibit as myocardial infarctions and heart failures.When the heart cannot pump enough blood to meet the needs of the body , chances of  Heart failure are extremely high.We can use the available medical data of patients quantify symptoms, body features, and clinical laboratory test values to perform statistical analysis aimed at highlighting patterns and correlations which could be undetectable by medical doctors. Machine learning can predict patients’ survival from their data and can characterize the most important features among those included in their medical records.
   The heart_failure_clinical_records data set used in this project was sourced from the UC Irvine Machine Learning Repository published in 2020. It can be found [here]("https://archive-beta.ics.uci.edu/ml/datasets/heart+failure+clinical+records"). This dataset contains the medical records of 299 patients who had heart failure, collected during their follow-up period. Each patient profile has 13 clinical features and our target feature is "death event".

- **Predictive Research Questions**

    In this project we are aiming explore the available data of the patients and analyze the features of importance.Carry out a binary classification to predict the survival of each patient having heart failure symptoms and to detect the important clinical features that may lead to heart failure.

- **Preliminary Analysis Plan**

  - Data download

  - Exploratory Data Analysis on the features

  - Data preprocessing

  - Classification- Using the best suitable classification model to predict the target.

  - Final report on the results

- ## License

    This dataset is licensed under a Creative Commons Attribution
    4.0 International (CC BY 4.0) license. This allows for the
    sharing and adaptation of the datasets for any purpose, provided
    that the appropriate credit is given.

## Source

[Link to the dataset](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records)

The original dataset version was collected by Tanvir Ahmad, Assia Munir, Sajjad Haider Bhatti, Muhammad Aftab, and Muhammad Ali Raza (Government College University, Faisalabad, Pakistan) and made available by them on FigShare under the Attribution 4.0 International (CC BY 4.0: freedom to share and adapt the material) copyright in July 2017.

The current version of the dataset was elaborated by Davide Chicco (Krembil Research Institute, Toronto, Canada) and donated to the University of California Irvine Machine Learning Repository under the same Attribution 4.0 International (CC BY 4.0) copyright in January 2020.

Davide Chicco, Giuseppe Jurman: "Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone". BMC Medical Informatics and Decision Making 20, 16 (2020). [Web Link](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5#citeas)
</div>

</div>
