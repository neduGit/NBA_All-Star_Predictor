# NBA All Star Predictor

Using a scraped data from [basketball-reference.com](https://www.basketball-reference.com) to create machine learning models to predict whether an NBA player makes an all-star nomination based on seasonal statistical performances. The data used is from the 1980-81 season to the 2023-24 season.

## Required Packages
These are the required Python packages to run the scripts:

- Matplotlib
- NumPy
- Pandas
- Scikit-learn (Sklearn)

##### Anaconda Installation

```
conda install matplotlib numpy pandas scikit-learn
```

##### pip Installation

```
pip install matplotlib numpy pandas scikit-learn
```

##### Data Scraper
A package for data scraping is required for `data_extraction.py` that scrapes the data from BasketballReference. It is only available through pip.

```
pip install basketball-reference-scraper
```

## Repo Contents:
 - `Evaluations` Performance metrics and confusion matrices of various models
 - `Scripts` Scripts to extract and transform data, run models
 - `Data` Processed training and test split data

## How to Use
First extract data from basketball reference by using a data scraping script, `data_extraction.py`. It takes around 6 hours for all the data to run. As such, the data is also provided in the `training_data` folder for easy access.

Prepare data for model training by running `data_processing.py` and `train_test_split.py`. The scripts clean and transform the data to a format ready for model training.

Train models and obtain evaluation metrics and confusion matrices on `decision_trees.py` and `mlpnn.py`. Both files are independent of each other.

A bash script has been compiled to run all the scripts efficiently. If you are on a Windows device ensure you have WSL or Git Bash.
