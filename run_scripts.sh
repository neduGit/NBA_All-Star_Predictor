#!/bin/bash

# extract data
echo "Scraping data from BasketballReference"
python /scripts/data_extraction.py
if [ $? -ne 0 ]; then
    echo "data_extraction.py failed. Exiting."
    exit 1
fi

# process data for training
echo "Cleaning and processing raw data"
python /scripts/data_processing.py
if [ $? -ne 0 ]; then
    echo "data_processing.py failed. Exiting."
    exit 1
fi

# train-test split on the processed data
echo "Train-test splitting"
python /scripts/train_test_split.py
if [ $? -ne 0 ]; then
    echo "train_test_split.py failed. Exiting."
    exit 1
fi

# decision_trees model
echo "Training and evaluating decision trees model"
python /scripts/decision_trees.py
if [ $? -ne 0 ]; then
    echo "decision_trees.py failed. Exiting."
    exit 1
fi

# neural network
echo "Training and evaluating neural network model"
python /scripts/mlpnn.py
if [ $? -ne 0 ]; then
    echo "mlpnn.py failed. Exiting."
    exit 1
fi

echo "All scripts executed"