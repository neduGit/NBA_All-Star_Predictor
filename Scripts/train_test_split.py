"""
Performs train test split on player_data.csv
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv('processed_data/player_data.csv')

    save_dir = "training_data"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_df, test_df = train_test_split(df, test_size=0.3)

    # Drops extra column to not be used for training
    train_df.drop(columns=["Unnamed: 0"], inplace=True)
    test_df.drop(columns=["Unnamed: 0"], inplace=True)

    train_df.to_csv(os.path.join(save_dir, "train.csv"))
    test_df.to_csv(os.path.join(save_dir, "test.csv"))

    print(f"Data saved to {save_dir}")

if __name__ == "__main__":
    main()
