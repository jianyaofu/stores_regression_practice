#!/usr/bin/env python
"""
Splits dataset into training set and evaluating set based on training fraction.
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def data_split(file_name, train_fraction=0.7):
    """
    Args:
        file_name: string; name of the dataset file
        train_fraction: float between 0.0 and 1.0, the proportion of the dataset in include in the training set
    Returns:
        None; creates training and evaluating sets in ./data/
    """
    file_path = os.path.join(os.getcwd(), file_name)
    df = pd.read_csv(file_path)
    train_df, eval_df = train_test_split(df, train_size=train_fraction)
    target_dir = os.path.join(os.getcwd(), "data")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    train_df.to_csv(os.path.join(target_dir, "train_data.csv"), index=False, header=False)
    eval_df.to_csv(os.path.join(target_dir, "eval_data.csv"), index=False, header=False)

    return

def main():
    # take inputs
    file_name, train_fraction = sys.argv[1], float(sys.argv[2])

    data_split(file_name, train_fraction)

if __name__ == '__main__':
    main()