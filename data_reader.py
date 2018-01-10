import pandas as pd


def read_data(file_path='result.cropped.tsv'):
    return pd.read_csv(file_path, sep='\t', dtype=str)
