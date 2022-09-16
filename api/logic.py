import numpy as np
import pandas as pd


def data_loader(file_path: str, col_name: str):
    output = pd.read_csv(file_path, delimiter="\t", header=None, names=col_name)
    return output
