__author__ = 'Chenliang Huang'

import pandas as pd

def load(filepath, cols = None, **kw):
    ftype = filepath.split('.')[-1]
    if ftype == 'csv':
        try:
            if cols:
                df = pd.read_csv(filepath, usecols=cols)
            else:
                df = pd.read_csv(filepath)
        except MemoryError as e:
            print(e)
        return df

#test
# filepath = r'..\twitter_disaster\data\train_cleaned.csv'
# cols = None
# df = load(filepath)