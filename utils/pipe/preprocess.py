import pandas as pd

def create_tabular_data(dataset):
    df = pd.DataFrame.from_records(pd.DataFrame.from_records(dataset)['profile'])
    return df