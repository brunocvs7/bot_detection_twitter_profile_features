import pandas as pd

def create_tabular_data(dataset):
    df = pd.DataFrame.from_records(pd.DataFrame.from_records(dataset)['profile'])
    return df

def build_features(df):
    list_columns_colors = df.filter(regex='color').columns.tolist()
    df = df.replace({'false':'FALSE', 'true':'TRUE', False:'FALSE', True:'TRUE'})
    df['name'] = df['name'].apply(lambda x: len(x) if x is not np.nan else 0)
    df['profile_location'] = df['profile_location'].apply(lambda x: 'TRUE' if x is not np.nan else 'FALSE')
    df['rate_friends_followers'] = df['friends_count']/df['followers_count']
    df['rate_friends_followers'] = df['rate_friends_followers'].map({np.inf:0, np.nan:0})
    df['unique_colors'] = df[list_columns_colors].stack().groupby(level=0).nunique()
    return df