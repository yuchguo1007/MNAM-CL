import pandas as pd


class Movielens():
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

    def data_process(self, base_dir, nrows=None, sample_rate=0.2, columns=None):
        df = pd.read_csv(base_dir, nrows=nrows).sample(frac=sample_rate)
        df['rating_binary'] = df['rating'].apply(lambda x: 1 if x > 3 else 0)
        df['title_length'] = df['title'].apply(lambda x: len(x.split(' ')))
        df = df.sort_values(by='timestamp', ascending=True)
        print(df.shape)
        train = df.iloc[:int(len(df) * 0.8)].copy()
        print("train shape: ", train.shape)
        test = df.iloc[int(len(df) * 0.8):].copy()
        print("test shape: ", test.shape)
        return train, test, df

    def get_user_feature(self, data):
        data_group = data[data['rating'] == 1]
        data_group = data_group[['user_id', 'movie_id']].groupby('user_id').agg(list).reset_index()
        data_group['user_hist'] = data_group['movie_id'].apply(lambda x: '|'.join([str(i) for i in x]))

        data = pd.merge(data_group.drop('movie_id', axis=1), data, on='user_id')
        data_group = data[['user_id', 'rating']].groupby('user_id').agg('mean').reset_index()
        data_group.rename(columns={'rating': 'user_mean_rating'}, inplace=True)
        data = pd.merge(data_group, data, on='user_id')
        return data

    def get_item_feature(self, data):
        data_group = data[['movie_id', 'rating']].groupby('movie_id').agg('mean').reset_index()
        data_group.rename(columns={'rating': 'item_mean_rating'}, inplace=True)
        data = pd.merge(data_group, data, on='movie_id')
        return data

    def enrich_test_feature(self, test_data, feature_src):
        test_data = pd.merge(test_data, feature_src[['movie_id', 'item_mean_rating']].drop_duplicates(), on='movie_id',
                             how='left').fillna(3)
        test_data = pd.merge(test_data, feature_src[['user_id', 'user_mean_rating']].drop_duplicates(), on='user_id',
                             how='left').fillna(3)
        return test_data

