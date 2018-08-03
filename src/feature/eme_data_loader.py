from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


def one_hotte(df):
    target_columns = [c for c in df.columns
                      if "ethnicity_" in c or
                         "has_children_" in c or
                         "faith_" in c or
                         "drink_" in c]
    df[target_columns] = df[target_columns].astype(str)
    df = pd.get_dummies(df, columns=target_columns, drop_first=True)
    return df


def get_id_columns(df):
    user_and_target_id_columns = ["user_id", "target_user_id"]
    return df[user_and_target_id_columns]


def get_ethnicity_columns(df):
    ethnicity_user = df.ethnicity_user
    ethnicity_target = df.ethnicity_target
    ethnicity_columns = [c for c in df.columns if "ethnicity_" in c]
    df.drop(ethnicity_columns, axis=1, inplace=True)
    df = df.assign(ethnicity_user=ethnicity_user,
                   ethnicity_target=ethnicity_target)
    return df


def drop_raws(df):
    at_columns = [c for c in df.columns if "_at" in c]
    distance_columns = [c for c in df.columns if "_distance" in c]
    is_columns = [c for c in df.columns if "is_" in c]
    has_columns = [c for c in df.columns
                   if "has_" in c and "has_children_user" != c]

    # exclude asian_user and asian_target because these are all 0
    # exclude income_user and income_target because these are almost all Null
    drop_targets = ["income_user", "income_target", "label"]
    drop_targets += at_columns
    drop_targets += distance_columns
    drop_targets += is_columns
    drop_targets += has_columns
    df.drop(drop_targets, axis=1, inplace=True)
    return df


def calculate_user_features(df):
    c_id = 'user_id'
    user_feature_columns = [c for c in df.columns
                            if '_user' in c and 'target_user_id' != c]
    user_features = df.groupby(c_id)[user_feature_columns].head(1)
    user_features[c_id] = df.loc[user_features.index].user_id
    # user_features.drop(c_id, axis=1, inplace=True)
    return one_hotte(user_features)


def calculate_target_features(df):
    c_id = 'target_user_id'
    target_feature_columns =\
        [c for c in df.columns.values if '_target' in c]
    target_features =\
        df.groupby(c_id)[target_feature_columns].head(1)
    target_features[c_id] =\
        df.loc[target_features.index].target_user_id
    # target_features.drop(c_id, axis=1, inplace=True)
    return one_hotte(target_features)


class OwnDataset(Dataset):
    def __init__(self, file_name, root_dir, subset=False, transform=None):
        super().__init__()
        self.file_name = file_name
        self.root_dir = root_dir
        self.transform = transform
        self.prepare_data()
        self.user_features_orig = self.user_features

    def __len__(self):
        return len(self.user_and_target_ids)

    def reset(self):
        self.user_features = self.user_features_orig

    def prepare_data(self):
        data_path = Path(self.root_dir, self.file_name)
        self.eme_data = pd.read_csv(data_path)

        self.user_and_target_ids = get_id_columns(self.eme_data)
        self.rewards = self.eme_data.label.astype(int)

        self.eme_data = get_ethnicity_columns(self.eme_data)
        self.eme_data = drop_raws(self.eme_data)

        self.user_features = calculate_user_features(self.eme_data)
        print(self.user_features.columns.values)
        self.target_features = calculate_target_features(self.eme_data)

    def __getitem__(self, idx):
        ids = self.user_and_target_ids.iloc[idx].values
        user_id, target_id = ids[0], ids[1]
        # reward = self.rewards.iloc[idx]

        user_feature = self.user_features[self.user_features.user_id == user_id]
        user_feature =\
            user_feature.copy().drop("user_id", axis=1).astype(np.float32).values

        target_feature =\
            self.target_features[self.target_features.target_user_id==target_id]
        target_feature =\
            target_feature.copy().drop("target_user_id", axis=1).astype(np.float32).values

        return torch.FloatTensor(user_feature)  #, torch.FloatTensor(target_feature)

    def get_reward(self, user_id, target_id):
        query_user = self.user_and_target_ids.user_id==user_id
        query_target = self.user_and_target_ids.target_user_id==target_id
        query = query_user & query_target
        idx = self.user_and_target_ids[query].index
        return self.rewards.loc[idx].values[0]

def loader(dataset, batch_size, shuffle=True):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4)
    return loader
