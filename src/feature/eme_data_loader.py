from pathlib import Path
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def get_id_columns(df):
    user_and_target_id_columns = ["user_id", "target_user_id"]
    return df[user_and_target_id_columns]


def extranct_interacted_user_rows(df):
    tmp = df[["user_id", "label"]].groupby('user_id').sum()
    interacted_user_id = tmp[tmp.label>0].reset_index()
    return df[df.user_id.isin(interacted_user_id.user_id)]


def get_ethnicity_columns(df):
    ethnicity_user = df.ethnicity_user
    ethnicity_target = df.ethnicity_target
    ethnicity_columns = [c for c in df.columns if "ethnicity_" in c]
    df.drop(ethnicity_columns, axis=1, inplace=True)
    df = df.assign(ethnicity_user=ethnicity_user,
                   ethnicity_target=ethnicity_target)
    return df


def calculate_user_features(df):
    c_id = 'user_id'
    user_feature_columns = [c for c in df.columns
                            if '_user' in c and 'target_user_id' != c]
    user_features = df.groupby(c_id)[user_feature_columns].head(1)
    user_features[c_id] = df.loc[user_features.index].user_id
    return user_features


def calculate_target_features(df):
    c_id = 'target_user_id'
    target_feature_columns =\
        [c for c in df.columns.values if '_target' in c]
    target_features = df[[c_id] + target_feature_columns]

    return target_features


def calcurate_target_clicked(df):
    result = df[['target_user_id', 'label']]\
        .groupby('target_user_id')\
        .agg(['sum', 'count'])\
        .reset_index()
    result.columns = ['target_user_id', 'label_sum', 'label_cnt']
    result = result.assign(label_rate=result.label_sum/result.label_cnt)
    result.index = df.groupby('target_user_id').head(1).index
    return result


def get_target_ids_for_train_input(squewed_user_target_labels,
                                   valued_target_idxs, n_high, n_low):
    # 全て返す
    return squewed_user_target_labels.index.values

    n_total = n_high + n_low
    high_rate_flag = squewed_user_target_labels.label > 0
    if len(valued_target_idxs) >= n_total:
        idxs = np.random.permutation(len(valued_target_idxs))[:n_total]
        return valued_target_idxs[idxs]

    query = ~squewed_user_target_labels.index.isin(valued_target_idxs)
    query &= high_rate_flag
    n_rest = n_total - len(valued_target_idxs)
    if n_rest == 1:
        hight = squewed_user_target_labels[query].sample(n_rest).index.values
        return np.concatenate([valued_target_idxs, hight])

    m_n_high = int(n_rest * n_high / n_total)
    m_n_low = n_rest - m_n_high
    hight = squewed_user_target_labels[query].sample(m_n_high, replace=True).index.values
    low = squewed_user_target_labels[
        squewed_user_target_labels.label == 0].sample(m_n_low, replace=True).index.values
    idxs = np.concatenate([valued_target_idxs, hight, low])
    return idxs


def get_target_ids_for_test_input(squewed_user_target_labels, n_high, n_low):
    # 全て返す
    return squewed_user_target_labels.index.values

    n_total = n_high + n_low
    high_rate_flag = squewed_user_target_labels.label > 0

    if sum(high_rate_flag) < n_high:
        hight = squewed_user_target_labels[high_rate_flag].index.values
        n_low = n_total - sum(high_rate_flag)
    else:
        hight = squewed_user_target_labels[high_rate_flag].sample(n_high).index.values
    low = squewed_user_target_labels[
        squewed_user_target_labels.label == 0].sample(n_low, replace=True).index.values
    idxs = np.concatenate([hight, low])
    return idxs


def get_target_ids_for_input(squewed_user_target_labels,
                             valued_target_idxs, n_high, n_low, train=True):
    if train:
        return get_target_ids_for_train_input(squewed_user_target_labels, valued_target_idxs, n_high, n_low)
    else:
        return get_target_ids_for_test_input(squewed_user_target_labels, n_high, n_low)


class OwnDataset(Dataset):
    def __init__(self, file_name, root_dir, n_high, n_low,
                 subset=False, transform=None, train=True, split_seed=555):
        super().__init__()
        print("Train:", train)
        self.file_name = file_name
        self.root_dir = root_dir
        self.transform = transform
        self.n_high = n_high
        self.n_low = n_low
        self._train = train
        self.split_seed = split_seed
        self.prepare_data()
        self.user_features_orig = self.user_features

    def __len__(self):
        return len(self.user_and_target_ids)

    def reset(self):
        self.user_features = self.user_features_orig

    def prepare_data(self):
        data_path = Path(self.root_dir, self.file_name)
        eme_data = pd.read_csv(data_path)

        extracted_interacted_rows = extranct_interacted_user_rows(eme_data)
        unique_user_ids = extracted_interacted_rows.user_id.unique()

        train_user_ids, test_user_ids = train_test_split(unique_user_ids,
                                                         random_state=self.split_seed,
                                                         shuffle=True,
                                                         test_size=0.2)
        if self._train:
            _data = eme_data[eme_data.user_id.isin(train_user_ids)]
            self.user_features = calculate_user_features(_data)
            self.user_and_target_ids = get_id_columns(_data)

            self.rewards = eme_data[["user_id", "target_user_id", "label"]]
            self.target_features_all = calculate_target_features(eme_data)  # _data
        else:
            _data = eme_data[eme_data.user_id.isin(test_user_ids)]
            self.user_and_target_ids = get_id_columns(_data)
            self.user_features = calculate_user_features(_data)

            self.rewards = eme_data[["user_id", "target_user_id", "label"]]
            self.target_features_all = calculate_target_features(eme_data)

        print("user", self.user_features.shape)
        print("target", len(self.target_features_all.target_user_id.unique()))

    def __getitem__(self, idx):
        ids = self.user_and_target_ids.iloc[idx].values
        current_user_id = ids[0]

        user_feature = self.user_features[self.user_features.user_id == current_user_id]
        user_feature = user_feature.copy().drop("user_id", axis=1)
        user_feature = user_feature.astype(np.float32).values
        user_feature = user_feature.reshape(-1)

        query = (self.rewards.user_id == current_user_id)
        query &= (self.rewards.label == 1)
        valued_target_idxs = self.rewards[query].index.values

        # TODO: 後で名前変えたる
        squewed_user_target_labels =\
            self.rewards.groupby("target_user_id").head(1)
        target_idxs = get_target_ids_for_input(
            squewed_user_target_labels, valued_target_idxs,
            self.n_high, self.n_low, self._train)
        target_features = self.target_features_all.loc[target_idxs].copy().reindex()
        target_ids = target_features.target_user_id.values

        target_features =\
            target_features.copy().drop("target_user_id", axis=1)
        target_features = target_features.astype(np.float32).values

        eliminate_teacher = self.target_features_all.loc[valued_target_idxs].copy().reindex()
        eliminate_teacher_ids = eliminate_teacher.target_user_id.values
        eliminate_teacher_val = target_ids == eliminate_teacher_ids[0]
        for v in eliminate_teacher_ids[1:]:
            eliminate_teacher_val += target_ids == v
        eliminate_teacher_val = eliminate_teacher_val.astype(np.float32)

        return (torch.FloatTensor(user_feature),
                torch.FloatTensor(target_features),
                current_user_id,
                target_ids,
                eliminate_teacher_val)

    def get_reward(self, current_user_id, target_ids):
        query_user = self.rewards.user_id == current_user_id
        query_target = self.rewards.target_user_id.isin(target_ids)
        query = (query_user) & (query_target)

        reward = self.rewards[query].label.values
        if len(reward) == 0:
            return 0.
        else:
            return float(reward.max())


def loader(dataset, batch_size, shuffle=True):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0)
    return loader
