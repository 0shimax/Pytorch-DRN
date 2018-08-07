from pathlib import Path
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def add_label(df):
    tmp = df[["yes_at", "smiled_at", "messaged_at"]]
    labels = []
    for y, s, m in tmp.values:
        if y!=y or s!=s or m!=m:
            labels.append(1)
        else:
            label = 1 if random.uniform(0, 1) > .5 else 0
            labels.append(label)
            # labels.append(0)
    return df.assign(label=labels)


def one_hotte(df):
    target_columns = [c for c in df.columns
                      if "has_children_" in c or
                         c == "asian_user" or
                         c == "asian_target" or
                         "smoke_" in c or
                         "drink_" in c]
    df[target_columns] = df[target_columns].astype(str)
    df = pd.get_dummies(df, columns=target_columns, drop_first=True)
    return df


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


def drop_raws(df):
    at_columns = [c for c in df.columns if "_at" in c
                  and c != "body_type_athletic_user"
                  and c != "body_type_athletic_target"]
    distance_columns = [c for c in df.columns if "_distance" in c]
    is_columns = [c for c in df.columns if "is_" in c]
    has_columns = [c for c in df.columns
                   if "has_" in c
                   and "has_children_user" != c
                   and "has_children_target" != c]

    # exclude asian_user and asian_target because these are all 0
    # exclude income_user and income_target because these are almost all Null
    drop_targets = ["label", "age_arrived_user", "age_arrived_target",
                    "country_target", "city_target", "state_target",
                    "country_user", "city_user", "state_user"]
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

    return one_hotte(target_features)


def calcurate_target_clicked(df):
    result = df[['target_user_id', 'label']]\
        .groupby('target_user_id')\
        .agg(['sum', 'count'])\
        .reset_index()
    result.columns = ['target_user_id', 'label_sum', 'label_cnt']
    result = result.assign(label_rate=result.label_sum/result.label_cnt)
    result.index = df.groupby('target_user_id').head(1).index
    return result


def get_target_ids_for_train_input(target_clicked_rate,
                                   valued_target_ids, n_high, n_low):
    n_total = n_high + n_low
    valued_target_flag = target_clicked_rate.target_user_id.isin(valued_target_ids)
    high_rate_flag = target_clicked_rate.label_rate > 0
    query = (valued_target_flag) & (high_rate_flag)
    valued_target_ids = target_clicked_rate[query].target_user_id.values
    if len(valued_target_ids) >= n_total:
        return valued_target_ids[:n_total]

    n_rest = n_total - len(valued_target_ids)
    m_n_high = int(n_rest * n_high / n_total)
    m_n_low = n_rest - m_n_high
    query = high_rate_flag & ~query
    hight = target_clicked_rate[query].sample(m_n_high).target_user_id.values
    low = target_clicked_rate[
        target_clicked_rate.label_rate == 0].sample(m_n_low).target_user_id.values
    ids = np.concatenate([valued_target_ids, hight, low])
    return ids


def get_target_ids_for_test_input(target_clicked_rate, n_high, n_low):
    n_total = n_high + n_low
    high_rate_flag = target_clicked_rate.label_rate > 0

    query = high_rate_flag
    hight = target_clicked_rate[query].sample(n_high).target_user_id.values
    low = target_clicked_rate[
        target_clicked_rate.label_rate == 0].sample(n_low).target_user_id.values
    ids = np.concatenate([hight, low])
    return ids


def get_target_ids_for_input(target_clicked_rate,
                             valued_target_ids, n_high, n_low, train=True):
    if train:
        return get_target_ids_for_train_input(target_clicked_rate, valued_target_ids, n_high, n_low)
    else:
        return get_target_ids_for_test_input(target_clicked_rate, n_high, n_low)


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
        eme_data = add_label(eme_data)

        extracted_interacted_rows = extranct_interacted_user_rows(eme_data)
        unique_user_ids = extracted_interacted_rows.user_id.unique()

        train_user_ids, test_user_ids = train_test_split(unique_user_ids,
                                                         random_state=self.split_seed,
                                                         shuffle=True,
                                                         test_size=0.2)
        if self._train:
            _data = eme_data[eme_data.user_id.isin(train_user_ids)]
            self.target_clicked_rate = calcurate_target_clicked(_data)
            self.user_and_target_ids = get_id_columns(_data)
            self.rewards = _data.label.astype(int)
            # _data = get_ethnicity_columns(_data)
            _data = drop_raws(_data)

            self.user_features = calculate_user_features(_data)
            self.user_features = self.user_features.fillna(self.user_features.median())

            self.target_features = calculate_target_features(_data)
            self.target_features = self.target_features.fillna(self.target_features.median())
        else:
            _data = eme_data[eme_data.user_id.isin(test_user_ids)]
            self.target_clicked_rate = calcurate_target_clicked(eme_data)
            self.user_and_target_ids = get_id_columns(_data)
            self.rewards = _data.label.astype(int)
            # _data = get_ethnicity_columns(_data)
            _data = drop_raws(_data)
            self.user_features = calculate_user_features(_data)
            self.user_features = self.user_features.fillna(self.user_features.median())

            # _eme_data = get_ethnicity_columns(eme_data)
            _eme_data = drop_raws(eme_data)
            self.target_features = calculate_target_features(_eme_data)
            self.target_features = self.target_features.fillna(self.target_features.median())

        # print(self.user_features.columns.values.tolist())
        # print(self.target_features.columns.values.tolist())
        # target_c = [c.replace('target', '') for c in self.target_features.columns.values.tolist()]
        # user_c = [c.replace('user', '') for c in self.user_features.columns.values.tolist()]
        # print([c for c in user_c if not c in target_c])
        # print([c for c in target_c if not c in user_c])
        # assert False

    def __getitem__(self, idx):
        ids = self.user_and_target_ids.iloc[idx].values
        current_user_id = ids[0]

        user_feature = self.user_features[self.user_features.user_id == current_user_id]
        user_feature = user_feature.copy().drop("user_id", axis=1)
        user_feature = user_feature.astype(np.float32).values
        user_feature = user_feature.reshape(-1)
        # print("user_feature", user_feature.shape)

        valued_target_ids =\
            self.user_and_target_ids[
                self.user_and_target_ids.user_id == current_user_id
            ].target_user_id.values
        target_ids = get_target_ids_for_input(
            self.target_clicked_rate, valued_target_ids,
            self.n_high, self.n_low, self._train)
        target_features = self.target_features[
            self.target_features.target_user_id.isin(target_ids)]

        target_features =\
            target_features.copy().drop("target_user_id", axis=1)
        target_features = target_features.astype(np.float32).values
        # print("target_features", target_features.shape)

        return (torch.FloatTensor(user_feature),
                torch.FloatTensor(target_features),
                current_user_id,
                target_ids)

    def get_reward(self, current_user_id, target_id):
        # target_id = self.target_features.target_user_id.iloc[target_action_id]

        # print("ids:", int(current_user_id), target_id)
        query_user = self.user_and_target_ids.user_id == current_user_id
        query_target = self.user_and_target_ids.target_user_id == target_id
        query = (query_user) & (query_target)

        idx = self.user_and_target_ids[query].index
        if len(idx) == 0:
            return 0.
        else:
            return float(self.rewards.loc[idx].values[0])

def loader(dataset, batch_size, shuffle=True):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4)
    return loader
