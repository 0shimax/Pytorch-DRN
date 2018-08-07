import pandas as pd
import numpy as np


def add_label(df):
    tmp = df[["yes_at", "smiled_at", "messaged_at"]]
    labels = []
    for y, s, m in tmp.values:
        if y is not np.nan or s is not np.nan or m is not np.nan:
            labels.append(1)
        else:
            labels.append(0)
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
    drop_targets = ["age_arrived_user", "age_arrived_target",
                    "country_target", "city_target", "state_target",
                    "country_user", "city_user", "state_user"]
    drop_targets += at_columns
    drop_targets += distance_columns
    drop_targets += is_columns
    drop_targets += has_columns
    df.drop(drop_targets, axis=1, inplace=True)
    return df


def main(in_file_path, out_file_path):
    ny_data = pd.read_csv(in_file_path)
    f = add_label(ny_data.copy())
    f = drop_raws(f)
    f = one_hotte(f)
    f = f.fillna(f.median())

    f.to_csv(out_file_path, index=False)


if __name__=='__main__':
    in_path = "./raw/eme_interactions_June-JULY2018_NY.csv"
    out_path = "./raw/eme_interactsions_June-July2018_NY_features.csv"
    main(in_path, out_path)
