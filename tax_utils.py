import codecs
import csv
import numpy as np
import pandas as pd

from gensim.models import KeyedVectors


def load_okpd_dict():
    f = codecs.open("original_data/okpd2.csv", "r", "cp1251")
    reader = csv.reader(f, delimiter=";")
    okpd_dict = {l[3]: l[1] for l in reader}
    return okpd_dict


def read_nigp_csv():
    maryland = dict()
    years = list(range(2012, 2018))
    for year in years:
        df = pd.read_csv(
            "original_data/"
            f"eMaryland_Marketplace_Bids_-_Fiscal_Year_{year}.csv")
        class_descriptions = df["NIGP Class with Description"].\
            str.replace("^[0-9]+ - ", "").str.strip() + " "
        item_descriptions = df["NIGP Class Item with Description"].\
            str.replace("^[0-9]+ - \d+ :", "").str.strip()
        # class_descriptions = class_descriptions + item_descriptions
        class_descriptions = class_descriptions + "::" + item_descriptions
        codes = df["NIGP 5 digit code"]
        # codes = df['NIGP Class']
        codes_dict = dict(zip(codes, class_descriptions))
        if year == 2012:
            maryland = codes_dict
        else:
            maryland.update(codes_dict)
    return maryland


def read_nigp_csv_pd():
    years = list(range(2012, 2018))
    combined_df = pd.DataFrame()
    for year in years:
        df = pd.read_csv(
            "original_data/"
            f"eMaryland_Marketplace_Bids_-_Fiscal_Year_{year}.csv")
        df = df[["NIGP Class with Description",
                 "NIGP Class Item with Description"]]
        df = df.drop_duplicates()

        nigp_class = df["NIGP Class with Description"].str.split(" - ")
        df["class0_code"] = nigp_class.str[0]
        df["class0_name"] = nigp_class.str[1]

        nigp_item = df[
            "NIGP Class Item with Description"].str.split(" - ").str[1]
        nigp_item = nigp_item.str.split(" : ")
        df["class1_code"] = nigp_item.str[0]
        df["class1_name"] = nigp_item.str[1]
        df = df[~df["class0_code"].isna()]
        df["class1_code"] = df["class0_code"] + "." + df["class1_code"]
        # df["class1_name"] = df["class1_name"].fillna("")
        columns = [col for col in df.columns if not col.startswith("class")]
        df.drop(columns, inplace=True, axis=1)
        combined_df = combined_df.append(df)
    combined_df = combined_df.drop_duplicates()
    combined_df = combined_df.reindex()
    combined_df.to_csv("nigp5_taxonomy.csv", index=None)
    return combined_df


def read_okpd_pd():
    okpd_dict = load_okpd_dict()
    df = pd.read_csv(
        "original_data/okpd2.csv", delimiter=";", encoding="cp1251")
    df = df[["Name", "Kod"]]
    df = df[~df["Kod"].isna()]
    df = df.drop_duplicates()
    codes = df["Kod"].str.split(".")
    for i in range(4):
        level_codes = codes[codes.str.len() > i]
        level_codes = level_codes.str[:i + 1].str.join(".")
        df[f"class{i}_code"] = level_codes
        # df[df["Kod"] == level_codes]["Name"].values[0]
        class_names = [okpd_dict[l] for l in level_codes if l]
        class_names = pd.Series(class_names, index=level_codes.index)
        df[f"class{i}_name"] = class_names
    columns = [col for col in df.columns if not col.startswith("class")]
    df.drop(columns, inplace=True, axis=1)
    return df


def get_averaged_vectors(model: KeyedVectors, split_sents: list):
    sents_filtered = [[w for w in sent if w in model.vocab]
                      for sent in split_sents]
    sents_averaged = np.array([np.sum([model[w] for w in sent], axis=0)
                               for sent in sents_filtered])
    sents_lengths = np.array([len(m) for m in sents_filtered])
    sents_averaged = sents_averaged.T / sents_lengths
    sents_averaged = sents_averaged.T
    return sents_averaged


def matrices_cosine_distance(u: np.ndarray, v: np.ndarray, get_dist=False):
    """
    beware huge matrices!!!
    """
    dot = np.matmul(u, v.T)
    u_norm = np.linalg.norm(u, axis=1)[np.newaxis].T
    v_norm = np.linalg.norm(v, axis=1)[np.newaxis].T
    norm = np.matmul(u_norm, v_norm.T)
    similarity = dot / norm
    # get distance instead of similarity (bigger values = bigger differences)
    if get_dist:
        similarity = 1 - similarity
    return similarity


def get_top_indices(input_matrix, top: int = 1):
    # not to sort all lines
    top_args = np.argpartition(input_matrix, top, axis=1)
    top_args = top_args[:, :top]
    # to get a 1-d array
    if top == 1:
        top_args = top_args.T[0]
    # hungarian algorithm
    # row_ind, col_ind = linear_sum_assignment(distance)
    # np.sum(distance[row_ind, col_ind])

    return top_args
