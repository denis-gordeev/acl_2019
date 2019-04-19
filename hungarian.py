import numpy as np
import pandas as pd

from nltk import word_tokenize
from scipy.optimize import linear_sum_assignment

import networkx as nx
from node2vec import Node2Vec

from tax_utils import read_nigp_csv_pd, get_averaged_vectors, read_okpd_pd
from tax_utils import matrices_cosine_distance, get_top_indices
from utils import annotate, load_embeddings


def get_vectors_from_df(df, lang, class_id):
    u = df[df["lang"] == lang][f"class{class_id}_vectors"].dropna()
    index = u.index
    u = u.values
    u = np.array([np.array(l) for l in u])
    return u, index


def get_vectors_from_name(name_split: pd.DataFrame, model):
    name_split = name_split.str.lower()
    name_split = name_split.apply(lambda x: word_tokenize(x))
    vectors = get_averaged_vectors(model, name_split)
    return vectors


def get_bottomup_vectors(taxonomy, i, model):
    all_codes = taxonomy[f"class{i}_code"].drop_duplicates().values
    bottom_code = "class{}_code".format(i + 1)
    bottom_vectors = "class{}_vectors".format(i + 1)
    level_code = f"class{i}_code"
    prev_code = f"class{i + 1}_code"
    level_name = f"class{i}_name"
    level_vectors = f"class{i}_vectors"
    taxonomy[level_vectors] = None
    taxonomy[level_vectors] = taxonomy[level_vectors].astype('object')
    for code_i, code in enumerate(all_codes):
        try:
            if type(code) != str:
                continue
            print("\t", code_i, len(all_codes), end="\r")
            # select all vectors of the bottom layer starting with the code
            # and average them; 10.02 <- average(10.02.21, 10.02.34)
            index = taxonomy[bottom_code].str.startswith(code, na=False)
            vectors = taxonomy.loc[index, bottom_vectors].dropna().values
            code_index = (taxonomy[level_code] == code)
            # &\ (taxonomy[prev_code].isna())
            try:
                code_index = taxonomy[code_index].index
            except Exception as ex:
                print(ex)
                continue
            if len(code_index) == 0:
                print(code)
                continue
            code_index = code_index[:1]
            if type(vectors) == list:
                vectors = np.array([np.array(v) for v in vectors])
            if vectors.shape[0] == 0:
                vectors = get_vectors_from_name(
                    taxonomy.loc[code_index, level_name], model)
                vectors = [v for v in vectors if not np.isnan(v).any()]
                vectors = np.array([np.array(v) for v in vectors])
                if len(vectors) == 0:
                    continue
            # if vectors.shape[0] == 1:
            #     vectors = vectors[0]
            # elif vectors.shape[0] == 0:
            #     vectors = np.nan
            vectors = np.array([np.array(v) for v in vectors])
            if vectors.shape[0] != 300:
                vectors = np.mean(vectors, axis=0)
            # where column "class{i}_code" == code
            # column class{i}_vectors = vectors
            # if type(vectors) != np.ndarray or vectors.shape[0] == 0 or\
            #         np.isnan(vectors).any():
            #     vectors = get_vectors_from_name(
            #         taxonomy.loc[code_index, level_name], model)
            # vectors = tuple([tuple(v) for v in vectors])
            # if len(vectors) != len(code_index):
            #     vectors = [vectors] * len(code_index)
            for c in code_index:
                taxonomy.loc[c, level_vectors] = [vectors]
        except Exception as ex:
            print(ex)
            continue
    return taxonomy


def get_taxonomy_level_vectors(taxonomy, i, model):
    class_taxonomy = taxonomy[[
        f"class{i}_name", f"class{i}_code"]].dropna().drop_duplicates()
    name_split = class_taxonomy[f"class{i}_name"]
    vectors = get_vectors_from_name(name_split, model)
    class_taxonomy[f"class{i}_vectors"] = list(vectors)
    class_taxonomy = class_taxonomy[f"class{i}_vectors"]
    taxonomy = pd.merge(
        taxonomy, class_taxonomy,
        left_index=True, right_index=True, how="left")
    return taxonomy


def load_taxonomy(lang: str, vector_method="topdown"):
    """
    get vectors for each class of the corresponding language
    """
    print(f"loading model {lang}")
    model = load_embeddings(lang)
    if lang == "en":
        taxonomy = read_nigp_csv_pd()
        class_range = 2
    elif lang == "ru":
        taxonomy = read_okpd_pd()
        class_range = 4
    if vector_method == "topdown":
        for i in range(class_range):
            print(i)
            # different categories may have the same name but different codes
            taxonomy = get_taxonomy_level_vectors(taxonomy, i, model)
    else:
        # range(3, -1, -1) -> [3, 2, 1, 0]
        for i in range(class_range - 1, -1, -1):
            print(i)
            if i == class_range - 1:
                taxonomy = get_taxonomy_level_vectors(taxonomy, i, model)
            else:
                taxonomy = get_bottomup_vectors(taxonomy, i, model)
    return taxonomy


def add_graph_vec_to_df(graph_df):
    graph_df = graph_df.reset_index(drop=True)
    graph = nx.Graph()
    # col_names vary only in ints 'class0_code', 'class1_code', 'class2_code'

    # the copy will be local only
    # but if use global df while debugging in the console
    cols = [col for col in graph_df.columns if "_code" in col]
    cols = sorted(cols)
    for col_i in range(len(cols) - 1):
        col = cols[col_i]
        col_next = cols[col_i + 1]
        print(col, col_next)
        all_nodes = set()
        for this_col in [col, col_next]:
            nodes = graph_df[this_col].drop_duplicates().dropna().values
            nodes = set(nodes)
            all_nodes.update(nodes)
        graph.add_nodes_from(all_nodes)
        edges = graph_df[[col, col_next]].values
        edges = [list(e) for e in edges]
        graph.add_edges_from(edges)

    node2vec = Node2Vec(
        graph, dimensions=300, walk_length=10, num_walks=10, workers=1)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    print(graph_df.columns)
    for col_i, col in enumerate(cols):
        vector_col = f"class{col_i}_vectors"
        code_col = f"class{col_i}_code"
        index = graph_df[vector_col].dropna().index
        for ind_i, ind in enumerate(index):
            print("\t", col, ind_i, end="\r")
            row = graph_df.loc[ind]
            if len(row.shape) > 1:
                row = row[:1]
            code = row[code_col]
            vector = row[vector_col]
            graph_vec = model[code]
            graph_df.loc[ind, vector_col] = vector + graph_vec
    return graph_df


def main():
    vector_methods = ["topdown", "bottomup"]
    algorithms = ["hungarian", "greedy"]
    done_vector_methods = ["bottomup"]
    for v_m in vector_methods:
        print(v_m)
        if v_m in done_vector_methods:
            print(v_m, "done")
            continue
        df = pd.DataFrame()
        langs = ["ru", "en"]
        for lang in langs:
            print(lang)
            taxonomy = load_taxonomy(lang, v_m)
            taxonomy["lang"] = lang
            taxonomy = add_graph_vec_to_df(taxonomy)
            df = df.append(taxonomy)
            df = df.reset_index(drop=True)

        for a in algorithms:
            print(a)
            match_df = None
            for level in range(2):
                vec_dict = dict()
                for lang in langs:
                    u, u_index = get_vectors_from_df(df, lang, level)
                    vec_dict[lang] = dict()
                    vec_dict[lang]["vectors"] = u
                    vec_dict[lang]["index"] = u_index

                if len(vec_dict[langs[0]]["vectors"]) >\
                        len(vec_dict[langs[1]]["vectors"]):
                    # reverse_langs
                    langs = langs[::-1]

                # Matches only for the smaller side of the matrix
                # if n != m; the output will be of the shape
                # min(n, m) * 1
                distance = matrices_cosine_distance(
                    vec_dict[langs[0]]["vectors"],
                    vec_dict[langs[1]]["vectors"],
                    get_dist=True)
                print(distance.shape)
                if level != 0:
                    distance = mask_distance_matrix(
                        distance, vec_dict, match_df, df, langs, level)

                if a == "hungarian":
                    row_ind, col_ind = linear_sum_assignment(distance)
                else:
                    col_ind = get_top_indices(distance)
                    row_ind = range(col_ind.shape[0])
                top_distance = distance[row_ind, col_ind]
                print("cumulative distance", np.sum(top_distance))

                value_index = vec_dict[langs[0]]["index"]
                sort_index = vec_dict[langs[1]]["index"]
                sort_index = sort_index[col_ind]
                indices = [value_index, sort_index]
                match_df = pd.DataFrame()
                for lang_i, lang in enumerate(langs):
                    for key in ["name", "code"]:
                        match_df[f"{lang}_{key}"] = df.loc[
                            indices[lang_i]][f"class{level}_{key}"].values
                match_df["score"] = top_distance
                # if level not in done_levels:
                match_df = annotate(match_df, a, level, v_m)


def mask_distance_matrix(dist_matrix, vec_dict, match_df, df, langs, level):
    index = vec_dict[langs[0]]["index"]
    value_index = vec_dict[langs[1]]["index"]
    for ind_i, ind in enumerate(index):
        print("\t", ind_i, end="\r")
        key_code = df.loc[ind]["class{}_code".format(level - 1)]
        key_code = key_code
        value_code = match_df[match_df[f"{langs[0]}_code"] == key_code]
        value_code = value_code[f"{langs[1]}_code"].values[0]
        mask = df[(df["lang"] == "en") &
                  (df["class{}_code".format(level - 1)] == value_code)]
        mask = mask.index
        mask = [vec not in mask for vec_j, vec in enumerate(value_index)]
        dist_matrix[ind_i][mask] = 1
    return dist_matrix
