import os
import itertools
import random
import pickle
import json
import re
import time
# import multiprocessing as mp
import numpy as np
from scipy import spatial
import pandas as pd
import tensorflow as tf
from gensim.models import KeyedVectors
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer

from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras.callbacks import EarlyStopping

from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import pymorphy2

from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

import networkx as nx
from node2vec import Node2Vec

from utils import read_nigp_csv_pd, get_averaged_vectors, read_okpd_pd,\
    matrices_cosine_distance, get_top_indices
from utils_d.utils import text_pipeline
from utils_d.ml_models import CnnLstm, train_lgb
from utils_d.ml_utils import Ensemble, predict_binary
# from utils_d.ml_utils import create_word_index


def get_now():
    """
    convert datetime to string
    """
    now = datetime.now().isoformat()
    now = now.split(".")[0]
    now = now.replace(":", "_")
    return now


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


def get_vectors_from_name(name_split: pd.DataFrame, model):
    name_split = name_split.str.lower()
    name_split = name_split.apply(lambda x: word_tokenize(x))
    vectors = get_averaged_vectors(model, name_split)
    return vectors


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


def load_embeddings(lang):
    path = (f'muse_embeddings/wiki.multi.{lang}.vec')
    model = KeyedVectors.load_word2vec_format(path)
    print("loaded")
    return model


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


def get_vectors_from_df(df, lang, class_id):
    u = df[df["lang"] == lang][f"class{class_id}_vectors"].dropna()
    index = u.index
    u = u.values
    u = np.array([np.array(l) for l in u])
    return u, index


def annotate(
        match_df, algorithm, level, vector_method="topdown", limit=100):
    checking = []
    annotation_values = ["1", "0", "0.5", "skip"]
    limit = 100
    # try:
    #     match_df = pd.read_csv(
    #         f"match_df_{level}_{algorithm}_{vector_method}.csv")
    # except Exception as ex:
    #     print(ex)
    for value_i, value in enumerate(match_df.values):
        if value_i >= limit:
            break
        if value_i % 10 == 0:
            match_df.to_csv(
                f"match_df_{level}_{algorithm}_{vector_method}.csv")
        print(value_i, "out of", len(match_df.values))
        print(value)
        annotation = None
        while annotation not in annotation_values:
            try:
                annotation = input(
                    "{} :".format(" or ".join(annotation_values)))
            except Exception as ex:
                print(ex)
                annotation = "0"
            if annotation not in annotation_values:
                print("wrong value")
        if annotation != "skip":
            checking.append(annotation)
    if len(checking) < match_df.shape[0]:
        checking += [None] * (match_df.shape[0] - len(checking))
    match_df["check"] = checking
    match_df.to_csv(
        f"annotations/match_df_{level}_{algorithm}_{vector_method}_vec.csv")
    print("accuracy is", match_df.check.astype(float).sum())
    return match_df


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


def get_nouns_from_model(model):
    from rnnmorph.predictor import RNNMorphPredictor
    rnn_morph_predictor = RNNMorphPredictor(language="ru")
    nouns = list()
    for j, w in enumerate(model.vocab):
        print("\t", j, end="\r")
        parse = rnn_morph_predictor.predict([w])[0]
        if parse.pos == "NOUN" and\
                "Case=Nom" in parse.tag and "Number=Sing" in parse.tag:
            nouns.append(w)
    nouns = set(nouns)
    return nouns


def get_vectors_from_prev_synset(df, i, lang="ru"):
    if lang == "ru":
        suffix = "_ru"
    else:
        suffix = ""
    i_vecs = f"class{i}_vectors{suffix}"
    if i < 9:
        upper_vecs = f"class{i+1}_vectors{suffix}"
    else:
        upper_vecs = f"class{i+1}_vectors"
    i_synsets = f"class{i}_synsets"
    df[i_vecs] = None
    # df[f"class{i}_vectors{lang}"] = df[f"class{i}_vectors"].astype('object')
    unique_synsets = df[i_synsets].drop_duplicates().dropna()
    # unique_synsets = unique_synsets[unique_synsets != '']
    # unique_synsets = df[f"class{i}_synsets"].dropna()
    index = unique_synsets.index
    print("\n\n")
    for u_i, u_s in enumerate(unique_synsets):
        print("\t", u_i, end="\r")
        # this level synsets having this homonym
        # (being hypernyms of the prev level)
        vectors = df[df[i_synsets] == u_s]
        vectors = vectors[upper_vecs]
        vectors = vectors.dropna()
        vectors = np.mean(vectors)
        df.at[index[u_i], i_vecs] = vectors
    return df


def wordnet_to_df():
    model = load_embeddings("en")
    hyponyms = [w for w in wordnet.all_synsets()]
    hyponyms = [w for w in hyponyms if not w.hyponyms()]
    hyponyms = [w for w in hyponyms if w.hypernyms()]
    hyponyms = [w for w in hyponyms if ".n." in w.name()]
    hyponyms = [h for h in hyponyms if h.name().split(".")[0] in model.vocab]

    top_level = 10
    df = pd.DataFrame()
    df[f"class{top_level}_synsets"] = hyponyms
    names = [h.name().split(".")[0] for h in hyponyms]
    df[f"class{top_level}_name"] = names
    # get_averaged_vectors requires a list of strings
    df[f"class{top_level}_vectors"] = [model[n] for n in names]
    # get hyponyms for English wordnet
    for i in range(top_level - 1, -1, -1):
        print("\t", i, end="\r")
        hypernyms = [h.hypernyms() if h else None
                     for h in df[f"class{i+1}_synsets"].values]

        hypernyms = [h[0] if h else None for h in hypernyms]
        df[f"class{i}_synsets"] = hypernyms
        df[f"class{i}_name"] = [h.name() if h else None for h in hypernyms]
        df = get_vectors_from_prev_synset(df, i, "en")
    print(df["class0_synsets"].drop_duplicates().dropna())
    lang = "ru"
    model_ru = load_embeddings(lang)
    if lang == "ru":
        nouns = get_nouns_from_model(model_ru)
    # get Non-English (Russian) wordnet words
    # first we get vectors from English words
    # after that we use hierarchical Russian vectors
    use_russian_vectors = False
    for i in range(top_level, -1, -1):
        print("\n")
        if i == top_level:
            vectors = df[f"class{i}_vectors"].dropna()
        else:
            if use_russian_vectors:
                vectors = df[f"class{i}_vectors_{lang}"].dropna()
            else:
                vectors = df[f"class{i}_vectors"].dropna()
        df[f"class{i}_{lang}"] = ""
        df[f"class{i}_sim"] = 0
        index = vectors.index
        for j, v in enumerate(vectors.values):
            print("\t", i, j, end="\r")
            if i != top_level:
                prev_word = df.loc[index[j]][f"class{i+1}_{lang}"]
            else:
                prev_word = ""
            most_similar = model_ru.most_similar([v], topn=100)
            most_similar = [m for m in most_similar
                            if m[0] in nouns and m[0] != prev_word]
            if not most_similar:
                continue
            most_similar = most_similar[0]
            ru_word = most_similar[0]
            similarity = most_similar[1]
            if similarity > 0.5:
                df.loc[index[j], f"class{i}_{lang}"] = ru_word
                df.loc[index[j], f"class{i}_sim"] = similarity
        # create Russian vectors for the next level
        if use_russian_vectors:
            if i == top_level:
                # get vectors for russian words
                df[f"class{i}_vectors_{lang}"] = [
                    model_ru[n] if n in model_ru else None
                    for n in df[f"class{i}_{lang}"].values]
            if i != 0:
                df = get_vectors_from_prev_synset(df, i - 1)
    i = 9
    closest = df[df[f"class{i}_sim"] > 0.8]
    closest[[f"class{i}_name", f"class{i}_{lang}", f"class{i}_sim"]]

    unique_ru = df[f"class{i}_ru"].drop_duplicates().dropna()
    unique_ru = unique_ru[unique_ru != '']
    for u_r in unique_ru:
        df[f"class{i}_ru"] == u_r

    i = 9
    group = df.groupby(f"class{i}_ru").count()
    group = group[group["class10_ru"] < 1000]["class10_ru"]
    words = group[group > 2].index
    for word in words:
        print(df[df[f"class{i}_ru"] == word][[
            "class10_ru", "class9_ru",
            "class10_name", "class9_name",
            "class9_sim"]])

    # save wordnet to csv
    df[[col for col in df.columns if
        any(w in col for w in ("_ru", "_sim", "_col", "name"))
        ]].to_csv(f"wordnet3_{lang}_without_duplicates.csv")
    return df


def score_khodak(path):
    with open("other_works/pawn/ru_matches.txt") as f:
        r = f.read()
    # split by POS
    r = r.split("#")
    # leave only NOUNS
    r = r[1]
    r = r.split("\n")
    r = [l.split("\t") for l in r]
    wordnet_list = list()
    key = None
    for l in r:
        if not l[0]:
            continue
        elif ":" in l[0]:
            key = l[1]
        else:
            wordnet_list.append([key] + l)
    khodak = pd.DataFrame(
        wordnet_list, columns=["ru", "score", "synset", "definition"])
    my_df = pd.read_csv("annotations/wordnet3_ru_without_duplicates.csv")
    t_p = 0
    f_p = 0
    t_n = 0
    f_n = 0
    for index, row in khodak.iterrows():
        if row["ru"] in my_df["class10_ru"]:
            print(row["ru"])
        else:
            print(row["ru"])
    pass


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


def get_ru_relations(
        w0_i, word_0, gbm, word_matrix, allowed_words,
        emb_norm, parts_of_speech,
        pos_filter=False):
    print("\t", w0_i, word_0, end="\r")
    emb_0 = word_matrix[w0_i]
    # distances = 1 - spatial.distance.cosine(emb_0, emb_1)
    if pos_filter:
        pos = parts_of_speech[w0_i]
        pos_mask = parts_of_speech == pos
    else:
        pos_mask = [i for i in range(len(allowed_words))]
    pos_matrix = word_matrix[pos_mask]
    pos_words = allowed_words[pos_mask]
    distances = np.matmul(emb_0, pos_matrix.T)

    norm = emb_norm[w0_i] * emb_norm[pos_mask].T
    distances = distances.T / norm
    distances = distances[0]
    embs_1 = [list(emb_0) + list(word_matrix[j]) + [distances[j]]
              for j in range(len(pos_words))]
    embs_1 = np.array(embs_1)
    preds = gbm.predict(embs_1)
    max_preds = np.argmax(preds, axis=1)
    scores = preds[np.arange(len(preds)), max_preds]
    word_hypernyms = []
    word_synonyms = []
    for s_i, s in enumerate(scores):
        if s >= 0.5 and max_preds[s_i] in (0, 1, 3):
            pred = max_preds[s_i]
            if pred == 0:
                word_1 = pos_words[s_i]
                word_hypernyms.append((word_0, word_1, s))
                print("hyp", word_0, word_1, s, pred)
            elif pred == 1:
                word_1 = pos_words[s_i]
                word_synonyms.append((word_0, word_1, s))
                print("syn", word_0, word_1, s, pred)
            # reverse hypernyms
            elif pred == 3:
                word_1 = pos_words[s_i]
                word_hypernyms.append((word_1, word_0, s))
                print("hyp", word_1, word_0, s, pred)
    return word_hypernyms, word_synonyms


def analyse_collocations(eng_emb):
    collocations = [w for w in eng_emb.vocab if "_" in w]
    collocations = [c for c in collocations
                    if all(w in eng_emb.vocab for w in c.split("_")) and
                    len(c.split("_")) > 1]
    col_vectors = np.array([eng_emb[c] for c in collocations])
    print(col_vectors.shape)
    col_vectors = np.array([eng_emb[c] for c in collocations])
    w_vectors = [np.mean([eng_emb[w] for w in c.split("_") if w], axis=0)
                 for c in collocations]
    w_vectors = np.array(w_vectors)
    # similarities; large is closer
    sim = [1 - spatial.distance.cosine(w_vectors[w_i], col_vectors[w_i])
           for w_i, w in enumerate(w_vectors)]
    print(np.median(sim))


def create_syn_combinations(input_iter):
    combinations = list(itertools.combinations(input_iter, 2))
    combinations = combinations[:int(len(input_iter) * 1.5)]
    combinations = [c + (1,) for c in combinations]
    return combinations


def train_gbm(x_train, x_test, y_train, y_test, name, k_fold=False):
    objective = 'multiclass'
    thresholds, weights = None, None
    # check whether target is binary
    if len(y_train.shape) > 1:
        categ_nums = np.unique(y_train).shape[0]
    else:
        categ_nums = None
        objective = "binary"
    gbm_params = {
        'objective': objective,
        'max_depth': 8,
        'num_leaves': 12,
        'subsample': 0.8,
        'learning_rate': 0.1,
        'estimators': 1000,
        'num_trees': 10000,
        'num_class': categ_nums,
        'early_stopping_rounds': 10,
        'verbose': 0}
    gbm, score, weights = train_lgb(
        x_train, x_test, y_train, y_test, k_fold=k_fold, params=gbm_params,
        n_splits=2, n_repeats=2)
    if not k_fold:
        gbm = gbm[0]
    else:
        gbm = Ensemble(gbm)
    pickle.dump(gbm, open(f"{name}gbm_{score}.pcl", "wb"))
    preds = gbm.predict(x_test)
    if categ_nums:
        for threshold in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
            print(threshold)

            mask = [i for i in range(len(preds)) if any(preds[i] > threshold)]
            mask_preds = preds[mask]
            mask_preds = np.argmax(mask_preds, axis=1)
            mask_y = y_test[mask]
            print("acc", accuracy_score(mask_y, mask_preds))
            print("f1", f1_score(mask_y, mask_preds))
            print(confusion_matrix(mask_y, mask_preds))
    else:
        for threshold in range(1, 10):
            threshold = threshold / 10
            print(threshold)
            mask_preds = preds > threshold
            mask_preds = mask_preds.astype(int)
            print("acc", accuracy_score(y_test, mask_preds))
            print("f1", f1_score(y_test, mask_preds))
    return gbm, thresholds, weights


def train_cnn(
        all_words, eng_emb, x_train, x_test, y_train, y_test, word_matrix):
    kwargs = {
        "voc_size": len(all_words) - 1,
        "sequence_len": 2,
        "vec_len": eng_emb.vector_size,
        "categ_nums": [3],
        "name": "w2v/",
        "use_generator": False,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "embedding_matrix": word_matrix,
        "layers_multiplier": 1,
        # "kernel_size": 1,
        # "pool_size": 1,
        "trainable_embeddings": False,
        "use_embeddings": False
    }
    cnn = CnnLstm(**kwargs)
    model = cnn.cnn_lstm_classification()
    return model


def april_new():
    lang = "en"
    # bad wording; but if False - then only
    #   hypernym: hyponym
    #   hypernym: random_word
    # wordnet_langs = ["fin", "pol"]
    # wordnet.synsets("kod_pocztowy", lang="pol")[0].hypernyms()
    parse_synonyms = True
    zaliznak_filter = True
    pos_filter = False
    nouns_only = False
    morphy_filter = True
    now = datetime.now().isoformat().split(".")[0]
    folder = f"models/{now}"
    if nouns_only:
        folder += "_nouns"
    os.mkdir(folder)
    eng_emb = load_embeddings(lang)
    # analyse_collocations(eng_emb)
    # 117659 words
    eng_words = {w for w in wordnet.all_synsets()}
    # 87943

    # eng_words = {w for w in eng_words if w.hyponyms() or w.hypernyms()}

    # 43647 synsets; 24099 words -> 63314
    eng_words = {w for w in eng_words if
                 w.name().split(".")[0] in eng_emb.vocab}
    pos_dataset = {(w.pos(), w.name().split(".")[0]) for w in eng_words}
    pos_df = pd.DataFrame(pos_dataset, columns=["y", "x"])
    pos_x = [eng_emb[w] for w in pos_df["x"].values]
    pos_x = np.array(pos_x)
    pos_y = pos_df["y"].astype("category")
    pos_y = pos_y.cat.codes.values
    pos_x_train, pos_x_test, pos_y_train, pos_y_test = train_test_split(
        pos_x, pos_y,
        test_size=0.2, random_state=42, stratify=pos_df["y"])
    pos_gbm = train_gbm(pos_x_train, pos_x_test, pos_y_train, pos_y_test,
                        f"{folder}/pos_")
    # eng_words = {w for w in eng_words if w.name().split(".")[0]}

    # sets are slow for random.sample
    sample_words = list({w.name().split(".")[0] for w in eng_words})
    wordnet_dict = dict()
    x = []
    # dataset consists of:
    # Class 0: hypernym hyponym
    # Class 1: hyponym hyponym
    # Class 2: hypernym random_word
    #          hyponym random_word
    # Class 3: hyponym hypernym
    # all three classes are +/- balanced
    for e_i, e in enumerate(eng_words):
        print("\t", e_i, end="\r")
        hyponyms = e.hyponyms()
        hypernyms = e.hypernyms()
        e = e.name().split(".")[0]
        if hyponyms:
            hyponyms = {h.name().split(".")[0] for h in hyponyms}
            hyponyms = {h for h in hyponyms if h in eng_emb}
            if e not in wordnet_dict:
                wordnet_dict[e] = hyponyms
            else:
                wordnet_dict[e].update(hyponyms)
            for h in hyponyms:
                # hypernym hyponym
                x.append((e, h, 0))
                if parse_synonyms:
                    # hyponym hypernym
                    x.append((h, e, 3))
            if parse_synonyms:
                # hyponym hyponym
                combinations = create_syn_combinations(hyponyms)
                x += combinations
        if hypernyms:
            hypernyms = {h.name().split(".")[0] for h in hypernyms}
            hypernyms = {h for h in hypernyms if h in eng_emb}
            for h in hypernyms:
                # hypernym hyponym
                x.append((h, e, 0))
                if parse_synonyms:
                    # hyponym hypernym
                    x.append((e, h, 3))
                if h not in wordnet_dict:
                    wordnet_dict[h] = {e}
                else:
                    wordnet_dict[h].add(e)
            # hyponym hyponym
            if parse_synonyms:
                combinations = create_syn_combinations(hypernyms)
                x += combinations

    x = set(x)
    # add some random words to the algorithm
    for e_i, e in enumerate(eng_words):
        print("\t", e_i, end="\r")
        related = {w.name().split(".")[0] for w in e.hypernyms()}
        related.update({w.name().split(".")[0] for w in e.hyponyms()})
        e = e.name().split(".")[0]
        word = random.choice(sample_words)
        if word not in related:
            if e_i % 2 == 0:
                x.add((e, word, 2))
            else:
                x.add((word, e, 2))

    df = pd.DataFrame(x, columns=[1, 2, "target"])
    df = df[df[1] != df[2]]
    df.groupby("target").count()[1]

    # 20378 words
    all_words = set(df[1].values).union(set(df[2].values))

    # transform words into their ids
    all_words = list(all_words)
    word_index = {w: i for i, w in enumerate(all_words)}
    word_matrix = [eng_emb[w] for w in all_words]
    word_matrix = np.array(word_matrix)
    with open(f'{folder}/word_index_syn{parse_synonyms}.json', 'w') as outfile:
        json.dump(word_index, outfile)
    wordnet_dict = {word_index[k]: {word_index[s] for s in v}
                    for k, v in wordnet_dict.items()}
    with open(f'{folder}/wordnet_dict_syn{parse_synonyms}.json', 'w')\
            as outfile:
        json.dump({k: list(v) for k, v in wordnet_dict.items()}, outfile)

    # words to their embeddings
    x = df[[1, 2]].values
    y = df["target"].values
    x = [[word_index[w] for w in row] for row in x]
    x = [[word_matrix[t] for t in row] for row in x]
    cosine = [1 - spatial.distance.cosine(r[0], r[1]) for r in x]
    x = [list(t[0]) + list(t[1]) + [cosine[t_i]] for t_i, t in enumerate(x)]
    x = np.array(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)

    # cnn_model = train_cnn(
    #     all_words, eng_emb, x_train, x_test, y_train, y_test, word_matrix)
    gbm = train_gbm(x_train, x_test, y_train, y_test, f"{folder}/main")

    ru_emb = load_embeddings("ru")

    # 200000 words
    allowed_words = list(ru_emb.vocab)

    # 448
    ru_collocations = [w for w in allowed_words if "_" in w]
    # 297
    ru_collocations = [w for w in ru_collocations if
                       len([l for l in w.split("_") if l]) > 1]
    # 146
    ru_collocations = [w for w in ru_collocations
                       if not re.fullmatch("[a-zA-Z_]+", w)]
    # 108
    ru_collocations = [w for w in ru_collocations
                       if re.fullmatch("[ёЁа-яА-Я_-]+", w)]
    # leave only words from opencorpora.org
    # 139674 words; open corpora and ru_emb preprocessings are not compatible
    # allowed_words = [w for w in allowed_words if
    #                  morph.word_is_known(w) or any(s in w for s in "_-")]
    # allowed_words = [w.replace("ё", "е") for w in allowed_words]

    # 163569 words
    allowed_words = [w for w in allowed_words
                     if re.fullmatch("[а-яА-Я]+[а-яА-Я_]+[а-яА-Я]+", w)]
    allowed_words = [w for w in allowed_words if len(w) > 3]
    if zaliznak_filter:
        with open("zaliznak.txt") as f:
            zaliznak = f.readlines()
        zaliznak = [l.strip() for l in zaliznak]
        allowed_words = [w for w in allowed_words
                         if w in zaliznak or "_" in w]
    if morphy_filter or nouns_only:
        morph = pymorphy2.MorphAnalyzer()
    if morphy_filter:
        morphed = [morph.parse(w)[0] for w in allowed_words]
        # Sgtm - singularia tantum; geox - geographical
        bad_tags = {"COMP", "Name", "plur", "Geox",
                    "NPRO",  # местоимение-существительное
                    "PREP",
                    "CONJ",
                    "PRCL",
                    "INTJ",
                    # non-nominative cases
                    "gent", "datv", "accs", "ablt", "loct",
                    "voct", "gen1", "gen2",
                    "acc2", "loc1", "loc2",
                    # names
                    "Surn", "Patr", "Orgn", "Trad",
                    # verb grammar
                    "past", "futr", "impr", "incl", "excl", "pssv"
                    }
        allowed_words = [w for w_i, w in enumerate(allowed_words)
                         if not any(t in morphed[w_i].tag for t in bad_tags)]
    if nouns_only:
        allowed_words = [w for w_i, w in enumerate(allowed_words)
                         if "NOUN" in morph.parse(w)[0].tag]
    # and
    # all(t in morphed[w_i].tag for t in good_tags)]
    # allowed_words = [w for w in allowed_words if len(w) < 17 or "_" in w]
    word_matrix = np.array([ru_emb[w] for w in allowed_words])
    allowed_words = np.array(allowed_words)
    emb_norm = np.linalg.norm(word_matrix, axis=1)[np.newaxis].T
    parts_of_speech = np.argmax(pos_gbm.predict(word_matrix), axis=1)

    ru_synonyms = []
    ru_hypernyms = []
    # irange = [(w0_i, word_0, gbm, word_matrix, allowed_words, emb_norm)
    #           for w0_i, word_0 in enumerate(allowed_words)]
    # pool = mp.pool.ThreadPool(4)
    for w0_i, word_0 in enumerate(allowed_words):
        word_hypernyms, word_synonyms = get_ru_relations(
            w0_i, word_0, gbm, word_matrix, allowed_words, emb_norm,
            parts_of_speech, pos_filter=pos_filter)
        ru_hypernyms += word_hypernyms
        ru_synonyms += word_synonyms
    print("allowed words", len(allowed_words))
    time.sleep(10)
    for filename, file in [(f"{folder}/synonyms", ru_synonyms),
                           (f"{folder}/hypernyms", ru_hypernyms)]:
        with open("{}_{}_{}".format(filename,
                                    len(allowed_words),
                                    now),
                  "a") as f:
            for line_i, line in enumerate(file):
                f.write("\t".join([str(w) for w in line]) + "\n")
    # pickle.dump(ru_synonyms, open("ru_synonyms_zaliznak.pcl", "wb"))
    # pickle.dump(ru_hypernyms, open("ru_hypernyms_zaliznak.pcl", "wb"))
    # In [27]: ru_wordnet = dict()
    #     ...: for r in ru_hypernyms:
    #     ...:     k, v = r
    #     ...:     if k not in ru_wordnet:
    #     ...:         ru_wordnet[k] = set([v])
    #     ...:     else:
    #     ...:         ru_wordnet[k].add(v)
    # for w1_i, word_1 in enumerate(allowed_words):
    #     if w0_i == w1_i:
    #         continue
    #     # The function is not symmetric
    #     # if w1_i > w0_i:
    #     #     continue
    #     emb_1 = word_matrix[w1_i]
    #     dist = distances[word_1]
    #     gbm_input = np.array(list(emb_0) + list(emb_1) + [dist])
    #     pred = gbm.predict([gbm_input])[0]
    #     max_pred = np.argmax(pred)
    #     score = pred[max_pred]
    #     pred = max_pred
    #     if score < 0.9:
    #         continue

    return None


def tf_model(x_train, y_train, x_test, y_test):
    """Model function for CNN."""
    # Input Layer; concatenated embeddings
    with tf.Session() as sess:
        batch_size = 128

        emb = tf.placeholder(shape=(None, 600), name='emb', dtype=tf.float32)
        dense_0 = tf.layers.dense(emb, 512, activation='relu')
        dense_1 = tf.layers.dense(dense_0, 256, activation='relu')
        dense_2 = tf.layers.dense(dense_1, 3, activation='relu')
        labels = tf.placeholder(shape=(None), name='labels', dtype=tf.int64)
        output = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=dense_2)

        cost = tf.reduce_mean(output)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
        accuracy, update_op = tf.metrics.accuracy(
            labels=labels, predictions=tf.argmax(dense_1, 1),
            name='accuracy')

        patience = 5
        train_loss_results = []
        train_accuracy_results = []
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for epoch in range(1000):
            epoch_loss = 0
            epoch_acc = 0
            epoch_range = int(len(x_train) / batch_size) + 1
            for i in range(epoch_range):
                batch_start = batch_size * i
                batch_end = batch_start + batch_size
                epoch_x = x_train[batch_start: batch_end]
                epoch_y = y_train[batch_start: batch_end]
                batch_opt, batch_cost, batch_acc = sess.run(
                    [optimizer, cost, accuracy],
                    feed_dict={emb: epoch_x, labels: epoch_y})
                epoch_loss += batch_cost
                epoch_acc += batch_acc
                print(f"Epoch: {epoch}; loss: {batch_cost}; "
                      f"acc: {batch_acc}", end="\r")
            epoch_acc /= epoch_range
            epoch_loss /= epoch_range
            print(f"\nEpoch: {epoch}; loss: {epoch_loss}; "
                  f"acc: {epoch_acc}")
            val_range = int(len(x_test) / batch_size) + 1
            val_loss = 0
            acc_val = 0
            for i in range(val_range):
                batch_start = batch_size * i
                batch_end = batch_start + batch_size
                val_x = x_test[batch_start: batch_end]
                val_y = y_test[batch_start: batch_end]
                batch_cost, epoch_acc = sess.run(
                    [cost, accuracy],
                    feed_dict={emb: val_x, labels: val_y})
                val_loss += batch_cost
                acc_val += epoch_acc
            val_loss /= val_range
            acc_val /= val_range
            if epoch == 0:
                tf.saved_model.simple_save(sess, 'w2v/tf/my-model',
                                           inputs={"emb": emb},
                                           outputs={"output": output})
            else:
                if val_loss < train_loss_results[-1]:
                    tf.saved_model.simple_save(sess, 'w2v/tf/my-model',
                                               inputs={"emb": emb},
                                               outputs={"output": output})
                else:
                    patience -= 1
            train_accuracy_results.append(acc_val)
            train_loss_results.append(val_loss)
            print("\n", epoch_loss, "Train accuracy:",
                  acc_val, "Val accuracy:", acc_val)
            if patience <= 0:
                break
        print("training complete")


def keras_model(x_train, y_train, x_test, y_test):
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                       verbose=0, mode='auto')
    inputs = Input(shape=(x_train.shape[1], ))

    dropout_1 = Dropout(0.2)(inputs)
    dense_1 = Dense(800, activation='relu')(dropout_1)

    dropout_2 = Dropout(0.2)(dense_1)
    dense_2 = Dense(512, activation='relu')(dropout_2)

    dropout_3 = Dropout(0.1)(dense_2)
    dense_3 = Dense(256, activation='relu')(dense_2)

    if len(y_train.shape) > 1:
        categ_nums = y_train.shape[1]
        activation = 'softmax'
        loss = 'sparse_categorical_crossentropy'
    else:
        categ_nums = 1
        activation = 'sigmoid'
        loss = "binary_crossentropy"
    outputs = Dense(categ_nums, activation=activation)(dense_3)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss, optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=256, epochs=2000,
              validation_data=(x_test, y_test),
              callbacks=[es])
    model.save("models/keras_cnn_lstm.h5")
    return model


def parse_hypernyms(filename="hypernyms_41800", khodak=None, vectorizer=None):

    f = open(filename)
    hypernyms = dict()
    reverse = dict()
    for i, line in enumerate(f):
        if i % 10000 == 0:
            print("\t", i, end="\r")
        line = line.strip()
        line = line.split("\t")
        score = float(line[-1])
        if score < 0.9:
            continue
        hyper = line[0]
        hypo = line[1]
        # if vectorizer:
        #     vec = vectorizer.transform([f"{hyper} {hypo}"]).todense()
        #     vec = vec[vec > 0]
        #     if vec.shape[1] > 1:
        #         new_score = score * vec.tolist()[0][1]
        #     else:
        #         continue
        #     if new_score < 0.8:
        #         continue
        #     else:
        #         print(hyper, hypo, score, new_score, vec.tolist()[0])
        if hyper == hypo:
            continue
        if khodak:
            if hyper not in khodak and hypo not in khodak:
                continue
        # vectorizer.fit([f"{hypo} {hyper}"])
            # print(hyper, hypo, score)
        if hypo not in reverse:
            reverse[hypo] = {hyper}
        else:
            reverse[hypo].add(hyper)
        if hyper not in hypernyms:
            # hypo = (hypo, score)  # !!!!!!
            hypernyms[hyper] = {(hypo, score)}
        else:
            hypernyms[hyper].add((hypo, score))
        # nature - bear
        # nature - animal
        # animal - bear
        # remove bear from nature
        for k, v in reverse.items():
            to_delete = list()
            for v_i, v_l in enumerate(v):
                if v_l in reverse:
                    joined = reverse[v_l] & v
                    if joined:
                        for hyper_k in joined:
                            if k in hypernyms[hyper_k]:
                                hypernyms[hyper_k].pop(k, None)
                                to_delete.append(hyper_k)
            reverse[k] = {v for v in reverse[k] if v not in to_delete}
        # to_delete = set()
        # for k, v in hypernyms.items():
        #     upper_hierarchy_words = dict()
        #     for word in v:
        #         if word in hypernyms:
        #             upper_hierarchy_words[word] = hypernyms[word]
        #             to_delete.add(word)
        #     hypernyms[k] = [w for w in v if w not in upper_hierarchy_word]
        #     for w in upper_hierarchy_word:
        #         print(k, word)
        #         hypernyms[k] =(hypernyms[w])
        # hypernyms = {k: v for k, v in hypernyms.items() if k not in to_delete}
        return hypernyms, reverse


def load_model():
    with open("other_works/pawn/ru_matches.txt") as f:
        khodak = f.readlines()
    khodak = [l.strip().split("\t") for l in khodak if ":" in l]
    khodak = [l[1] for l in khodak if len(l) == 2]
    khodak = set(khodak)
    hypernyms, reverse = parse_hypernyms(khodak, vectorizer=None)
    lines = []
    for k, v in reverse.items():
        lines.append("{} {}".format(k, " ".join(v)))
    for k, v in hypernyms.items():
        lines.append("{} {}".format(k, " ".join(v)))
    vectorizer = TfidfVectorizer()
    vectorizer.fit(lines)
    hypernyms_v, reverse_v = parse_hypernyms(khodak, vectorizer=vectorizer)

    to_match = [(w, k) for k,v  in reverse.items() for w in v]
    random.shuffle(to_match)
    match_df = pd.DataFrame(to_match)
    annotate(match_df, "wordnet_nouns", 0, limit=200)


def create_synset_names_dict(synsets):
    synset_names = dict()
    for k, v in synsets.items():
        name = "".join(k.split(".")[:-2])
        if name in synset_names:
            synset_names[name].update(set(v))
        else:
            synset_names[name] = set(v)
    return synset_names


def get_synset_pos(synset):
    pos = synset.split(".")[-2]
    if pos == "v":
        pos = [0, 0, 1]
    elif pos == "n":
        pos = [0, 1, 0]
    else:
        pos = [1, 0, 0]
    return pos


def create_synset_dataset(synsets, syn_names, all_lemmas, emb, syn_mtx,
                          syn_ind,
                          add_zeroes=True):
    X = []
    Y = []

    for k, v in synsets.items():
        name = "".join(k.split(".")[:-2])
        pos = get_synset_pos(k)
        meaning = k.split(".")[-1]
        meaning = int(meaning) / 60
        v = {lemma for lemma in v if lemma in emb and lemma != name}
        name_lemmas = syn_names[name]
        name_lemmas = {lemma for lemma in name_lemmas if lemma in emb and
                       lemma != name}
        union = name_lemmas & v
        difference = name_lemmas.difference(v)
        add_vec = [meaning] + pos
        k_vec = syn_mtx[syn_ind[k]]
        for lemma in union:
            lemma_vec = emb[lemma]
            X.append(np.concatenate([lemma_vec, k_vec, add_vec]))
            Y.append(1)
        for lemma in difference:
            lemma_vec = emb[lemma]
            X.append(np.concatenate([lemma_vec, k_vec, add_vec]))
            Y.append(0)
        left_random = len(union) - len(difference)
        if add_zeroes:
            if left_random > 0:
                for i in range(left_random):
                    random_lemma = random.choice(all_lemmas)
                    if random_lemma not in v:
                        r_lem_vec = emb[random_lemma]
                        X.append(np.concatenate([r_lem_vec, k_vec, add_vec]))
                        Y.append(0)
    return X, Y


def create_all_lemmas(synsets, emb):
    all_lemmas = list(set([w for v in synsets.values() for w in v]))
    all_lemmas = [l for l in all_lemmas if l in emb]
    return all_lemmas


def k_fold_training(X_, Y_, tr_function, num_folds=5):
    folds = KFold(n_splits=num_folds, random_state=2319)
    models = []
    for fold_, (train_index, test_index) in enumerate(
            folds.split(X_, Y_)):
        X_train_fold, X_valid = X_[train_index], X_[test_index]
        Y_train_fold, Y_valid = Y_[train_index], Y_[test_index]
        model = tr_function(X_train_fold, Y_train_fold, X_valid, Y_valid)
        models.append(model)
    return models


def april_khodak():
    eng_emb = load_embeddings("en")
    ru_emb = load_embeddings("ru")

    USE_FOREIGN = True
    ADD_ZEROES = True
    USE_KFOLD = True
    TFIDF = True

    stops = set(stopwords.words('english'))
    # get synsets + their lemmas
    # 117659 synsets
    synsets = {w.name(): [l.name().lower().replace(".", "")
                          for l in w.lemmas()]
               for w in wordnet.all_synsets()}
    synsets = {k: v for k, v in synsets.items() if v}
    # synsets = {k: [w for w in v if w in eng_emb] for k, v in synsets.items()}
    # 69094 synsets
    synsets = {k: set(v) for k, v in synsets.items() if v}

    synset_list = list(synsets.keys())
    synset_index = {w: i for i, w in enumerate(synset_list)}

    with open(f'synsets/synset_index.json', 'w') as outfile:
        json.dump(synset_index, outfile)

    synset_matrix = np.zeros(shape=(len(synset_index), eng_emb.vector_size))

    # no_embedding_synsets = set()
    definitions = dict()
    for s in synsets:
        definition = wordnet.synset(s).definition()
        definition = text_pipeline(definition, lemmatize=False)
        definition = [w for w in definition if w not in stops and w in eng_emb]
        # if not definition:
        #     print(wordnet.synset(s).definition())
        definitions[s] = definition
    vocabulary = {w for d in definitions.values() for w in d}
    vocabulary = {v: v_i for v_i, v in enumerate(vocabulary)}
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x,
        lowercase=False,
        vocabulary=vocabulary.keys())
    vectorizer = vectorizer.fit(definitions.values())

    for s_i, s in enumerate(synset_list):
        print("\t", s_i, s, len(synset_list), end="\r")
        lemmas = synsets[s]
        weights = (0.1, 0.9)
        # averaged; e.g. pass_away <- np.mean(eng_emb[["pass", "away"]])
        # secondary_lemmas = False
        no_lemmas = False
        emb_lemmas = [l for l in lemmas if l in eng_emb]
        if not emb_lemmas:
            lemmas = [l.split("_") for l in lemmas if "_" in l]
            lemmas = [l for l in lemmas if
                      all(w in eng_emb for w in l)]
            if lemmas:
                # gensim KeyedVectors also accept lists of words
                embeddings = [np.mean(eng_emb[l], axis=0) for l in lemmas]
                embeddings = np.mean(embeddings, axis=0)
                weights = (0.1, 0.9)
            else:
                no_lemmas = True
        else:
            lemmas = emb_lemmas
            # gensim KeyedVectors also accept lists of words
            embeddings = eng_emb[lemmas]
            embeddings = np.average(embeddings, axis=0)
        definition = definitions[s]
        if definition:
            def_weights = None
            if TFIDF:
                def_weights = vectorizer.transform([definition]).toarray()[0]
                def_weights = [def_weights[vocabulary[w]] for w in definition]

            definition_emb = np.average(eng_emb[definition], axis=0,
                                        weights=def_weights)
            if not no_lemmas:
                embeddings = np.average([embeddings, definition_emb],
                                        weights=weights, axis=0)
            else:
                embeddings = definition_emb
        # I havent checked the case when there are no lemmas and no definition
        # but there isnt such a case

        synset_matrix[s_i] = embeddings
    # prepare dataset
    synset_names = create_synset_names_dict(synsets)

    all_lemmas = create_all_lemmas(synsets, eng_emb)
    X, Y = create_synset_dataset(
        synsets, synset_names, all_lemmas, eng_emb, synset_matrix,
        synset_index, add_zeroes=ADD_ZEROES)

    lang_embeddings = {"fi": "fin",
                       # "pl": "pol",
                       }
    if USE_FOREIGN:
        for emb_lng, wn_lng in lang_embeddings.items():
            lang_emb = load_embeddings(emb_lng)
            lang_synsets = {s: wordnet.synset(s).lemma_names(wn_lng)
                            for s in synsets
                            if wordnet.synset(s).lemma_names(wn_lng)}
            lang_name_synsets = create_synset_names_dict(lang_synsets)
            lang_lemmas = create_all_lemmas(lang_synsets, lang_emb)

            lang_X, lang_Y = create_synset_dataset(
                lang_synsets, lang_name_synsets, lang_lemmas, lang_emb,
                synset_matrix, synset_index,
                add_zeroes=ADD_ZEROES)
            X += lang_X
            Y += lang_Y

    X = np.array(X)
    Y = np.array(Y)
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)
    if USE_KFOLD:
        ensemble = train_gbm(x_train, x_test,
                             y_train, y_test, f"synsets/main2",
                             k_fold=USE_KFOLD)
        nn_models = k_fold_training(x_train, y_train, keras_model, num_folds=4)
        ensemble.models += nn_models
    else:
        gbm = train_gbm(x_train, x_test,
                        y_train, y_test, f"synsets/main2",
                        k_fold=USE_KFOLD)
        gbm = gbm[0]
        nn_model = keras_model(x_train, y_train, x_test, y_test)
    if USE_KFOLD:
        # threshold between [0.1, 0.9]
        ensemble_threshold = [f1_score(y_test,
                                       predict_binary(ensemble,
                                                      x_test,
                                                      threshold=t / 10))
                              for t in range(1, 10)]
        ensemble_threshold = np.argmax(ensemble_threshold)
        adjust_min = ensemble_threshold * 10
        ensemble_adjust = [f1_score(y_test,
                                    predict_binary(ensemble,
                                                   x_test,
                                                   threshold=t / 100))
                           for t in range(adjust_min, adjust_min + 20)]
        ensemble_adjust = np.argmax(ensemble_adjust)
        ensemble_threshold = round(
            ensemble_threshold * 0.1 + ensemble_adjust * 0.01, 3)
    with open("other_works/pawn/ru_matches.txt") as f:
        khodak = f.readlines()
    khodak = [l.strip().split("\t") for l in khodak]

    # convert data to
    # ад  chaos.n.01          1
    # ад  conflagration.n.01  0
    # ад  gehenna.n.01        1
    # ад  hell.n.01           1
    # ад  hell.n.04           0
    khodak_data = []
    ru_word = ""
    pos = ""
    for l in khodak:
        if len(l) == 2 and ":" in l[0]:
            ru_word = l[1]
        elif len(l) == 2 and l[0] == "#":
            pos = l[1]
        elif len(l) == 3:
            synset = l[1]
            target = l[0]
            if synset in synsets:
                khodak_data.append([ru_word, synset, target, pos])
    # not_found = len([l for l in khodak_data if l[0] not in ru_emb])
    khodak_data = [l for l in khodak_data if l[0] in ru_emb]
    khodak_df = pd.DataFrame(
        khodak_data, columns=["ru", "synset", "target", "pos"])
    khodak_df["target"] = khodak_df["target"].astype(int)

    f1_scores = []
    pos_s = ["", "n", "a", "v"]
    for pos in pos_s:
        if pos == "":
            pred_df = khodak_df
        else:
            pred_df = khodak_df[khodak_df["pos"] == pos]

        ru_input = ru_emb[pred_df["ru"].values]
        # embeddings for all ru words
        eng_input = [synset_index[s] for s in pred_df["synset"].values]
        # embeddings for all eng words
        eng_input = synset_matrix[eng_input]

        syn_pos = [get_synset_pos(s) for s in pred_df["synset"].values]

        meaning = [[int(s.split(".")[-1]) / 60]
                   for s in pred_df["synset"].values]

        input_data = np.concatenate([ru_input, eng_input, meaning, syn_pos],
                                    axis=1)

        # preds = np.argmax(preds, axis=1).astype(bool)  # &\
        # np.max(preds, axis=1) > 0.6
        # preds = preds.astype(int)
        # print(pos, "gbm", f1_score(pred_df["target"].values, preds))
        if USE_KFOLD:
            preds = predict_binary(
                ensemble, input_data, threshold=ensemble_threshold)
            f1 = f1_score(pred_df["target"].values, preds)
            f1_scores.append([f1, pos, "avg", ensemble_threshold])
        else:

            gbm_preds = gbm.predict(input_data)
            nn_preds = nn_model.predict(input_data).T[0]
            avg_preds = np.mean([gbm_preds, nn_preds], axis=0)
            models = ["gbm", "nn", "avg"]
            for threshold in (0.2, 0.3, 0.4, 0.5):
                for model_i, preds in enumerate(
                        [gbm_preds, nn_preds, avg_preds]):
                    model = models[model_i]
                    threshold_preds = preds > threshold
                    threshold_preds = threshold_preds.astype(int)
                    f1 = f1_score(pred_df["target"].values, threshold_preds)
                    f1_scores.append([f1, pos, model, threshold])
                    # print(f1, pos, model, threshold)
    f1_df = pd.DataFrame(f1_scores,
                         columns=["f1", "pos", "model", "threshold"])

    if not USE_KFOLD:
        group_cols = ["pos", "model", "threshold"]
        for pos in pos_s:
            print(f1_df[f1_df.pos == pos].groupby(group_cols).max())
    else:
        print(f1_df[["pos", "f1"]])
    pred_df["preds"] = preds


if __name__ == "__main__":
    # main()
    april_new()
