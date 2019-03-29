import itertools
import random
import pickle
import json
import re
import multiprocessing as mp
import numpy as np
from scipy import spatial
import pandas as pd
import tensorflow as tf
from gensim.models import KeyedVectors

from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras.callbacks import EarlyStopping

from nltk import word_tokenize
from nltk.corpus import wordnet
import pymorphy2

from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import networkx as nx
from node2vec import Node2Vec

from utils import read_nigp_csv_pd, get_averaged_vectors, read_okpd_pd,\
    matrices_cosine_distance, get_top_indices
from utils_d.ml_models import CnnLstm, train_lgb
from utils_d.ml_utils import create_word_index


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
    annotation_values = ["1", "0", "0.5"]
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


def get_ru_relations(w0_i, word_0, gbm, word_matrix, allowed_words, emb_norm):
    print("\t", w0_i, end="\r")
    emb_0 = word_matrix[w0_i]
    # distances = 1 - spatial.distance.cosine(emb_0, emb_1)
    distances = np.matmul(emb_0, word_matrix.T)
    norm = emb_norm[w0_i] * emb_norm.T
    distances = distances.T / norm
    distances = distances[0]
    embs_1 = [list(emb_0) + list(word_matrix[j]) + [distances[j]]
              for j in range(len(allowed_words))]
    embs_1 = np.array(embs_1)
    preds = gbm.predict(embs_1)
    max_preds = np.argmax(preds, axis=1)
    scores = preds[np.arange(len(preds)), max_preds]
    word_hypernyms = []
    word_synonyms = []
    for s_i, s in enumerate(scores):
        if s > 0.5 and max_preds[s_i] in (0, 1, 3):
            pred = max_preds[s_i]
            if pred == 0:
                word_1 = allowed_words[s_i]
                word_hypernyms.append((word_0, word_1, s))
                print("hyp", word_0, word_1, s)
            elif pred == 1:
                word_1 = allowed_words[s_i]
                word_synonyms.append((word_0, word_1, s))
                print("syn", word_0, word_1, s)
            # reverse hypernyms
            elif pred == 3:
                word_1 = allowed_words[s_i]
                word_hypernyms.append((word_1, word_0, s))
                print("hyp", word_1, word_0, s)
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


def april_new():
    lang = "en"
    eng_emb = load_embeddings(lang)
    # analyse_collocations(eng_emb)
    # 117659 words
    eng_words = {w for w in wordnet.all_synsets()}
    # 87943
    eng_words = {w for w in eng_words if w.hyponyms() or w.hypernyms()}
    # 43647 synsets; 24099 words
    eng_words = {w for w in eng_words if
                 w.name().split(".")[0] in eng_emb.vocab}
    # eng_words = {w for w in eng_words if w.name().split(".")[0]}
    wordnet_dict = dict()
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
        if hypernyms:
            hypernyms = {h.name().split(".")[0] for h in hypernyms}
            hypernyms = {h for h in hypernyms if h in eng_emb}
            for h in hypernyms:
                if h not in wordnet_dict:
                    wordnet_dict[h] = {e}
                else:
                    wordnet_dict[h].add(e)
    # 20378 words
    all_words = set(wordnet_dict.keys())
    for v in wordnet_dict.values():
        all_words.update(set(v))
    # random choise does not accept sets
    # transform words into their ids
    all_words = list(all_words)
    word_index = {w: i for i, w in enumerate(all_words)}
    word_matrix = [eng_emb[w] for w in all_words]
    word_matrix = np.array(word_matrix)
    with open('w2v/word_index.json', 'w') as outfile:
        json.dump(word_index, outfile)
    wordnet_dict = {word_index[k]: {word_index[s] for s in v}
                    for k, v in wordnet_dict.items()}
    with open('w2v/wordnet_dict.json', 'w') as outfile:
        json.dump({k: list(v) for k, v in wordnet_dict.items()}, outfile)
    x = []
    y = []
    # dataset consists of:
    # Class 0: hypernym hyponym
    # Class 1: hyponym hyponym
    # Class 2: hypernym random_word
    #          hyponym random_word
    # Class 3: hyponym hypernym
    # all three classes are +/- balanced
    j = 0
    for hypernym, hyponyms in wordnet_dict.items():
        print("\t", j, len(wordnet_dict), end="\r")
        j += 1
        for h in hyponyms:
            x.append((hypernym, h))
            y.append(0)
            x.append((h, hypernym))
            y.append(3)
        combinations = list(itertools.combinations(hyponyms, 2))

        # to keep the dataset balanced
        random.shuffle(combinations)
        combinations = combinations[:len(hyponyms) * 2]
        x += combinations
        y += [1] * len(combinations)
        for i in range(len(combinations)):
            random_words = random.sample(all_words, 10)
            random_words = [word_index[r] for r in random_words]
            random_words = [w for w in random_words
                            if w not in hyponyms and w != hypernym]
            x.append((hypernym, random_words[0]))
            y.append(2)
            x.append((random.sample(hyponyms, 1)[0], random_words[1]))
            y.append(2)
    # words to their embeddings
    x = [[word_matrix[t] for t in row] for row in x]
    cosine = [1 - spatial.distance.cosine(r[0], r[1]) for r in x]
    x = [list(t[0]) + list(t[1]) + [cosine[t_i]] for t_i, t in enumerate(x)]
    x = np.array(x)
    # x = np.array([np.concatenate(row) for row in x])
    y = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)

    # kwargs = {
    #     "voc_size": len(all_words) - 1,
    #     "sequence_len": 2,
    #     "vec_len": eng_emb.vector_size,
    #     "categ_nums": [3],
    #     "name": "w2v/",
    #     "use_generator": False,
    #     "x_train": x_train,
    #     "x_test": x_test,
    #     "y_train": y_train,
    #     "y_test": y_test,
    #     "embedding_matrix": word_matrix,
    #     "layers_multiplier": 1,
    #     # "kernel_size": 1,
    #     # "pool_size": 1,
    #     "trainable_embeddings": False,
    #     "use_embeddings": False
    # }
    # cnn = CnnLstm(**kwargs)
    # model = cnn.cnn_lstm_classification()
    gbm_params = {
        'objective': 'multiclass',
        'max_depth': 8,
        'num_leaves': 12,
        'subsample': 0.8,
        'learning_rate': 0.1,
        'estimators': 1000,
        'num_trees': 10000,
        'num_class': 4,
        'early_stopping_rounds': 10,
        'verbose': 0}
    gbm, score = models, score = train_lgb(
        x_train, x_test, y_train, y_test, k_fold=False, params=gbm_params)
    gbm = gbm[0]
    pickle.dump(gbm, open(f"gbm_{score}.pcl", "wb"))
    preds = gbm[0].predict(x_test)
    for threshold in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        print(threshold)
        mask = [i for i in range(len(preds)) if any(preds[i] > threshold)]
        mask_preds = preds[mask]
        mask_preds = np.argmax(mask_preds, axis=1)
        mask_y = y_test[mask]
        print(accuracy_score(mask_y, mask_preds))
        print(confusion_matrix(mask_y, mask_preds))

    ru_emb = load_embeddings("ru")
    morph = pymorphy2.MorphAnalyzer()

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

    # morphed = [morph.parse(w)[0] for w in allowed_words]
    # Sgtm - singularia tantum
    # bad_tags = {"Name", "plur", "Geox", "Sgtm"}
    # good_tags = {"NOUN", "nomn", "sing"}
    # allowed_words = [w for w_i, w in enumerate(allowed_words)
    #                  if not any(t in morphed[w_i].tag for t in bad_tags) and
    #                  all(t in morphed[w_i].tag for t in good_tags)]
    # allowed_words = [w for w in allowed_words if len(w) < 17 or "_" in w]
    word_matrix = np.array([ru_emb[w] for w in allowed_words])
    emb_norm = np.linalg.norm(word_matrix, axis=1)[np.newaxis].T

    ru_synonyms = []
    ru_hypernyms = []
    # irange = [(w0_i, word_0, gbm, word_matrix, allowed_words, emb_norm)
    #           for w0_i, word_0 in enumerate(allowed_words)]
    # pool = mp.pool.ThreadPool(4)

    for w0_i, word_0 in enumerate(allowed_words):
        word_hypernyms, word_synonyms = get_ru_relations(
            w0_i, word_0, gbm, word_matrix, allowed_words, emb_norm)
        ru_hypernyms += word_hypernyms
        ru_synonyms += word_synonyms
    pickle.dump(ru_synonyms, open("ru_synonyms_2.pcl", "wb"))
    pickle.dump(ru_hypernyms, open("ru_hypernyms_2.pcl", "wb"))
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
    inputs = Input(shape=(601, ))
    dropout_1 = Dropout(0.1)(inputs)
    dense_1 = Dense(512, activation='relu')(dropout_1)
    dropout_2 = Dropout(0.1)(dense_1)
    dense_2 = Dense(256, activation='relu')(dropout_2)
    outputs = Dense(3, activation='softmax')(dense_2)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=128, epochs=2000,
              validation_data=(x_test, y_test),
              callbacks=[es])
    model.save("w2v/keras_cnn_lstm.h5")


if __name__ == "__main__":
    main()
