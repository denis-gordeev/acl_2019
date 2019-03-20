import numpy as np
import pandas as pd

from nltk import word_tokenize
from nltk.corpus import wordnet
from scipy.optimize import linear_sum_assignment
from gensim.models import KeyedVectors
import networkx as nx
from node2vec import Node2Vec

from utils import read_nigp_csv_pd, get_averaged_vectors, read_okpd_pd,\
    matrices_cosine_distance, get_top_indices


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


if __name__ == "__main__":
    main()


