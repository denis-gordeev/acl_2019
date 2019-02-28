import numpy as np
import pandas as pd

from nltk import word_tokenize
from nltk.corpus import wordnet
from scipy.optimize import linear_sum_assignment
from gensim.models import KeyedVectors

from utils import read_nigp_csv_pd, get_averaged_vectors, read_okpd_pd,\
    matrices_cosine_distance, get_top_indices


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
                taxonomy.loc[code_index, level_vectors] = [vectors]
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
    match_df.to_csv(f"match_df_{level}_{algorithm}_{vector_method}.csv")
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
    for i in range(top_level - 1, -1, -1):
        print("\t", i, end="\r")
        hypernyms = [h.hypernyms() if h else None
                     for h in df[f"class{i+1}_synsets"].values]

        hypernyms = [h[0] if h else None for h in hypernyms]
        df[f"class{i}_synsets"] = hypernyms
        df[f"class{i}_name"] = [h.name() if h else None for h in hypernyms]
        df[f"class{i}_vectors"] = None
        df[f"class{i}_vectors"] = df[f"class{i}_vectors"].astype('object')
        # unique_synsets = df[f"class{i}_synsets"].drop_duplicates().dropna()
        unique_synsets = df[f"class{i}_synsets"].dropna()
        index = unique_synsets.index
        print("\n\n")
        for u_i, u_s in enumerate(unique_synsets):
            print("\t", u_i, end="\r")
            # prev class synsets having this homonym
            vectors = df[df[f"class{i}_synsets"] == u_s][f"class{i+1}_vectors"]
            vectors = vectors.dropna()
            vectors = np.mean(vectors)
            df.loc[index[u_i], f"class{i}_vectors"] = vectors
    df["class0_synsets"].drop_duplicates().dropna()
    model_ru = load_embeddings("ru")

    nouns = get_nouns_from_model(model_ru)

    for i in range(top_level, -1, -1):
        print("\n")
        vectors = df[f"class{i}_vectors"].dropna()
        df[f"class{i}_ru"] = ""
        df[f"class{i}_sim"] = 0
        index = vectors.index
        for j, v in enumerate(vectors.values):
            print("\t", i, j, end="\r")
            most_similar = model_ru.most_similar([v], topn=20)
            most_similar = [m for m in most_similar if m[0] in nouns]
            if not most_similar:
                continue
            most_similar = most_similar[0]
            ru_word = most_similar[0]
            similarity = most_similar[1]
            if similarity > 0.5:
                df.loc[index[j], f"class{i}_ru"] = ru_word
                df.loc[index[j], f"class{i}_sim"] = similarity
    closest = df[df[f"class{i}_sim"] > 0.7]
    closest[[f"class{i}_names", f"class{i}_ru", f"class{i}_sim"]]
    return df


def main():
    vector_methods = ["topdown", "bottomup"]
    algorithms = ["hungarian", "greedy"]
    done_vector_methods = ["topdown"]
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


