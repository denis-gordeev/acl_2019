import itertools
import os
import re
import time
import random
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import spatial
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
import pymorphy2
from nltk.corpus import wordnet

from utils import load_embeddings, get_now, annotate
from utils_d.ml_models import CnnLstm, train_gbm


def score_khodak(path):
    """
    unfinished
    """
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
    # t_p = 0
    # f_p = 0
    # t_n = 0
    # f_n = 0
    for index, row in khodak.iterrows():
        if row["ru"] in my_df["class10_ru"]:
            print(row["ru"])
        else:
            print(row["ru"])
    pass


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
    now = get_now()
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
        x, y, test_size=0.1, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=42)
    # cnn_model = train_cnn(
    #     all_words, eng_emb, x_train, x_test, y_train, y_test, word_matrix)
    gbm = train_gbm(x_train, x_val, y_train, y_val, f"{folder}/main")

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
        # hypernyms = {k: v for k, v in hypernyms.items()
        # if k not in to_delete}
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

    to_match = [(w, k) for k, v in reverse.items() for w in v]
    random.shuffle(to_match)
    match_df = pd.DataFrame(to_match)
    annotate(match_df, "wordnet_nouns", 0, limit=200)
