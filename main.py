import random
import json
import pickle
import os

from collections import Counter
from typing import List
# import multiprocessing as mp
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras.callbacks import EarlyStopping
import umap
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from string import punctuation
import fastText

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
# from sklearn.model_selection import RepeatedStratifiedKFold

import wget

from utils import load_embeddings, train_gbm
from utils_d.utils import text_pipeline
from utils_d.ml_utils import predict_binary, fit_pca, plot_pca

from SIF.src import params
from SIF.src.data_io import seq2weight
# from SIF.src.data_io import sentences2idx
from SIF.src.SIF_embedding import SIF_embedding
# from utils_d.ml_utils import create_word_index


def keras_model(x_train, y_train, x_test, y_test):
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                       verbose=0, mode='auto')
    inputs = Input(shape=(x_train.shape[1], ))

    norm_1 = Dropout(0.2)(inputs)
    # norm_1 = BatchNormalization()(inputs)
    dense_1 = Dense(800, activation='relu')(norm_1)

    norm_2 = Dropout(0.2)(dense_1)
    # norm_2 = BatchNormalization()(dense_1)
    dense_2 = Dense(512, activation='relu')(norm_2)

    norm_3 = Dropout(0.1)(dense_2)
    # norm_3 = BatchNormalization()(dense_2)
    dense_3 = Dense(256, activation='relu')(norm_3)

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


def construct_feature_vector(lemma, emb, synset_vec, add_vec, augment=True):
    lemma_vec = emb[lemma]
    vectors = [lemma_vec, synset_vec]
    if augment:
        vectors.append(add_vec)
    # diff_vec = lemma_vec / synset_vec
    # vectors.append(diff_vec)
    vector = np.concatenate(vectors)
    return vector


def create_synset_dataset(synsets, syn_names, all_lemmas, emb, syn_mtx,
                          syn_ind, pos_limit="",
                          add_zeroes=True):
    X = []
    Y = []

    for k, v in synsets.items():
        name = "".join(k.split(".")[:-2])
        pos = get_synset_pos(k)

        # dog.n.01
        if pos_limit and k.split(".")[-2] != pos_limit:
            continue
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
            vector = construct_feature_vector(lemma, emb, k_vec, add_vec)
            X.append(vector)
            Y.append(1)
        for lemma in difference:
            vector = construct_feature_vector(lemma, emb, k_vec, add_vec)
            X.append(vector)
            Y.append(0)
        left_random = len(union) - len(difference)
        if add_zeroes:
            if left_random > 0:
                for i in range(left_random):
                    random_lemma = random.choice(all_lemmas)
                    if random_lemma not in v:
                        vector = construct_feature_vector(
                            random_lemma, emb, k_vec, add_vec)
                        X.append(vector)
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


def sif_embeddings(emb, sentences: List):
    counter = Counter()
    for s in sentences:
        for w in s:
            counter[w] += 1
    weightpara = 1e-3
    We = emb.vectors
    # {index: word} dict
    words = {i: w for i, w in enumerate(emb.vocab)}
    # {word: index} dict
    words_reverse = {w: i for i, w in words.items()}
    total = 0

    word2weight = dict()
    for k, v in counter.items():
        word2weight[k] = v
        total += v
    for k, v in word2weight.items():
        word2weight[k] = weightpara / (weightpara + v / total)
    weight4ind = dict()
    for ind, word in words.items():
        if word in word2weight:
            weight4ind[ind] = word2weight[word]
        else:
            weight4ind[ind] = 1.0
    max_sent = max([len(s) for s in sentences])
    x = np.zeros(shape=(len(sentences), max_sent), dtype=int)
    m = np.zeros(shape=(len(sentences), max_sent), dtype=int)
    for s_i, s in enumerate(sentences):
        sent_indices = [words_reverse[w] for w in s]
        x[s_i, :len(sent_indices)] = sent_indices
        m[s_i, :len(sent_indices)] = 1
    w = seq2weight(x, m, weight4ind)

    param = params.params()
    param.rmpc = 1
    # get SIF embedding
    embedding = SIF_embedding(We, x, w, param)
    return embedding


def get_f1s(range, divisor, model, x_test, y_test):
    threshold = []
    for t in range:
        print(f"threshold {t: 10}", end="\r")
        f1 = [f1_score(y_test,
                       predict_binary(model, x_test, threshold=t / divisor))]
        if threshold:
            if f1 < threshold[-1]:
                break
        threshold.append(f1)
    threshold = np.argmax(threshold)
    return threshold


def calculate_threshold(model, x_test, y_test):
    threshold = get_f1s(range(1, 10), 10, model, x_test, y_test)
    adjust_min = threshold * 10
    adjust = get_f1s(
        range(adjust_min, adjust_min + 20), 100, model, x_test, y_test)
    threshold = round(threshold * 0.1 + adjust * 0.01, 3)
    return threshold


def load_khodak_dataset(lang, lang_emb, synsets):
    if lang not in ("ru", "fr"):
        return None
    with open(f"other_works/pawn/{lang}_matches.txt") as f:
        khodak = f.readlines()
    khodak = [l.strip().split("\t") for l in khodak]

    # convert data to
    # ад  chaos.n.01          1
    # ад  conflagration.n.01  0
    # ад  gehenna.n.01        1
    # ад  hell.n.01           1
    # ад  hell.n.04           0
    khodak_data = []
    lang_word = ""
    pos = ""
    for l in khodak:
        if len(l) == 2 and ":" in l[0]:
            lang_word = l[1]
        elif len(l) == 2 and l[0] == "#":
            pos = l[1]
        elif len(l) == 3:
            synset = l[1]
            target = l[0]
            if synset in synsets:
                khodak_data.append([lang_word, synset, target, pos])
    # not_found = len([l for l in khodak_data if l[0] not in ru_emb])
    khodak_data = [l for l in khodak_data if l[0] in lang_emb]
    khodak_df = pd.DataFrame(
        khodak_data, columns=[lang, "synset", "target", "pos"])
    khodak_df["target"] = khodak_df["target"].astype(int)
    return khodak_df


def dim_reducted(X, pca_model, scaler, umap_model=None, USE_UMAP=False):
    scaled = scaler.transform(X)
    vectors = pca_model.transform(scaled)[:, :30]
    if USE_UMAP:
        X_umap = umap_model.transform(X)
        vectors = np.concatenate([vectors, X_umap], axis=1)
    X = np.concatenate([X, vectors], axis=1)
    return X


def plot_synsets(synset_matrix, synset_list):
    labels = [s.split(".")[-2] for s in synset_list]
    labels_dict = {"n": "nouns",
                   "a": "adjectives",
                   "v": "verbs",
                   "s": "adjective satellites",
                   "r": "adverbs"}
    for l_i, l in enumerate(labels):
        labels[l_i] = labels_dict[l]
    vectors, _, _, _ = fit_pca(synset_matrix)
    index = [i for i in range(len(labels))]
    random.shuffle(index)
    index = index[:3000]
    labels = [labels[i] for i in index]
    vectors = vectors[index]
    vectors = vectors[:, :30]
    vectors = TSNE(n_components=2).fit_transform(vectors)
    plot_pca(vectors, labels, y_name="Parts of speech",
             path="wordnet_", show_figures=False)


def april_khodak():
    USE_FOREIGN = True
    ADD_ZEROES = True
    USE_KFOLD = True
    TFIDF = True
    USE_SIF = True
    USE_MUSE = True
    POS_LIMIT = None
    USE_PCA = False
    FINE_TUNE_KHODAK = False
    USE_UMAP = False
    if USE_SIF:
        TFIDF = False

    eng_emb = load_embeddings("en", muse=USE_MUSE)

    stops = set(stopwords.words('english'))
    # get synsets + their lemmas
    # 117659 synsets
    synsets = {w.name(): [l.name().lower().replace(".", "")
                          for l in w.lemmas()]
               for w in wordnet.all_synsets()}
    synsets = {k: v for k, v in synsets.items() if v}
    # synsets = {k: [w for w in v if w in eng_emb] for k, v in synsets.items()}
    # 117659 synsets
    synsets = {k: set(v) for k, v in synsets.items() if v}

    # no_embedding_synsets = set()
    definitions = dict()
    for s in synsets:
        definition = wordnet.synset(s).definition()
        definition = text_pipeline(definition, lemmatize=False)
        definition = [w for w in definition if w not in stops and w in eng_emb]
        # if not definition:
        #     print(wordnet.synset(s).definition())
        if definition:
            definitions[s] = definition
    # 117589
    synsets = {k: v for k, v in synsets.items() if k in definitions}

    synset_list = list(synsets.keys())
    synset_index = {w: i for i, w in enumerate(synset_list)}

    with open(f'synsets/synset_index.json', 'w') as outfile:
        json.dump(synset_index, outfile)

    if TFIDF:
        vocabulary = {w for d in definitions.values() for w in d}
        vocabulary = {v: v_i for v_i, v in enumerate(vocabulary)}

        vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,
            lowercase=False,
            vocabulary=vocabulary.keys())
        vectorizer = vectorizer.fit(definitions.values())
    synset_matrix = np.zeros(
        shape=(len(synset_index), eng_emb.vector_size))
    # np.save('models/synset_mtx.npy', synset_matrix)
    if USE_SIF:
        synset_matrix = sif_embeddings(eng_emb, definitions.values())
    for s_i, s in enumerate(synset_list):
        print("\t", s_i, s, len(synset_list), end="\r")
        lemmas = synsets[s]
        weights = (0.2, 0.8)
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
            else:
                no_lemmas = True
        else:
            lemmas = emb_lemmas
            # gensim KeyedVectors also accept lists of words
            embeddings = eng_emb[lemmas]
            embeddings = np.average(embeddings, axis=0)

        def_weights = None
        if TFIDF:
            definition = definitions[s]
            def_weights = vectorizer.transform(
                [definition]).toarray()[0]
            def_weights = [def_weights[vocabulary[w]]
                           for w in definition]
            definition_emb = np.average(eng_emb[definition], axis=0,
                                        weights=def_weights)
        else:
            def_weights = None
            definition_emb = synset_matrix[s_i]
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
        synset_index, add_zeroes=ADD_ZEROES, pos_limit=POS_LIMIT)

    lang_embeddings = {"fi": "fin",
                       # "pl": "pol",
                       # "sv": "swe"
                       }
    if USE_FOREIGN:
        for emb_lng, wn_lng in lang_embeddings.items():
            lang_emb = load_embeddings(emb_lng, muse=USE_MUSE)
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
    if USE_PCA:
        pca_v, v_scaled, pca_model, scaler = fit_pca(X)
        if USE_UMAP:
            umap_model = umap.UMAP(n_neighbors=20, min_dist=0.1,
                                   metric='correlation')
            umap_model.fit(X)
        else:
            umap_model = None
        X = dim_reducted(X, pca_model, scaler, umap_model=umap_model,
                         USE_UMAP=USE_UMAP)
    Y = np.array(Y)
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(
        X, Y, test_size=0.1, random_state=42)

    if USE_KFOLD:
        ensemble = train_gbm(x_train, x_val,
                             y_train, y_val, f"synsets/main2",
                             k_fold=USE_KFOLD)
        nn_models = k_fold_training(x_train, y_train, keras_model, num_folds=4)
        ensemble = ensemble[0]
        ensemble.models += nn_models
    else:
        gbm = train_gbm(x_train, x_val,
                        y_train, y_val, f"synsets/main2",
                        k_fold=USE_KFOLD)
        gbm = gbm[0]
        nn_model = keras_model(x_train, y_train, x_val, y_val)
    if USE_KFOLD:
        # threshold between [0.1, 0.9]
        ensemble_threshold = calculate_threshold(ensemble, x_test, y_test)

    for lang in ["ru", "fr"]:
        print(lang)
        lang_emb = load_embeddings(lang, muse=USE_MUSE)
        khodak_df = load_khodak_dataset(lang, lang_emb, synsets)
        f1_scores = []
        pos_s = ["", "n", "a", "v"]
        for pos in pos_s:
            if pos == "":
                pred_df = khodak_df
            else:
                pred_df = khodak_df[khodak_df["pos"] == pos]

            lang_input = lang_emb[pred_df[lang].values]
            # embeddings for all ru words
            eng_input = [synset_index[s] for s in pred_df["synset"].values]
            # embeddings for all eng words
            eng_input = synset_matrix[eng_input]

            syn_pos = [get_synset_pos(s) for s in pred_df["synset"].values]

            meaning = [[int(s.split(".")[-1]) / 60]
                       for s in pred_df["synset"].values]

            input_data = np.concatenate(
                [lang_input, eng_input, meaning, syn_pos],
                axis=1)
            if USE_PCA:
                input_data = dim_reducted(
                    input_data, pca_model, scaler,
                    umap_model=umap_model,
                    USE_UMAP=USE_UMAP)
            # preds = np.argmax(preds, axis=1).astype(bool)  # &\
            # np.max(preds, axis=1) > 0.6
            # preds = preds.astype(int)
            # print(pos, "gbm", f1_score(pred_df["target"].values, preds))
            if USE_KFOLD:
                target = pred_df["target"].values

                if FINE_TUNE_KHODAK:
                    half = int(input_data.shape[0] / 2)
                    ensemble_threshold = calculate_threshold(
                        ensemble, input_data[:half], target[:half])
                    input_data = input_data[half:]
                    target = target[half:]
                preds = predict_binary(
                    ensemble, input_data, threshold=ensemble_threshold)
                f1 = f1_score(target, preds)
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
                        f1 = f1_score(
                            pred_df["target"].values, threshold_preds)
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
        # pred_df["preds"] = preds


def save_ensemble(ensemble, folder="models"):
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass
    pickle.dump(ensemble, open(f"{folder}/ensemble.pcl", "wb"))
    for m_i, m in enumerate(ensemble.models):
        if "keras" in str(m.__class__):
            m.save(f"{folder}/_keras_{m_i}")
        else:
            pickle.dump(m, open(f"{folder}/__lgbm_{m_i}", "wb"))


def create_wordnets():
    langs = ['af', 'ar', 'bg', 'bn', 'bs', 'ca', 'cs', 'da', 'de', 'el', 'en',
             'es', 'et', 'fa', 'fi', 'fr', 'he', 'hi', 'hr', 'hu', 'id', 'it',
             'ko', 'lt', 'lv', 'mk', 'ms', 'nl', 'no', 'pl', 'pt', 'ro', 'ru',
             'sk', 'sl', 'sq', 'sv', 'ta', 'th', 'tl', 'tr', 'uk', 'vi', 'zh']
    punct = punctuation.replace("_", "")
    punct += "\xa0—„…»゛–"
    ensemble = pickle.load(open("models/best2/ensemble.pcl", "rb"))
    nn_model = ensemble.models[4]
    threshold = 0.41

    with open("synsets/synset_index.json") as f:
        r = f.read()
        synset_index = json.loads(r)
    reverse_index = {v: k for k, v in synset_index.items()}
    syn_pos = [get_synset_pos(s) for s in synset_index.keys()]

    meaning = [[int(s.split(".")[-1]) / 60]
               for s in synset_index.keys()]

    synset_mtx = np.load('synsets/synset_mtx.npy')
    synset_mtx = np.concatenate([synset_mtx, syn_pos, meaning], axis=1)
    synset_zeros = np.zeros(shape=(synset_mtx.shape[0],
                                   300 + synset_mtx.shape[1])
                            )
    synset_zeros[:, 300:] = synset_mtx
    synset_mtx = synset_zeros

    # predicts the language of the word
    lang_model = fastText.load_model("muse_embeddings/lid.176.bin")
    for lang in langs:
        print(f"---{lang}---")
        if os.path.exists(f"wordnets_constructed/coloc_{lang}"):
            continue
        f_lang = open(f"wordnets_constructed/coloc_{lang}", "a")
        url = "https://dl.fbaipublicfiles.com/fasttext/"\
              f"vectors-aligned/wiki.{lang}.align.vec"
        wget.download(url)
        filename = f"wiki.{lang}.align.vec"
        new_filename = f"muse_embeddings/{filename}"
        os.rename(filename, new_filename)
        lang_emb = load_embeddings(lang, muse=False)
        # 1888418
        allowed_vocab = lang_emb.vocab
        print(len(allowed_vocab))
        # 1597995
        allowed_vocab = [w for w in allowed_vocab
                         if not any(p in w for p in punct)]
        allowed_vocab = list(allowed_vocab)

        langs = lang_model.predict(allowed_vocab)
        langs = langs[0]
        langs = [l[0].replace("__label__", "") for l in langs]
        # 1037723
        allowed_vocab = [w for w_i, w in enumerate(allowed_vocab)
                         if langs[w_i] == lang]
        # 1037102
        allowed_vocab = [w for w in allowed_vocab if len(w) >= 3]
        print(len(allowed_vocab))
        # allowed_vocab = [w for w in allowed_vocab if "_" in w]

        for v_i, v in enumerate(allowed_vocab):
            print(f"{v_i:<5} {v:<20}", end="\r")
            if v_i > 1000:
                break
            emb_v = lang_emb[v]
            synset_mtx[:, :300] = emb_v
            preds = predict_binary(ensemble,
                                   synset_mtx,
                                   threshold=0.51)
            synsets = np.where(preds == 1)[0]
            # preds = nn_model.predict(synset_mtx).T[0]
            # synsets = np.where(preds > 0.95)[0]
            if len(synsets) > 10:
                continue
            synsets = [reverse_index[s] for s in synsets]
            for s in synsets:
                f_lang.write("{}\t{}\n".format(v, s))
        os.remove(new_filename)
        f_lang.close()


if __name__ == "__main__":
    # main()
    # april_khodak()
    create_wordnets()
