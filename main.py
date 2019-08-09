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
from bert_serving.client import BertClient

from utils import load_embeddings, train_gbm
from utils_d.utils import text_pipeline
from utils_d.ml_utils import predict_binary, fit_pca, plot_pca

from SIF.src import params
from SIF.src.data_io import seq2weight
# from SIF.src.data_io import sentences2idx
from SIF.src.SIF_embedding import SIF_embedding
# from utils_d.ml_utils import create_word_index


# bert client
bc = None


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


def construct_feature_vector(lemma, synset_vec, add_vec,
                             all_lemmas,
                             augment=True, emb=None,
                             all_lemmas_vecs=None):
    if emb:
        lemma_vec = emb[lemma]
    # No embeddings loaded; use Bert
    elif type(all_lemmas_vecs) is np.ndarray and lemma in all_lemmas:
        lemma_vec = all_lemmas_vecs[all_lemmas[lemma]]
    else:
        lemma_vec = get_bert_embeddings([lemma])[0]
    vectors = [lemma_vec, synset_vec]
    if augment:
        vectors.append(add_vec)
    # diff_vec = lemma_vec / synset_vec
    # vectors.append(diff_vec)
    vector = np.concatenate(vectors)
    return vector


def create_synset_dataset(synsets, syn_names, all_lemmas, syn_mtx,
                          syn_ind, pos_limit="",
                          emb=None,
                          add_zeroes=True,
                          all_lemmas_vecs=None):
    X = []
    Y = []
    all_lemmas_keys = list(all_lemmas.keys())
    for k, v in synsets.items():
        name = "".join(k.split(".")[:-2])
        pos = get_synset_pos(k)

        # dog.n.01
        if pos_limit and k.split(".")[-2] != pos_limit:
            continue
        meaning = k.split(".")[-1]
        meaning = int(meaning) / 60
        if emb:
            v = {lemma for lemma in v if lemma in emb}
        # v = {lemma for lemma in v if lemma != name}
        name_lemmas = syn_names[name]
        if emb:
            name_lemmas = {lemma for lemma in name_lemmas if lemma in emb}
        # name_lemmas = {lemma for lemma in name_lemmas if lemma != name}
        union = name_lemmas & v
        difference = name_lemmas.difference(v)
        add_vec = [meaning] + pos
        k_vec = syn_mtx[syn_ind[k]]
        for lemma in union:
            vector = construct_feature_vector(
                lemma, k_vec, add_vec, all_lemmas, emb=emb,
                all_lemmas_vecs=all_lemmas_vecs)
            X.append(vector)
            Y.append(1)
        for lemma in difference:
            vector = construct_feature_vector(
                lemma, k_vec, add_vec, all_lemmas, emb=emb,
                all_lemmas_vecs=all_lemmas_vecs)
            X.append(vector)
            Y.append(0)
        left_random = len(union) - len(difference)
        if add_zeroes:
            if left_random > 0:
                for i in range(left_random):
                    random_lemma = random.choice(all_lemmas_keys)
                    if random_lemma not in v:
                        vector = construct_feature_vector(
                            random_lemma, k_vec, add_vec, all_lemmas, emb=emb,
                            all_lemmas_vecs=all_lemmas_vecs)
                        X.append(vector)
                        Y.append(0)
    return X, Y


def create_all_lemmas(synsets, emb=None):
    all_lemmas = list(set([w for v in synsets.values() for w in v]))
    if emb:
        all_lemmas = [l for l in all_lemmas if l in emb]
    all_lemmas = {l: l_i for l_i, l in enumerate(all_lemmas)}
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


def sif_embeddings(emb, sentences: dict):
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


def load_khodak_dataset(lang, synsets, lang_emb=None):
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
    if type(lang_emb) == np.ndarray or lang_emb:
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


def get_bert_embeddings(sentences: List[str]):
    if bc:
        local_bc = bc
    else:
        local_bc = BertClient(ip='10.8.0.3')  # ip address of the GPU machine
    embeddings = local_bc.encode(sentences)
    return embeddings


def april_khodak():
    USE_BERT = True
    USE_FOREIGN = False
    ADD_ZEROES = True
    USE_KFOLD = False
    TFIDF = False
    USE_SIF = False
    USE_MUSE = False
    POS_LIMIT = None
    USE_PCA = False
    FINE_TUNE_KHODAK = False
    USE_UMAP = False
    if USE_SIF:
        TFIDF = False
    if not USE_BERT:
        eng_emb = load_embeddings("en", muse=USE_MUSE)
        vector_size = eng_emb.vector_size
    else:
        global bc
        bc = BertClient(ip='10.8.0.3')
        vector_size = 768
        eng_emb = None
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
        if not USE_BERT:
            definition = [w for w in definition if
                          w not in stops and w in eng_emb]
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

    if TFIDF and not USE_BERT:
        vocabulary = {w for d in definitions.values() for w in d}
        vocabulary = {v: v_i for v_i, v in enumerate(vocabulary)}

        vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,
            lowercase=False,
            vocabulary=vocabulary.keys())
        vectorizer = vectorizer.fit(definitions.values())
    synset_matrix = np.zeros(
        shape=(len(synset_index), vector_size))
    # np.save('models/synset_mtx.npy', synset_matrix)
    if USE_BERT:
        sentences = list(definitions.values())
        sentences = [" ".join(s) for s in sentences]
        synset_matrix = get_bert_embeddings(sentences)
    elif USE_SIF:
        synset_matrix = sif_embeddings(eng_emb, definitions.values())
    if not USE_BERT:
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
    if USE_BERT:
        all_lemmas_vecs = get_bert_embeddings(list(all_lemmas.keys()))
    else:
        all_lemmas_vecs = None
    X, Y = create_synset_dataset(
        synsets, synset_names, all_lemmas, synset_matrix,
        synset_index, add_zeroes=ADD_ZEROES, pos_limit=POS_LIMIT, emb=eng_emb,
        all_lemmas_vecs=all_lemmas_vecs)

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
                lang_synsets, lang_name_synsets, lang_lemmas,
                synset_matrix, synset_index,
                add_zeroes=ADD_ZEROES, emb=lang_emb)
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
        if not USE_BERT:
            print(lang)
            lang_emb = load_embeddings(lang, muse=USE_MUSE)
        else:
            lang_emb = None
        khodak_df = load_khodak_dataset(lang, synsets, lang_emb=lang_emb)
        f1_scores = []
        pos_s = ["", "n", "a", "v"]
        for pos in pos_s:
            if pos == "":
                pred_df = khodak_df
            else:
                pred_df = khodak_df[khodak_df["pos"] == pos]
            if lang_emb:
                lang_input = lang_emb[pred_df[lang].values]
            else:
                # should have used unique and then map, but I'm too lazy
                # the difference is about 10x times
                lang_input = list(pred_df[lang].values)
                lang_input = get_bert_embeddings(lang_input)
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
    version = 2
    langs = ['af', 'ar', 'bg', 'bn', 'bs', 'ca', 'cs', 'da', 'de', 'el', 'en',
             'es', 'et', 'fa', 'fi', 'fr', 'he', 'hi', 'hr', 'hu', 'id', 'it',
             'ko', 'lt', 'lv', 'mk', 'ms', 'nl', 'no', 'pl', 'pt', 'ro', 'ru',
             'sk', 'sl', 'sq', 'sv', 'ta', 'th', 'tl', 'tr', 'uk', 'vi', 'zh']
    punct = punctuation.replace("_", "")
    punct += "\xa0—„…»゛–"
    ensemble = pickle.load(open("models/best2/ensemble.pcl", "rb"))
    nn_model = ensemble.models[4]

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
    batch_mtx = np.zeros(shape=(2000000, synset_mtx.shape[1]))  # 2 mln rows
    batch_mtx_len = batch_mtx.shape[0]
    # predicts the language of the word
    lang_model = fastText.load_model("muse_embeddings/lid.176.bin")
    for lang in langs:
        if os.path.exists(
                f"wordnets_constructed/colocations_{lang}_{version}"):
            continue
        print(f"---{lang}---")
        # if os.path.exists(f"wordnets_constructed/all_{lang}_2"):
        #     continue
        url = "https://dl.fbaipublicfiles.com/fasttext/"\
              f"vectors-aligned/wiki.{lang}.align.vec"
        wget.download(url)
        filename = f"wiki.{lang}.align.vec"
        new_filename = f"muse_embeddings/{filename}"
        os.rename(filename, new_filename)
        lang_emb = load_embeddings(lang, muse=False)
        if lang not in ("ru", "fr", "en"):
            os.remove(new_filename)
        # 1888418
        allowed_vocab = lang_emb.vocab
        print("allowed vocab len", len(allowed_vocab))
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
        print("final allowed vocab len", len(allowed_vocab))
        collocations = [w for w in allowed_vocab if "_" in w and
                        len([l for l in w.split("_") if l]) > 1]
        print("collocations len", len(collocations))
        for dataset_i, dataset in enumerate([allowed_vocab, collocations]):
            coloc = "colocations" if dataset_i == 1 else "all"
            f_lang_name = f"wordnets_constructed/{coloc}_{lang}_{version}"
            if os.path.exists(f_lang_name):
                with open(f_lang_name) as f:
                    processed = f.readlines()
                processed = [t.split("\t")[0] for t in processed]
                if processed:
                    last_processed = dataset.index(processed[-1])
                    dataset = dataset[last_processed + 1:]
            else:
                last_processed = 0

            f_lang = open(f_lang_name, "a")
            last_batch_ind = 0
            batch_dict = dict()

            for v_i, v in enumerate(dataset):
                print(f"{v_i:<5} {v:<20}", end="\r")
                if v_i + last_processed > 10000:
                    break
                emb_v = lang_emb[v]
                synset_mtx[:, :300] = emb_v

                prelim_preds = nn_model.predict(synset_mtx)
                prelim_preds = prelim_preds.T[0]

                selected_preds = np.where(prelim_preds > 0.1)[0]
                preds_len = selected_preds.shape[0]

                if preds_len >= batch_mtx_len - last_batch_ind or\
                        v_i == len(dataset) - 1:
                    # then predict the batch and reset it
                    preds = predict_binary(ensemble,
                                           # to save processor time (doubtful?)
                                           batch_mtx[:last_batch_ind],
                                           threshold=0.6)
                    synsets = np.where(preds == 1)[0]
                    for batch_k, batch_v in batch_dict.items():
                        batch_synsets = []
                        low_lim, up_lim = batch_v[:2]
                        for s_i, s in enumerate(synsets):
                            if low_lim <= s < up_lim:
                                s -= low_lim
                                batch_synsets.append(s)
                            # the lower limit is always respected
                            else:
                                synsets = synsets[s_i:]
                                break
                        if len(batch_synsets) > 10:
                            continue
                        batch_synsets = [
                            reverse_index[s] for s in batch_synsets]
                        for s in batch_synsets:
                            print(batch_k, s)
                            f_lang.write("{}\t{}\n".format(batch_k, s))
                    # reset the batch
                    last_batch_ind = 0
                    batch_dict = dict()
                # add new rows to the batch matrix

                # update upper limit
                upper_lim = last_batch_ind + preds_len
                # set batch to new values
                batch_mtx[
                    last_batch_ind: upper_lim] = synset_mtx[selected_preds]
                # memorize batch info
                batch_dict[v] = (last_batch_ind, upper_lim, selected_preds)
                # update last_batch_ind
                last_batch_ind = upper_lim

            f_lang.close()


if __name__ == "__main__":
    # main()
    april_khodak()
    # create_wordnets()
