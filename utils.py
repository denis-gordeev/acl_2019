import pickle

from datetime import datetime

import numpy as np

from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score, f1_score

from sklearn.metrics import confusion_matrix
from utils_d.ml_models import train_lgb
from utils_d.ml_utils import Ensemble


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


def get_now():
    """
    convert datetime to string
    """
    now = datetime.now().isoformat()
    now = now.split(".")[0]
    now = now.replace(":", "_")
    return now


def load_embeddings(lang, muse=True):
    if muse:
        path = (f'muse_embeddings/wiki.multi.{lang}.vec')
    else:
        path = (f'muse_embeddings/wiki.{lang}.align.vec')
    model = KeyedVectors.load_word2vec_format(path)
    print("loaded")
    return model


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
        'verbose': -1,
        "silent": True}
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
