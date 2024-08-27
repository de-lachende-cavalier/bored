import spacy

from tqdm import tqdm
import multiprocessing as mp
from itertools import islice
import random
import os

from sklearn.preprocessing import OrdinalEncoder

import torch
import numpy as np
import pandas as pd


def get_dataframe_for_pretraining(pretrain_df, batch_size=1000, num_processes=None):
    if os.path.isfile("data/prepared.parquet"):
        print("[+] Loading data from prepared.parquet.")
        return pd.read_parquet("data/prepared.parquet")

    # the best model for NER is trf (https://spacy.io/models/en), but it's too slow without a GPU
    nlp = spacy.load("en_core_web_sm")

    if num_processes is None:
        num_processes = mp.cpu_count()

    results = []
    for doc in tqdm(
        nlp.pipe(pretrain_df["text"], batch_size=batch_size, n_process=num_processes),
        total=len(pretrain_df["text"]),
        desc="Processing pretraining data",
    ):
        result = process_doc(doc)
        results.extend(result)

    out_df = pd.DataFrame(results)
    out_df.to_parquet("data/prepared.parquet")

    return out_df


def process_doc(doc):
    out = []
    for sent_idx, sent in enumerate(doc.sents):
        for token in sent:
            out.append(
                {
                    "sentence_id": sent_idx,
                    "token": token.text,
                    "pos": token.pos_,
                    "dep": token.dep_,
                    "ner_tag": token.ent_type_ if token.ent_type_ != "" else "NONE",
                }
            )
    return out


def encode_categorical(prepared_df, cols_to_encode):
    columns_to_keep = list(set(prepared_df.columns) - set(cols_to_encode))
    encoded_df = prepared_df[columns_to_keep].copy()

    mappings = {}
    for col in cols_to_encode:
        ordinal_encoder = OrdinalEncoder(dtype=np.int8)

        encoded_col = ordinal_encoder.fit_transform(prepared_df[[col]])
        col_name = ordinal_encoder.get_feature_names_out()[0]
        encoded_df[col_name] = encoded_col.flatten()

        # keep track of the mappigs for later (note that the encoding proceeds in the order in which the features appear in the list => the first element is encoded as 0, the second as 1, etc.)
        mappings[col_name] = ordinal_encoder.categories_[0]

    return encoded_df, mappings


def construct_traintest_dataframe(nlp, target_entity, data_dict, train_size=50):
    n_disambigs_target = len(data_dict[target_entity])
    valid_ents = [
        ent
        for ent in data_dict.keys()
        if (ent != target_entity) and (len(data_dict[ent]) >= n_disambigs_target)
    ]

    if train_size:
        # if a training size is specified, only choose enough entities to match it
        valid_ents = random.sample(valid_ents, min(train_size, len(valid_ents)))
    valid_ents.append(target_entity)  # include the target entity to process it

    train_dict = {}
    for ent in valid_ents:
        # we want the same number of disambiguations for each entity in the training set
        train_dict[ent] = dict(islice(data_dict[ent].items(), n_disambigs_target))

    df = _process_text_snips(nlp, train_dict)

    disambig_index_map = {
        entity: {
            key: idx
            for idx, key in enumerate(
                sorted(df[df["entity"] == entity]["disambig_key"].unique())
            )
        }
        for entity in df["entity"].unique()
    }

    df["disambig_label"] = df.apply(
        lambda row: disambig_index_map[row["entity"]][row["disambig_key"]], axis=1
    )

    return df


def get_mlp_data(X, y, born_clf, features, mappings):
    columns = mappings["ner_tag"]

    mlp_X = []
    mlp_y = []
    for i, x in enumerate(X):
        x_explanation = pd.DataFrame(
            born_clf.explain(x).toarray(), index=features, columns=columns
        )
        most_likely_y = x_explanation.sum().idxmax()

        mlp_features = x_explanation[most_likely_y].to_list()
        num_ner_tag = list(mappings["ner_tag"]).index(most_likely_y)
        mlp_features.append(num_ner_tag)

        mlp_X.append(mlp_features)
        # HACK: this adjustment makes the function work with single data points
        mlp_y.append(y[i] if isinstance(y, list) else y)

    return torch.tensor(mlp_X), torch.tensor(mlp_y)


def _process_text_snips(nlp, d, entity=None, disambig_key=None):
    if isinstance(d, dict):
        dfs = []
        for k, v in d.items():
            if entity is None:
                # top level, so k is the entity name
                df = _process_text_snips(nlp, v, entity=k)
            else:
                # inside an entity, so k is the disambig key
                df = _process_text_snips(nlp, v, entity=entity, disambig_key=k)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)
    elif isinstance(d, list):
        dfs = [
            _process_text_snips(nlp, item, entity=entity, disambig_key=disambig_key)
            for item in d
        ]
        return pd.concat(dfs, ignore_index=True)
    elif isinstance(d, str):
        doc = nlp(d)
        processed = process_doc(doc)
        df = pd.DataFrame(processed)
        df["entity"] = entity
        df["disambig_key"] = disambig_key
        return df
    else:
        return pd.DataFrame()
