import pandas as pd
import spacy
from tqdm import tqdm
import multiprocessing as mp
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import os


def prepare_data(df, batch_size=1000, num_processes=None):
    if os.path.isfile("data/prepared.parquet"):
        print("[+] Loading data from prepared.parquet.")
        return pd.read_parquet("data/prepared.parquet")

    # the best model for NER is trf (https://spacy.io/models/en), but it's too slow without a GPU
    nlp = spacy.load("en_core_web_sm")

    if num_processes is None:
        num_processes = mp.cpu_count()

    results = []
    for doc in tqdm(
        nlp.pipe(df["text"], batch_size=batch_size, n_process=num_processes),
        total=len(df["text"]),
        desc="Processing pretraining data",
    ):
        result = _process_doc(doc)
        results.extend(result)

    out_df = pd.DataFrame(results)
    out_df.to_parquet("data/prepared.parquet")

    return out_df


def _process_doc(doc):
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


def process_data(prepared_df, cols_to_encode):
    columns_to_keep = list(set(prepared_df.columns) - set(cols_to_encode))
    encoded_df = prepared_df[columns_to_keep].copy()

    ordinal_mappings = {}
    for col in cols_to_encode:
        ordinal_encoder = OrdinalEncoder(dtype=np.int8)

        encoded_col = ordinal_encoder.fit_transform(prepared_df[[col]])
        col_name = ordinal_encoder.get_feature_names_out()[0]
        encoded_df[col_name] = encoded_col.flatten()

        # keep track of the mappigs for later (note that the encoding proceeds in the order in which the features appear in the list => the first element is encoded as 0, the second as 1, etc.)
        ordinal_mappings[col_name] = ordinal_encoder.categories_[0]

    return encoded_df, ordinal_mappings
