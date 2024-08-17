import pandas as pd
import spacy
from tqdm import tqdm
import multiprocessing as mp
import os


def prepare_data(df, batch_size=1000, num_processes=None):
    if os.path.isfile("data/prepared.parquet"):
        print("[+] Loading data from prepared.parque...")
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
