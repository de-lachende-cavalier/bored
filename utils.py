import pandas as pd


def expand_conll_dataframe(df):
    """Expands the CoNLL dataset provided by HuggingFace. (A necessary step for further processing.)"""

    data = {
        "sentence_id": [],
        "token": [],
        "pos_tag": [],
        "chunk_tag": [],
        "ner_tag": [],
    }

    for i, row in df.iterrows():
        tokens = row["tokens"]
        pos_tags = row["pos_tags"]
        chunk_tags = row["chunk_tags"]
        ner_tags = row["ner_tags"]
        min_length = min(len(tokens), len(pos_tags), len(chunk_tags), len(ner_tags))
        tokens = tokens[:min_length]
        pos_tags = pos_tags[:min_length]
        chunk_tags = chunk_tags[:min_length]
        ner_tags = ner_tags[:min_length]

        data["sentence_id"].extend([i] * min_length)
        data["token"].extend(tokens)
        data["pos_tag"].extend(pos_tags)
        data["chunk_tag"].extend(chunk_tags)
        data["ner_tag"].extend(ner_tags)

    expanded_df = pd.DataFrame(data)
    return expanded_df
