from datasets import load_dataset
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import os
import shutil

from .utils import count_links, save_checkpoint


def has_alpha(s):
    """The string should not be made of only punctuation and digits."""
    return any(c.isalpha() for c in s.split(" (")[0])


def filter_out_pages(
    page_titles, highly_ambiguous_entities, threshold, checkpoint_file
):
    for page_title in page_titles:
        if page_title in highly_ambiguous_entities:
            # skip the ones already considered
            continue

        link_count = count_links(page_title)
        if link_count > threshold:
            if has_alpha(page_title):
                highly_ambiguous_entities[page_title] = link_count
                save_checkpoint(checkpoint_file, highly_ambiguous_entities)


def delete_non_ambiguous_entities():
    """We only want to keep the entities that are actually ambiguous (i.e., the ones for which we have at least two subdirs)."""
    path = "data/snippets"
    for disambig in os.listdir(path):
        disambig_path = os.path.join(path, disambig)

        if not os.path.isdir(disambig_path):
            continue

        subdirs = [
            d
            for d in os.listdir(disambig_path)
            if os.path.isdir(os.path.join(disambig_path, d))
        ]

        # we don't need to disambiguate a non-ambiguous entity, i.e., one for which there are less than 2 subdirs
        if len(subdirs) < 2:
            shutil.rmtree(disambig_path)


def delete_dirs_with_different_entities():
    """This function removes the directories that do not have any mention of the disambiguation entity in their text.

    This can happen because of the way pages were initially filtered and because following links doesn't guarantee that the title string will be in a certain page.
    """
    path = "data/snippets"
    for disambig in os.listdir(path):
        disambig_path = os.path.join(path, disambig)
        if not os.path.isdir(disambig_path):
            continue

        subdirs_to_keep = []
        for subdir in os.listdir(disambig_path):
            subdir_path = os.path.join(disambig_path, subdir)
            if not os.path.isdir(subdir_path):
                continue

            entity_name = disambig.split("(")[0].strip()
            if entity_name in subdir:
                subdirs_to_keep.append(subdir)
            else:
                shutil.rmtree(subdir_path)

        # we might be left with only one subdir after the deletions
        if len(subdirs_to_keep) < 2:
            shutil.rmtree(disambig_path)


def process_pretrain_dataset(train_cut, dev_cut, test_cut, cleanup_cache=True):
    # https://huggingface.co/datasets/lucadiliello/wikipedia_512_pretraining
    dataset_name = "lucadiliello/wikipedia_512_pretraining"
    splits = ["train", "dev", "test"]
    dataset = load_dataset(dataset_name)

    all_data = []

    for split, cut_percentage in zip(splits, [train_cut, dev_cut, test_cut]):
        # the cuts should be in the (0, 1] range
        samples_to_keep = int(len(dataset[split]) * cut_percentage)
        resized_split = dataset[split].shuffle(seed=42).select(range(samples_to_keep))
        df = resized_split.to_pandas()
        all_data.append(df)
    # we do not care which split the data comes from
    combined_df = pd.concat(all_data, ignore_index=True)

    output_file = "data/pretrain/combined.parquet"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pq.write_table(pa.Table.from_pandas(combined_df), output_file)

    if cleanup_cache:
        # to avoid needlessly occupy space on disk
        dataset.cleanup_cache_files()
