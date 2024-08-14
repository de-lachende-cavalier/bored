import os
import shutil

import pandas as pd

import requests
import tarfile
import io

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


def get_pretrain_dataset(cut):
    # https://huggingface.co/datasets/jordiclive/wikipedia-summary-dataset
    data_url = (
        "https://thijsai.ams3.digitaloceanspaces.com/wiki-summary-dataset/raw.tar.gz"
    )

    response = requests.get(data_url)
    if response.status_code != 200:
        print(f"Failed to download the file. Status code: {response.status_code}")

    fbytes = io.BytesIO(response.content)
    with tarfile.open(fileobj=fbytes, mode="r:gz") as tar:
        # there's only one file in the archive (raw.txt)
        file_name = tar.getnames()[0]
        extracted_file = tar.extractfile(file_name)
        if extracted_file:
            content = extracted_file.read().decode("utf-8")
            # the last line is empty
            lines = content.split("\n")[:-1]

            clean_summaries = _preprocess_pretrain(lines)

            df = pd.DataFrame(clean_summaries, columns=["summary"])
            # shuffle the data before saving
            df = df.sample(frac=cut).reset_index(drop=True)
            df.to_parquet("data/pretrain.parquet")


def _preprocess_pretrain(lines):
    summaries = []
    for line in lines:
        parts = line.split("|||")
        summary = parts[1].strip()
        if summary.endswith("may refer to:"):
            # some summaries are actually from disambiguation pages => we throw them away
            continue
        summaries.append(summary)

    return summaries
