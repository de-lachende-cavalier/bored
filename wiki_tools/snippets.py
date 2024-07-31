import re
import os
import random

from nltk import sent_tokenize
from difflib import SequenceMatcher

from .config import WIKI, wiki_rate_limiter
from .utils import get_all_titles_from_disambiguation


def is_snippet_similar(new_snippet, previous_snippets, similarity_threshold=0.70):
    for prev_snippet in previous_snippets:
        similarity = SequenceMatcher(None, new_snippet, prev_snippet).ratio()
        if similarity > similarity_threshold:
            return True
    return False


def has_sufficient_content(
    sentences, num_snippets_per_page, max_sentences, tolerance=10
):
    # we want the page to have enough content so that, if every snipper were made up of max_sentences, we could still get unique snippets
    # we add a tolerance value to account for any page-related weirdness
    required_n_sentences = max_sentences * num_snippets_per_page + tolerance
    if len(sentences) < required_n_sentences:
        return False
    return True


@wiki_rate_limiter
def extract_snippet(
    page_title,
    previous_snippets,
    num_snippets_per_page,
    min_length=100,
    max_sentences=4,
):
    page = WIKI.page(page_title)
    content = page.text
    sentences = sent_tokenize(content)
    if not has_sufficient_content(sentences, num_snippets_per_page, max_sentences):
        return None

    entity = page_title.split(" (")[0]
    pattern = re.compile(r"\b" + re.escape(entity) + r"\b")
    matches = list(pattern.finditer(content))

    if not matches:
        return None

    random.shuffle(matches)

    return find_suitable_snippet(
        matches,
        sentences,
        pattern,
        content,
        min_length,
        max_sentences,
        previous_snippets,
    )


def find_suitable_snippet(
    matches, sentences, pattern, content, min_length, max_sentences, previous_snippets
):
    for match in matches:
        match_pos = match.start()
        sentence_index = find_sentence_index(sentences, content, match_pos)

        if sentence_index is not None:
            snippet = create_snippet(
                sentences, sentence_index, max_sentences, pattern, min_length
            )

            if snippet:
                if len(previous_snippets) >= 1 and is_snippet_similar(
                    snippet, previous_snippets
                ):
                    return None
                else:
                    return snippet

    return None


def find_sentence_index(sentences, content, match_pos):
    """Find the index of the sentence containing the match position."""
    for i, sentence in enumerate(sentences):
        sentence_start = content.find(sentence)
        if sentence_start <= match_pos < sentence_start + len(sentence):
            return i
    return None


def create_snippet(sentences, start_index, max_sentences, pattern, min_length):
    for num_sentences in range(max_sentences, 1, -1):
        snippet = " ".join(sentences[start_index : start_index + num_sentences])
        if re.search(pattern, snippet) and len(snippet) >= min_length:
            return snippet
    return None


def save_snippet(disambiguation_page, page_title, snippet, count):
    directory = os.path.join(
        "data",
        "snippets",
        disambiguation_page.replace(" ", "_"),
        page_title.replace(" ", "_"),
    )

    os.makedirs(directory, exist_ok=True)

    file_path = os.path.join(directory, f"{count}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(snippet)


def get_snippets(
    disambiguation_page, num_snippets_per_page, min_length=100, max_sentences=4
):
    titles = get_all_titles_from_disambiguation(disambiguation_page)
    for page_title in titles:
        if "(disambiguation)" in page_title:
            continue

        previous_snips = []
        for count in range(0, num_snippets_per_page):
            snippet = extract_snippet(
                page_title,
                previous_snips,
                num_snippets_per_page,
                min_length=min_length,
                max_sentences=max_sentences,
            )
            if not snippet:
                # if extract_snippet returns a None, skip this page entirely
                break
            previous_snips.append(snippet)

        # only save the snippets if there are exactly num_snippets_per_page
        if len(previous_snips) < num_snippets_per_page:
            continue
        for count, snippet in enumerate(previous_snips):
            save_snippet(disambiguation_page, page_title, snippet, count)
