import re
import os
import random

from nltk import sent_tokenize

from .config import WIKI, wiki_rate_limiter
from .utils import get_all_titles_from_disambiguation


@wiki_rate_limiter
def extract_snippet(page_title, max_sentences=4):
    page = WIKI.page(page_title)
    content = page.text
    sentences = sent_tokenize(content)
    entity = page_title.split(" (")[0]

    # use word boundaries for exact match, case-sensitive
    pattern = re.compile(r"\b" + re.escape(entity) + r"\b")
    matches = list(pattern.finditer(content))

    if not matches:
        return None

    match = random.choice(matches)
    match_pos = match.start()

    for i, sentence in enumerate(sentences):
        sentence_start = content.find(sentence)
        if sentence_start <= match_pos < sentence_start + len(sentence):
            # choose a number of sentences to concatenate
            num_sentences = random.randint(1, min(max_sentences, len(sentences) - i))
            snippet = " ".join(sentences[i : i + num_sentences])

            # check if the entity is in the snippet
            if re.search(pattern, snippet):
                return snippet

            break


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


@wiki_rate_limiter
def create_snippets(disambiguation_page, num_snippets_per_page, max_sentences=4):
    titles = get_all_titles_from_disambiguation(disambiguation_page)
    for page_title in titles:
        for count in range(1, num_snippets_per_page + 1):
            if "(disambiguation)" in page_title:
                continue
            snippet = extract_snippet(page_title, max_sentences)
            if snippet:
                save_snippet(disambiguation_page, page_title, snippet, count)
