import re
import os
import random

from nltk import sent_tokenize

from .config import WIKI


def extract_snippet(page_title, max_sentences=4):
    page = WIKI.page(page_title)

    content = page.text
    sentences = sent_tokenize(content)
    entity = page.title.split(" (")[0]

    # find all occurrences of the entity in the text, case-sensitive and exact
    pattern = re.compile(r"\b" + re.escape(entity) + r"\b")
    matches = list(pattern.finditer(content))
    if not matches:
        return None

    match = random.choice(matches)
    start_index = match.start()
    end_index = match.end()

    for i, sentence in enumerate(sentences):
        sentence_start = content.find(sentence)
        sentence_end = sentence_start + len(sentence)
        if start_index >= sentence_start and end_index <= sentence_end:
            # choose a number of sentences to concatatenate
            num_sentences = random.randint(1, max_sentences)
            # a snippet is said concatenation of sentences
            snippet = " ".join(sentences[i : i + num_sentences])
            break

    if entity in snippet:
        return snippet
    return None


def save_snippet(disambiguation_page, page_title, snippet, count):
    sanitized_disambiguation = re.sub(r"[^\w\s]", "_", disambiguation_page)
    sanitized_title = re.sub(r"[^\w\s]", "_", page_title)

    directory = os.path.join("data", sanitized_disambiguation, sanitized_title)

    os.makedirs(directory, exist_ok=True)

    file_path = os.path.join(directory, f"{count}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(snippet)


def get_snippets_for(disambiguation_page, page_title, num_snippets, max_sentences=4):
    for count in range(1, num_snippets + 1):
        snippet = extract_snippet(page_title, max_sentences)
        if snippet:
            save_snippet(disambiguation_page, page_title, snippet, count)
