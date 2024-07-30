import json
import os

from .config import WIKI, wiki_rate_limiter


@wiki_rate_limiter
def get_disambiguation_page_titles():
    all_disambiguation_links_ak = WIKI.page(
        "Wikipedia:Links_to_(disambiguation)_pages/A-K"
    ).links
    all_disambiguation_links_lz = WIKI.page(
        "Wikipedia:Links_to_(disambiguation)_pages/L-Z"
    ).links

    all_disambiguation_links = {
        **all_disambiguation_links_ak,
        **all_disambiguation_links_lz,
    }

    # ns (namespace) 0 corresponds to main/article content: https://en.wikipedia.org/wiki/Wikipedia:Namespace
    return [title for title, page in all_disambiguation_links.items() if page.ns == 0]


def load_checkpoint(filepath):
    """Load existing entities from a JSON checkpoint file."""
    if os.path.exists(filepath):
        with open(filepath, "r") as infile:
            return json.load(infile)
    return {}


def save_checkpoint(filepath, data):
    """Save entities to a JSON checkpoint file."""
    with open(filepath, "w") as outfile:
        json.dump(data, outfile)


@wiki_rate_limiter
def count_links(disambiguation_page):
    page = WIKI.page(disambiguation_page)
    return len(page.links) if page.exists() else 0


@wiki_rate_limiter
def get_all_titles_from_disambiguation(disambiguation_page):
    return [
        title
        for title, page in WIKI.page(disambiguation_page).links.items()
        if page.ns == 0
    ]
