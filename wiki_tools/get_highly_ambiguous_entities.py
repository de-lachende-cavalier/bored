from .config import WIKI


def get_disambiguation_pages():
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


def count_links(disambiguation_page):
    page = WIKI.page(disambiguation_page)
    return len(page.links) if page.exists() else 0
