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
