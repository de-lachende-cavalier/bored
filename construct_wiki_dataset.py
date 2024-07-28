from dotenv import load_dotenv
import wikipediaapi
import os
import time

from utils import load_checkpoint, save_checkpoint

# remember to create the .env file and set the WIKI_USER_AGENT environment variable!
load_dotenv()

LANG = "en"
USER_AGENT = os.getenv("WIKI_USER_AGENT")
if USER_AGENT is None:
    raise ValueError(
        "The environment variable 'WIKI_USER_AGENT' is not set! See https://foundation.wikimedia.org/wiki/Policy:User-Agent_policy for guidelines on how to do so."
    )
WIKI = wikipediaapi.Wikipedia(language=LANG, user_agent=USER_AGENT)


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
    disambiguation_pages = [
        title for title, page in all_disambiguation_links.items() if page.ns == 0
    ]

    return disambiguation_pages


def count_links(disambiguation_page):
    page = WIKI.page(disambiguation_page)
    if page.exists():
        return len(page.links)
    return 0


if __name__ == "__main__":
    json_file = "highly_ambiguous_entities.json"
    highly_ambiguous_entities = load_checkpoint(json_file)
    if highly_ambiguous_entities:
        print(f"[+] Resuming from checkpoint file {json_file}.")
    else:
        print("[+] No checkpoint file found, starting fresh.")

    # https://www.mediawiki.org/wiki/Wikimedia_REST_API#Terms_and_conditions
    # we'll use 100 reqs/s just to be on the safe side
    rate_limit_interval = 0.01

    print("[+] Getting all the disambiguation pages...")
    disambiguation_pages = get_disambiguation_pages()

    # ignore all the pages already stored
    pages_to_process = [
        page_title
        for page_title in disambiguation_pages
        if page_title not in highly_ambiguous_entities
    ]
    threshold = 10
    print(f"[+] Filtering for the ones with at least {threshold} disambiguations...")
    for page_title in pages_to_process:
        link_count = count_links(page_title)
        if link_count > threshold:
            highly_ambiguous_entities[page_title] = link_count
            save_checkpoint(json_file, highly_ambiguous_entities)

        time.sleep(rate_limit_interval)

    print(f"[+] Processing complete! Results saved to {json_file}")
