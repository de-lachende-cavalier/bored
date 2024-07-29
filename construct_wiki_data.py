import time

from wiki_tools import *

print("[+] Gathering highly ambiguous entities...")

json_file = "data/highly_ambiguous_entities.json"

highly_ambiguous_entities = load_checkpoint(json_file)
if highly_ambiguous_entities:
    print(f"\t[+] Resuming from checkpoint file {json_file}.")
else:
    print("\t[+] No checkpoint file found, starting fresh.")

# https://www.mediawiki.org/wiki/Wikimedia_REST_API#Terms_and_conditions
# we'll use 100 reqs/s just to be on the safe side
rate_limit_interval = 0.01

print("\t[+] Getting all the disambiguation pages...")
disambiguation_pages = get_disambiguation_pages()

threshold = 10
print(f"\t[+] Filtering for the ones with at least {threshold} disambiguations...")
for page_title in disambiguation_pages:
    if page_title not in highly_ambiguous_entities:
        link_count = count_links(page_title)
        if link_count > threshold:
            if has_alpha(page_title):
                highly_ambiguous_entities[page_title] = link_count
                save_checkpoint(json_file, highly_ambiguous_entities)
        time.sleep(rate_limit_interval)

print(f"\t[+] Results saved to {json_file}.")

print("[+] Gathering text snippets for each entity...")

print("[+] All done!")
