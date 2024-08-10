from wiki_tools import *

print("[+] Gathering highly ambiguous entities...")

json_file = "data/highly_ambiguous_entities.json"
highly_ambiguous_entities = load_checkpoint(json_file)
if highly_ambiguous_entities:
    print(f"\t[+] Resuming from checkpoint file {json_file}...")
else:
    print("\t[+] No checkpoint file found, starting fresh...")

print("\t[+] Getting all the disambiguation page titles...")
disambiguation_page_titles = get_disambiguation_page_titles()

# as of 30/07/2024, all the highly ambiguous entities (threshold 10) make up precisely 11008 entries
all_ambiguous_in_checkpoint = len(highly_ambiguous_entities) == 11008
if not all_ambiguous_in_checkpoint:
    threshold = 10
    print(f"\t[+] Filtering for the ones with at least {threshold} disambiguations...")
    filter_out_pages(
        disambiguation_page_titles, highly_ambiguous_entities, threshold, json_file
    )
    print(f"\t[+] Results saved to {json_file}.")
print(f"[+] All the ambiguous entities are already in the checkpoint file.")

print("[+] Gathering text snippets for each entity...")

highly_ambiguous_entities = load_checkpoint(json_file)
for disambiguation_page in highly_ambiguous_entities:
    get_snippets(disambiguation_page, num_snippets_per_page=5)

print("[+] Cleaning up the data...")

delete_non_ambiguous_entities()
delete_dirs_with_different_entities()

print("[+] All done!")
