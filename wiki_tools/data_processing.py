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
