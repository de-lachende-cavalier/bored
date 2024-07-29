def has_alpha(s):
    """The string should not be made of only punctuation and digits."""
    return any(c.isalpha() for c in s.split(" (")[0])
