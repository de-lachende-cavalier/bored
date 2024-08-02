import wikipediaapi
import time
import os

from dotenv import load_dotenv

# remember to create the .env file and set the WIKI_USER_AGENT environment variable!
load_dotenv()


class RateLimiter:
    def __init__(self, max_per_second):
        self.delay = 1 / max_per_second

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            time.sleep(self.delay)
            return func(*args, **kwargs)

        return wrapper


_LANG = "en"
_USER_AGENT = os.getenv("WIKI_USER_AGENT")
if _USER_AGENT is None:
    raise ValueError(
        "The environment variable 'WIKI_USER_AGENT' is not set! See https://foundation.wikimedia.org/wiki/Policy:User-Agent_policy for guidelines on how to do so."
    )

WIKI = wikipediaapi.Wikipedia(language=_LANG, user_agent=_USER_AGENT)
# in case one needs the page in HTML format (particularly useful if there are tables et alia)
WIKI_HTML = wikipediaapi.Wikipedia(
    language=_LANG,
    user_agent=_USER_AGENT,
    extract_format=wikipediaapi.ExtractFormat.HTML,
)


# https://www.mediawiki.org/wiki/Wikimedia_REST_API#Terms_and_conditions
wiki_rate_limiter = RateLimiter(200)
