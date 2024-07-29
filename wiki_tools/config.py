import wikipediaapi
import os

from dotenv import load_dotenv

# remember to create the .env file and set the WIKI_USER_AGENT environment variable!
load_dotenv()

_LANG = "en"
_USER_AGENT = os.getenv("WIKI_USER_AGENT")
if _USER_AGENT is None:
    raise ValueError(
        "The environment variable 'WIKI_USER_AGENT' is not set! See https://foundation.wikimedia.org/wiki/Policy:User-Agent_policy for guidelines on how to do so."
    )

WIKI = wikipediaapi.Wikipedia(language=_LANG, user_agent=_USER_AGENT)
