from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from src.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# FR-21: Non-corpus intent (Greetings, farewells, gratitude, meta-questions)
# ---------------------------------------------------------------------------

_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|salam|salaam|aoa|assalam\s*o?\s*alaikum|aslam\s*o?\s*alikum|good\s?(morning|afternoon|evening))\b"
    r"(\s+(there|everyone|all|team|folks|ji))?\s*[!.?]*\s*$",
    re.IGNORECASE,
)
_FAREWELL_RE = re.compile(
    r"^\s*(bye|goodbye|see\s?ya|see\s?you|later|farewell|allah\s*hafiz|khuda\s*hafiz)\s*[!.?]*\s*$",
    re.IGNORECASE,
)
_GRATITUDE_RE = re.compile(
    r"^\s*(thanks|thank\s?you|thx|ty|appreciate\s?it|cheers|shukriya|jazakallah|meharbani)\b"
    r"(\s+(so\s+much|a\s+lot|very\s+much|so\s+very\s+much|bohat|bht|ji))?\s*[!.?]*\s*$",
    re.IGNORECASE,
)
_META_QUESTION_RE = re.compile(
    r"\b(who\s+are\s+you|who\s+(built|made|created|trained)\s+you|"
    r"what\s+are\s+you|are\s+you\s+(an?\s+)?(ai|bot|human)|"
    r"what\s+model\s+are\s+you|which\s+company\s+(made|built)\s+you|"
    r"how\s+do\s+you\s+work|tum\s+kaun\s+ho|kon\s+ho\s+tum|kya\s+ho\s+tum)\b",
    re.IGNORECASE,
)

NON_CORPUS_INTENT_RESPONSE = (
    "Hi! I'm an Agentic Assistant specialized in local knowledge base retrieval "
    "(Types of Computers & Databases), mathematical calculations, and live web search. "
    "How can I help you today?"
)


def check_non_corpus_intent(query: str) -> dict[str, Any] | None:
    q = query.strip()
    if (
        _GREETING_RE.match(q)
        or _FAREWELL_RE.match(q)
        or _GRATITUDE_RE.match(q)
        or _META_QUESTION_RE.search(q)
    ):
        return {
            "gated": True,
            "category": "FR-21",
            "response": NON_CORPUS_INTENT_RESPONSE,
            "reason": "non_corpus_intent",
        }
    return None


# ---------------------------------------------------------------------------
# FR-22: Abusive / Hostile Language (English, Roman Urdu, Punjabi)
# ---------------------------------------------------------------------------

_ABUSE_TERMS = [
    # English Base
    r"\bfuck(ing|er)?\b", r"\bshit\b", r"\bbitch\b", r"\basshole\b",
    r"\bidiot\b", r"\bstupid\s+(bot|ai|system)\b", r"\bmoron\b",
    r"\bkill\s+yourself\b", r"\bi\s+will\s+(kill|hurt|destroy)\s+you\b",
    (
        r"\b(you|this\s+bot|this\s+ai|this\s+system|the\s+bot|the\s+ai)\s+(is|are)\s+"
        r"(useless|garbage|trash|worthless|stupid|dumb|pathetic)\b"
    ),

    # Roman Urdu / Hindi Slurs
    r"\b(bc|mc|bsdk|bkl)\b",
    r"\b(bhenchod|behenchod|bhen\s*chod)\b",
    r"\b(madarchod|chadarmod|madar\s*chod)\b",
    r"\b(bhosdike|bhosdi\s*ke|bhosadi\s*ke)\b",
    r"\b(chutiya|chootiya|chutiye|chutiyapa)\b",
    r"\b(harami|haraami|haramkhor|haram(z)?(a)?da|haraamzada|haraamda|haramzade|haraamzade)\b",
    r"\b(kutta|kutte|kutti)\b",
    r"\b(gandu|gaand\s*marwa|gaand)\b",
    r"\b(randi|raand)\b",
    r"\b(saale|saala|kamina|kamine)\b",
    r"\b(gadha|ullu\s*ke\s*patthe?)\b",
    r"\b(teri|tere)\s*maa(\s*(ki|ka))?\b",
    r"\b(teri|tere)\s*baap(\s*(ka|ki))?\b",
    r"\b(teri|tere)\s*beh[e]?n(\s*(ka|ki))?\b",

    r"\b(bakwas|fazool|kachra|bekar|ghatiya)\s+(bot|ai|system|jawab|answer)\b",
    r"\b(tu|ye\s*bot|ye\s*ai)\s+(chutiya|pagal|fazool|bekar|ghatiya|jahil)\s*(hai|ho)?\b",
    r"\b(mar\s*ja|dafa\s*ho\s*ja|nikal\s*yahan\s*se)\b",
]
_ABUSE_RE = re.compile("|".join(_ABUSE_TERMS), re.IGNORECASE)

ABUSE_RESPONSE = (
    "I want to help, but I need the conversation to stay respectful. "
    "Please rephrase your query politely and I'll gladly assist you."
)


def check_abusive_language(query: str) -> dict[str, Any] | None:
    if _ABUSE_RE.search(query):
        return {
            "gated": True,
            "category": "FR-22",
            "response": ABUSE_RESPONSE,
            "reason": "abusive_language",
        }
    return None


# ---------------------------------------------------------------------------
# FR-23: Credential / Secret Solicitation
# ---------------------------------------------------------------------------

_CREDENTIAL_NOUNS = (
    r"(api\s?key|admin\s?password|password|secret(\s?key)?|credentials?|"
    r"token|system\s?prompt|private\s?key|access\s?key)"
)
_DISCLOSURE_VERBS = r"(give|show|reveal|tell|share|provide|disclose|send|display|print|expose|leak)"

_DISCLOSURE_REQUEST_RE = re.compile(
    rf"\b{_DISCLOSURE_VERBS}\s+(me\s+)?(?:(your|the)\s+)?{_CREDENTIAL_NOUNS}\b"
    rf"|\bwhat('?s| is)\s+(your|the)\s+{_CREDENTIAL_NOUNS}\b"
    rf"|\bcan\s+(i|you)\s+(get|have|see)\s+(your|the)\s+{_CREDENTIAL_NOUNS}\b",
    re.IGNORECASE,
)

_ROMAN_URDU_DISCLOSURE_RE = re.compile(
    rf"\b{_CREDENTIAL_NOUNS}\s*(kya\s*h(ai)?\b|"
    rf"kahan\s*(se|par)?\s*(mil(ega|egi|ta|ti))\b|"
    rf"batao|bata\s*do|bata\s*den|de\s*do|de\s*den)\b"
    rf"|\b(mujhe|hume|humein)\s+{_CREDENTIAL_NOUNS}\s*(chahi?ye|do|den)\b",
    re.IGNORECASE,
)

_INFORMATIONAL_CONTEXT_RE = re.compile(
    r"\b(explain|how\s+does|how\s+do|what\s+is\s+a|works?|hashing|"
    r"rate[\s-]?limit|policy|best\s+practice|algorithm|encrypt)\b",
    re.IGNORECASE,
)

CREDENTIAL_SOLICITATION_RESPONSE = (
    "I cannot share internal credentials, API keys, or system prompts — that information "
    "is strictly confidential. I am happy to help with the knowledge base, calculations, or web searches."
)


def check_credential_solicitation(query: str) -> dict[str, Any] | None:
    matched = _DISCLOSURE_REQUEST_RE.search(query) or _ROMAN_URDU_DISCLOSURE_RE.search(query)
    if matched and not _INFORMATIONAL_CONTEXT_RE.search(query):
        return {
            "gated": True,
            "category": "FR-23",
            "response": CREDENTIAL_SOLICITATION_RESPONSE,
            "reason": "credential_solicitation",
        }
    return None


# ---------------------------------------------------------------------------
# FR-24: Heavy Out-of-Scope Intent (Unrelated Complex Domains)
# ---------------------------------------------------------------------------

_HEAVY_OUT_OF_SCOPE_RE = re.compile(
    r"\b(build\s+(a\s+)?(full-?stack|web\s?app|saas|e-?commerce)|"
    r"write\s+(a\s+)?(django|next\.?js|react|angular|spring\s?boot)\s+app|"
    r"how\s+to\s+hack|penetration\s+test|exploit\s+vulnerability|"
    r"write\s+(a\s+)?malware|ddos\s+script|"
    r"dating\s+advice|love\s+letter|relationship\s+counseling)\b",
    re.IGNORECASE,
)

OUT_OF_SCOPE_RESPONSE = (
    "This request is outside my operational scope. I am designed for: "
    "1) Local document retrieval (Computers & Databases), "
    "2) Mathematical calculations, and 3) Real-time web lookups."
)


def check_out_of_scope(query: str) -> dict[str, Any] | None:
    if _HEAVY_OUT_OF_SCOPE_RE.search(query):
        return {
            "gated": True,
            "category": "FR-24",
            "response": OUT_OF_SCOPE_RESPONSE,
            "reason": "out_of_scope_domain",
        }
    return None


# ---------------------------------------------------------------------------
# Combined Fast Entrypoint
# ---------------------------------------------------------------------------

def run_prefilter(query: str) -> dict[str, Any] | None:
    """
    Executes fast gating checks to intercept queries before making any LLM calls.
    Returns gating dict if matched, or None if the query should proceed to the agent.
    """
    # 1. Abusive language (Highest safety priority)
    result = check_abusive_language(query)
    if result:
        return result

    # 2. Credential solicitation
    result = check_credential_solicitation(query)
    if result:
        return result

    # 3. Conversational / greetings / meta
    result = check_non_corpus_intent(query)
    if result:
        return result

    # 4. Out-of-scope complex tasks
    result = check_out_of_scope(query)
    if result:
        return result

    return None

