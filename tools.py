import re

FILLER_WORDS = [
    "um",
    "uh",
    "like",
    "basically",
    "actually",
    "you know",
    "so yeah",
    "i mean",
]


def detect_filler_words(answer: str) -> dict:
    """Count filler words (um, like, basically, etc.) in a candidate's answer."""
    text = answer.lower()

    found = {}

    for phrase in FILLER_WORDS:
        pattern = rf"\b{re.escape(phrase)}\b"
        count = len(re.findall(pattern, text))

        if count > 0:
            found[phrase] = count

    total = sum(found.values())

    return {
        "total": total,
        "found": found,
        "verdict": "clean" if total <= 2 else "needs improvement",
    }