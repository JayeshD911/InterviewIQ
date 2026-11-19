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


STAR_SIGNALS = {
    "situation": [
        "when",
        "at my",
        "in my previous",
        "while working",
        "during",
    ],
    "task": [
        "i needed to",
        "my task",
        "i was responsible",
        "the goal was",
        "i had to",
    ],
    "action": [
        "i decided",
        "i implemented",
        "i built",
        "i led",
        "i created",
        "i took",
    ],
    "result": [
        "as a result",
        "this led to",
        "we achieved",
        "the outcome",
        "ended up",
        "improved",
        "increased",
        "reduced",
    ],
}

def check_star_structure(answer: str) -> dict:
    """Check whether a behavioral answer covers Situation, Task, Action, Result."""
    text = answer.lower()

    present = {
        stage: any(signal in text for signal in signals)
        for stage, signals in STAR_SIGNALS.items()
    }

    missing = [
        stage
        for stage, is_present in present.items()
        if not is_present
    ]

    return {
        "present": present,
        "missing": missing,
        "all_present": not missing,
    }


def score_relevance(answer: str, expected_keywords: list) -> dict:
    """Score how many expected concepts appear in the candidate's answer."""
    text = answer.lower()

    covered = [
        keyword
        for keyword in expected_keywords
        if keyword.lower() in text
    ]

    missed = [
        keyword
        for keyword in expected_keywords
        if keyword.lower() not in text
    ]

    if expected_keywords:
        score = round(100 * len(covered) / len(expected_keywords))
    else:
        score = 0

    return {
        "score": score,
        "covered": covered,
        "missed": missed,
    }

