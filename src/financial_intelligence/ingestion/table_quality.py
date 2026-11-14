"""Deterministic financial table review sampling."""

import random
import re
from collections import defaultdict


def select_review_pages(
    chunks: list[dict], top_n: int = 5, random_n: int = 3
) -> list[tuple[int, str, int]]:
    """Select numeric-heavy and random pages for review."""

    pages = defaultdict(list)

    for chunk in chunks:
        pages[chunk["page_number"]].append(chunk["text"])

    # Count numerical tokens on each page.
    scores = {}

    for page_number, texts in pages.items():
        full_text = " ".join(texts)
        scores[page_number] = len(re.findall(r"\d+(?:[,.]\d+)*", full_text))

    ranked_pages = sorted(scores, key=lambda page: (-scores[page], page))

    selected = ranked_pages[:top_n]
    remaining = sorted(set(pages) - set(selected))

    random_selection = random.Random(42).sample(
        remaining, min(random_n, len(remaining))
    )

    return [(page, "numeric-heavy", scores[page]) for page in selected] + [
        (page, "random", scores[page]) for page in sorted(random_selection)
    ]
