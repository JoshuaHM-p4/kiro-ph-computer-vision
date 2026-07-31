"""Tagalog slang → English description translator.

Maps Filipino slang terms to English descriptions suitable for use as
text prompts in segmentation models (e.g., SAM). The English descriptions
are phrased as object categories the model can segment.

Usage:
    from translator import translate, get_all_slang

    english = translate("pogi")       # "handsome person / face"
    all_terms = get_all_slang()       # dict of all mappings
"""

from __future__ import annotations

from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Slang dictionary
# ---------------------------------------------------------------------------

TAGALOG_SLANG: Dict[str, str] = {
    "pogi": "handsome person / face",
    "ganda": "beautiful person",
    "chibog": "food / snack",
    "tsismis": "cell phone",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def translate(slang: str) -> Optional[str]:
    """Translate a Tagalog slang term to its English description.

    Args:
        slang: A Tagalog slang word (case-insensitive).

    Returns:
        The English description if the term is known, or None if not found.
    """
    return TAGALOG_SLANG.get(slang.strip().lower())


def translate_or_passthrough(term: str) -> str:
    """Translate if known, otherwise return the term unchanged.

    Useful when feeding user input directly to a segmentation model — unknown
    terms can still work as English prompts.

    Args:
        term: A slang term or English phrase.

    Returns:
        The English translation if the term is in the dictionary, otherwise
        the original term.
    """
    result = translate(term)
    return result if result is not None else term


def get_all_slang() -> Dict[str, str]:
    """Return a copy of the full slang dictionary."""
    return dict(TAGALOG_SLANG)


def is_known(term: str) -> bool:
    """Check if a term exists in the slang dictionary."""
    return term.strip().lower() in TAGALOG_SLANG
