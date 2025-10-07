import torch
import pandas as pd
from bs4 import BeautifulSoup
from bs4 import MarkupResemblesLocatorWarning
import warnings
import re
from typing import Any

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# Define weighted keywords. Higher weight means a stronger spam indicator.
KEYWORD_RULES = {
    "coinbase": 0.9,
    "support number": 0.5,
    "contact support": 0.5,
    "tollfree": 0.3,
    "dial": 0.15,
    "technical support": 0.2,
    "phone number": 0.25,
    "directly contact": 0.9,
    "call directly": 0.9,
    "call us": 0.2,
    "call now": 0.9,
    "call today": 0.9,
    "robinhood": 0.9,
    "customer service": 0.9,
    "help desk": 0.9,
    "live agent": 0.9,
    "course2day": 0.9,
    "udemy": 0.9,
    "coursera": 0.9
}

# Define regex patterns to find common scam attributes like phone numbers.
PHONE_REGEX_PATTERNS = [
    # North American phone number patterns
    re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    # International phone numbers (various formats)
    re.compile(
        r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}[-.\s]?\d{2,4}\b"
    ),
    # Numbers with extensions
    re.compile(
        r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\s*(?:ext|extension|xtn)[\.]?\s*\d+\b",
        re.IGNORECASE,
    ),
]

# Keywords followed by numbers (e.g., "call us at 123...")
KEYWORD_NUMBER_PATTERNS = [
    re.compile(
        r"\b(call|contact|phone|support|helpline)[\s:]*(\d{5,})",
        re.IGNORECASE,
    ),
    re.compile(r"\b(dial|ring|number)[\s:]*(\d{5,})", re.IGNORECASE),
]


def get_device():
    """
    Returns the best available device for PyTorch.
    Priority: MPS (Apple Silicon) > CUDA > CPU
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def clean_text(text: Any) -> str:
    """
    Cleans text by removing HTML tags, unescaping entities, and normalizing whitespaces.
    Handles non-string inputs gracefully.
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""

    # Use BeautifulSoup for robust HTML tag removal
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    # Normalize whitespace and remove leading/trailing spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_phone_number(text: str) -> bool:
    """
    Check if text contains any phone number patterns.
    """
    for pattern in PHONE_REGEX_PATTERNS:
        if pattern.search(text):
            return True
    return False


def contains_keyword_with_numbers(text: str) -> bool:
    """
    Check if text contains keywords followed by numbers.
    """
    for pattern in KEYWORD_NUMBER_PATTERNS:
        if pattern.search(text):
            return True
    return False


def contains_specified_keywords(text: str) -> bool:
    """
    Check if text contains any of our specified keywords.
    """
    text_lower = text.lower()
    for keyword in KEYWORD_RULES.keys():
        if keyword in text_lower:
            return True
    return False
