"""Regression tests for the name matcher in redact_pymupdf_fast.py.

The bug these exist for: redaction matched whole whitespace-split word tokens, while the verify
pass matched plain page text. A label with no space after it - "Contact Name:Elena" - extracts as
the single token "Name:Elena", so redaction could never match "Elena" and verify reported it on
every pass: a name was listed for redaction, survived it, and came back as a leftover with no way
out. The two passes now agree because redaction reads the character stream instead of tokens.

All fixtures use the fabricated demo cast from demo_data/ - never real extracted values.

Run with: pytest tests/
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz
import pytest

import redact_pymupdf_fast as R


def matches(name, text):
    pattern = R.build_name_match_regex(name)
    return [m.group(0) for m in pattern.finditer(R.normalise_for_match(text))]


# --- names glued to neighbouring punctuation: the bug itself ---

def test_name_glued_to_label_by_colon_is_matched():
    """The original failure: no space after "Name:", so the name sits inside a larger token."""
    assert matches("Elena", "Contact Name:Elena Vasquez, Head of Research") == ["Elena"]


def test_name_glued_into_an_email_local_part_is_matched():
    assert matches("Priya", "priya.nandakumar@example.com") == ["priya"]


def test_name_with_trailing_dot_matches_itself():
    """"E.V." cleaned to "e.v" under the old end-stripping, so the entry could never match."""
    assert matches("E.V.", "Reviewed by E.V. Vasquez") == ["E.V."]


# --- the guarantee that must survive the fix: never redact inside a word ---

@pytest.mark.parametrize("name,text", [
    ("Del", "Delivery schedule attached"),
    ("Win", "Window frames and glazing"),
    ("Mar", "Marketing budget approved"),
    ("Ana", "Analysis of the results"),
])
def test_name_inside_a_longer_word_is_never_matched(name, text):
    assert matches(name, text) == []


def test_short_name_does_not_match_inside_a_longer_name():
    """Listing one name must never redact a different, longer name that contains it."""
    assert matches("Croix", "Marcus Delacroix") == []
    assert matches("Combe", "Samuel Whitcombe") == []


# --- punctuation and spacing a PDF varies on its own ---

def test_curly_and_straight_apostrophes_match_one_list_entry():
    """A document can typeset the same surname both ways; one list entry has to catch both."""
    assert matches("O'Whitmore", "A. O’Whitmore, 2024") == ["O'Whitmore"]
    assert matches("O'Whitmore", "A. O'Whitmore, 2024") == ["O'Whitmore"]


def test_multi_word_name_tolerates_odd_spacing():
    for spacing in ["Priya Nandakumar", "Priya  Nandakumar",
                    "Priya Nandakumar", "Priya\nNandakumar"]:
        assert matches("Priya Nandakumar", spacing), f"failed on {spacing!r}"


# --- end to end over a real PDF page ---

def _page_with(text):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), text, fontsize=11)
    return doc, page


def test_find_name_rects_covers_only_the_name_not_the_label():
    """The rect must sit over the name alone - blacking out "Contact Name:" as well would be a
    needless loss of surrounding document."""
    doc, page = _page_with("Contact Name:Elena Vasquez")
    rects = R.find_name_rects(page, [R.build_name_match_regex("Elena")])
    assert len(rects) == 1

    label_rects = page.search_for("Contact Name:")
    assert label_rects, "sanity check: the label should be findable"
    assert not rects[0].intersects(label_rects[0])
    doc.close()


def test_find_name_rects_actually_removes_the_name_from_the_output():
    doc, page = _page_with("Contact Name:Elena Vasquez")
    for rect in R.find_name_rects(page, [R.build_name_match_regex("Elena")]):
        page.add_redact_annot(rect, fill=(0, 0, 0))
    page.apply_redactions()

    text = page.get_text()
    assert "Elena" not in text
    assert "Contact Name:" in text
    doc.close()
