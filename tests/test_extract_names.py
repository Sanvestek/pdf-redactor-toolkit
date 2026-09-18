"""Regression tests for extract_names.py.

Each test here is a scenario that was manually found, fixed, and verified during real usage -
this file exists so the next code change gets checked automatically instead of relying on
someone remembering to re-run every scenario by hand (which is exactly how three real bugs
almost shipped this session: a digit-check that ate real surnames, an exact-vs-substring header
match, and a title-decomposition gap).

Run with: pytest tests/
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz
import pytest

import extract_names as en


# --- extract_names_from_text: the spaCy NER + POS-tag path ---

def test_citation_list_keeps_surnames_drops_numbers():
    """Dense citation-style text ('Harker C.1, Lindqvist F.2, ...') - surnames must survive even
    though they're glued to a trailing citation number; the bare number must not."""
    text = (
        "Authors: Harker C.1, Lindqvist F.2, Nakamura J.3, Prescott A.4, Costello P.5, Kowalski G.6, "
        "Ashworth J.7, Penhallow M.8, Radcliffe J.9."
    )
    names = en.extract_names_from_text(text)
    for surname in ["Harker", "Lindqvist", "Nakamura", "Prescott", "Costello", "Kowalski", "Ashworth",
                     "Penhallow", "Radcliffe"]:
        assert surname in names, f"lost real surname {surname!r}"
    for junk in ["1", "2", "3", "12", "C.1", "F.2"]:
        assert junk not in names, f"kept citation-number junk {junk!r}"


def test_business_prose_rejects_section_headers_and_jargon():
    """Section headers / job-plan jargon must not be mistaken for PERSON entities, but a real
    name in the same text must still be caught."""
    text = (
        "Software Development: Backend development is conducted in stage 1. "
        "Project Management: Remains largely the same. Contact: Adam Brennan for details."
    )
    names = en.extract_names_from_text(text)
    assert "Adam Brennan" in names
    assert "Adam" in names
    assert "Brennan" in names
    for junk in ["Software Development", "Backend", "Development", "Project Management",
                 "Remains", "Contact"]:
        assert junk not in names, f"kept false positive {junk!r}"


def test_standalone_common_word_rejected_but_paired_surname_kept():
    """'Track'/'Smartphone'/'Scientist' alone are common-English-word false positives with no
    name context. 'Hale' inside 'Michael Hale' must survive even though 'Hale' is also in the
    common-word list (it's a frequency list, so it contains common names too)."""
    text = (
        "Track Record of Enterprise Ireland Funding. This is a Smartphone based application. "
        "Michael Hale [Research Scientist] will oversee the project."
    )
    names = en.extract_names_from_text(text)
    assert "Michael Hale" in names
    assert "Michael" in names
    assert "Hale" in names
    for junk in ["Track", "Smartphone", "Scientist"]:
        assert junk not in names, f"kept standalone common-word junk {junk!r}"


def test_title_does_not_survive_decomposition():
    """A title swept into a PERSON span ('Prof Adrian Fenwick') must not leave 'Prof' behind as
    its own standalone candidate, while the real name still comes through intact.

    Note: uses an exact previously-verified sentence, not a paraphrase - spaCy's small NER model
    turned out to be sensitive to the *exact* surrounding wording, not just "how much" context is
    present (confirmed: a reworded-but-similar-length version of this same sentence made spaCy
    miss the name entirely). That's a real fragility in the model itself, not this project's
    code, but it does mean regression tests here need exact known-good text, not approximations."""
    text = (
        "Immunogens were assessed via colonoscopy and biopsy screening. Reagents were prepared "
        "under simultaneous dosing protocols. Prof Adrian Fenwick, Lecturer, oversaw the milestone "
        "deliverable for the feasibility study. Biomarker specificity for the assay exceeded "
        "prevalence estimates in the cohort."
    )
    names = en.extract_names_from_text(text)
    assert "Adrian" in names
    assert "Fenwick" in names
    assert "Prof" not in names


def test_ocr_style_run_on_junk_rejected():
    """Long run-together blobs (a classic OCR word-segmentation failure) and generic technical
    prose must be rejected; a real name in the same text must survive."""
    text = (
        "However, current cleaning practices in hospitals are inadequate. Colonies of "
        "microorganisms counting on petri plates does not meet the required level of "
        "performance for commercial use. Contact John Smith for details."
    )
    names = en.extract_names_from_text(text)
    assert "John Smith" in names
    for junk in ["However", "colonies", "petri", "inadequate.", "commercial use."]:
        assert junk not in names


# --- looks_like_a_name: the structural validator ---

@pytest.mark.parametrize("junk", [
    "and disinfection practices play a critical",
    "commercial use.",
    "nationalhospitalconfirmedthatthelackof",
    "develo",
    "programs,",
    "weights)",
    "Assistant25",
    "Date:",
    "CF20191266",
])
def test_looks_like_a_name_rejects_junk(junk):
    assert en.looks_like_a_name(junk) is False


@pytest.mark.parametrize("name", [
    "Adrian Fenwick", "Alice Kennedy", "Harker", "O'Toole", "John Doe", "J.", "J. L.",
    "Radcliffe", "Jean-Claude", "Benedetti", "Smith Jr.",
])
def test_looks_like_a_name_accepts_real_names(name):
    assert en.looks_like_a_name(name) is True


# --- extract_names_from_tables: the word-position table extractor ---

def _make_table_pdf(path, headers, rows, gridded=True):
    doc = fitz.open()
    page = doc.new_page()
    col_x = [50, 250, 400, 500][:len(headers)]
    row_y = [60 + 30 * i for i in range(len(rows) + 1)]
    if gridded:
        for x in col_x:
            page.draw_line((x, row_y[0]), (x, row_y[-1]))
        for y in row_y:
            page.draw_line((col_x[0], y), (col_x[-1] + 20, y))
    else:
        page.draw_line((col_x[0], row_y[0] + 5), (col_x[-1] + 100, row_y[0] + 5))
    for r, row in enumerate([headers] + rows):
        for c, val in enumerate(row):
            page.insert_text((col_x[c] + 3, row_y[r] + 18), val, fontsize=9)
    doc.save(path)


def test_table_extraction_finds_names_in_applicant_table(tmp_path):
    pdf = tmp_path / "applicant.pdf"
    _make_table_pdf(str(pdf), ["Name", "Role", "Institution"],
                     [["Alice Kennedy", "Tech Transfer Advisor", "UCD"],
                      ["Adrian Fenwick", "Tech Transfer Advisor", "DCU"]])
    found = en.extract_names_from_tables(str(pdf))
    assert "Alice Kennedy" in found
    assert "Adrian Fenwick" in found


def test_table_extraction_works_without_gridlines(tmp_path):
    """A header underline with no full grid must still be detected (the 'text' strategy case)."""
    pdf = tmp_path / "borderless.pdf"
    _make_table_pdf(str(pdf), ["Name", "Role", "Institution"],
                     [["Diane Reyes", "Lead Applicant", "Springfield Institute"]],
                     gridded=False)
    found = en.extract_names_from_tables(str(pdf))
    assert "Diane Reyes" in found


def test_table_extraction_ignores_non_person_name_columns(tmp_path):
    """'Funding Body Name' and similar columns are about organisations, not people, even though
    they contain 'name' as a substring - must produce nothing."""
    pdf = tmp_path / "funding.pdf"
    _make_table_pdf(str(pdf), ["Funding Body Name", "Grant Ref", "Amount"],
                     [["Enterprise Ireland", "DEMO-2024-9999", "150000"]])
    found = en.extract_names_from_tables(str(pdf))
    assert found == {}


def test_table_extraction_aligns_multiword_header_phrase(tmp_path):
    """'Full Name' is a two-word header; data below must align to where 'Full' starts, not just
    'Name' - otherwise a short first name gets clipped off (the exact bug that dropped 'Fiona')."""
    pdf = tmp_path / "fullname.pdf"
    doc = fitz.open()
    page = doc.new_page()
    headers = ["#", "Full Name", "Affiliation"]
    col_x = [50, 120, 350]
    y = 60
    for i, h in enumerate(headers):
        page.insert_text((col_x[i], y), h, fontsize=10)
    page.draw_line((50, y + 5), (500, y + 5))
    y += 25
    for i, val in enumerate(["1", "Fiona Novak", "Enterprise Ireland"]):
        page.insert_text((col_x[i], y), val, fontsize=10)
    doc.save(str(pdf))

    found = en.extract_names_from_tables(str(pdf))
    assert "Fiona Novak" in found


# --- extract_names_from_labels: the inline "Label: Value" form extractor ---

def test_label_form_finds_name_and_ignores_multiword_label(tmp_path):
    """'Lead Applicant Name: Mark Whitfield   Institution Name: Springfield Institute' on one
    line - a real document layout spaCy's NER completely failed on (it produced no PERSON entity
    at all here, folding the real name into a bogus ORG span instead). The extracted value must
    be exactly 'Mark Whitfield', not swallow the next label's first word ('Institution')."""
    pdf = tmp_path / "labelform.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 60), "Reference No: DEMO-2024-0001-I    Requested Budget: e250,000.00",
                      fontsize=9)
    page.insert_text((50, 90),
                      "Lead Applicant Name: Mark Whitfield    Institution Name: Springfield "
                      "Institute", fontsize=9)
    doc.save(str(pdf))

    found = en.extract_names_from_labels(str(pdf))
    assert "Mark Whitfield" in found
    assert "Mark Whitfield Institution" not in found


def test_multiword_label_rejected_when_no_person_entity_exists():
    """spaCy tags the label itself ('Lead Applicant Name') as a false-positive PERSON here (it's
    followed immediately by a colon in the source text) - must be rejected regardless of it being
    multi-word and Title-Case, which otherwise passes every other check."""
    text = "Lead Applicant Name: Mark Whitfield Institution Name: Springfield Institute of Technology."
    names = en.extract_names_from_text(text)
    assert "Lead Applicant Name" not in names


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
