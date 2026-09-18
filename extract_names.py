#;encoding=utf-8
# Extracts candidate PERSON names from a PDF (or a directory of PDFs) using
# spaCy NER, for use as a --names_file input to redact_pymupdf_fast.py.
#
# Replaces the old extract_names.py / extract_names_batch.py / extract_names_post.py
# split. Single-file vs. directory mode is auto-detected from the input path
# (same convention as redact_pymupdf_fast.py); --merge controls whether an
# existing output file is added to (building a cumulative master list) or
# overwritten fresh (useful when re-checking a "redacted" output folder for
# leftover names).

import argparse
import os
import re
import sys

import spacy
import pytesseract
import fitz  # PyMuPDF
from pypdf import PdfReader
from pdf2image import convert_from_path

# Reused, not reimplemented, so the exceptions file means the same thing here as it does at
# redaction time - one file, one definition of "never treat this as sensitive".
from redact_pymupdf_fast import load_exceptions_from_file

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Error: spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm", file=sys.stderr)
    sys.exit(1)

# Common English words (top ~10k by web frequency; source: first20hours/google-10000-english,
# MIT licensed). Used to reject standalone single-word candidates like "Track"/"Scientist"/
# "Equipment" that structurally look exactly like a real single-word name (Title Case, no
# digits) - no grammar-based signal can tell those apart, only actual word frequency can. Note
# this list is frequency-based, not a pure dictionary, so it also contains common first/last
# names ("Scott", "Grace", "Max", "Hope") - see where it's used for how that's handled.
_WORDLIST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "common_english_words.txt")
try:
    with open(_WORDLIST_PATH, "r", encoding="utf-8") as _f:
        COMMON_ENGLISH_WORDS = {line.strip().lower() for line in _f if line.strip()}
except FileNotFoundError:
    COMMON_ENGLISH_WORDS = set()

# --- Stoplist for common false positives ---
COMMON_STOPLIST = {
    "App", "Store", "Applicant", "Artificial", "Intelligence", "Business", "Software",
    "Clinical", "Operations", "Data", "Digital", "Enterprise", "Hospital", "Transfer",
    "Office", "Learning", "Machine", "Neuroscience", "Technology", "Therapy", "Therapis",
    "Mental", "Lead", "Commercial", "Confidential", "Connect", "Report", "Template",
    "Vitae", "Curriculum", "DELIVERABLES", "Signatures", "Unclear", "Vantera", "Basic",
    "SOM", "Mind", "Msc", "Revs", "Salary", "Hire", "St", "Dublin", "Background", "Generalist",
    "Problem", "Solving", "Producer", "Stakeholder", "Started", "Interim", "Branded",
    "Analyse", "Additionall", "Application", "Theory", "Queen", "Regulatory", "Compliance",
    "Project", "Form", "Name", "Info", "User", "WP", "Dev", "T2", "T4", "Performance",
    "Harpsichord", "Oboe", "Piano", "College", "License", "Literary", "Corvyx", "Halcyra",
    "Northglen", "Brightloop", "Meridian", "PSD2", "Personal", "Additionall", "CF", "Final",
    "et", "al", "ie", "and", "phone", "senor", "dashboardM2", "tim", "et al.",
    "B2B", "B2C", "CV", "NA", "CFP_Compiled_WPV_FINAL", "Std", "WPV", "FINAL",
    # Titles: unambiguously never a real surname (unlike e.g. "Scott"), so unlike the wordlist
    # file, safe to reject unconditionally - including when swept into a PERSON span and
    # decomposed into components, which the wordlist file deliberately does not check.
    "Prof", "Dr", "Mr", "Mrs", "Ms", "Miss",
}

# Known multi-letter initials that are safe to redact despite looking short
KNOWN_INITIALS = {"J. L.", "J. S.", "S. W.", "C. N.", "L. C.", "L. T."}

# Closed-class English function words. Unlike technical/content-word stoplists (open-ended,
# document-specific - there's no bounding a list of every possible jargon term), this set is
# small, finite, and universal: a real person's name never legitimately contains "however",
# "the", "is", "since" etc, in any document, regardless of subject matter.
FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "nor", "of", "to", "in", "on", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "from", "up", "down", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "not", "only", "own", "same",
    "so", "than", "too", "very", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "will", "would",
    "should", "can", "could", "may", "might", "must", "shall", "that", "this", "these",
    "those", "it", "its", "as", "if", "because", "since", "however", "therefore",
    "although", "though", "while", "whereas", "unless", "until", "also", "either",
    "neither", "yet", "still", "openly", "heavily", "you", "your", "we", "our", "us",
}

MAX_NAME_WORDS = 4        # "Jean-Claude van der Berg Jr." territory; a full sentence is not this
MAX_NAME_LENGTH = 40      # characters, whole phrase
MAX_SINGLE_WORD_LENGTH = 22  # a real surname is virtually never longer than this


def looks_like_a_name(candidate: str) -> bool:
    """Final structural sanity check, independent of whatever produced the candidate (NER label,
    POS tag, table column position, OCR text) - all of which can be fooled, especially by garbled
    OCR output where spaCy's models become unreliable across the board. This checks the string in
    isolation instead: real names are short, each word starts with a capital letter (or is a
    short acronym-style initial), contain no digits, and never include an English function word
    or a long run-on blob with no spaces (a classic OCR word-segmentation failure).
    """
    candidate = candidate.strip()
    if not candidate:
        return False

    words = candidate.split()
    if len(words) > MAX_NAME_WORDS:
        return False
    if len(candidate) > MAX_NAME_LENGTH:
        return False
    if len(words) == 1 and len(words[0]) > MAX_SINGLE_WORD_LENGTH:
        return False
    if re.search(r'[,;()]$', candidate):
        return False
    if candidate.rstrip().endswith(":"):
        return False
    last_word = words[-1]
    if last_word.endswith(".") and len(last_word.rstrip(".")) > 3:
        return False

    for w in words:
        core = w.strip(".,;:()'\"")
        if not core:
            return False
        if any(c.isdigit() for c in core):
            return False
        if core.lower() in FUNCTION_WORDS:
            return False
        if core.isupper():
            if len(core) > 4:
                return False
            continue
        if core[0].isupper() and re.fullmatch(r"[A-Za-z'-]+", core):
            continue
        return False

    return True


def _is_numeric_noise(token: str) -> bool:
    """True if token is a number/ID/form-label artifact rather than plausible name text.

    Names always contain at least one letter and never contain a digit (catches things like
    "Assistant25" or a grant reference "CF20191266" that slip past a letters-only check) or end
    in a colon (a form-field label like "Date:" or "Citizenship:", not a name).
    """
    if not any(c.isalpha() for c in token):
        return True
    if any(c.isdigit() for c in token):
        return True
    if token.rstrip().endswith(":"):
        return True
    if re.search(r'[\d\.-]+[PTM]', token) or re.search(r'\d-[\d\.]+', token):
        return True
    return False


def matches_exception(value: str, exceptions) -> bool:
    """True if any exception phrase appears as a whole-word match inside value (case-insensitive).

    Word-boundary containment, not exact equality: "appendix" as an exception must also reject
    "Appendix Sch" and "Appendix Schematic" - the same span spaCy actually produced from a
    document heading - not just a candidate that's the literal single word "Appendix" and nothing
    else. This mirrors how redact_pdf() itself uses exceptions - protecting anything overlapping
    the phrase - so a phrase excluded from redaction is now also excluded from ever cluttering
    the review list in the first place.
    """
    if not exceptions:
        return False
    value_lower = value.lower()
    for exc in exceptions:
        if re.search(r'\b' + re.escape(exc.lower()) + r'\b', value_lower):
            return True
    return False


def extract_text_pypdf(pdf_path: str) -> str:
    """Extracts text from a PDF using pypdf (best for text-based PDFs)."""
    try:
        reader = PdfReader(pdf_path)
        return " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    except Exception as e:
        print(f"pypdf extraction failed for {pdf_path}: {e}", file=sys.stderr)
        return ""


def extract_text_ocr(pdf_path: str) -> str:
    """Converts PDF pages to images and uses Tesseract OCR (for scanned PDFs)."""
    try:
        images = convert_from_path(pdf_path)
        return " ".join(pytesseract.image_to_string(image) for image in images)
    except Exception as e:
        print(f"OCR failed for {pdf_path}: {e}", file=sys.stderr)
        return ""


def extract_text(pdf_path: str) -> str:
    text = extract_text_pypdf(pdf_path)
    if not text.strip():
        text = extract_text_ocr(pdf_path)
    return text


def extract_names_from_text(text: str, exceptions=None) -> set:
    """Runs spaCy NER, filters against the stoplist, and deconstructs multi-word names into components.

    Requires each accepted span/component to be tagged PROPN by spaCy's own POS tagger, not just
    labeled PERSON by the (much less reliable) NER model. This is the single biggest precision
    lever available: the NER model regularly mistags section headers, form labels, and technical
    terms as PERSON, but the POS tagger - a separate model in the same pipeline, already computed,
    previously never consulted - almost never tags those as proper nouns. A real surname like
    "Harker" gets PROPN; a mistagged "Development" or "COMMERCIALISATION" does not.

    `exceptions`, if given, is the same phrase list redact_pdf() already uses to protect text from
    redaction - applying it here too means something you've told the tool to never treat as
    sensitive never shows up as a candidate in the first place, not just gets silently protected
    later at redaction time.
    """
    if not text.strip():
        return set()

    doc = nlp(text)
    final_names = set()

    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue

        # If nothing in this span is a proper noun, the PERSON label itself is almost certainly wrong.
        if not any(tok.pos_ == "PROPN" for tok in ent):
            continue

        # A colon immediately after the span (in the ORIGINAL text, not the entity's own captured
        # text - a colon is never part of the entity itself) means this is a form-field label like
        # "Lead Applicant Name:", not free-flowing content - reject regardless of word count, since
        # a multi-word Title-Case label can otherwise pass every other check here.
        if text[ent.end_char:ent.end_char + 2].lstrip().startswith(":"):
            continue

        name = ent.text.strip()

        # Whether to add the FULL phrase as its own candidate (e.g. "Adrian Fenwick"). This does not
        # gate decomposition below - a whole span like "Harker C.1" fails this (trailing citation
        # digit), but the surname "Harker" inside it is still a valid, separately-checked component.
        # A standalone single word ("Track", "Scientist") gets extra scrutiny against the common-
        # word list, since it has no surrounding name context to lend it credibility. A multi-word
        # span doesn't need this - "Michael Hale" is unambiguous even though "Hale" alone is in
        # the wordlist too (it's frequency-based, so it also contains common first/last names).
        is_standalone_common_word = (
            len(name.split()) == 1 and name.lower() in COMMON_ENGLISH_WORDS
        )
        add_whole = (
            not _is_numeric_noise(name)
            and (len(name) >= 2 or re.fullmatch(r'[A-Z]\.', name))
            and name not in COMMON_STOPLIST
            and name.title() not in COMMON_STOPLIST
            and name.upper() not in COMMON_STOPLIST
            and looks_like_a_name(name)
            and not is_standalone_common_word
            and not matches_exception(name, exceptions)
        )
        if add_whole:
            final_names.add(name)

        # Deconstruct into components using spaCy's own tokens (not a hand-rolled regex split) so
        # each component's POS tag is available: only keep tokens spaCy itself calls a proper noun,
        # or a bare capital-letter initial (e.g. "L.").
        for tok in ent:
            comp = tok.text.strip()
            if not comp:
                continue
            if _is_numeric_noise(comp):
                continue
            if comp in COMMON_STOPLIST or comp.title() in COMMON_STOPLIST:
                continue
            if len(comp) == 1 and not re.fullmatch(r'[A-Z]', comp):
                continue
            if tok.pos_ != "PROPN" and not re.fullmatch(r'[A-Z]\.?', comp):
                continue
            if not looks_like_a_name(comp):
                continue
            if matches_exception(comp, exceptions):
                continue
            final_names.add(comp)

    # Drop stray single-letter entries, then restore known-safe multi-letter initials
    final_names = {n for n in final_names if not (re.fullmatch(r'[A-Z]\.?', n) and len(n) <= 2)}
    final_names |= KNOWN_INITIALS
    return final_names


HEADER_ROW_Y_TOLERANCE = 3.0   # points; words within this of each other are "the same row"
NAME_COLUMN_ROW_LIMIT = 25     # max rows below a header to treat as part of its table
NAME_COLUMN_Y_LIMIT = 350.0    # max vertical distance (points) below a header to still count

# Header phrases containing "name" that are NOT about a person - "Technology name/Supplier
# details", "Funding Body Name", "Project Name", "File Name" etc. all match "name" as a
# substring but have nothing to do with redacting anyone. Checked against the matched word
# itself plus up to two preceding words in the same header row (same header phrase/cell).
NON_PERSON_NAME_HEADER_WORDS = {
    "technology", "product", "supplier", "company", "funding", "file", "project", "grant",
    "domain", "device", "brand", "body", "fund", "drug", "software", "hardware", "asset",
    "account", "record", "document", "reference", "programme", "program", "scheme", "study",
    "trial", "server", "database", "field", "column", "variable", "class",
}


def extract_names_from_tables(pdf_path: str, exceptions=None) -> dict:
    """Finds a column headed "Name" (or similar) by its literal word position on the page, then
    reads whatever text sits in that same horizontal band on the rows below it - independent of
    PyMuPDF's find_tables(), which turned out not to recognize some real-world table layouts at
    all (confirmed on an actual document: it found zero tables in the relevant region, in any
    strategy). This works directly off word coordinates instead, the same low-level approach the
    boilerplate-text detector elsewhere in this project already uses.

    This exists because spaCy's NER leans on sentence-level grammar to recognize a PERSON span -
    a bare name sitting alone in a table cell (no verb, no surrounding sentence) is exactly the
    case it's weakest at, even though the column header already tells us unambiguously what that
    cell is. Returns {name: context}, where context is the rest of that row's text.
    """
    results = {}
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return results

    for page in doc:
        words = page.get_text("words")  # (x0, y0, x1, y1, text, block, line, word_no)
        if not words:
            continue

        rows = {}
        for w in words:
            key = round(w[1] / HEADER_ROW_Y_TOLERANCE) * HEADER_ROW_Y_TOLERANCE
            rows.setdefault(key, []).append(w)
        for key in rows:
            rows[key].sort(key=lambda w: w[0])

        header_rows = []
        for y_key, row_words in rows.items():
            for i, w in enumerate(row_words):
                label = w[4].strip().rstrip(":").lower()
                if "name" not in label or len(row_words) < 2:
                    continue
                # Check this word plus up to 2 preceding words (same header phrase) for a
                # non-person context, e.g. "Funding Body Name" or "Technology name/Supplier".
                phrase = " ".join(row_words[j][4] for j in range(max(0, i - 2), i + 1)).lower()
                if any(kw in phrase for kw in NON_PERSON_NAME_HEADER_WORDS):
                    continue
                header_rows.append((y_key, i, row_words))
                break

        for header_y, name_idx, header_words in header_rows:
            # Widen to the start of the whole header phrase, not just the word containing "name":
            # a header like "Full Name" has data below aligned to "Full"'s x-position, not
            # "Name"'s - anchoring only to "Name" cut off short values like a first name that
            # falls under "Full" and to the left of "Name" (confirmed: this is what was dropping
            # first names entirely under a "Full Name" column). Walk left while the gap to the
            # previous word is small enough to still be the same header cell, not a new column.
            PHRASE_GAP_TOLERANCE = 20.0
            left_idx = name_idx
            while left_idx > 0:
                gap = header_words[left_idx][0] - header_words[left_idx - 1][2]
                if gap > PHRASE_GAP_TOLERANCE:
                    break
                left_idx -= 1

            col_x0 = header_words[left_idx][0] - 2
            col_x1 = (
                header_words[name_idx + 1][0] - 2
                if name_idx + 1 < len(header_words)
                else header_words[name_idx][2] + 150
            )

            data_rows = sorted(y for y in rows if header_y < y <= header_y + NAME_COLUMN_Y_LIMIT)
            for y_key in data_rows[:NAME_COLUMN_ROW_LIMIT]:
                row_words = rows[y_key]
                name_words = [w[4] for w in row_words if col_x0 <= w[0] < col_x1]
                if not name_words:
                    continue
                value = " ".join(name_words).strip()
                if not value or _is_numeric_noise(value):
                    continue
                if value in COMMON_STOPLIST or value.title() in COMMON_STOPLIST:
                    continue
                if not looks_like_a_name(value):
                    continue
                if matches_exception(value, exceptions):
                    continue
                other_words = [w[4] for w in row_words if not (col_x0 <= w[0] < col_x1)]
                results.setdefault(value, " ".join(other_words)[:200])

    doc.close()
    return results


MAX_LABEL_VALUE_WORDS = 4  # a name after "Label:" is short; stop collecting regardless past this


def extract_names_from_labels(pdf_path: str, exceptions=None) -> dict:
    """Detects inline "Label: Value" forms - e.g. one line reading "Lead Applicant Name: Jane
    Doe  Institution Name: Springfield Institute..." - as opposed to a column table. This is a
    different real-world layout than extract_names_from_tables() (a column header with data rows
    below it): here the label and its value sit on the *same* line, often with a second unrelated
    "Label: Value" pair right after it. Confirmed necessary on an actual document: spaCy's NER
    didn't just mislabel this text, it produced no PERSON entity for the real name at all (it
    folded "Jane Doe" into a bogus ORG span together with the next label instead) - so there
    was no PERSON span for any text-based filter to work with in the first place.
    """
    results = {}
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return results

    for page in doc:
        words = page.get_text("words")
        if not words:
            continue

        rows = {}
        for w in words:
            key = round(w[1] / HEADER_ROW_Y_TOLERANCE) * HEADER_ROW_Y_TOLERANCE
            rows.setdefault(key, []).append(w)
        for key in rows:
            rows[key].sort(key=lambda w: w[0])

        for row_words in rows.values():
            for i, w in enumerate(row_words):
                text_w = w[4]
                if not text_w.endswith(":"):
                    continue
                label = text_w.rstrip(":").lower()
                if "name" not in label:
                    continue
                # Same non-person-context guard as the table header check: "Funding Body Name:"
                # etc is about an organisation, not a person, even though it contains "name".
                phrase = " ".join(row_words[j][4] for j in range(max(0, i - 2), i + 1)).lower()
                if any(kw in phrase for kw in NON_PERSON_NAME_HEADER_WORDS):
                    continue

                # Collect the words after this label, stopping at the next "Label:" or row end.
                # Looks a couple of words ahead, not just at the very next word, since a label can
                # itself be multi-word ("Institution Name:") - without the lookahead, "Institution"
                # gets swallowed into the value before its own trailing colon is even reached.
                value_words = []
                j = i + 1
                while j < len(row_words) and len(value_words) < MAX_LABEL_VALUE_WORDS:
                    lookahead = row_words[j:j + 2]
                    if any(w2[4].endswith(":") for w2 in lookahead):
                        break
                    value_words.append(row_words[j][4])
                    j += 1

                value = " ".join(value_words).strip()
                if not value or _is_numeric_noise(value):
                    continue
                if value in COMMON_STOPLIST or value.title() in COMMON_STOPLIST:
                    continue
                if not looks_like_a_name(value):
                    continue
                if matches_exception(value, exceptions):
                    continue
                context = " ".join(w2[4] for w2 in row_words)[:200]
                results.setdefault(value, context)

    doc.close()
    return results


def collect_pdfs(input_path: str):
    """Yields PDF paths: the file itself, or every .pdf under a directory tree."""
    if os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for fname in files:
                if fname.lower().endswith(".pdf"):
                    yield os.path.join(root, fname)
    else:
        yield input_path


def clean_names_file(input_path, output_path):
    """Re-validates every line of an existing names file against looks_like_a_name() and rewrites
    it with the junk stripped out. For files built before that check existed (or that picked up
    OCR-garbled text some other way) - a fix to extraction only prevents new junk, it doesn't
    retroactively clean what's already been saved."""
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    kept = sorted({n for n in lines if looks_like_a_name(n)})
    removed = [n for n in lines if n not in set(kept)]

    with open(output_path, "w", encoding="utf-8") as f:
        for name in kept:
            f.write(name + "\n")

    print(f"Cleaned {input_path} -> {output_path}: kept {len(kept)}, removed {len(removed)}.")
    if removed:
        print(f"Removed entries ({len(removed)} total, first 20 shown):", file=sys.stderr)
        for r in removed[:20]:
            print(f"  {r!r}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Extract candidate PERSON names from a PDF file or a directory of PDFs."
    )
    parser.add_argument("input_path", help="Path to a single PDF/directory, or (with --clean) an "
                        "existing names file to re-validate.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output text file, one name per line. Default: names_to_redact.txt "
                        "for normal extraction; input_path itself (in place) for --clean.")
    parser.add_argument("--merge", action="store_true",
                        help="Load existing names from --output first and add to them, instead of "
                             "overwriting. Use this to build up a cumulative master list across "
                             "multiple runs (e.g. MASTER_NAMES_LIST.txt). Omit --merge for a fresh "
                             "run, e.g. checking a 'redacted' output folder for leftover names.")
    parser.add_argument("--clean", action="store_true",
                        help="Instead of extracting from a PDF, re-validate an existing names "
                        "file (input_path) against the current filters and strip out entries "
                        "that don't structurally look like a name (full sentences, OCR-garbled "
                        "run-on blobs, form labels, etc).")
    parser.add_argument("--exceptions_file", default=None,
                        help="Optional path to a text file with phrases that must NOT be treated "
                        "as sensitive - the same file redact_pymupdf_fast.py's --exceptions_file "
                        "uses. Applying it here too means a phrase you've already told the tool "
                        "to ignore never shows up as a candidate to review in the first place.")
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"Error: input path not found: {args.input_path}", file=sys.stderr)
        sys.exit(1)

    if args.clean:
        clean_names_file(args.input_path, args.output or args.input_path)
        return

    args.output = args.output or "names_to_redact.txt"
    exceptions = set(p.lower() for p in load_exceptions_from_file(args.exceptions_file))

    master_names = set()
    if args.merge and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            master_names = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(master_names)} existing names from {args.output} (--merge).")

    pdf_count = 0
    for pdf_path in collect_pdfs(args.input_path):
        pdf_count += 1
        text = extract_text(pdf_path)
        found = set()
        if text.strip():
            found = extract_names_from_text(text, exceptions=exceptions)
        else:
            print(f"Warning: no text extracted from {pdf_path}", file=sys.stderr)

        table_names = extract_names_from_tables(pdf_path, exceptions=exceptions)
        label_names = extract_names_from_labels(pdf_path, exceptions=exceptions)
        found |= table_names.keys()
        found |= label_names.keys()

        master_names.update(found)
        print(f"  {pdf_path}: +{len(found)} names ({len(table_names)} from table columns, "
              f"{len(label_names)} from inline labels)")

    if pdf_count == 0:
        print(f"No PDF files found at {args.input_path}.", file=sys.stderr)
        sys.exit(1)

    name_list = sorted(master_names)
    with open(args.output, "w", encoding="utf-8") as f:
        for name in name_list:
            f.write(name + "\n")

    print(f"\nDone. Processed {pdf_count} PDF(s). {len(name_list)} unique names written to {args.output}.")


if __name__ == "__main__":
    main()
