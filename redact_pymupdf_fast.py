#;encoding=utf-8
# Fast PDF Redaction Script using PyMuPDF (fitz)
# This script handles names (loaded from a file), emails, phone numbers, LinkedIn URLs, 
# and now, Global Boilerplate Graphics and Deep Content/Coordinate Repetitive Text.

import os
import re
import sys
import argparse
import fitz # PyMuPDF
from datetime import datetime, UTC 
import numpy as np # Used for basic math safety checks

# --- Configuration Constants ---
REDACTION_COLOR = (0, 0, 0) # RGB for black
REDACTION_TEXT = "REDACTED"
EMAIL_REGEX = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', re.IGNORECASE)
PHONE_REGEX = re.compile(r'(\+\d{1,3}[-. ]?)?(\(\d{2,5}\)|\d{2,5})([-. ]?\d{2,5}){2,4}', re.IGNORECASE)
LINKEDIN_REGEX = re.compile(r'(?:https?://)?(?:www\.)?(linkedin\.com/|linked\.in/)\S*', re.IGNORECASE)

# --- Configuration Constants for Boilerplate Elements ---
HEADER_ZONE_HEIGHT = 40.0 # Vertical zone (in points) from top/bottom edges for analysis
FOOTER_ZONE_HEIGHT = 60.0
COORD_TOLERANCE = 3.0 # Tolerance for coordinate matching graphics

# Constants for Deep Text Repetition Analysis 
TEXT_Y_TOLERANCE = 1.0 # Very tight tolerance for matching Y-coordinates of words/spans
REPETITION_THRESHOLD = 0.9 # Minimum frequency (90%) for a (text, y-coord) pair to be considered boilerplate

# --- Global Storage for Boilerplate Templates ---
# Stores bounding boxes (x0, y0, x1, y1) of graphics found in the header/footer of Page 1.
global_boilerplate_coords = [] 
# Stores (Y0_KEY, TEXT_STRING) tuples identified as highly repetitive boilerplate text.
global_boilerplate_text_patterns = set()

MIN_HEADER_FOOTER_IMAGE_SIZE = 40.0  # tweakable

# --- Signature detection configuration ---

SIGNATURE_TRIGGER_KEYWORDS = {
    "regards", "regard", "sincerely", "faithfully", "cheers", "Yours Sincerely, ", "Yours Sincerely,", "Yours Sincerely", "Yours Sincerely ", "sincerely", "Sincerely", "Sincerely ", "sincerely ", "sincerely,", "Sincerely,", "Sincerely, ", "sincerely, "
}

SIGNATURE_MIN_OFFSET = 5.0       # pts below trigger line
SIGNATURE_MAX_OFFSET = 120.0     # max distance below trigger line
MIN_SIGNATURE_WIDTH   = 80.0     # minimum width in points
MIN_SIGNATURE_AR      = 2.5      # width/height >= 2.5
MAX_SIGNATURE_WIDTH_FRAC = 0.8   # must be <= 80% of page width
#MIN_TRIGGER_Y_FRACTION = 0.10     # trigger word must be in bottom half - DISABLED TO ALLOW HIGHER SIGNATURES TO BE DETECTED



def find_header_footer_images(page):
    """
    Heuristic per-page detector for header/footer *images* that are
    likely logos or decorative graphics.

    Rules:
      - image centre is in top or bottom HEADER/FOOTER zone
      - image is at least MIN_HEADER_FOOTER_IMAGE_SIZE in width or height
      - (optionally) no overlapping text blocks -> "non-text graphics"
    """
    page_height = page.rect.height
    candidates = []

    for info in page.get_image_info(xrefs=True):
        # Clamp bbox to the page rect in case it bleeds slightly off-page
        bbox = fitz.Rect(info["bbox"]) & page.rect

        # Skip tiny images like bullets
        if max(bbox.width, bbox.height) < MIN_HEADER_FOOTER_IMAGE_SIZE:
            continue

        y_mid = (bbox.y0 + bbox.y1) / 2.0

        in_header = y_mid < HEADER_ZONE_HEIGHT
        in_footer = y_mid > (page_height - FOOTER_ZONE_HEIGHT)

        if not (in_header or in_footer):
            continue

        # If you truly want "non-text graphics only", keep this check.
        # If you're happy to wipe letterhead text too, you can comment this out.
        #if rect_has_text(page, bbox):
        #    continue

        candidates.append(bbox)

    return candidates

def find_signature_images(page):
    """
    Over-zealous but robust signature remover:
    If a sign-off trigger phrase is found, redact a fixed rectangular "signature zone"
    under/overlapping the trigger line. This avoids relying on PDF object types
    (image vs vector vs XObject), which is why signatures sometimes slip through.
    """

    # Tune these safely
    SIGN_ZONE_OVERLAP_ABOVE = 8.0        # include the trigger line itself if needed
    SIGN_ZONE_BELOW = 140.0              # how far below the trigger line to redact
    SIGN_ZONE_MAX_HEIGHT_FRAC = 0.30     # never redact more than 30% of page height
    SIGN_ZONE_MAX_WIDTH_FRAC = 1.00      # 1.00 = full width; reduce if you want
    MIN_TRIGGER_Y_FRAC = 0.40            # only consider triggers in lower 60% of page

    # Triggers (lowercase)
    TRIGGERS = [
        "yours sincerely",
        "yours faithfully",
        "kind regards",
        "best regards",
        "regards",
        "sincerely",
        "faithfully",
        "yours",
    ]

    width = page.rect.width
    height = page.rect.height

    # --- Find trigger lines using dict text (more reliable for multi-word phrases) ---
    text = page.get_text("dict")
    trigger_line_rects = []

    for block in text.get("blocks", []):
        for line in block.get("lines", []):
            line_text = "".join(span.get("text", "") for span in line.get("spans", "")).strip()
            if not line_text:
                continue
            lt = " ".join(line_text.lower().split())

            # ignore triggers too high up the page (reduces false positives)
            line_bbox = fitz.Rect(line["bbox"])
            #if line_bbox.y0 < height * MIN_TRIGGER_Y_FRAC:  ##< DISBALED TO TEST DETECTION OF SIGNATURES HIGHER IN PAGE
            #    continue

            if any(t in lt for t in TRIGGERS):
                trigger_line_rects.append(line_bbox)

    if not trigger_line_rects:
        return []

    # --- Build signature-zone redaction rectangles ---
    zones = []
    for lr in trigger_line_rects:
        # Start slightly above the trigger line to allow overlap and to nuke the phrase if needed
        y0 = max(0.0, lr.y0 - SIGN_ZONE_OVERLAP_ABOVE)

        # End below the line, but clamp to a maximum fraction of page height
        max_zone_height = height * SIGN_ZONE_MAX_HEIGHT_FRAC
        y1 = min(height, y0 + min(SIGN_ZONE_BELOW + (lr.height), max_zone_height))

        # Width: full width by default
        zone_width = width * SIGN_ZONE_MAX_WIDTH_FRAC
        x0 = 0.0
        x1 = min(width, x0 + zone_width)

        zones.append(fitz.Rect(x0, y0, x1, y1))

    # Merge overlapping zones to avoid multiple redaction bands on same page
    zones.sort(key=lambda r: (r.y0, r.x0))
    merged = []
    gap = 5.0
    for r in zones:
        if not merged:
            merged.append(r)
            continue
        last = merged[-1]
        last_expanded = fitz.Rect(last.x0, last.y0 - gap, last.x1, last.y1 + gap)
        if last_expanded.intersects(r):
            merged[-1] = last | r
        else:
            merged.append(r)

    return merged



def find_email_rects_from_words(words):
    """
    Detect email addresses that may be broken across multiple words/lines.
    Returns a list of fitz.Rect objects covering the full email.
    """
    rects = []
    max_tokens = 5  # more than enough for any sane email split
    n = len(words)

    for i in range(n):
        concat = ""
        for j in range(i, min(i + max_tokens, n)):
            token = words[j][4]
            concat += token

            candidate = concat.strip()
            # strip trailing punctuation that might cling to the email
            candidate_clean = candidate.rstrip(".,);:?!")

            if "@" not in candidate_clean:
                # can't be an email yet
                continue

            # require *full* match, not substring
            if EMAIL_REGEX.fullmatch(candidate_clean):
                # union of all word boxes from i..j
                xs0 = [words[k][0] for k in range(i, j + 1)]
                ys0 = [words[k][1] for k in range(i, j + 1)]
                xs1 = [words[k][2] for k in range(i, j + 1)]
                ys1 = [words[k][3] for k in range(i, j + 1)]

                rect = fitz.Rect(min(xs0), min(ys0), max(xs1), max(ys1))
                rects.append(rect)
                break  # stop extending this window; move to next i

    return rects

def parse_page_spec(spec):
    """
    Parse a page spec string like '3,5-7, 10' into a set of 1-based page numbers.
    """
    pages = set()
    if not spec:
        return pages

    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            start_s, end_s = part.split("-", 1)
            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            pages.update(range(start, end + 1))
        else:
            try:
                pages.add(int(part))
            except ValueError:
                continue

    return pages


# =======================================================
# 📌 Manual name-search: a targeted "is this specific name really gone" check, independent of
# the automated extraction pipeline. No spaCy dependency, so it works even when that's broken.
# =======================================================

def build_name_search_pattern(name):
    """A regex that matches `name` regardless of the spacing between its words - a normal single
    space, multiple spaces, a literal newline (a name split across a line break in the extracted
    text), or nothing at all (the two words run together, e.g. "JaneDoe" - a real, observed
    PDF text-extraction artifact) all match the same pattern, since \\s* allows zero or more
    whitespace characters between each word."""
    words = name.strip().split()
    if not words:
        return None
    return re.compile(r'\s*'.join(re.escape(w) for w in words), re.IGNORECASE)


def _context_snippet(text, start, end, window=40):
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    return " ".join(text[lo:hi].split())


def search_document_for_names(pdf_path, names):
    """For each name, searches BOTH the page text and the document metadata (Info dict, XMP,
    embedded-file names) for any occurrence, tolerant of spacing variations. Returns
    {name: {"text_matches": [...], "metadata_matches": [...]}}; each text match has page/matched
    text/context, each metadata match has which field and a snippet of that field's value.
    """
    results = {name: {"text_matches": [], "metadata_matches": []} for name in names}
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return {"__error__": f"Could not open {pdf_path}: {e}"}

    meta_sources = []
    for key, value in (doc.metadata or {}).items():
        if value:
            meta_sources.append((f"metadata:{key}", value))
    xml_meta = doc.get_xml_metadata() or ""
    if xml_meta:
        meta_sources.append(("metadata:xmp", xml_meta))
    for embedded_name in doc.embfile_names():
        meta_sources.append(("metadata:embedded-file-name", embedded_name))

    for name in names:
        pattern = build_name_search_pattern(name)
        if pattern is None:
            continue

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            for m in pattern.finditer(text):
                results[name]["text_matches"].append({
                    "page": page_num,
                    "matched_text": m.group(0),
                    "context": _context_snippet(text, m.start(), m.end()),
                })

        for field, value in meta_sources:
            for m in pattern.finditer(value):
                results[name]["metadata_matches"].append({
                    "field": field,
                    "matched_text": m.group(0),
                    "context": _context_snippet(value, m.start(), m.end()),
                })

    doc.close()
    return results


def search_names_in_path(input_path, names):
    """search_document_for_names() over a single PDF or every PDF under a folder. Returns
    {name: [{"file": str, "text_matches": [...], "metadata_matches": [...]}]} - one entry per
    file that had at least one match for that name, so a hit in a batch of many documents still
    tells you exactly which file to open.
    """
    results = {name: [] for name in names}

    if os.path.isdir(input_path):
        pdf_paths = []
        for dirpath, _, filenames in os.walk(input_path):
            for fname in filenames:
                if fname.lower().endswith(".pdf"):
                    pdf_paths.append(os.path.join(dirpath, fname))
    else:
        pdf_paths = [input_path]

    for pdf_path in pdf_paths:
        per_file = search_document_for_names(pdf_path, names)
        if "__error__" in per_file:
            print(per_file["__error__"], file=sys.stderr)
            continue
        for name, matches in per_file.items():
            if matches["text_matches"] or matches["metadata_matches"]:
                results[name].append({"file": pdf_path, **matches})

    return results


# =======================================================
# 📌 STEP 1: DEFINE FUNCTION TO LOAD NAMES FROM A FILE
# =======================================================
def load_names_from_file(filepath):
    """Loads a list of names, one per line, from a given text file."""
    names = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f: 
                name = line.strip()
                if name:
                    names.append(name)
        return names
    except FileNotFoundError:
        # A raised exception, not sys.exit() - this is also called as a library function (e.g. from
        # the web UI), where sys.exit()'s SystemExit would silently kill the request thread instead
        # of being caught as a normal error and reported back to the user.
        print(f"Error: Name list file not found at '{filepath}'. Please run name extraction first.", file=sys.stderr)
        raise FileNotFoundError(f"Name list file not found at '{filepath}'. Please run name extraction first.")

def load_exceptions_from_file(filepath):
    """Loads a list of exception phrases (one per line) that must NOT be redacted."""
    phrases = []
    if not filepath:
        return phrases
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                phrase = line.strip()
                if phrase:
                    phrases.append(phrase)
    except FileNotFoundError:
        print(f"Warning: exceptions file not found at '{filepath}'. Continuing without exceptions.", file=sys.stderr)
    return phrases



# =======================================================
# 📌 STEP 2A: PRE-ANALYSIS: IDENTIFY GLOBAL BOILERPLATE GRAPHICS
# =======================================================
def get_global_boilerplate_templates(doc, start_page, end_page):
    start_index = max(1, start_page)
    end_index   = min(doc.page_count, end_page)

    rect_counts = {}  # quantised rect -> occurrence count

    for page_num in range(start_index, end_index + 1):
        page = doc.load_page(page_num - 1)
        height = page.rect.height
        header_zone = fitz.Rect(0, 0, page.rect.width, HEADER_ZONE_HEIGHT)
        footer_zone = fitz.Rect(0, height - FOOTER_ZONE_HEIGHT, page.rect.width, height)

        # IMAGES + DRAWINGS
        elements = []
        elements += [fitz.Rect(img["bbox"]) for img in page.get_image_info(xrefs=True)]
        elements += [path["rect"] for path in page.get_drawings()]

        for r in elements:
            if r not in header_zone and r not in footer_zone:
                continue

            # quantise to make matching robust
            key = (
                round(r.x0, 1),
                round(r.y0, 1),
                round(r.x1, 1),
                round(r.y1, 1),
            )
            rect_counts[key] = rect_counts.get(key, 0) + 1

    # keep only those present on most pages in the sample
    pages_sampled = end_index - start_index + 1
    global global_boilerplate_coords
    global_boilerplate_coords = [
        key for key, count in rect_counts.items()
        if count / pages_sampled >= 0.7  # 70% of pages -> boilerplate
    ]


# =======================================================
# 📌 STEP 2B: PRE-ANALYSIS: IDENTIFY GLOBAL BOILERPLATE TEXT (Deep Content Repetition)
# =======================================================
def rect_has_text(page, rect):
    """Return True if any text block overlaps the given rectangle."""
    for block in page.get_text("blocks"):
        block_rect = fitz.Rect(block[:4])
        text = (block[4] or "").strip()
        if text and block_rect.intersects(rect):
            return True
    return False


def get_global_boilerplate_text_templates(doc, start_page, end_page):
    """
    Analyzes a specified page range at the word/span level to find text content 
    that is highly repetitive at a precise Y-coordinate within the boilerplate zones.
    """
    start_index = max(1, start_page) # Start at page 1 (index 0 is often unique cover)
    end_index = min(doc.page_count, end_page + 1) # Range is exclusive, so add 1 to end_page
    
    pages_to_sample = end_index - start_index
    
    if pages_to_sample < 2:
        # Need at least two pages in the analysis range to check for repetition
        return
        
    # Maps (y0_key, text_content) -> set(page_numbers_seen_on)
    content_y0_counts = {}
    
    for page_num in range(start_index, end_index):
        page = doc.load_page(page_num - 1) # page_num is 1-based, load_page is 0-based index
        height = page.rect.height
        
        header_limit = HEADER_ZONE_HEIGHT
        footer_limit = height - FOOTER_ZONE_HEIGHT
        
        # Get words: (x0, y0, x1, y1, text, block_no, line_no, word_no)
        words = page.get_text('words')
        
        for word in words:
            text_content = word[4].strip()
            if not text_content: continue
            
            y0 = word[1] # Top coordinate of the word
            
            # Check if the word is in the header or footer zone
            is_in_header = y0 < header_limit
            is_in_footer = word[3] > footer_limit # Check bottom coordinate against footer limit
            
            if is_in_header or is_in_footer:
                # Create a key by rounding the Y0 coordinate to the tight tolerance
                y0_key = round(y0 / TEXT_Y_TOLERANCE) * TEXT_Y_TOLERANCE
                
                # The pattern is the combination of Y-position and the text content
                pattern = (y0_key, text_content)
                
                if pattern not in content_y0_counts:
                    content_y0_counts[pattern] = set()
                # Store the page number the pattern was found on
                content_y0_counts[pattern].add(page_num)

    
    global global_boilerplate_text_patterns
    
    # Identify patterns that appear on > REPETITION_THRESHOLD of sampled pages
    for pattern, pages_set in content_y0_counts.items():
        if len(pages_set) / pages_to_sample >= REPETITION_THRESHOLD:
            global_boilerplate_text_patterns.add(pattern)

    if global_boilerplate_text_patterns:
        print(f"Found {len(global_boilerplate_text_patterns)} deep boilerplate text patterns from Pages {start_page}-{end_page} (Repetition > {REPETITION_THRESHOLD:.0%}).", file=sys.stderr)

def batch_redact_tree(
    input_root,
    output_root,
    names_file,
    boilerplate_graphics,
    boilerplate_text,
    analysis_start_page,
    analysis_end_page,
    exceptions_file=None,
    full_page_redact_pages=None,
    redact_names=True,
    redact_emails=True,
    redact_phones=True,
    emails_file=None,
    phones_file=None,
):
    """
    Walk input_root for PDFs and create matching structure under output_root.

    Example:
      input_root / "DEMO-2024-0001/Panel_Recommendations_WPV.pdf"
    -> output_root / "DEMO-2024-0001/Panel_Recommendations_WPV_REDACTED.pdf"
    """
    for dirpath, _, filenames in os.walk(input_root):
        for fname in filenames:
            if not fname.lower().endswith(".pdf"):
                continue

            in_full = os.path.join(dirpath, fname)
            rel = os.path.relpath(in_full, input_root)

            base, ext = os.path.splitext(os.path.basename(fname))
            out_name = f"{base}_REDACTED{ext}"

            out_dir = os.path.join(output_root, os.path.dirname(rel))
            os.makedirs(out_dir, exist_ok=True)
            out_full = os.path.join(out_dir, out_name)

            print(f"\n=== Redacting: {in_full} ===", file=sys.stderr)
            redact_pdf(
                in_full,
                out_full,
                names_file,
                boilerplate_graphics,
                boilerplate_text,
                analysis_start_page,
                analysis_end_page,
                exceptions_file,
                full_page_redact_pages=full_page_redact_pages,
                redact_names=redact_names,
                redact_emails=redact_emails,
                redact_phones=redact_phones,
                emails_file=emails_file,
                phones_file=phones_file,
            )



# =======================================================
# 📌 STEP 3: MAIN REDACTION FUNCTION
# =======================================================
def redact_pdf(input_file, output_file, names_file, boilerplate_graphics, boilerplate_text, analysis_start_page, analysis_end_page, exceptions_file=None, full_page_redact_pages=None, redact_names=True, redact_emails=True, redact_phones=True, emails_file=None, phones_file=None, manual_page_rects=None, manual_global_terms=None, automated=True,):
    """
    manual_page_rects / manual_global_terms / automated support the interactive click-to-redact
    flow in the web UI (see /redact_interactive in redactor_webui.py):

      - manual_page_rects: {page_number_1based: [[x0, y0, x1, y1], ...]} - explicit rectangles to
        black out, in the page's *displayed* coordinate space (points, top-left origin - i.e.
        exactly what page.rect / page.search_for use). These are the "redact just this occurrence"
        (red) selections.
      - manual_global_terms: [str, ...] - literal strings searched across every page with
        page.search_for (case-insensitive) and redacted wherever found. These are the "redact
        every occurrence" (yellow) selections. Document metadata is always scrubbed on save
        regardless (see STEP 6 below), so those terms are covered there too.
      - automated: when False, every heuristic strategy (name tokens, email/phone/LinkedIn
        regexes, boilerplate graphics/text, signature zones) is skipped - only full-page
        redaction, manual rectangles, manual global terms and the exceptions filter run. This is
        what the interactive flow uses: it redacts exactly what the user picked, nothing else.
    """

    if full_page_redact_pages is None:
        full_page_redact_pages = set()
    if manual_page_rects is None:
        manual_page_rects = {}
    if manual_global_terms is None:
        manual_global_terms = []

    # Emails/phones: if a curated exact-value file is given, redact ONLY those specific values
    # (matched like names, via exact search) - nothing else, even if the file is empty (an empty
    # curated list means "nothing curated yet", not "no list was given"). Only when no file path
    # is given at all do we fall back to the blanket regex scan below, which catches anything
    # matching the pattern regardless of review.
    emails_file_given = bool(redact_emails and emails_file)
    phones_file_given = bool(redact_phones and phones_file)
    emails_to_redact = load_names_from_file(emails_file) if emails_file_given else []
    phones_to_redact = load_names_from_file(phones_file) if phones_file_given else []

    if redact_names:
        names_to_redact = load_names_from_file(names_file)
        if not names_to_redact:
            print(f"Warning: No names found in '{names_file}'.", file=sys.stderr)
    else:
        names_to_redact = []

    exceptions_phrases = load_exceptions_from_file(exceptions_file)

    # 1. Open the PDF document
    try:
        doc = fitz.open(input_file)
    except Exception as e:
        print(f"Error opening PDF: {e}", file=sys.stderr)
        return

    # 2. Identify global boilerplate elements BEFORE the main page loop
    if boilerplate_graphics:
        get_global_boilerplate_templates(doc, analysis_start_page, analysis_end_page)
    
    if boilerplate_text:
        get_global_boilerplate_text_templates(doc, analysis_start_page, analysis_end_page)
    
    total_redactions = 0
    
    # Pre-compile the word-boundary regexes for names
    name_regexes = {}
    for name in names_to_redact:
        pattern = re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE)
        name_regexes[name] = pattern

    # Build token-level name patterns to avoid in-word redaction
    single_token_names = set()
    multi_token_names = {}  # length -> set of tuples

    for name in names_to_redact:
        parts = [p.lower() for p in name.split() if p]
        if not parts:
            continue
        if len(parts) == 1:
            single_token_names.add(parts[0])
        else:
            multi_token_names.setdefault(len(parts), set()).add(tuple(parts))

    max_name_len = max(multi_token_names.keys(), default=1)


    # 3. Iterate through every page
    print(f"Starting redaction of {doc.page_count} pages...", file=sys.stderr)
    for page_num, page in enumerate(doc):
        page_number = page_num +1 #1-based
        # List to hold all found unique redaction areas (Rectangles)
        redact_rects_set = set()
        
        # Get the plain text from the page once for entity search 
        text = page.get_text()

        # --- Sub-Strategy 0: full-page redaction if requested ---
        if page_number in full_page_redact_pages:
            # Add a single rectangle covering the entire page.
            # Other strategies can still add more rects, but this one
            # will already cover everything.
            redact_rects_set.add(tuple(page.rect))


        # --- Sub-Strategy M (MANUAL SELECTIONS from the interactive click-to-redact UI) ---
        # These run regardless of `automated`: they ARE the whole point of the interactive flow.
        # manual_page_rects are already in this page's displayed coordinate space.
        for r in manual_page_rects.get(page_number, []):
            redact_rects_set.add(tuple(fitz.Rect(r)))
        # manual_global_terms: literal doc-wide search (case-insensitive), same mechanism as a
        # curated email/phone value. Metadata is scrubbed wholesale on save (STEP 6) either way.
        for term in manual_global_terms:
            if not term or not term.strip():
                continue
            for r in page.search_for(term):
                redact_rects_set.add(tuple(r))

        # A. Redact names using word tokens (never partial words)
        words = page.get_text("words")  # (x0, y0, x1, y1, text, block, line, word)

        # --- Sub-Strategy A, B, C, D, E, F: automated heuristics. Skipped entirely when
        # automated=False (the interactive flow redacts only the manual selections above). ---
        if automated:
          if redact_names:
            tokens = []
            for w in words:
                raw = w[4]
                # normalize: strip punctuation around the word
                clean = re.sub(r"^[^\w'-]+|[^\w'-]+$", "", raw).lower()
                tokens.append(clean)

            # single-token names
            for i, tok in enumerate(tokens):
                if tok and tok in single_token_names:
                    rect = fitz.Rect(words[i][:4])
                    redact_rects_set.add(tuple(rect))

            # multi-token names (e.g., "Jane Doe")
            for n, patterns in multi_token_names.items():
                if n == 1 or len(tokens) < n:
                    continue
                for i in range(len(tokens) - n + 1):
                    seq = tuple(tokens[i:i + n])
                    if any(not t for t in seq):
                        continue
                    if seq in patterns:
                        r0 = words[i]
                        r1 = words[i + n - 1]
                        rect = fitz.Rect(r0[0], r0[1], r1[2], r1[3])
                        redact_rects_set.add(tuple(rect))

          # B0/curated. Emails: a curated exact-value list (from review) takes priority and is matched
          # exactly, like names - nothing outside that list gets touched. Without one, fall back to the
          # word-window regex scan (handles multi-line emails) for blanket detection.
          if emails_to_redact:
            for value in emails_to_redact:
                for r in page.search_for(value):
                    redact_rects_set.add(tuple(r))
          elif redact_emails and not emails_file_given:
            for email_rect in find_email_rects_from_words(words):
                redact_rects_set.add(tuple(email_rect))

          # Phones: same curated-list-first approach.
          if phones_to_redact:
            for value in phones_to_redact:
                for r in page.search_for(value):
                    redact_rects_set.add(tuple(r))

          # --- B. Redact other entities using SPAN-LEVEL text extraction (Most Robust) ---
          # LinkedIn is always checked. Email/phone regexes only run here in blanket mode - i.e. when
          # no curated list was given for that category at all - since an empty curated list means
          # "nothing checked yet", not "fall back to blanket". A curated list is already exact.
          entity_patterns = []
          if redact_emails and not emails_file_given:
            entity_patterns.append(("Email", EMAIL_REGEX))
          if redact_phones and not phones_file_given:
            entity_patterns.append(("Phone", PHONE_REGEX))
          entity_patterns.append(("LinkedIn", LINKEDIN_REGEX))

          all_found_entities = set()

          # Extract text at the span level
          text_dict = page.get_text("dict")
          for block in text_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    # Join all text in the line's spans into a single string
                    line_text = "".join(span["text"] for span in line["spans"])

                    if not line_text:
                        continue

                    for _, regex in entity_patterns:
                        # Find all matches for the regex within this reconstructed line
                        matches = regex.finditer(line_text)
                        for match in matches:
                            all_found_entities.add(match.group(0))

          # Now, search for the found entities on the page
          for entity_string in all_found_entities:
            # We must use page.search_for on the EXACT string found
            rects = page.search_for(entity_string)
            redact_rects_set.update(tuple(r) for r in rects)

          # C. Redact LinkedIn via link annotations (handles wrapped / hidden URLs) - KEEP THIS
          for link in page.get_links():
            uri = (link.get("uri") or "").lower()
            if uri and ("linkedin.com" in uri or "linked.in/" in uri):
                try:
                    link_rect = fitz.Rect(link["from"])
                    redact_rects_set.add(tuple(link_rect))
                except Exception:
                    pass


          # --- Sub-Strategy D (GLOBAL BOILERPLATE GRAPHIC REDACTION) ---
          if boilerplate_graphics and global_boilerplate_coords:

            elements_to_check = []
            # 1. Add IMAGES
            for img in page.get_image_info(xrefs=True):
                elements_to_check.append(fitz.Rect(img['bbox']))
            # 2. Add VECTOR DRAWINGS
            for path in page.get_drawings():
                elements_to_check.append(path['rect'])

            for bbox in elements_to_check:

                # Check this graphic element against all global templates
                for template_coords in global_boilerplate_coords:
                    template_bbox = fitz.Rect(template_coords)

                    # Check if the current bbox coordinates are within the defined tolerance of the template
                    x0_match = abs(bbox.x0 - template_bbox.x0) < COORD_TOLERANCE
                    y0_match = abs(bbox.y0 - template_bbox.y0) < COORD_TOLERANCE
                    x1_match = abs(bbox.x1 - template_bbox.x1) < COORD_TOLERANCE
                    y1_match = abs(bbox.y1 - template_bbox.y1) < COORD_TOLERANCE

                    if x0_match and y0_match and x1_match and y1_match:
                        redact_rects_set.add(tuple(bbox))
                        break # Found a match, move to the next element

          # --- Sub-Strategy D2 (LOCAL HEADER/FOOTER IMAGE REDACTION) ---
          if boilerplate_graphics:
            for img_rect in find_header_footer_images(page):
                redact_rects_set.add(tuple(img_rect))


          # --- Sub-Strategy E (GLOBAL BOILERPLATE TEXT REDACTION - Deep Word Match) ---
          if boilerplate_text and global_boilerplate_text_patterns:

            words = page.get_text('words')

            for word in words:
                text_content = word[4].strip()
                if not text_content: continue

                y0 = word[1] # Top coordinate of the word
                word_rect = fitz.Rect(word[:4])

                # Create a key by rounding the Y0 coordinate to the tight tolerance
                y0_key = round(y0 / TEXT_Y_TOLERANCE) * TEXT_Y_TOLERANCE

                # Check if this word's content and its precise Y-coordinate match a global boilerplate pattern
                pattern = (y0_key, text_content)

                if pattern in global_boilerplate_text_patterns:
                    # Redact the word
                    redact_rects_set.add(tuple(word_rect))

          # --- Sub-Strategy F (SIGNATURE IMAGE REDACTION) ---
          for sig_rect in find_signature_images(page):
            redact_rects_set.add(tuple(sig_rect))

        # --- Sub-Strategy X: apply exceptions ("do not redact" phrases) ---
        if exceptions_phrases:
            exception_rects = []
            for phrase in exceptions_phrases:
                for ex_rect in page.search_for(phrase):
                    exception_rects.append(ex_rect)

            if exception_rects:
                kept = set()
                for r in redact_rects_set:
                    rect = fitz.Rect(r)
                    # drop any redaction that intersects an exception phrase
                    if any(rect.intersects(er) for er in exception_rects):
                        continue
                    kept.add(r)
                redact_rects_set = kept


        
        # Convert unique tuples back to fitz.Rect objects
        final_rects = [fitz.Rect(r) for r in redact_rects_set]
        
        # 4. Apply redaction annotation to each found instance
        for rect in final_rects:
            if rect.is_infinite:
                continue

            # Inflate degenerate rectangles
            if rect.width <= 1.0 or rect.height <= 1.0:
                rect.x1 = rect.x0 + max(1.0, rect.width)
                rect.y1 = rect.y0 + max(1.0, rect.height)
            
            annot_text = REDACTION_TEXT
            if rect.width < 10 or rect.height < 10:
                annot_text = ""

            page.add_redact_annot(
                rect,
                text=annot_text,
                fill=REDACTION_COLOR,
                cross_out=False
            )
        
        # 5. Apply the queued redactions to the page's content stream
        if final_rects:
            page.apply_redactions()
            total_redactions += len(final_rects)

    # =======================================================
    # 📌 STEP 6: METADATA CLEANUP
    # =======================================================
    now_utc = datetime.now(UTC).strftime('%Y%m%d%H%M%S')
    new_metadata = {
        'title': 'Redacted Document',
        'author': 'Redacted',
        'subject': 'Cleaned of sensitive data',
        'creator': 'PyMuPDF Script',
        'producer': 'PyMuPDF Script',
        'creationDate': 'D:' + now_utc + 'Z00\'00\'',
        'modDate': 'D:' + now_utc + 'Z00\'00\'',
    }
    doc.set_metadata(new_metadata)

    # Clear XMP metadata (a separate XML stream that set_metadata() above does not touch,
    # and which can otherwise still carry the original author/title/producer).
    doc.set_xml_metadata("")

    # Remove any embedded file attachments (e.g. source docs someone attached to the PDF) -
    # they bypass all page-level redaction since they're opaque blobs, not page content.
    for embfile_name in doc.embfile_names():
        doc.embfile_del(embfile_name)

    # Warn if the PDF has optional content groups (togglable layers). Hidden layers can hold
    # sensitive content that survives page-level redaction; this tool doesn't auto-flatten them
    # since blindly doing so can corrupt complex PDFs, so a manual check is safer.
    ocgs = doc.get_ocgs()
    if ocgs:
        layer_names = [info.get("name", "?") for info in ocgs.values()]
        print(f"Warning: document has {len(ocgs)} optional content layer(s) not checked by this "
              f"tool: {layer_names}. Inspect manually if any may contain sensitive content.",
              file=sys.stderr)

    # 7. Save the modified document
    print(f"Saving redacted document to {output_file}...", file=sys.stderr)
    doc.save(
        output_file,
        garbage=4,  # Remove unused objects
        deflate=True, # Recompress streams
        clean=True, # Clean up cross-references and stream lengths
    )
    
    # 8. Print success message
    print(f"\n✅ Success! {total_redactions} unique items redacted across {doc.page_count} pages.")
    print(f"Output saved to {output_file}.", file=sys.stderr)
    
    doc.close()

def main():
    parser = argparse.ArgumentParser(description="Redact names and entities from a PDF file using PyMuPDF (fitz).")
    parser.add_argument("input_file", help="Path to the input PDF file OR directory.")
    parser.add_argument("output_file", help="Path to the output redacted PDF file OR directory.")
    parser.add_argument("--names_file", 
                        default="names_to_redact.txt", 
                        help="Path to the text file containing names (one per line) to redact. (Default: names_to_redact.txt)")
    parser.add_argument("--boilerplate_graphics", 
                        action="store_true", 
                        help="Enable Global Boilerplate Redaction for repetitive IMAGES and GRAPHICS (logos/lines) found in header/footer zones.")
    parser.add_argument("--boilerplate_text", 
                        action="store_true", 
                        help="Enable Deep Boilerplate Text Redaction. Targets specific, highly repetitive TEXT content (e.g., page numbers, document titles) at consistent vertical coordinates.")
    parser.add_argument("--analysis_start_page", 
                        type=int, 
                        default=1, 
                        help="Start page (1-based) for boilerplate analysis (default: 1).")
    parser.add_argument("--analysis_end_page", 
                        type=int, 
                        default=10, 
                        help="End page (1-based, inclusive) for boilerplate analysis (default: 10).")
    parser.add_argument("--exceptions_file",
                        default=None,
                        help="Optional path to a text file with phrases that must NOT be redacted (one per line).")
    parser.add_argument("--full_page_redact",
                        default="",
                        help="Comma-separated list of pages or ranges to fully redact, "
                        "e.g. '3,5-7,12' (1-based page numbers).")
    parser.add_argument("--no-redact-names",
                        dest="redact_names", action="store_false", default=True,
                        help="Skip name redaction (on by default).")
    parser.add_argument("--no-redact-emails",
                        dest="redact_emails", action="store_false", default=True,
                        help="Skip email redaction (on by default).")
    parser.add_argument("--no-redact-phones",
                        dest="redact_phones", action="store_false", default=True,
                        help="Skip phone/fax number redaction (on by default). Fax numbers use the "
                        "same digit pattern as phone numbers, so there is no separate fax flag.")
    parser.add_argument("--emails_file",
                        default=None,
                        help="Optional path to a text file of exact email addresses (one per line) to "
                        "redact. If given, ONLY these are redacted - not a blanket regex scan. Without "
                        "it, every email-looking string is redacted (default).")
    parser.add_argument("--phones_file",
                        default=None,
                        help="Optional path to a text file of exact phone/fax numbers (one per line) to "
                        "redact. If given, ONLY these are redacted - not a blanket regex scan. Without "
                        "it, every phone-looking string is redacted (default).")

    args = parser.parse_args()
    full_page_pages = parse_page_spec(args.full_page_redact)

    # Basic input validation for page numbers
    if args.analysis_start_page < 1:
        print("Error: analysis_start_page must be 1 or greater.", file=sys.stderr)
        sys.exit(1)
    if args.analysis_end_page < args.analysis_start_page:
        print("Error: analysis_end_page must be greater than or equal to analysis_start_page.", file=sys.stderr)
        sys.exit(1)

    input_path = args.input_file
    output_path = args.output_file

    if os.path.isdir(input_path):
        # Directory mode
        batch_redact_tree(
            input_path,
            output_path,
            args.names_file,
            args.boilerplate_graphics,
            args.boilerplate_text,
            args.analysis_start_page,
            args.analysis_end_page,
            args.exceptions_file,
            full_page_redact_pages=full_page_pages,
            redact_names=args.redact_names,
            redact_emails=args.redact_emails,
            redact_phones=args.redact_phones,
            emails_file=args.emails_file,
            phones_file=args.phones_file,
        )
    else:
        # Single-file mode (backwards compatible)
        redact_pdf(
            input_path,
            output_path,
            args.names_file,
            args.boilerplate_graphics,
            args.boilerplate_text,
            args.analysis_start_page,
            args.analysis_end_page,
            args.exceptions_file,
            full_page_redact_pages=full_page_pages,
            redact_names=args.redact_names,
            redact_emails=args.redact_emails,
            redact_phones=args.redact_phones,
            emails_file=args.emails_file,
            phones_file=args.phones_file,
        )


if __name__ == "__main__":
    main()