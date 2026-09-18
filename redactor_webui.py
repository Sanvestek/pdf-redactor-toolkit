#;encoding=utf-8
# Rough local web UI for the redaction pipeline: extract -> review/curate -> redact -> verify.
#
# Design intent: this is a thin UI layer. Every action below calls the *actual* functions from
# extract_names.py / redact_pymupdf_fast.py directly (no re-implemented logic, no shelling out to
# argv strings) so that tweaks to the underlying extraction/redaction behaviour are picked up here
# automatically. Only a change to those functions' signatures would require an update in this file.
#
# Single local user, one PDF job at a time - run() is blocking per request, no auth, no concurrency
# control. Not meant to be exposed beyond localhost.
#
# Review/curate covers all three candidate kinds (name/email/phone), not just names: emails and
# phones are found with the exact same regexes redact_pymupdf_fast.py uses at redaction time
# (imported from it directly, not re-implemented). Checked emails/phones are saved to their own
# exact-value files (emails_to_redact.txt / phones_to_redact.txt), which redact_pymupdf_fast.py now
# treats as authoritative when present - matched exactly like names, not by a blanket regex scan.
# Redact's category checkboxes only control whether that category is attempted at all; whether it
# runs in curated (exact list) or blanket (regex) mode depends solely on whether an emails_file/
# phones_file was actually supplied.

import contextlib
import io
import os
import re
import sys

from flask import Flask, abort, jsonify, redirect, request, send_file, url_for
from jinja2 import Template
from werkzeug.utils import secure_filename

import redact_pymupdf_fast as redactor

try:
    import extract_names as extractor
    EXTRACTION_AVAILABLE = True
    EXTRACTION_ERROR = None
except Exception as e:  # e.g. spaCy not compatible with the running Python version
    extractor = None
    EXTRACTION_AVAILABLE = False
    EXTRACTION_ERROR = f"{type(e).__name__}: {e}"

# When bundled by PyInstaller, __file__/cwd tricks that work in dev can point into the frozen
# app's internal folder instead of the folder the user actually launched it from. Pin both the
# static asset lookup and the working directory to where the executable itself lives, so bundled
# assets (static/, common_english_words.txt) and user-writable defaults (names_to_redact.txt,
# uploads/) resolve the same way whether running from source or from the packaged app.
if getattr(sys, "frozen", False):
    _APP_DIR = os.path.dirname(sys.executable)
    _ASSET_DIR = getattr(sys, "_MEIPASS", _APP_DIR)
    os.chdir(_APP_DIR)
else:
    _APP_DIR = os.path.dirname(os.path.abspath(__file__))
    _ASSET_DIR = _APP_DIR

app = Flask(__name__, static_folder=os.path.join(_ASSET_DIR, "static"))

SERVER_START_DIR = os.getcwd()  # "Repo" shortcut in the folder browser - captured once at
                                 # startup, since CWD doesn't change over the server's lifetime.
UPLOAD_DIR = os.path.join(SERVER_START_DIR, "uploads")  # where "upload from my computer" lands -
                                                         # gitignored; a real local path the rest
                                                         # of the pipeline can then read normally.

KEY_SEP = "\x1f"  # joins (kind, value) into one form field value; never appears in real text

# Confidence scoring: a plain additive point system, not a black-box score - every rule below is
# independently readable, and the numbers are editable live in the UI (Review section), not
# buried constants. See compute_confidence() for how these combine.
DEFAULT_CONFIDENCE_WEIGHTS = {
    "table_bonus": 4,        # found in a table's "Name" column - structural certainty, not a guess
    "email_bonus": 4,        # email regex is inherently precise
    "phone_shape_bonus": 3,  # digit count + punctuation look like a real phone, not a grant ref/ISBN
    "multiword_bonus": 2,    # "Adrian Fenwick" is far less likely to be a false positive than "Adrian"
    "repeat_bonus_cap": 3,   # +1 per repeat occurrence in the document, capped here
    "ambiguous_penalty": 3,  # single word that's ALSO a common English word (the "Scott" case)
    "single_ner_penalty": 1, # single word guessed by spaCy alone, no table/repeat backing it up
    "high_threshold": 4,
    "medium_threshold": 1,
}

state = {
    "candidates": [],                         # [{"kind","value","context","source","count"}, ...]
    "names_file_path": "names_to_redact.txt",
    "emails_file_path": "emails_to_redact.txt",
    "phones_file_path": "phones_to_redact.txt",
    "exceptions_file_path": "name_exceptions.txt",
    "verify_items": [],
    "last_input_path": "",                    # most recent Extract/Redact source, for form defaults
    "last_output_path": "",
    "search_results": {},                     # {name: [{"file","text_matches","metadata_matches"}, ...]}
    "flash": None,                            # {"text","level"} - one-shot banner, shown once then cleared
    "last_saved_candidates": [],              # snapshot of candidates as of the last successful Save - the
                                               # "undo" baseline and the unsaved-changes comparison point
    "confidence_weights": dict(DEFAULT_CONFIDENCE_WEIGHTS),
    "log": "",
}


def item_key(item):
    return f'{item["kind"]}{KEY_SEP}{item["value"]}'


def set_flash(text, level="success"):
    """One-shot confirmation banner - set here, read and cleared by index() on the next render, so
    every action (save, add, search, redact...) can tell the user in plain language what actually
    happened, not just leave them to infer it from a page reload."""
    state["flash"] = {"text": text, "level": level}


def derive_sibling_paths(names_path):
    """Given a names-file path, derives matching emails/phones paths using the same naming
    convention, server-side - so this applies the moment Extract runs, not only when someone
    happens to type live into Review's "Save names to" field (a page loaded with a value already
    filled in from a previous run never fires that kind of live-typing event at all).

    If the path contains the word "name" (the common case - "gnames_to_redact.txt", the default
    "names_to_redact.txt"), swaps it in place. Otherwise (a custom filename with no literal "name"
    in it, e.g. "grah2_to_redact.txt") falls back to inserting "_emails"/"_phones" before the
    extension, so the three files are still distinct and still share the one prefix either way.
    """
    if re.search(r'name', names_path, re.IGNORECASE):
        def swap(replacement):
            def repl(m):
                return replacement.capitalize() if m.group(0)[0].isupper() else replacement
            return re.sub(r'name', repl, names_path, flags=re.IGNORECASE)
        return swap("email"), swap("phone")

    base, ext = os.path.splitext(names_path)
    return f"{base}_emails{ext}", f"{base}_phones{ext}"


def is_ambiguous(item):
    """A standalone single-word name that's also a common English word (e.g. "Hale" alone) -
    structurally identical to a real single-word name, so it needs a human's judgement call, not
    an algorithm's. Multi-word names ("Michael Hale") are never ambiguous by this definition."""
    return (
        EXTRACTION_AVAILABLE
        and item["kind"] == "name"
        and len(item["value"].split()) == 1
        and item["value"].lower() in extractor.COMMON_ENGLISH_WORDS
    )


def looks_like_a_phone(value):
    """True if a matched phone-regex value is structurally phone-shaped, as opposed to a grant
    reference, ISBN, or similar digit-string that the phone regex also happens to match (a known,
    real false-positive source - see PHONE_REGEX's own history in this project). A reference code
    shaped like "2021-1778" (a recent year, then a dash, then a short number) is a strong negative
    signal - phone numbers don't start with what looks like a calendar year. Otherwise: a
    plausible total digit count (7-15, the real-world range) plus either a leading "+" (country
    code) or normal phone punctuation (spaces/dashes/dots/parens) counts as phone-shaped."""
    v = value.strip()
    if re.fullmatch(r'(19|20)\d{2}-\d+', v):
        return False
    digits = re.sub(r'\D', '', v)
    if not (7 <= len(digits) <= 15):
        return False
    if v.startswith('+'):
        return True
    if re.search(r'[-. ()]', v):
        return True
    return len(digits) >= 9


def compute_confidence(item, weights):
    """Plain additive scoring - see DEFAULT_CONFIDENCE_WEIGHTS for what each term means. A
    manually-added item always scores High: it doesn't need heuristics, you already vouched for it."""
    if item.get("source") == "manual" or item.get("confirmed"):
        return "High"

    word_count = len(item["value"].split())
    score = 0
    if item.get("source") == "table":
        score += weights["table_bonus"]
    if item["kind"] == "email":
        score += weights["email_bonus"]
    if item["kind"] == "phone" and looks_like_a_phone(item["value"]):
        score += weights["phone_shape_bonus"]
    if item["kind"] == "name" and word_count >= 2:
        score += weights["multiword_bonus"]
    score += min(max(item.get("count", 1) - 1, 0), weights["repeat_bonus_cap"])
    if is_ambiguous(item):
        score -= weights["ambiguous_penalty"]
    if item["kind"] == "name" and word_count == 1 and item.get("source") == "ner":
        score -= weights["single_ner_penalty"]

    if score >= weights["high_threshold"]:
        return "High"
    if score >= weights["medium_threshold"]:
        return "Medium"
    return "Low"


def annotate(items, weights):
    """Returns a copy of items with 'confidence' and 'ambiguous' computed fresh against the
    current weights, without mutating the stored candidates - so changing the weights in the UI
    re-labels the existing list immediately without needing to re-extract."""
    out = []
    for it in items:
        copy = dict(it)
        copy["confidence"] = compute_confidence(it, weights)
        copy["ambiguous"] = is_ambiguous(it)
        out.append(copy)
    return out


def run_captured(fn):
    """Runs fn(), capturing anything it prints to stdout/stderr into state['log']."""
    buf = io.StringIO()
    ok = True
    result = None
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            result = fn()
        except SystemExit as e:
            # A stray sys.exit() somewhere in the underlying library code (written for CLI use,
            # not import) would otherwise propagate past `except Exception` uncaught and silently
            # kill this request's thread - empty response, no flash, no log line, nothing. Treat it
            # as any other failure instead.
            buf.write(f"\nERROR: underlying code called sys.exit({e.code!r}) instead of raising.\n")
            ok = False
        except Exception as e:
            buf.write(f"\nERROR: {type(e).__name__}: {e}\n")
            ok = False
    state["log"] = buf.getvalue()
    return result, ok


def extract_names_with_context(text, exceptions=None):
    """Same filtering as extractor.extract_names_from_text, plus an example sentence per name."""
    final_names = extractor.extract_names_from_text(text, exceptions=exceptions)
    context = {}
    if text.strip():
        doc = extractor.nlp(text)
        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue
            sentence = ent.sent.text.strip().replace("\n", " ")
            candidates = [ent.text.strip()]
            candidates += [c.strip() for c in re.split(r'\s|(?<=\w)\.(?=\s*\w)', ent.text.strip())]
            for cand in candidates:
                if cand in final_names and cand not in context:
                    context[cand] = sentence
    return final_names, context


def find_regex_matches_with_context(text, regex, window=40):
    """Yields (matched_string, surrounding_text) for every match of regex in text."""
    results = []
    for m in regex.finditer(text):
        start = max(0, m.start() - window)
        end = min(len(text), m.end() + window)
        context = " ".join(text[start:end].split())
        results.append((m.group(0), context))
    return results


def run_extraction(input_path, seed_items=None, do_names=True, do_emails=True, do_phones=True,
                    exceptions=None):
    """Scans one PDF or a directory of PDFs for names (spaCy) and/or emails/phones (regex).

    Returns a deduplicated, sorted list of {"kind", "value", "context", "source", "count"} dicts.
    "source" and "count" feed the confidence score (see compute_confidence): source is one of
    "table" (structural column match - most trustworthy), "ner" (spaCy guess), "regex" (email/
    phone pattern match), or "seed"/"manual" (you already told it this is real). "count" is how
    many times that exact value was actually seen across the scanned document(s).

    `exceptions`, if given, is applied during extraction itself - a phrase already excluded from
    redaction never becomes a review candidate in the first place, not just gets silently
    protected later. Only applies to names; emails/phones are unaffected.
    """
    by_key = {}  # (kind, value) -> {"context","source","count","confirmed"}

    def record(kind, value, context, source, occurrences=1, confirmed=False):
        key = (kind, value)
        entry = by_key.get(key)
        if entry is None:
            by_key[key] = {"context": context, "source": source, "count": occurrences,
                            "confirmed": confirmed}
        else:
            entry["count"] += occurrences
            if source == "table":
                entry["source"] = "table"  # the more trustworthy source wins if found both ways
            # Once confirmed (e.g. a merged-in name from an already-saved file), stays confirmed
            # even if this same value also turns up again via a fresh, unconfirmed NER/regex hit -
            # re-extracting must never silently downgrade something you already vouched for.
            entry["confirmed"] = entry["confirmed"] or confirmed

    for it in (seed_items or []):
        record(it["kind"], it["value"], it.get("context", ""), it.get("source", "seed"),
               it.get("count", 1), confirmed=it.get("confirmed", False))

    count = 0
    for pdf_path in extractor.collect_pdfs(input_path):
        count += 1
        print(f"Reading {pdf_path} ...")
        text = extractor.extract_text(pdf_path)
        if not text.strip():
            print(f"  Warning: no text extracted from {pdf_path}")

        found_this_pdf = 0
        if do_names:
            if text.strip():
                names, ctx = extract_names_with_context(text, exceptions=exceptions)
                for n in names:
                    record("name", n, ctx.get(n, ""), "ner", occurrences=max(1, text.count(n)))
                found_this_pdf += len(names)
            # Table-column and inline "Label: Value" names bypass spaCy/regex entirely (a
            # structural signal, not a text guess), so they're checked regardless of whether
            # ordinary text extraction succeeded.
            table_names = extractor.extract_names_from_tables(pdf_path, exceptions=exceptions)
            label_names = extractor.extract_names_from_labels(pdf_path, exceptions=exceptions)
            for n, ctx_val in table_names.items():
                record("name", n, ctx_val, "table")
            for n, ctx_val in label_names.items():
                record("name", n, ctx_val, "table")
            found_this_pdf += len(table_names) + len(label_names)
        if not text.strip():
            print(f"  +{found_this_pdf} candidate(s) (table columns only, no page text)")
            continue
        if do_emails:
            email_matches = find_regex_matches_with_context(text, redactor.EMAIL_REGEX)
            for value, ctx in email_matches:
                record("email", value, ctx, "regex")
            found_this_pdf += len(email_matches)
        if do_phones:
            phone_matches = find_regex_matches_with_context(text, redactor.PHONE_REGEX)
            for value, ctx in phone_matches:
                record("phone", value, ctx, "regex")
            found_this_pdf += len(phone_matches)
        print(f"  +{found_this_pdf} candidate(s)")

    if count == 0:
        raise RuntimeError(f"No PDF files found at {input_path}")

    items = [
        {"kind": k, "value": v, "context": meta["context"], "source": meta["source"],
         "count": meta["count"], "confirmed": meta["confirmed"]}
        for (k, v), meta in by_key.items()
    ]
    items.sort(key=lambda it: (it["kind"], it["value"].lower()))
    print(f"\nScanned {count} PDF(s). {len(items)} unique candidate(s).")
    return items


TEMPLATE = Template("""
<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PDF Redactor</title>
<style>
  :root {
    --bg: #f4f6f9; --surface: #ffffff; --border: #e1e5ea; --text: #1c2430; --text-dim: #5b6572;
    --accent: #2f6fed; --accent-dark: #1f52c2; --accent-bg: #eaf0fe;
    --success: #1a7f37; --success-bg: #e7f7ed; --success-border: #a9dfbb;
    --warn: #8a6300; --warn-bg: #fff6dd; --warn-border: #f0d78c;
    --error: #b3261e; --error-bg: #fdecec; --error-border: #f3b7b2;
    --radius: 10px; --radius-sm: 6px;
    --shadow: 0 1px 3px rgba(20,30,50,0.06), 0 4px 14px rgba(20,30,50,0.06);
    --shadow-lg: 0 8px 30px rgba(20,30,50,0.18);
    font-synthesis: none;
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    max-width: 960px; margin: 0 auto; padding: 0 1.25em 3em; color: var(--text); background: var(--bg);
    line-height: 1.45;
  }
  h1 { font-size: 1.5em; font-weight: 700; margin: 0.9em 0 0.15em; letter-spacing: -0.01em; }
  .subtitle { color: var(--text-dim); font-size: 0.92em; margin: 0 0 1em; }
  section {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 1.3em 1.6em; margin-bottom: 1.4em; box-shadow: var(--shadow);
  }
  h2 { margin-top: 0; font-size: 1.15em; font-weight: 650; color: var(--text); }
  h2::before { content: ""; }
  nav {
    position: sticky; top: 0.7em; z-index: 150; display: flex; flex-wrap: wrap; gap: 0.4em;
    background: var(--surface); border: 1px solid var(--border); border-radius: 999px;
    padding: 0.4em; margin: 0.6em 0 1.4em; box-shadow: var(--shadow);
  }
  nav a {
    color: var(--text-dim); text-decoration: none; font-size: 0.88em; font-weight: 600;
    padding: 0.45em 0.9em; border-radius: 999px; transition: background 0.15s, color 0.15s;
  }
  nav a:hover { background: var(--accent-bg); color: var(--accent-dark); }
  label { display: inline-block; min-width: 140px; color: var(--text-dim); font-size: 0.92em; }
  input[type=text], input[type=number], textarea {
    width: 400px; font: inherit; font-size: 0.92em; color: var(--text);
    border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 0.45em 0.6em;
    background: #fbfcfe; transition: border-color 0.15s, box-shadow 0.15s;
  }
  input[type=text]:focus, input[type=number]:focus, textarea:focus {
    outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-bg);
  }
  input[type=checkbox] { accent-color: var(--accent); }
  .row { margin-bottom: 0.6em; }
  table { width: 100%; border-collapse: collapse; }
  td, th { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); font-size: 0.9em; }
  th { color: var(--text-dim); font-weight: 650; font-size: 0.82em; text-transform: uppercase; letter-spacing: 0.03em; }
  tbody tr:hover { background: #f8fafc; }
  td.ctx { color: var(--text-dim); }
  td.kind { text-transform: capitalize; color: var(--text-dim); }
  td.conf-high { color: var(--success); font-weight: 700; }
  td.conf-medium { color: var(--warn); font-weight: 700; }
  td.conf-low { color: var(--error); font-weight: 700; }
  h3 { margin: 1em 0 0.4em 0; font-size: 1em; font-weight: 650; }
  pre {
    background: #0f1420; color: #d7dce4; padding: 1em; overflow-x: auto; max-height: 300px;
    border-radius: var(--radius-sm); font-size: 0.85em;
  }
  .warn {
    background: var(--warn-bg); border: 1px solid var(--warn-border); color: var(--warn);
    padding: 0.85em 1.1em; border-radius: var(--radius-sm); margin-bottom: 1.2em;
  }
  .items-table { max-height: 350px; overflow-y: auto; display: block; }
  button {
    font: inherit; padding: 0.5em 1.1em; border-radius: var(--radius-sm); border: 1px solid var(--border);
    background: var(--surface); color: var(--text); cursor: pointer; font-weight: 600; font-size: 0.9em;
    transition: background 0.15s, border-color 0.15s, transform 0.05s;
  }
  button:hover { border-color: var(--accent); color: var(--accent-dark); }
  button:active { transform: translateY(1px); }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  button[type=submit] { background: var(--accent); border-color: var(--accent); color: #fff; }
  button[type=submit]:hover { background: var(--accent-dark); border-color: var(--accent-dark); color: #fff; }
  .path-row { display: flex; gap: 0.4em; align-items: center; }
  .browse-btn { padding: 0.4em 0.7em; }
  .filter-box { width: 300px; margin-bottom: 0.5em; }
  #browse-panel {
    display: none; position: fixed; top: 8%; left: 50%; transform: translateX(-50%);
    width: 500px; max-height: 70vh; overflow-y: auto; background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); box-shadow: var(--shadow-lg); padding: 1.2em; z-index: 1000;
  }
  #browse-panel .path-display { font-family: ui-monospace, monospace; font-size: 0.85em; word-break: break-all;
    background: var(--bg); padding: 0.5em 0.6em; border-radius: var(--radius-sm); margin-bottom: 0.6em; }
  #browse-panel ul { list-style: none; padding: 0; margin: 0.5em 0 0.75em 0; max-height: 40vh; overflow-y: auto; }
  #browse-panel li { padding: 5px 6px; cursor: pointer; border-radius: var(--radius-sm); }
  #browse-panel li:hover { background: var(--accent-bg); }
  .browse-shortcuts { display: flex; gap: 0.4em; margin-bottom: 0.6em; }
  .crumb { cursor: pointer; text-decoration: none; color: var(--accent); font-weight: 600; }
  .crumb:hover { color: var(--accent-dark); text-decoration: underline; }
  .crumb-sep { color: var(--text-dim); }
  #browse-overlay { display: none; position: fixed; inset: 0; background: rgba(15,20,32,0.35); z-index: 999; }
  #loading-overlay { display: none; position: fixed; inset: 0; background: rgba(244,246,249,0.9);
    z-index: 1500; flex-direction: column; align-items: center; justify-content: center; gap: 1em; }
  .spinner { width: 46px; height: 46px; border: 5px solid var(--border); border-top-color: var(--accent);
    border-radius: 50%; animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  #loading-message { font-weight: 650; color: var(--text); max-width: 320px; text-align: center; }
  .flash-stack { position: sticky; top: 0; z-index: 200; padding-top: 0.8em; }
  .flash { margin-bottom: 0.5em; padding: 0.75em 1.1em;
    border-radius: var(--radius-sm); font-weight: 600; box-shadow: var(--shadow); font-size: 0.92em; }
  .flash-success { background: var(--success-bg); color: var(--success); border: 1px solid var(--success-border); }
  .flash-warn { background: var(--warn-bg); color: var(--warn); border: 1px solid var(--warn-border); }
  .flash-error { background: var(--error-bg); color: var(--error); border: 1px solid var(--error-border); }
  .flash form { display: inline; margin-left: 0.6em; }
  .flash button.link-btn { padding: 0.2em 0.7em; font-size: 0.85em; background: rgba(255,255,255,0.6); }
  .flash a { color: inherit; margin-left: 0.6em; }
  .flash { position: relative; padding-right: 2.4em; }
  .dismiss-btn { position: absolute; top: 0.6em; right: 0.7em; background: none; border: none;
    font-size: 1em; cursor: pointer; color: inherit; opacity: 0.55; padding: 0.2em 0.4em; }
  .dismiss-btn:hover { opacity: 1; }

  /* --- Section 5: interactive click-to-redact viewer --- */
  #iact-pages { border: 1px solid var(--border); border-radius: var(--radius-sm); background: #4b5563; max-height: 75vh; overflow: auto;
    padding: 12px; margin: 0.8em 0; }
  .iact-page-wrap { position: relative; margin: 0 auto 12px auto; background: #fff;
    box-shadow: 0 2px 10px rgba(0,0,0,0.35); width: max-content; }
  .iact-page-wrap canvas { display: block; }
  .iact-textlayer { position: absolute; inset: 0; overflow: hidden; opacity: 0.25;
    line-height: 1; text-align: initial; transform-origin: 0 0;
    -webkit-text-size-adjust: none; text-size-adjust: none; forced-color-adjust: none; }
  .iact-textlayer span, .iact-textlayer br { position: absolute; white-space: pre; cursor: text;
    transform-origin: 0 0; color: transparent; }
  .iact-textlayer ::selection { background: rgba(47,111,237,0.35); }
  .iact-textlayer .endOfContent { display: block; position: absolute; inset: 100% 0 0;
    z-index: -1; cursor: default; user-select: none; }
  .iact-hl-layer { position: absolute; inset: 0; pointer-events: none; }
  .iact-hl { position: absolute; pointer-events: auto; cursor: pointer;
    box-sizing: border-box; }
  .iact-hl.occurrence { background: rgba(220,0,0,0.35); border: 1px solid rgba(150,0,0,0.9); }
  .iact-hl.global { background: rgba(255,210,0,0.40); border: 1px solid rgba(180,140,0,0.9); }
  .iact-hl.preview { background: rgba(255,210,0,0.22); border: 1px dashed rgba(180,140,0,0.7); }
  #iact-popup { position: fixed; z-index: 1200; display: none; background: var(--surface);
    border: 1px solid var(--border); border-radius: var(--radius-sm); box-shadow: var(--shadow-lg);
    padding: 0.7em 0.8em; max-width: 320px; font-size: 0.85em; }
  #iact-popup .sel-text { font-family: ui-monospace, monospace; background: var(--bg); padding: 0.35em 0.45em;
    border-radius: var(--radius-sm); word-break: break-all; margin-bottom: 0.5em; max-height: 4em; overflow: auto; }
  #iact-popup button { display: block; width: 100%; margin: 0.3em 0; text-align: left; }
  .iact-selrow { display: flex; align-items: center; gap: 0.5em; padding: 4px 0;
    border-bottom: 1px solid var(--border); font-size: 0.88em; }
  .iact-badge { font-size: 0.72em; font-weight: 700; padding: 2px 7px; border-radius: 999px; color: #fff; }
  .iact-badge.occurrence { background: var(--error); }
  .iact-badge.global { background: #b8860b; }
  .iact-selrow .val { flex: 1; font-family: ui-monospace, monospace; word-break: break-all; }
</style></head>
<body>
<div class="flash-stack">
{% if flash %}
<div class="flash flash-{{ flash.level }}">{{ flash.text }}
  <button type="button" class="dismiss-btn" onclick="this.closest('.flash').remove()" title="Dismiss" aria-label="Dismiss">✕</button>
</div>
{% endif %}
<div class="flash flash-warn" id="unsaved-changes-banner" hidden>
  ⚠ <strong>Unsaved changes:</strong> <span id="unsaved-changes-summary"></span> if you saved now -
  the curated list files on disk (used by Redact) don't reflect this yet.
  <button type="button" class="link-btn" onclick="document.getElementById('save-curated-btn').scrollIntoView({behavior:'smooth', block:'center'}); document.getElementById('save-curated-btn').focus();">Go save them</button>
  <button type="button" class="link-btn" onclick="undoUnsavedChanges()">Undo (reset checkboxes to last saved)</button>
</div>
</div>
<h1>PDF Redactor</h1>
<p class="subtitle">Redact names, emails, phone numbers, and boilerplate from PDFs - locally, in your browser.</p>
<nav><a href="#extract">1. Extract</a><a href="#review">2. Review &amp; Curate</a>
<a href="#redact">3. Redact</a><a href="#verify">4. Verify</a>
<a href="#interactive">5. Interactive redaction</a><a href="#log">Log</a></nav>

{% if not EXTRACTION_AVAILABLE %}
<div class="warn">Extraction is unavailable in this Python environment
({{ EXTRACTION_ERROR }}). This usually means spaCy doesn't support the running
Python version - see the README. Redaction (section 3) still works fully; you
can load a names file by hand there.</div>
{% endif %}

<section id="extract">
<h2>1. Extract candidates</h2>
<form method="post" action="/extract">
  <div class="row"><label>PDF file or folder</label><div class="path-row">
    <input type="text" id="extract_input_path" name="input_path" value="{{ state.last_input_path }}" required>
    <button type="button" class="browse-btn" onclick="openBrowse('extract_input_path')">📁</button></div></div>
  <div class="row"><label>Save names to</label><div class="path-row">
    <input type="text" id="extract_output_path" name="output_path" value="{{ state.names_file_path }}">
    <button type="button" class="browse-btn" onclick="openBrowse('extract_output_path')">📁</button></div></div>
  <div class="row"><label></label><input type="checkbox" name="merge" id="merge"> <label for="merge" style="min-width:auto">Merge into existing file instead of overwriting</label></div>
  <div class="row"><label>Exceptions file</label><div class="path-row">
    <input type="text" id="extract_exceptions_file" name="exceptions_file" value="{{ state.exceptions_file_path }}">
    <button type="button" class="browse-btn" onclick="openBrowse('extract_exceptions_file')">📁</button></div>
    (optional - a phrase in here never becomes a candidate to review, not just protected later at redaction time)</div>
  <div class="row"><label>Look for</label>
    <input type="checkbox" name="extract_names" checked> Names
    <input type="checkbox" name="extract_emails" checked> Emails
    <input type="checkbox" name="extract_phones" checked> Phone / fax numbers
    (unchecking a category skips scanning for it entirely, saving time)</div>
  <button type="submit" {{ "" if EXTRACTION_AVAILABLE else "disabled" }}>Run extraction</button>
</form>
</section>

{% macro name_search_tool(default_path, return_to) %}
<div style="margin-top:1.5em; border-top:1px solid #ddd; padding-top:1em">
<h3>Search for a specific name (text + metadata)</h3>
<p style="color:#555;font-size:0.85em">A targeted, manual check - separate from the automated extraction above. Searches the raw
document text AND metadata (Title/Author/XMP/embedded file names) for any occurrence of the name(s) below, tolerant of
spacing: "John Smith", "JohnSmith" (no space - a real observed PDF text-extraction artifact), and a name split across a
line break in the extracted text all count as a match. Doesn't use spaCy, so it works even when extraction is unavailable
- good for checking an already-redacted output for anything that slipped through, or double-checking one specific name
by hand.</p>
<form method="post" action="/review/search_document">
  <input type="hidden" name="return_to" value="{{ return_to }}">
  <div class="row"><label>Search in</label><div class="path-row">
    <input type="text" id="search_doc_path_{{ return_to }}" name="search_path" value="{{ default_path }}" required>
    <button type="button" class="browse-btn" onclick="openBrowse('search_doc_path_{{ return_to }}')">📁</button></div></div>
  <label style="vertical-align:top">Names to search for</label>
  <textarea name="search_names" rows="3" style="width:400px" placeholder="One per line&#10;Jane Doe&#10;John Smith" required></textarea>
  <button type="submit">Search</button>
</form>
{% if state.search_results %}
<form method="post" action="/review/search_add" style="margin-top:1em">
  <input type="hidden" name="return_to" value="{{ return_to }}">
  <table class="curation-table" id="search-results-table-{{ return_to }}">
  <tr><th><input type="checkbox" class="select-all" title="Select/deselect all"></th><th>Name searched</th><th>Status</th><th>Where found</th></tr>
  {% for name, matches in state.search_results.items() %}
  <tr>
    <td><input type="checkbox" name="add_names" value="{{ name }}" {{ "checked" if matches else "" }}></td>
    <td>{{ name }}</td>
    {% if matches %}
    <td class="conf-low" style="color:#cf222e" data-sort="1">⚠ found ({{ matches|length }} file(s))</td>
    <td>
      {% for m in matches %}
      <div style="margin-bottom:0.3em"><strong>{{ m.file }}</strong>
        {% for tm in m.text_matches %}
          <br><span style="color:#555">page {{ tm.page }}, matched "{{ tm.matched_text }}": {{ tm.context }}</span>
        {% endfor %}
        {% for mm in m.metadata_matches %}
          <br><span style="color:#555">{{ mm.field }}, matched "{{ mm.matched_text }}": {{ mm.context }}</span>
        {% endfor %}
      </div>
      {% endfor %}
    </td>
    {% else %}
    <td style="color:#1a7f37" data-sort="0">✓ not found</td>
    <td>-</td>
    {% endif %}
  </tr>
  {% endfor %}
  </table>
  <p style="color:#555;font-size:0.85em">Names found are checked by default. Add the checked names to the curated
  list below so Redact will actually remove them - a search hit on its own doesn't change what gets redacted until
  you do this (or the equivalent add in Review).</p>
  <button type="submit">Add selected names to curated list</button>
</form>
{% endif %}
</div>
{% endmacro %}

<section id="review">
<h2>2. Review &amp; curate ({{ (high_items + medium_items + low_items)|length }} candidates)</h2>
<p style="color:#555">Nothing here is written to disk until you click "Save curated list" below -
extraction results only live in this page's memory until then, and are lost if the server restarts.
Checked items are saved to their type's file below; unchecked ones are simply left out. In Redact,
these three files are then matched exactly (like names always have been) instead of a blanket regex
scan - so what you check here is exactly what gets redacted, nothing more.</p>

<details style="margin-bottom:1em">
<summary style="cursor:pointer">Confidence scoring settings (click to adjust)</summary>
<form method="post" action="/settings/confidence" class="row" style="margin-top:0.5em">
  <div class="row"><label title="Points for a table-column match">Table bonus</label>
    <input type="number" name="table_bonus" value="{{ weights.table_bonus }}" style="width:60px"></div>
  <div class="row"><label title="Points for an email regex match">Email bonus</label>
    <input type="number" name="email_bonus" value="{{ weights.email_bonus }}" style="width:60px"></div>
  <div class="row"><label title="Points for a phone number that's digit-count and punctuation plausible (not a grant ref/ISBN)">Phone-shape bonus</label>
    <input type="number" name="phone_shape_bonus" value="{{ weights.phone_shape_bonus }}" style="width:60px"></div>
  <div class="row"><label title="Points for a 2+ word name">Multi-word bonus</label>
    <input type="number" name="multiword_bonus" value="{{ weights.multiword_bonus }}" style="width:60px"></div>
  <div class="row"><label title="Max points from repeat occurrences">Repeat bonus (cap)</label>
    <input type="number" name="repeat_bonus_cap" value="{{ weights.repeat_bonus_cap }}" style="width:60px"></div>
  <div class="row"><label title="Points subtracted for a single word that's also a common English word">Ambiguous penalty</label>
    <input type="number" name="ambiguous_penalty" value="{{ weights.ambiguous_penalty }}" style="width:60px"></div>
  <div class="row"><label title="Points subtracted for an unbacked single-word spaCy guess">Single-guess penalty</label>
    <input type="number" name="single_ner_penalty" value="{{ weights.single_ner_penalty }}" style="width:60px"></div>
  <div class="row"><label title="Score at/above this = High">High threshold</label>
    <input type="number" name="high_threshold" value="{{ weights.high_threshold }}" style="width:60px"></div>
  <div class="row"><label title="Score at/above this = Medium; below = Low">Medium threshold</label>
    <input type="number" name="medium_threshold" value="{{ weights.medium_threshold }}" style="width:60px"></div>
  <button type="submit">Apply (re-labels the list below immediately)</button>
</form>
</details>

<form method="post" action="/review/save">
  <div class="row"><label>Save names to</label><div class="path-row">
    <input type="text" id="review_names_path" name="names_path" value="{{ state.names_file_path }}" oninput="deriveSiblingPaths()">
    <button type="button" class="browse-btn" onclick="openBrowse('review_names_path')">📁</button></div></div>
  <div class="row"><label>Save emails to</label><div class="path-row">
    <input type="text" id="review_emails_path" name="emails_path" value="{{ state.emails_file_path }}">
    <button type="button" class="browse-btn" onclick="openBrowse('review_emails_path')">📁</button></div></div>
  <div class="row"><label>Save phones to</label><div class="path-row">
    <input type="text" id="review_phones_path" name="phones_path" value="{{ state.phones_file_path }}">
    <button type="button" class="browse-btn" onclick="openBrowse('review_phones_path')">📁</button></div></div>
  <p style="color:#555;font-size:0.85em">Typing in "Save names to" auto-fills the other two by swapping "name" for "email"/"phone"
  in whatever you typed (e.g. "graam_l_names_to_redact.txt" -&gt; "graam_l_emails_to_redact.txt") - edit either field afterward
  if you want a different filename. These same three paths carry over as the defaults in Redact below, and stay set once saved.</p>

  <input type="text" class="filter-box" placeholder="Search across ALL tiers below..." data-filters-all="curation-table" style="width:100%">

  {% for tier, items, default_checked in [
      ("High", high_items, true),
      ("Medium", medium_items, false),
      ("Low", low_items, false),
  ] %}
  {% for kind_label, kind_key in [("Names", "name"), ("Emails", "email"), ("Phones / fax", "phone")] %}
  {% set kind_items = items|selectattr("kind", "equalto", kind_key)|list %}
  {% if kind_items %}
  {% set table_id = tier|lower ~ "-" ~ kind_key ~ "-table" %}
  <h3 class="conf-{{ tier|lower }}">{{ tier }} confidence - {{ kind_label }} ({{ kind_items|length }}){% if tier != "High" %}
    - not pre-checked, review before including{% endif %}</h3>
  <input type="text" class="filter-box" placeholder="Filter..." data-filters="{{ table_id }}">
  <div class="items-table">
  <table class="curation-table" id="{{ table_id }}">
  <tr><th><input type="checkbox" class="select-all"{{ " checked" if default_checked else "" }}></th>
      <th>Value</th><th>Confidence</th><th>Found in context</th></tr>
  {% for item in kind_items %}
  <tr><td><input type="checkbox" name="keep" value="{{ item.kind }}{{ KEY_SEP }}{{ item.value }}" data-was-saved="{{ '1' if item_key(item) in last_saved_keys else '0' }}"{{ " checked" if default_checked else "" }}></td>
      <td>{{ item.value }}{% if item.ambiguous %} <span title="Also a common English word - double-check">⚠</span>{% endif %}</td>
      <td class="conf-{{ item.confidence|lower }}">{{ item.confidence }}</td>
      <td class="ctx">{{ item.context }}</td></tr>
  {% endfor %}
  </table>
  </div>
  {% endif %}
  {% endfor %}
  {% endfor %}
  <p style="color:#555;font-size:0.85em">Tip: click a column header to sort by it (click again to reverse). Click a header
  checkbox to select/deselect all in that table. Shift-click a row's checkbox to set every row between it and your last
  click (within the same table) to the same state. Checking and saving a Medium/Low item marks it confirmed - it shows
  as High from then on. The search box at the top searches every table on this page at once, in case what you're
  looking for is sitting in a different tier than you expected.</p>
  <button type="submit" id="save-curated-btn">Save curated list</button>
</form>
<form method="post" action="/review/add" style="margin-top:1em">
  <label style="vertical-align:top">Add known people/emails/phones</label>
  <textarea name="bulk_items" rows="4" style="width:400px" placeholder="One per line - paste a whole known-people list at once if you like. Each line is auto-classified as a name, email, or phone number by matching the exact same patterns Redact uses.&#10;Jane Smith&#10;jane.smith@example.com&#10;+353 1 234 5678"></textarea>
  <button type="submit">Add all</button>
</form>

{{ name_search_tool(state.last_input_path, "review") }}
</section>

<section id="redact">
<h2>3. Redact</h2>
<form method="post" action="/redact">
  <div class="row"><label>Input file or folder</label><div class="path-row">
    <input type="text" id="redact_input_path" name="input_path" value="{{ state.last_input_path }}" required>
    <button type="button" class="browse-btn" onclick="openBrowse('redact_input_path')">📁</button></div></div>
  <div class="row"><label>Output file or folder</label><div class="path-row">
    <input type="text" id="redact_output_path" name="output_path" required>
    <button type="button" class="browse-btn" onclick="openBrowse('redact_output_path')">📁</button></div></div>
  <div class="row"><label>Names file</label><div class="path-row">
    <input type="text" id="redact_names_file" name="names_file" value="{{ state.names_file_path }}">
    <button type="button" class="browse-btn" onclick="openBrowse('redact_names_file')">📁</button></div></div>
  <div class="row"><label>Emails file</label><div class="path-row">
    <input type="text" id="redact_emails_file" name="emails_file" value="{{ state.emails_file_path }}">
    <button type="button" class="browse-btn" onclick="openBrowse('redact_emails_file')">📁</button></div>
    (optional - if this file exists, redacts exactly those values; otherwise falls back to scanning for anything email-shaped)</div>
  <div class="row"><label>Phones file</label><div class="path-row">
    <input type="text" id="redact_phones_file" name="phones_file" value="{{ state.phones_file_path }}">
    <button type="button" class="browse-btn" onclick="openBrowse('redact_phones_file')">📁</button></div>
    (optional - same idea: exact list if present, blanket scan otherwise)</div>
  <div class="row"><label>Exceptions file</label><div class="path-row">
    <input type="text" id="redact_exceptions_file" name="exceptions_file" value="{{ state.exceptions_file_path }}">
    <button type="button" class="browse-btn" onclick="openBrowse('redact_exceptions_file')">📁</button></div>
    (protects a specific phrase from ANY rule - still useful for names, e.g. a surname inside an unrelated compound phrase)</div>
  <div class="row"><label>Redact categories</label>
    <input type="checkbox" name="redact_names" checked> Names
    <input type="checkbox" name="redact_emails" checked> Emails
    <input type="checkbox" name="redact_phones" checked> Phone / fax numbers
    (LinkedIn links are always redacted)</div>
  <div class="row"><label></label><input type="checkbox" name="boilerplate_graphics" checked> repeated header/footer graphics</div>
  <div class="row"><label></label><input type="checkbox" name="boilerplate_text" checked> repeated header/footer text</div>
  <div class="row"><label>Analysis pages</label>
    <input type="number" name="start_page" value="1" style="width:60px"> to
    <input type="number" name="end_page" value="10" style="width:60px">
    (range sampled to learn what counts as boilerplate)</div>
  <div class="row"><label>Full-page redact</label><input type="text" name="full_page_redact" placeholder='e.g. 8,10,12-13'>
    (single-file mode only - same page list applies to every file in folder mode)</div>
  <button type="submit">Run redaction</button>
</form>
</section>

<section id="verify">
<h2>4. Verify (check redacted output for leftovers)</h2>
<form method="post" action="/verify">
  <div class="row"><label>Redacted file/folder</label><div class="path-row">
    <input type="text" id="verify_input_path" name="verify_input_path" value="{{ state.last_output_path }}" required>
    <button type="button" class="browse-btn" onclick="openBrowse('verify_input_path')">📁</button></div></div>
  <button type="submit" {{ "" if EXTRACTION_AVAILABLE else "disabled" }}>Check for leaks</button>
</form>
{% if verify_items %}
<form method="post" action="/verify/promote">
  <p>{{ verify_items|length }} candidate(s) still found. Check any real leaks and promote them back into curation:</p>
  <input type="text" class="filter-box" placeholder="Filter (type/value/context)..." data-filters="verify-table">
  <table class="curation-table" id="verify-table">
  <tr><th><input type="checkbox" class="select-all" title="Select/deselect all"></th><th>Type</th><th>Value</th><th>Confidence</th><th>Found in context</th></tr>
  {% for item in verify_items %}
  <tr><td><input type="checkbox" name="promote" value="{{ item.kind }}{{ KEY_SEP }}{{ item.value }}"></td>
      <td class="kind">{{ item.kind }}</td><td>{{ item.value }}</td>
      <td class="conf-{{ item.confidence|lower }}" data-sort="{{ {'High': 3, 'Medium': 2, 'Low': 1}[item.confidence] }}">{{ item.confidence }}</td>
      <td class="ctx">{{ item.context }}</td></tr>
  {% endfor %}
  </table>
  <p style="color:#555;font-size:0.85em">Tip: header checkbox selects/deselects all. Shift-click a row's
  checkbox to range-select from your last click.</p>
  <button type="submit">Add checked candidates to curation list</button>
</form>
{% endif %}
{{ name_search_tool(state.last_output_path, "verify") }}
</section>

<section id="interactive">
<h2>5. Interactive redaction (click to select)</h2>
<p style="color:#555">Open a PDF below, then <strong>select text directly on the page</strong>
(click-drag to highlight, like selecting text anywhere). A small menu appears with two choices:</p>
<ul style="color:#555;font-size:0.9em">
  <li><strong style="color:#cf222e">Redact this occurrence</strong> - blacks out only the spot you
  selected (shown <span style="background:rgba(220,0,0,0.35);padding:0 4px">red</span>).</li>
  <li><strong style="color:#b8860b">Redact every occurrence + metadata</strong> - blacks out that
  exact text everywhere in the document (shown
  <span style="background:rgba(255,210,0,0.4);padding:0 4px">yellow</span>); other matches on the
  visible pages are outlined as a preview. Document metadata is always scrubbed on save regardless.</li>
</ul>
<p style="color:#555;font-size:0.85em">Click any highlight to remove it. When you're done, set an
output path and click <strong>Redact &rarr; save output as</strong> - redaction runs through the
same engine as section 3 (content removed from the page stream, not covered over; metadata
scrubbed; old revisions dropped) and the viewer reloads on the new file. This pass redacts
<em>only</em> what you selected here - nothing is auto-detected.</p>

<div class="row"><label>PDF to open</label><div class="path-row">
  <input type="text" id="iact_input" name="iact_input" value="{{ state.last_output_path or state.last_input_path }}">
  <button type="button" class="browse-btn" onclick="openBrowse('iact_input')">📁</button>
  <button type="button" onclick="iactLoad()">Load</button></div></div>

<div id="iact-pages"><p style="color:#eee;padding:1em">No PDF loaded.</p></div>

<div id="iact-selections" style="margin:0.6em 0">
  <strong style="font-size:0.9em">Pending redactions</strong>
  <div id="iact-selection-list"><p style="color:#777;font-size:0.85em">None yet - select some text above.</p></div>
</div>

<div class="row"><label>Save output as</label><div class="path-row">
  <input type="text" id="iact_output" name="iact_output" placeholder="e.g. panel_REDACTED.pdf">
  <button type="button" class="browse-btn" onclick="openBrowse('iact_output')">📁</button>
  <button type="button" onclick="iactRedact()">Redact &rarr; save output as</button></div></div>
<div id="iact-status" style="font-size:0.9em;margin-top:0.4em"></div>

<div id="iact-popup">
  <div class="sel-text" id="iact-popup-text"></div>
  <button type="button" onclick="iactAddSelection('occurrence')">🟥 Redact this occurrence</button>
  <button type="button" onclick="iactAddSelection('global')">🟨 Redact every occurrence + metadata</button>
  <button type="button" onclick="iactHidePopup()">Cancel</button>
</div>
</section>

<section id="preview">
<h2>Preview a PDF</h2>
<p style="color:#555">Opens in a new tab using your browser's built-in PDF viewer. Works for any
PDF path on disk, not just redaction output - handy for spot-checking input files too.</p>
<form method="get" action="/view" target="_blank">
  <label>PDF path</label><div class="path-row">
    <input type="text" id="preview_path" name="path" value="{{ state.last_output_path }}" required>
    <button type="button" class="browse-btn" onclick="openBrowse('preview_path')">📁</button>
    <button type="submit">Open</button></div>
</form>
</section>

<div id="loading-overlay"><div class="spinner"></div><div id="loading-message">Working...</div></div>
<div id="browse-overlay" onclick="closeBrowse()"></div>
<div id="browse-panel">
  <div class="browse-shortcuts">
    <button type="button" class="browse-btn" onclick="browseTo(browseRepo)">📦 Repo</button>
    <button type="button" class="browse-btn" onclick="browseTo(browseHome)">🏠 Home</button>
    <button type="button" class="browse-btn" onclick="browseTo('/')">💻 Filesystem</button>
    <button type="button" class="browse-btn" onclick="document.getElementById('browse-upload-input').click()">⬆ Upload from my computer</button>
  </div>
  <input type="file" id="browse-upload-input" accept="application/pdf,.pdf" hidden>
  <div class="path-display" id="browse-path"></div>
  <input type="text" id="browse-filter" class="filter-box" style="width:100%" placeholder="Filter this folder...">
  <ul id="browse-list"></ul>
  <p style="color:#555;font-size:0.8em;margin:0.4em 0">"Upload from my computer" opens your OS file
  picker and copies the chosen PDF into <code>./uploads/</code>, then fills this field with that
  copy's path. Use it to grab a PDF from anywhere the folder list above can't easily reach.</p>
  <button type="button" onclick="chooseCurrentFolder()">Use this folder</button>
  <button type="button" onclick="closeBrowse()">Cancel</button>
</div>

<section id="log">
<h2>Log</h2>
<pre>{{ state.log or "(nothing run yet)" }}</pre>
</section>

<script>
document.querySelectorAll('table.curation-table').forEach(function (table) {
  var selectAll = table.querySelector('.select-all');
  var lastIndex = null;

  // Only rows currently visible (not hidden by a filter box) count for shift-click ranging and
  // "select all" - otherwise a filtered-out row between two clicks would silently get toggled too.
  function rowBoxes() {
    return Array.from(table.querySelectorAll('tr input[type=checkbox]'))
      .filter(function (cb) {
        if (cb.classList.contains('select-all')) return false;
        var tr = cb.closest('tr');
        return tr && tr.style.display !== 'none';
      });
  }

  if (selectAll) {
    selectAll.addEventListener('change', function () {
      rowBoxes().forEach(function (cb) { cb.checked = selectAll.checked; });
      updateCheckboxDirtyState();
    });
  }

  table.addEventListener('click', function (e) {
    var cb = e.target;
    if (!(cb.tagName === 'INPUT' && cb.type === 'checkbox') || cb.classList.contains('select-all')) return;
    var boxes = rowBoxes();
    var idx = boxes.indexOf(cb);
    if (e.shiftKey && lastIndex !== null && idx !== -1) {
      var start = Math.min(lastIndex, idx), end = Math.max(lastIndex, idx);
      for (var i = start; i <= end; i++) { boxes[i].checked = cb.checked; }
    }
    lastIndex = idx;
    updateCheckboxDirtyState();
  });
});

// Unsaved-changes indicator: "if I clicked Save right now, would the files on disk change?" -
// answered purely client-side by comparing each checkbox's LIVE checked state against
// data-was-saved (server-rendered from the last successful Save, not from the page's default
// checked/unchecked tier state - those aren't the same thing: a fresh extraction's High-tier
// items are checked by default but were never actually saved). Updates instantly on every
// checkbox change, select-all toggle, and shift-click range-set - no server round-trip, no stale
// count, since only the browser knows what's currently ticked.
var keepBoxes = Array.from(document.querySelectorAll('input[name="keep"]'));

function updateCheckboxDirtyState() {
  var banner = document.getElementById('unsaved-changes-banner');
  var summary = document.getElementById('unsaved-changes-summary');
  if (!banner) return;
  var added = 0, removed = 0;
  keepBoxes.forEach(function (cb) {
    var wasSaved = cb.dataset.wasSaved === '1';
    if (cb.checked && !wasSaved) added++;
    else if (!cb.checked && wasSaved) removed++;
  });
  var dirty = added > 0 || removed > 0;
  banner.hidden = !dirty;
  if (dirty) {
    var parts = [];
    if (added) parts.push(added + ' item' + (added !== 1 ? 's' : '') + ' would be added');
    if (removed) parts.push(removed + ' item' + (removed !== 1 ? 's' : '') + ' would be removed');
    summary.textContent = parts.join(', ');
  }
}

function undoUnsavedChanges() {
  keepBoxes.forEach(function (cb) { cb.checked = cb.dataset.wasSaved === '1'; });
  updateCheckboxDirtyState();
}

document.addEventListener('change', function (e) {
  if (e.target && e.target.name === 'keep') updateCheckboxDirtyState();
});

updateCheckboxDirtyState();  // e.g. right after an extraction, High-tier items are pre-checked
                              // but not yet saved - the banner should reflect that immediately.

// Sortable columns: click a header to sort by it (ascending), click again to reverse. Uses a
// cell's data-sort attribute if present (e.g. Confidence: High/Medium/Low needs severity order,
// not alphabetical - "High" < "Low" < "Medium" alphabetically would be wrong), otherwise just the
// visible text. The checkbox column (index 0) is never sortable.
document.querySelectorAll('table.curation-table').forEach(function (table) {
  var headerCells = Array.from(table.rows[0].cells);
  var sortState = { col: null, dir: 1 };

  headerCells.forEach(function (th, colIndex) {
    if (colIndex === 0) return;
    th.style.cursor = 'pointer';
    th.title = 'Click to sort';
    th.addEventListener('click', function () {
      var dir = (sortState.col === colIndex) ? -sortState.dir : 1;
      sortState = { col: colIndex, dir: dir };
      var dataRows = Array.from(table.rows).slice(1);
      dataRows.sort(function (rowA, rowB) {
        var cellA = rowA.cells[colIndex], cellB = rowB.cells[colIndex];
        var a = (cellA.dataset.sort !== undefined ? cellA.dataset.sort : cellA.textContent).trim().toLowerCase();
        var b = (cellB.dataset.sort !== undefined ? cellB.dataset.sort : cellB.textContent).trim().toLowerCase();
        var na = parseFloat(a), nb = parseFloat(b);
        var cmp = (!isNaN(na) && !isNaN(nb)) ? (na - nb) : a.localeCompare(b);
        return cmp * dir;
      });
      dataRows.forEach(function (tr) { table.tBodies[0] ? table.tBodies[0].appendChild(tr) : table.appendChild(tr); });
    });
  });
});

// Filter boxes: each has data-filters="<table id>" and hides rows where neither the Type,
// Value, nor Context column contains the typed text (case-insensitive).
document.querySelectorAll('.filter-box').forEach(function (box) {
  var table = document.getElementById(box.getAttribute('data-filters'));
  if (!table) return;
  box.addEventListener('input', function () {
    var q = box.value.toLowerCase();
    Array.from(table.querySelectorAll('tr')).slice(1).forEach(function (tr) {
      var text = tr.textContent.toLowerCase();
      tr.style.display = (!q || text.indexOf(q) !== -1) ? '' : 'none';
    });
  });
});

// The one "search across ALL tiers" box: same filtering logic, but applied to every table with
// the given class at once (rather than one table via a specific id) - so a name/email/phone sitting
// in a tier or kind table you didn't expect still turns up.
document.querySelectorAll('[data-filters-all]').forEach(function (box) {
  var tables = document.querySelectorAll('table.' + box.getAttribute('data-filters-all'));
  box.addEventListener('input', function () {
    var q = box.value.toLowerCase();
    tables.forEach(function (table) {
      Array.from(table.querySelectorAll('tr')).slice(1).forEach(function (tr) {
        var text = tr.textContent.toLowerCase();
        tr.style.display = (!q || text.indexOf(q) !== -1) ? '' : 'none';
      });
    });
  });
});

// Auto-derives the emails/phones save-paths from whatever's typed in the names save-path, by
// swapping the word "name" for "email"/"phone" (case-preserved on the first letter) wherever it
// appears - so a custom prefix like "graam_l_" only needs to be typed once, not three times.
function deriveSiblingPaths() {
  var namesInput = document.getElementById('review_names_path');
  var emailsInput = document.getElementById('review_emails_path');
  var phonesInput = document.getElementById('review_phones_path');
  var v = namesInput.value;
  if (!/name/i.test(v)) return;  // no recognizable pattern - leave the other two alone

  function derive(replacement) {
    return v.replace(/name/gi, function (match) {
      var isUpper = match[0] === match[0].toUpperCase();
      return isUpper ? (replacement[0].toUpperCase() + replacement.slice(1)) : replacement;
    });
  }
  emailsInput.value = derive('email');
  phonesInput.value = derive('phone');
}

// Folder/file browser: a shared panel any "📁" button can open against its own text input. A real
// native "Open File" dialog isn't reachable from a web page - browsers deliberately hide the true
// filesystem path of anything picked that way, and this app needs real local paths (including
// folders, and paths that don't exist yet for Save/output fields), not an uploaded copy. So this
// is a custom in-page equivalent instead: quick-access shortcuts, a clickable breadcrumb, and a
// filter box, styled after a native file picker's sidebar/address-bar/search.
var browseTargetId = null;
var browseCurrentPath = '';
var browseParentPath = '';
var browseHome = '';
var browseRepo = '';
var browseEntries = { dirs: [], files: [] };

function openBrowse(targetId) {
  browseTargetId = targetId;
  var input = document.getElementById(targetId);
  var start = input.value.trim();
  document.getElementById('browse-overlay').style.display = 'block';
  document.getElementById('browse-panel').style.display = 'block';
  browseTo(start);
}

function closeBrowse() {
  document.getElementById('browse-overlay').style.display = 'none';
  document.getElementById('browse-panel').style.display = 'none';
}

function renderBreadcrumb(path) {
  var el = document.getElementById('browse-path');
  el.innerHTML = '';
  var parts = path.split('/').filter(function (p) { return p.length; });
  var acc = '';
  var root = document.createElement('span');
  root.textContent = '/';
  root.className = 'crumb';
  root.onclick = function () { browseTo('/'); };
  el.appendChild(root);
  parts.forEach(function (part) {
    acc += '/' + part;
    var target = acc;
    el.appendChild(document.createTextNode(' '));
    var sep = document.createElement('span');
    sep.className = 'crumb-sep';
    sep.textContent = '/';
    var crumb = document.createElement('span');
    crumb.className = 'crumb';
    crumb.textContent = part;
    crumb.onclick = function () { browseTo(target); };
    el.appendChild(crumb);
    el.appendChild(document.createTextNode(' '));
    el.appendChild(sep);
  });
}

function renderBrowseList() {
  var filter = document.getElementById('browse-filter').value.trim().toLowerCase();
  var list = document.getElementById('browse-list');
  list.innerHTML = '';

  var up = document.createElement('li');
  up.textContent = '⬆ .. (parent folder)';
  up.onclick = function () { browseTo(browseParentPath); };
  list.appendChild(up);

  browseEntries.dirs
    .filter(function (name) { return !filter || name.toLowerCase().indexOf(filter) !== -1; })
    .forEach(function (name) {
      var li = document.createElement('li');
      li.textContent = '📁 ' + name;
      li.onclick = function () { browseTo(browseCurrentPath + '/' + name); };
      list.appendChild(li);
    });
  browseEntries.files
    .filter(function (name) { return !filter || name.toLowerCase().indexOf(filter) !== -1; })
    .forEach(function (name) {
      var li = document.createElement('li');
      li.textContent = '📄 ' + name;
      li.onclick = function () {
        document.getElementById(browseTargetId).value = browseCurrentPath + '/' + name;
        closeBrowse();
      };
      list.appendChild(li);
    });
}

function browseTo(path) {
  fetch('/browse?path=' + encodeURIComponent(path))
    .then(function (r) { return r.json(); })
    .then(function (data) {
      browseCurrentPath = data.path;
      browseParentPath = data.parent;
      browseHome = data.home;
      browseRepo = data.repo;
      browseEntries = { dirs: data.dirs || [], files: data.files || [] };
      renderBreadcrumb(data.path);
      document.getElementById('browse-filter').value = '';
      renderBrowseList();
    });
}

document.getElementById('browse-filter').addEventListener('input', renderBrowseList);

// "Upload from my computer": the browser's native file picker IS the real OS dialog, it just
// can't hand back a usable path - so POST the file to /upload, which saves it under ./uploads/
// and returns where it landed, and drop that path into the field the panel was opened for.
document.getElementById('browse-upload-input').addEventListener('change', function () {
  var file = this.files[0];
  this.value = '';  // so re-picking the same file still fires 'change'
  if (!file) return;
  var fd = new FormData();
  fd.append('file', file);
  showLoadingOverlay('Uploading ' + file.name + '...');
  fetch('/upload', { method: 'POST', body: fd })
    .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, data: d }; }); })
    .then(function (res) {
      hideLoadingOverlay();
      if (!res.ok || res.data.error) { alert('Upload failed: ' + (res.data.error || 'unknown error')); return; }
      if (browseTargetId) document.getElementById(browseTargetId).value = res.data.path;
      closeBrowse();
    })
    .catch(function (e) { hideLoadingOverlay(); alert('Upload failed: ' + e); });
});

function chooseCurrentFolder() {
  if (browseTargetId) {
    document.getElementById(browseTargetId).value = browseCurrentPath;
  }
  closeBrowse();
}

function showLoadingOverlay(message) {
  document.getElementById('loading-message').textContent = message;
  document.getElementById('loading-overlay').style.display = 'flex';
}
function hideLoadingOverlay() {
  document.getElementById('loading-overlay').style.display = 'none';
}

// Long-running actions (spaCy extraction, page-by-page redaction, folder-wide search) go through
// a normal full-page POST, so the browser just sits there with no feedback for however long the
// PDF work takes. Show a blocking spinner overlay the instant the form is actually submitted (not
// on click - a required field failing validation must not show it) so it's clear something is
// happening, and disable the button so a slow request can't be fired twice. The full-page reload
// that follows tears the overlay down on its own.
function attachLoadingSpinner(action, message) {
  document.querySelectorAll('form[action="' + action + '"]').forEach(function (form) {
    form.addEventListener('submit', function () {
      showLoadingOverlay(message);
      var btn = form.querySelector('button[type=submit]');
      if (btn) btn.disabled = true;
    });
  });
}

attachLoadingSpinner('/extract', 'Extracting names, emails and phone numbers - this can take a while for large PDFs or folders...');
attachLoadingSpinner('/redact', 'Redacting - writing the output file(s) now...');
attachLoadingSpinner('/verify', 'Scanning the redacted output for anything that slipped through...');
attachLoadingSpinner('/review/search_document', 'Searching text and metadata for these names...');
</script>

<script src="/static/pdfjs/pdf.js"></script>
<script>
// ===== Section 5: interactive click-to-redact =====
// Renders the PDF with pdf.js (vendored under /static/pdfjs/), lays a selectable text layer over
// each page, and lets the user turn a text selection into a pending redaction - either this one
// spot (red) or every occurrence of that text in the document (yellow). "Redact -> save output
// as" POSTs the selections to /redact_interactive, which runs the same redact_pdf() engine with
// automated=False so ONLY these selections are applied. Nothing is sent until that button.
(function () {
  if (!window.pdfjsLib) { return; }
  pdfjsLib.GlobalWorkerOptions.workerSrc = '/static/pdfjs/pdf.worker.js';

  var pagesEl = document.getElementById('iact-pages');
  var popupEl = document.getElementById('iact-popup');
  var listEl = document.getElementById('iact-selection-list');
  var statusEl = document.getElementById('iact-status');
  if (!pagesEl) { return; }

  var selections = [];   // {id, page, mode:'occurrence'|'global', text, rects:[{x0,y0,x1,y1}]}
  var pageWraps = [];     // index 0-based -> {wrap, hlLayer, textLayer, w, h}
  var pendingSel = null;  // {page, text, rects} awaiting a mode choice from the popup
  var seq = 0;
  var loadedPath = '';

  function setStatus(msg, isError) {
    statusEl.textContent = msg || '';
    statusEl.style.color = isError ? '#a40e26' : '#116329';
  }

  async function iactLoad() {
    var path = document.getElementById('iact_input').value.trim();
    if (!path) { setStatus('Enter a PDF path first.', true); return; }
    setStatus('Loading ' + path + ' ...');
    popupHide();
    selections = [];
    renderSelectionList();
    pagesEl.innerHTML = '';
    pageWraps = [];
    try {
      var doc = await pdfjsLib.getDocument({ url: '/view?path=' + encodeURIComponent(path) }).promise;
      var avail = Math.max(320, pagesEl.clientWidth - 32);
      for (var n = 1; n <= doc.numPages; n++) {
        var page = await doc.getPage(n);
        var base = page.getViewport({ scale: 1 });
        var scale = Math.min(2.0, avail / base.width);
        var vp = page.getViewport({ scale: scale });

        var wrap = document.createElement('div');
        wrap.className = 'iact-page-wrap';
        wrap.style.width = vp.width + 'px';
        wrap.style.height = vp.height + 'px';
        wrap.dataset.page = String(n);

        var canvas = document.createElement('canvas');
        canvas.width = Math.floor(vp.width);
        canvas.height = Math.floor(vp.height);
        wrap.appendChild(canvas);

        var textLayer = document.createElement('div');
        textLayer.className = 'iact-textlayer';
        // pdf.js 3.x positions text spans with `calc(var(--scale-factor) * Npx)` - without this
        // the whole text layer collapses to zero size and selection geometry breaks.
        textLayer.style.setProperty('--scale-factor', String(scale));
        wrap.appendChild(textLayer);

        var hlLayer = document.createElement('div');
        hlLayer.className = 'iact-hl-layer';
        wrap.appendChild(hlLayer);

        pagesEl.appendChild(wrap);
        pageWraps.push({ wrap: wrap, hlLayer: hlLayer, textLayer: textLayer,
                         w: vp.width, h: vp.height });

        await page.render({ canvasContext: canvas.getContext('2d'), viewport: vp }).promise;
        var tc = await page.getTextContent();
        await pdfjsLib.renderTextLayer({ textContentSource: tc, container: textLayer,
                                         viewport: vp, textDivs: [] }).promise;
      }
      loadedPath = path;
      setStatus('Loaded ' + doc.numPages + ' page(s). Select text on a page to redact it.');
    } catch (e) {
      setStatus('Could not load PDF: ' + e, true);
    }
  }
  window.iactLoad = iactLoad;

  // --- turning a text selection into a pending redaction ---
  pagesEl.addEventListener('mouseup', function () {
    setTimeout(handleSelection, 0);  // let the browser finish updating the selection
  });

  function handleSelection() {
    var sel = window.getSelection();
    if (!sel || sel.isCollapsed || !sel.rangeCount) { popupHide(); return; }
    var text = sel.toString().replace(/\s+/g, ' ').trim();
    if (!text) { popupHide(); return; }

    var range = sel.getRangeAt(0);
    var node = range.startContainer;
    var wrapEl = (node.nodeType === 1 ? node : node.parentElement);
    wrapEl = wrapEl && wrapEl.closest('.iact-page-wrap');
    if (!wrapEl) { return; }
    var pageNo = parseInt(wrapEl.dataset.page, 10);
    var wrapRect = wrapEl.getBoundingClientRect();

    var rects = [];
    var clientRects = range.getClientRects();
    for (var i = 0; i < clientRects.length; i++) {
      var r = clientRects[i];
      if (r.width < 1 || r.height < 1) { continue; }
      // keep only rects that sit within this page
      if (r.bottom < wrapRect.top || r.top > wrapRect.bottom) { continue; }
      rects.push(normRect(r, wrapRect));
    }
    if (!rects.length) { return; }

    pendingSel = { page: pageNo, text: text, rects: rects };
    document.getElementById('iact-popup-text').textContent = text;
    var last = clientRects[clientRects.length - 1];  // viewport coords; popup is position:fixed
    popupEl.style.display = 'block';
    var pw = popupEl.offsetWidth || 300, ph = popupEl.offsetHeight || 140;
    popupEl.style.left = Math.max(8, Math.min(window.innerWidth - pw - 8, last.left)) + 'px';
    var below = last.bottom + 6;
    popupEl.style.top = (below + ph < window.innerHeight ? below : Math.max(8, last.top - ph - 6)) + 'px';
  }

  function normRect(r, wrapRect) {
    return {
      x0: (r.left - wrapRect.left) / wrapRect.width,
      y0: (r.top - wrapRect.top) / wrapRect.height,
      x1: (r.right - wrapRect.left) / wrapRect.width,
      y1: (r.bottom - wrapRect.top) / wrapRect.height,
    };
  }

  function popupHide() { popupEl.style.display = 'none'; pendingSel = null; }
  window.iactHidePopup = function () { popupHide(); if (window.getSelection) window.getSelection().removeAllRanges(); };

  window.iactAddSelection = function (mode) {
    if (!pendingSel) { return; }
    selections.push({ id: ++seq, page: pendingSel.page, mode: mode,
                      text: pendingSel.text, rects: pendingSel.rects });
    popupHide();
    if (window.getSelection) { window.getSelection().removeAllRanges(); }
    renderSelectionList();
    renderHighlights();
  };

  function removeSelection(id) {
    selections = selections.filter(function (s) { return s.id !== id; });
    renderSelectionList();
    renderHighlights();
  }

  function renderSelectionList() {
    if (!selections.length) {
      listEl.innerHTML = '<p style="color:#777;font-size:0.85em">None yet - select some text above.</p>';
      return;
    }
    listEl.innerHTML = '';
    selections.forEach(function (s) {
      var row = document.createElement('div');
      row.className = 'iact-selrow';
      var badge = document.createElement('span');
      badge.className = 'iact-badge ' + s.mode;
      badge.textContent = (s.mode === 'global' ? 'ALL + META' : 'THIS SPOT');
      var val = document.createElement('span');
      val.className = 'val';
      val.textContent = s.text + '  (p' + s.page + ')';
      var rm = document.createElement('button');
      rm.type = 'button';
      rm.textContent = '✕';
      rm.title = 'Remove';
      rm.onclick = function () { removeSelection(s.id); };
      row.appendChild(badge); row.appendChild(val); row.appendChild(rm);
      listEl.appendChild(row);
    });
  }

  function addBox(layer, nr, cls) {
    var d = document.createElement('div');
    d.className = 'iact-hl ' + cls;
    d.style.left = (nr.x0 * 100) + '%';
    d.style.top = (nr.y0 * 100) + '%';
    d.style.width = ((nr.x1 - nr.x0) * 100) + '%';
    d.style.height = ((nr.y1 - nr.y0) * 100) + '%';
    layer.appendChild(d);
    return d;
  }

  function renderHighlights() {
    pageWraps.forEach(function (pw, idx) {
      var pageNo = idx + 1;
      pw.hlLayer.innerHTML = '';

      // preview: outline every visible occurrence of each 'global' term on this page
      var globalTerms = selections.filter(function (s) { return s.mode === 'global'; })
                                  .map(function (s) { return s.text.toLowerCase(); });
      if (globalTerms.length) {
        var wrapRect = pw.wrap.getBoundingClientRect();
        var spans = pw.textLayer.querySelectorAll('span');
        for (var i = 0; i < spans.length; i++) {
          var t = (spans[i].textContent || '').toLowerCase();
          if (!t.trim()) { continue; }
          if (globalTerms.some(function (g) { return t.indexOf(g) !== -1; })) {
            var box = addBox(pw.hlLayer, normRect(spans[i].getBoundingClientRect(), wrapRect), 'preview');
            box.style.pointerEvents = 'none';
          }
        }
      }

      // the actual pending selections
      selections.forEach(function (s) {
        if (s.page !== pageNo) { return; }
        s.rects.forEach(function (nr) {
          var box = addBox(pw.hlLayer, nr, s.mode);
          box.title = 'Click to remove this redaction';
          box.onclick = function () { removeSelection(s.id); };
        });
      });
    });
  }

  window.iactRedact = async function () {
    if (!selections.length) { setStatus('Select something to redact first.', true); return; }
    var inputPath = document.getElementById('iact_input').value.trim() || loadedPath;
    var outputPath = document.getElementById('iact_output').value.trim();
    if (!inputPath) { setStatus('No input PDF.', true); return; }
    if (!outputPath) { setStatus('Set an output path ("Save output as").', true); return; }
    if (outputPath === inputPath) { setStatus('Choose an output path different from the input.', true); return; }

    showLoadingOverlay('Redacting your ' + selections.length + ' selection(s) and writing the output...');
    try {
      var res = await fetch('/redact_interactive', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input_path: inputPath, output_path: outputPath, selections: selections }),
      });
      var data = await res.json();
      hideLoadingOverlay();
      if (!res.ok || !data.ok) { setStatus('Redaction failed: ' + (data.log || res.statusText), true); return; }
      document.getElementById('iact_input').value = data.output_path;
      selections = [];
      renderSelectionList();
      await iactLoad();
      setStatus('Redacted. Output saved to ' + data.output_path + ' - viewer reloaded on it.');
    } catch (e) {
      hideLoadingOverlay();
      setStatus('Redaction request failed: ' + e, true);
    }
  };
})();
</script>

</body></html>
""")


@app.route("/")
def index():
    weights = state["confidence_weights"]
    annotated = annotate(state["candidates"], weights)
    high_items = [it for it in annotated if it["confidence"] == "High"]
    medium_items = [it for it in annotated if it["confidence"] == "Medium"]
    low_items = [it for it in annotated if it["confidence"] == "Low"]
    annotated_verify = annotate(state["verify_items"], weights)
    flash = state["flash"]
    state["flash"] = None
    # Which items are ACTUALLY on disk right now, keyed the same way the "keep" checkboxes are -
    # lets the template mark each checkbox with whether saving right now would change anything for
    # it, which is the only thing that live JS dirty-tracking can't know on its own (a checkbox's
    # on-page-load checked state isn't necessarily the same as "is this saved").
    last_saved_keys = {item_key(it) for it in state["last_saved_candidates"]}
    return TEMPLATE.render(
        state=state, EXTRACTION_AVAILABLE=EXTRACTION_AVAILABLE, EXTRACTION_ERROR=EXTRACTION_ERROR,
        KEY_SEP=KEY_SEP, high_items=high_items, medium_items=medium_items, low_items=low_items,
        verify_items=annotated_verify, weights=weights, flash=flash,
        item_key=item_key, last_saved_keys=last_saved_keys,
    )


@app.route("/extract", methods=["POST"])
def do_extract():
    if not EXTRACTION_AVAILABLE:
        state["log"] = f"Extraction unavailable: {EXTRACTION_ERROR}"
        set_flash(f"Extraction unavailable: {EXTRACTION_ERROR}", "error")
        return redirect(url_for("index") + "#log")
    input_path = request.form["input_path"].strip()
    state["last_input_path"] = input_path
    merge = "merge" in request.form
    output_path = request.form.get("output_path", "").strip() or state["names_file_path"]
    exceptions_file = request.form.get("exceptions_file", "").strip() or state["exceptions_file_path"]
    do_names = "extract_names" in request.form
    do_emails = "extract_emails" in request.form
    do_phones = "extract_phones" in request.form

    def task():
        exceptions = set(p.lower() for p in redactor.load_exceptions_from_file(exceptions_file))
        seed = []
        if merge and os.path.exists(output_path):
            with open(output_path, encoding="utf-8") as fh:
                seed = [{"kind": "name", "value": line.strip(), "context": "(from existing file)",
                         "confirmed": True}  # already saved once - a re-extraction shouldn't
                        for line in fh if line.strip()]                                # un-confirm it
            print(f"Loaded {len(seed)} existing names from {output_path} (merge).")
        return run_extraction(input_path, seed_items=seed,
                               do_names=do_names, do_emails=do_emails, do_phones=do_phones,
                               exceptions=exceptions)

    result, ok = run_captured(task)
    if ok and result is not None:
        state["exceptions_file_path"] = exceptions_file
        state["candidates"] = result
        state["names_file_path"] = output_path
        state["emails_file_path"], state["phones_file_path"] = derive_sibling_paths(output_path)
        set_flash(f"Extraction complete: {len(result)} candidate(s) ready for review below.")
    else:
        set_flash("Extraction failed - see the log for details.", "error")
    return redirect(url_for("index") + "#review")


@app.route("/review/save", methods=["POST"])
def review_save():
    names_path = request.form.get("names_path", "").strip() or state["names_file_path"]
    emails_path = request.form.get("emails_path", "").strip() or state["emails_file_path"]
    phones_path = request.form.get("phones_path", "").strip() or state["phones_file_path"]
    keep_keys = set(request.form.getlist("keep"))

    kept_items = [it for it in state["candidates"] if item_key(it) in keep_keys]
    # unchecked items of any kind are simply left out of their file below - that omission alone
    # is what keeps them from being redacted, once Redact is pointed at these exact-match files.

    # Checking (and saving) an item is treated as your confirmation it's real, regardless of which
    # confidence tier it started in - this is the "clicking accept moves it to High" mechanic:
    # it takes effect on the next page view/reload, since confidence is computed fresh from stored
    # signals each time, not baked in at extraction time.
    for it in kept_items:
        it["confirmed"] = True

    def task():
        for kind, path, label in (
            ("name", names_path, "names"),
            ("email", emails_path, "emails"),
            ("phone", phones_path, "phones"),
        ):
            values = sorted({it["value"] for it in kept_items if it["kind"] == kind})
            with open(path, "w", encoding="utf-8") as fh:
                for value in values:
                    fh.write(value + "\n")
            print(f"Saved {len(values)} {label} to {path}.")

    _, ok = run_captured(task)
    state["candidates"] = kept_items
    state["names_file_path"] = names_path
    state["emails_file_path"] = emails_path
    state["phones_file_path"] = phones_path
    if ok:
        # Only advance the "last saved" baseline on an actual successful write - a failed save
        # must keep flagging as unsaved, since the files on disk weren't actually touched.
        state["last_saved_candidates"] = [dict(it) for it in kept_items]
        set_flash(f"Saved {len(kept_items)} checked item(s) to the curated lists.")
    else:
        set_flash("Save failed - see the log for details.", "error")
    return redirect(url_for("index") + "#review")


@app.route("/review/add", methods=["POST"])
def review_add():
    """Bulk-adds known people/emails/phones typed or pasted in directly - the 'seed with what you
    already know' path, as an alternative (or complement) to extracting from a PDF. Kind is
    auto-detected per line using the exact same regexes Redact matches against, so classification
    stays consistent with actual redaction behaviour."""
    bulk_text = request.form.get("bulk_items", "")
    existing_keys = {item_key(it) for it in state["candidates"]}
    added = 0

    for line in bulk_text.splitlines():
        value = line.strip()
        if not value:
            continue
        if redactor.EMAIL_REGEX.fullmatch(value):
            kind = "email"
        elif redactor.PHONE_REGEX.fullmatch(value):
            kind = "phone"
        else:
            kind = "name"

        key = f"{kind}{KEY_SEP}{value}"
        if key in existing_keys:
            continue
        state["candidates"].append({"kind": kind, "value": value, "context": "(manually added)",
                                     "source": "manual", "count": 1})
        existing_keys.add(key)
        added += 1

    state["candidates"].sort(key=lambda it: (it["kind"], it["value"].lower()))
    if added:
        state["log"] = f"Added {added} manually-entered item(s)."
        set_flash(f"Added {added} manually-entered item(s) to the curated list.")
    else:
        set_flash("Nothing added - those item(s) were already in the curated list, or the box was empty.", "warn")
    return redirect(url_for("index") + "#review")


@app.route("/review/search_document", methods=["POST"])
def review_search_document():
    """Manual 'is this name really gone' check - independent of the extraction pipeline above, so
    it works even without spaCy. Searches raw page text AND document metadata for each typed-in
    name, tolerant of spacing differences (join/split/line-break variants)."""
    search_path = request.form.get("search_path", "").strip()
    names = [n.strip() for n in request.form.get("search_names", "").splitlines() if n.strip()]

    def task():
        return redactor.search_names_in_path(search_path, names)

    result, ok = run_captured(task)
    return_to = request.form.get("return_to", "review")
    if return_to not in ("review", "verify"):
        return_to = "review"
    if ok and result is not None:
        state["search_results"] = result
        found = sum(1 for matches in result.values() if matches)
        if found:
            set_flash(f"Search complete: {found} of {len(names)} name(s) found in {search_path}.", "warn")
        else:
            set_flash(f"Search complete: none of the {len(names)} name(s) were found in {search_path}.")
    else:
        set_flash("Search failed - see the log for details.", "error")
    return redirect(url_for("index") + f"#{return_to}")


@app.route("/review/search_add", methods=["POST"])
def review_search_add():
    """Promotes names the manual text/metadata search confirmed present straight into the curated
    candidate list - the search is often how a leftover gets *found* in the first place, so it
    should be one click from there into the list that actually drives redaction, rather than
    requiring the same name be re-typed into the bulk-add box above."""
    chosen = [n.strip() for n in request.form.getlist("add_names") if n.strip()]
    existing_keys = {item_key(it) for it in state["candidates"]}
    added = []
    for name in chosen:
        key = f"name{KEY_SEP}{name}"
        if key in existing_keys:
            continue
        state["candidates"].append({"kind": "name", "value": name, "context": "(confirmed via name search)",
                                     "source": "manual", "count": 1})
        existing_keys.add(key)
        added.append(name)
    state["candidates"].sort(key=lambda it: (it["kind"], it["value"].lower()))

    return_to = request.form.get("return_to", "review")
    if return_to not in ("review", "verify"):
        return_to = "review"
    if added:
        set_flash(f"Added {len(added)} name(s) to the curated list: {', '.join(added)}.")
    else:
        set_flash("No names added - the selected name(s) were already in the curated list, or none were checked.", "warn")
    return redirect(url_for("index") + f"#{return_to}")


@app.route("/settings/confidence", methods=["POST"])
def update_confidence_weights():
    """Live-updates the confidence scoring weights. Re-labels the existing candidate list
    immediately (via annotate() in index()) - no re-extraction needed."""
    for key in DEFAULT_CONFIDENCE_WEIGHTS:
        raw = request.form.get(key, "").strip()
        if raw:
            try:
                state["confidence_weights"][key] = int(raw)
            except ValueError:
                pass
    set_flash("Confidence weights updated - tiers below are re-labeled immediately.")
    return redirect(url_for("index") + "#review")


@app.route("/browse")
def browse_dir():
    """Directory listing for the in-page folder/file browser. Local-only tool, same trust model
    as every other path field here (they already accept arbitrary local paths typed by hand)."""
    path = request.args.get("path", "").strip() or os.getcwd()
    if not os.path.isdir(path):
        path = os.path.dirname(path.rstrip("/")) or os.getcwd()
    if not os.path.isdir(path):
        path = os.getcwd()
    path = os.path.abspath(path)

    # Quick-access shortcuts, like the sidebar in a native file picker - a real OS "Open File"
    # dialog isn't reachable from a web page (browsers deliberately hide the real filesystem path
    # of anything picked that way), so this is the closest equivalent: one click from wherever you
    # are back to the folder the server was launched from, your home directory, or the filesystem
    # root (useful in WSL specifically - Windows drives are reachable under /mnt/c etc. from there).
    home = os.path.expanduser("~")
    repo_root = SERVER_START_DIR

    try:
        entries = os.listdir(path)
    except Exception as e:
        return jsonify({"path": path, "parent": os.path.dirname(path) or path,
                        "dirs": [], "files": [], "error": str(e), "home": home, "repo": repo_root})

    dirs = sorted(e for e in entries if not e.startswith(".") and os.path.isdir(os.path.join(path, e)))
    files = sorted(e for e in entries if e.lower().endswith(".pdf") and os.path.isfile(os.path.join(path, e)))
    parent = os.path.dirname(path.rstrip("/")) or path
    return jsonify({"path": path, "parent": parent, "dirs": dirs, "files": files,
                    "home": home, "repo": repo_root})


@app.route("/upload", methods=["POST"])
def upload_pdf():
    """Accepts one PDF from the browser's native file picker (the real OS Open dialog), saves it
    into ./uploads/, and hands back the on-disk path - which then flows through Extract/Redact/etc.
    exactly like any typed path. This is how you pick a file from somewhere the server's own
    directory listing can't easily reach (a Windows drive, another user's home, a network mount).
    Nothing is deleted automatically; ./uploads/ is gitignored so real case files never get
    committed, but you may want to clear it out yourself periodically."""
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "No file received."}), 400
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files can be uploaded."}), 400

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_name = secure_filename(f.filename) or "upload.pdf"
    if not safe_name.lower().endswith(".pdf"):
        safe_name += ".pdf"
    dest = os.path.join(UPLOAD_DIR, safe_name)
    if os.path.exists(dest):  # don't silently overwrite an earlier upload of the same name
        base, ext = os.path.splitext(safe_name)
        n = 2
        while os.path.exists(os.path.join(UPLOAD_DIR, f"{base}_{n}{ext}")):
            n += 1
        dest = os.path.join(UPLOAD_DIR, f"{base}_{n}{ext}")

    f.save(dest)
    return jsonify({"path": dest, "name": os.path.basename(dest)})


@app.route("/redact", methods=["POST"])
def do_redact():
    f = request.form
    input_path = f["input_path"].strip()
    state["last_input_path"] = input_path
    output_path = f["output_path"].strip()
    names_file = f.get("names_file", "").strip() or state["names_file_path"]
    exceptions_file = f.get("exceptions_file", "").strip() or None
    # emails_file/phones_file are optional (blanket regex mode when absent) - unlike names_file,
    # a path that doesn't exist yet should silently mean "no curated list", not crash the job.
    emails_file = f.get("emails_file", "").strip() or None
    if emails_file and not os.path.isfile(emails_file):
        emails_file = None
    phones_file = f.get("phones_file", "").strip() or None
    if phones_file and not os.path.isfile(phones_file):
        phones_file = None
    boilerplate_graphics = "boilerplate_graphics" in f
    boilerplate_text = "boilerplate_text" in f
    redact_names = "redact_names" in f
    redact_emails = "redact_emails" in f
    redact_phones = "redact_phones" in f
    start_page = int(f.get("start_page") or 1)
    end_page = int(f.get("end_page") or 10)
    full_page_pages = redactor.parse_page_spec(f.get("full_page_redact", ""))

    def task():
        if os.path.isdir(input_path):
            redactor.batch_redact_tree(
                input_path, output_path, names_file, boilerplate_graphics, boilerplate_text,
                start_page, end_page, exceptions_file, full_page_redact_pages=full_page_pages,
                redact_names=redact_names, redact_emails=redact_emails, redact_phones=redact_phones,
                emails_file=emails_file, phones_file=phones_file,
            )
        else:
            redactor.redact_pdf(
                input_path, output_path, names_file, boilerplate_graphics, boilerplate_text,
                start_page, end_page, exceptions_file, full_page_redact_pages=full_page_pages,
                redact_names=redact_names, redact_emails=redact_emails, redact_phones=redact_phones,
                emails_file=emails_file, phones_file=phones_file,
            )

    _, ok = run_captured(task)
    if not os.path.isdir(input_path):
        state["last_output_path"] = output_path
    if ok:
        set_flash(f"Redaction complete - output saved to {output_path}.")
    else:
        set_flash("Redaction failed - see the log for details.", "error")
    return redirect(url_for("index") + "#log")


@app.route("/view")
def view_pdf():
    path = request.args.get("path", "").strip()
    if not path or not path.lower().endswith(".pdf") or not os.path.isfile(path):
        abort(404)
    return send_file(path, mimetype="application/pdf")


@app.route("/redact_interactive", methods=["POST"])
def do_redact_interactive():
    """Backs the "5. Interactive redaction" section. The browser sends the input/output paths and
    a list of on-page selections the user made in the pdf.js viewer; each selection is either

      - mode "occurrence": redact just this spot - carries one or more rectangles, normalised to
        0..1 of the rendered page size, top-left origin (which is exactly PyMuPDF's page.rect
        coordinate space once multiplied back up by page.rect.width/height).
      - mode "global": redact every occurrence of this text in the document - carries the matched
        string (searched doc-wide server-side) plus its own clicked rectangle as a fallback.

    Redaction runs through the same redact_pdf() as everything else, with automated=False so ONLY
    these manual selections are applied (no name/email/boilerplate heuristics). Metadata is still
    scrubbed on save exactly as in a normal run.
    """
    data = request.get_json(silent=True) or {}
    input_path = (data.get("input_path") or "").strip()
    output_path = (data.get("output_path") or "").strip()
    selections = data.get("selections") or []

    if not input_path.lower().endswith(".pdf") or not os.path.isfile(input_path):
        return jsonify({"ok": False, "log": f"Input PDF not found: {input_path}"}), 400
    if not output_path.lower().endswith(".pdf"):
        return jsonify({"ok": False, "log": "Output path must end in .pdf"}), 400
    if not selections:
        return jsonify({"ok": False, "log": "No selections were made - nothing to redact."}), 400

    import fitz

    manual_page_rects = {}
    manual_global_terms = set()
    try:
        doc = fitz.open(input_path)
        page_sizes = [(p.rect.width, p.rect.height) for p in doc]
        doc.close()
    except Exception as e:
        return jsonify({"ok": False, "log": f"Could not open {input_path}: {e}"}), 400

    for sel in selections:
        try:
            page_no = int(sel.get("page"))
        except (TypeError, ValueError):
            continue
        if not (1 <= page_no <= len(page_sizes)):
            continue
        pw, ph = page_sizes[page_no - 1]
        for rn in sel.get("rects") or []:
            try:
                x0 = float(rn["x0"]) * pw
                y0 = float(rn["y0"]) * ph
                x1 = float(rn["x1"]) * pw
                y1 = float(rn["y1"]) * ph
            except (KeyError, TypeError, ValueError):
                continue
            manual_page_rects.setdefault(page_no, []).append(
                [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
            )
        if sel.get("mode") == "global":
            term = (sel.get("text") or "").strip()
            if term:
                manual_global_terms.add(term)

    if not manual_page_rects and not manual_global_terms:
        return jsonify({"ok": False, "log": "Selections carried no usable coordinates or text."}), 400

    def task():
        redactor.redact_pdf(
            input_path, output_path, names_file=None,
            boilerplate_graphics=False, boilerplate_text=False,
            analysis_start_page=1, analysis_end_page=10,
            redact_names=False, redact_emails=False, redact_phones=False,
            automated=False,
            manual_page_rects=manual_page_rects,
            manual_global_terms=sorted(manual_global_terms),
        )

    _, ok = run_captured(task)
    if ok:
        state["last_input_path"] = input_path
        state["last_output_path"] = output_path
        n_occ = sum(len(v) for v in manual_page_rects.values())
        set_flash(f"Interactive redaction complete - {n_occ} region(s) and "
                  f"{len(manual_global_terms)} whole-document term(s) redacted to {output_path}.")
        return jsonify({"ok": True, "output_path": output_path, "log": state["log"]})
    return jsonify({"ok": False, "log": state["log"]}), 500


@app.route("/verify", methods=["POST"])
def do_verify():
    if not EXTRACTION_AVAILABLE:
        state["log"] = f"Extraction unavailable: {EXTRACTION_ERROR}"
        set_flash(f"Extraction unavailable: {EXTRACTION_ERROR}", "error")
        return redirect(url_for("index") + "#log")
    input_path = request.form["verify_input_path"].strip()

    def task():
        return run_extraction(input_path)  # checks all three kinds by default

    result, ok = run_captured(task)
    if ok and result is not None:
        state["verify_items"] = result
        if result:
            set_flash(f"Verify complete: {len(result)} possible leftover(s) found - review below.", "warn")
        else:
            set_flash("Verify complete: no leftover names/emails/phones found.")
    else:
        set_flash("Verify failed - see the log for details.", "error")
    return redirect(url_for("index") + "#verify")


@app.route("/verify/promote", methods=["POST"])
def verify_promote():
    chosen_keys = set(request.form.getlist("promote"))
    verify_by_key = {item_key(it): it for it in state["verify_items"]}
    existing_keys = {item_key(it) for it in state["candidates"]}
    promoted = 0
    for key in chosen_keys:
        if key not in existing_keys and key in verify_by_key:
            state["candidates"].append(verify_by_key[key])
            promoted += 1
    state["candidates"].sort(key=lambda it: (it["kind"], it["value"].lower()))
    if promoted:
        set_flash(f"Added {promoted} confirmed leak(s) back into the curated list.")
    else:
        set_flash("Nothing added - the checked item(s) were already in the curated list, or none were checked.", "warn")
    return redirect(url_for("index") + "#review")


def _find_free_port(preferred=5000, host="127.0.0.1"):
    import socket
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    return preferred


def _open_browser_soon(url, delay=1.0):
    import threading
    import webbrowser

    def _open():
        try:
            webbrowser.open(url)
        except Exception:
            pass  # non-fatal - the console message below still has the URL to open by hand

    threading.Timer(delay, _open).start()


if __name__ == "__main__":
    if getattr(sys, "frozen", False):
        try:
            sys.stdout.reconfigure(line_buffering=True)
            sys.stderr.reconfigure(line_buffering=True)
        except Exception:
            pass

    host = "127.0.0.1"
    port = _find_free_port(5000, host)
    url = f"http://{host}:{port}"

    banner = f"""
==============================================
  PDF Redactor
  Running at {url}
  Opening in your browser - if it doesn't,
  copy the address above into one.
  Close this window (or Ctrl+C) to stop.
==============================================
"""
    print(banner, file=sys.stderr)

    _open_browser_soon(url)

    try:
        app.run(host=host, port=port, debug=False)
    except Exception:
        import traceback
        traceback.print_exc()
        if getattr(sys, "frozen", False):
            input("\nSomething went wrong - see the error above. Press Enter to close...")
        raise
