# PDF Redactor

[![License: AGPL v3](https://img.shields.io/github/license/Sanvestek/pdf-redaction-toolkit)](LICENSE)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![Build Windows exe](https://github.com/Sanvestek/pdf-redaction-toolkit/actions/workflows/build-windows-exe.yml/badge.svg?branch=exe)](https://github.com/Sanvestek/pdf-redaction-toolkit/actions/workflows/build-windows-exe.yml)

A free, local-first PDF redaction tool — no uploads, no cloud — that actually removes content instead of drawing a black box on top of it.

If you searched for a way to redact a PDF for free, remove personal information
from a PDF, or you specifically need an offline PDF redaction / PDF PII removal
tool that never uploads your document anywhere, that's exactly what this is.

Redacts names, emails, phone numbers, LinkedIn links, signatures, and repeated
header/footer boilerplate from PDFs using PyMuPDF. Redaction happens at the
content-stream level (not a black box drawn on top), and document metadata is
scrubbed on save.

<!--
  Replace this with a real screenshot or short GIF of the web UI once you have
  one — see the "Screenshot / GIF" section of docs/ (or the PR/commit that added
  this comment) for exactly how to capture and drop it in. Recommended shot: the
  Review page's confidence-tiered candidate list (section 2 in the web UI), or a
  short GIF of the interactive click-to-redact flow (section 5).
-->
![PDF Redactor web UI — reviewing auto-detected names, emails and phone numbers by confidence before redacting](docs/screenshot.png)

> This repo contains no real documents or personal data, in the working tree or
> in its git history — only the tool's code and a fabricated demo case (see
> below). It was extracted from a private working repo that does process real
> documents; nothing from that repo (including its history) was copied here.

## Try it on the demo data

`demo_data/raw/` has three small synthetic PDFs (a fake grant application, a fake
consolidated evaluation, a fake panel recommendation) with matching redaction
lists, so you can run the whole pipeline immediately after setup:

```
python redact_pymupdf_fast.py demo_data/raw demo_data/redacted \
  --names_file demo_data/names_to_redact.txt \
  --emails_file demo_data/emails_to_redact.txt \
  --phones_file demo_data/phones_to_redact.txt \
  --exceptions_file name_exceptions.txt \
  --boilerplate_graphics --boilerplate_text
```

Compare `demo_data/raw/` and `demo_data/redacted/` afterwards to see what changed.
Every name, email, phone number, and institution in `demo_data/` is made up
(`demo_data/generate_demo_pdfs.py` regenerates it from scratch) — it's not derived
from or related to any real document.

## Portable Windows build (no Python required)

Every push to the `exe` branch builds a standalone `PDF Redactor.exe` via GitHub
Actions - no Python install needed on the machine running it. Grab it from the
repo's **Actions** tab -> pick the latest "Build Windows demo exe" run -> download
the `PDF-Redactor-windows` artifact (a zip). Unzip it anywhere and double-click
`PDF Redactor.exe` inside; your browser opens to the app automatically.

It starts blank (no PDFs or names lists baked in) - see
`packaging/DEMO_README.txt`, which ships alongside the exe, for what it does and
its limitations. To trigger a build without pushing a change, use the "Run
workflow" button on that same Actions page.

## Setup

```
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

OCR fallback (for scanned/image-only PDFs) also requires the `tesseract` and
`poppler` system binaries to be installed and on your PATH.

The web UI's interactive redaction viewer (section 5) uses **pdf.js**, vendored in
`static/pdfjs/` (`pdf.js` + `pdf.worker.js`, the legacy UMD build of
[pdfjs-dist](https://github.com/mozilla/pdf.js) 3.11.174, Apache-2.0). It's served
locally by Flask — no CDN, works offline. To upgrade, drop in the matching pair from
a newer `pdfjs-dist` legacy build.

> **Known issue:** name extraction (`extract_names.py`) depends on spaCy,
> which does not yet support Python 3.14 (fails on import with a pydantic
> `ConfigError`). Use Python 3.11 or 3.12 for that step until spaCy adds
> support. `redact_pymupdf_fast.py` itself has no such restriction.

## Workflow

### 1. Build a name list

```
python extract_names.py INPUT_PDF_OR_FOLDER -o names_to_redact.txt
```

- `INPUT_PDF_OR_FOLDER` can be a single PDF or a directory (scanned recursively).
- Add `--merge` to add newly-found names to an existing output file instead of
  overwriting it — use this to build up one cumulative list across many runs,
  e.g. `python extract_names.py ./new_pdfs -o MASTER_NAMES_LIST.txt --merge`.
- **Review the output file by hand before redacting.** NER is a heuristic —
  it will miss some names and flag some false positives (see `name_exceptions.txt`
  below for handling the latter).

### 2. Redact

Single file:

```
python redact_pymupdf_fast.py INPUT.pdf OUTPUT.pdf \
  --names_file names_to_redact.txt \
  --exceptions_file name_exceptions.txt \
  --boilerplate_graphics --boilerplate_text
```

Whole folder (mirrors the input tree into the output folder, appending `_REDACTED`):

```
python redact_pymupdf_fast.py INPUT_FOLDER OUTPUT_FOLDER \
  --names_file MASTER_NAMES_LIST.txt \
  --exceptions_file name_exceptions.txt \
  --boilerplate_graphics --boilerplate_text \
  --analysis_start_page 1 --analysis_end_page 10
```

Redact specific pages in full (e.g. a page that's a scanned image with no
extractable text — names/entities on it can't be found by text search):

```
python redact_pymupdf_fast.py INPUT.pdf OUTPUT.pdf \
  --names_file names_to_redact.txt --full_page_redact "8,10,12-13"
```

Flag reference:

| Flag | Purpose |
|---|---|
| `--names_file` | Text file, one name per line, to redact as whole-word matches (default: `names_to_redact.txt`) |
| `--exceptions_file` | Text file, one phrase per line, that must never be redacted even if it matches another rule |
| `--boilerplate_graphics` | Auto-detect and strip repeated header/footer logos/images |
| `--boilerplate_text` | Auto-detect and strip repeated header/footer text (page titles, running headers) |
| `--analysis_start_page` / `--analysis_end_page` | Page range sampled to learn what counts as "boilerplate" (default 1–10) |
| `--full_page_redact` | Comma/range list of pages to black out entirely, e.g. `"8,10,12-13"` |

Emails, phone numbers, and LinkedIn links/URLs are always redacted; this isn't
behind a flag.

### 3. Verify

Re-run name extraction against the redacted output to check nothing leaked
through:

```
python extract_names.py OUTPUT_FOLDER -o post_redaction_check.txt
```

If `post_redaction_check.txt` comes back non-empty, those names survived
redaction and need investigating (usually an OCR/scan page, or a name not in
the original list).

## Interactive redaction (web UI, section 5)

An alternative to the list-driven pipeline for one-off/manual work. Open a PDF in
the in-page pdf.js viewer, **select text directly on the page**, and choose per
selection:

- **Redact this occurrence** — blacks out only the selected spot (shown red).
- **Redact every occurrence + metadata** — blacks out that exact text everywhere
  in the document (shown yellow). Metadata is scrubbed on save regardless.

Click a highlight to remove it. "Redact → save output as" runs the same
`redact_pdf()` engine with `automated=False`, so **only** the manual selections
are applied (no name/email/boilerplate heuristics), then reloads the viewer on the
output. Same save semantics as everything else (content removed from the stream,
metadata scrubbed, old revisions dropped).

Prototype caveats: all pages render up front (slow for very large PDFs); the
yellow "other occurrences" outlines are a per-line preview only — the backend
`page.search_for` is what actually decides doc-wide matches, and it matches literal
(case-insensitive) text, so a term broken by a hard line break or hyphenation
won't be caught everywhere. Heavily rotated pages may be slightly misaligned.

## What "redacted" means here, precisely

- **Text and vector content**: removed from the page content stream via
  PyMuPDF's `apply_redactions()` — not overlaid.
- **Images**: pixels inside the redacted rectangle are blanked in place.
- **Standard metadata** (Title/Author/Subject/Creator/Producer/dates):
  overwritten to generic values.
- **XMP metadata** (the separate XML metadata stream) and any **embedded file
  attachments** are stripped.
- **Old revisions**: the output is a full rewrite (`garbage=4, clean=True`),
  so no prior incremental-save history survives in the file.

**Not currently handled** — the tool will warn on stderr if it finds optional
content layers (OCGs), but does not auto-flatten them, since doing so
automatically can corrupt complex PDFs. If a document has hidden/toggleable
layers, inspect it manually before treating the output as final. AcroForm
field values and document outline/bookmark titles also aren't scanned.

## Known limitations

- Signature detection (`find_signature_images` in `redact_pymupdf_fast.py`)
  is a heuristic: it blacks out a zone below any line containing a sign-off
  word ("regards", "sincerely", etc.), even if no actual signature is present
  there. It's intentionally over-inclusive; tighten `SIGNATURE_TRIGGER
  KEYWORDS` / the zone-size constants if it's redacting too much.
- Boilerplate detection is a statistical heuristic (things repeated on ≥70–90%
  of sampled pages). Always spot-check a redacted batch, especially the first
  and last few pages of each document.

## Contributing

If you use this tool on your own documents locally, keep any real extracted
name/email/phone lists and real PDFs out of commits to this repo — everything
here is meant to stay fabricated demo data. `scripts/pre-commit` guards
against accidentally committing anything that looks like real extracted data
(disallowed filenames like a stray PDF or `*_to_redact*.txt`, plus a content
scan of the staged diff for email/phone-shaped strings that aren't the
`@example.com` / `+353 1 555 0xxx` demo patterns); enable it as a real git
hook with:

```
ln -sf ../../scripts/pre-commit .git/hooks/pre-commit
```

`.claude/settings.json` also wires the same script in as a Claude Code
PreToolUse hook (`scripts/claude_git_guard.py`), so `git commit`/`git push`
run through Claude Code are checked even if the git hook above was never
installed, or was bypassed with `--no-verify`.

## License

PDF Redactor — a local-first PDF redaction tool.
Copyright (C) 2026 Sanvestek

Licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0)
or later. This is copyleft: you're free to use, modify, and redistribute this
tool (including running a modified version as a hosted service), but any
distributed or hosted modified version must also be released under the AGPL-3.0
with its source available. See the [LICENSE](LICENSE) file for the full text.
