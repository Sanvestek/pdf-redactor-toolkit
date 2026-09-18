# PyInstaller spec for the PDF Redactor demo bundle.
#
# Build with (from the repo root):
#   pyinstaller packaging/pdf_redactor.spec --noconfirm --clean
#
# This ships only app code plus the non-personal support files needed to run on a
# blank machine: the vendored pdf.js viewer (static/), the common-English-words
# dictionary used for confidence scoring, and the generic name_exceptions.txt
# starter list. No PDFs, no curated names/emails/phones lists, no uploads/ - those
# are case data, not app code, and are deliberately left out (see .gitignore for
# why they're not even in source control).
#
# Output mode is "onedir" (a folder containing the .exe + its libraries), not
# "onefile" - startup is near-instant since nothing needs to be unpacked to a temp
# directory on every launch, which matters a lot with spaCy/numpy's large payload.

import os

from PyInstaller.utils.hooks import collect_all

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(SPEC), ".."))

datas = [
    (os.path.join(REPO_ROOT, "static"), "static"),
    (os.path.join(REPO_ROOT, "common_english_words.txt"), "."),
    (os.path.join(REPO_ROOT, "name_exceptions.txt"), "."),
]
binaries = []
hiddenimports = ["spacy.lang.en"]

# Packages whose plugins/data/lang-modules are looked up dynamically (via catalogue's
# registry, entry points, or string-based imports) rather than being visible to
# PyInstaller's static import analysis - each needs its data + hidden imports pulled
# in explicitly or the frozen app breaks on first use even though `pip install` works.
for pkg in [
    "spacy",
    "spacy_legacy",
    "spacy_loggers",
    "srsly",
    "thinc",
    "blis",
    "wasabi",
    "weasel",
    "catalogue",
    "confection",
    "cymem",
    "murmurhash",
    "preshed",
    "cloudpathlib",
    "smart_open",
    "typer",
    "en_core_web_sm",
    "fitz",  # PyMuPDF
]:
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

a = Analysis(
    [os.path.join(REPO_ROOT, "redactor_webui.py")],
    pathex=[REPO_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PDF Redactor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="PDF Redactor",
)
