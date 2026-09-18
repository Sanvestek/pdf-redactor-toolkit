PDF Redactor - portable demo build
===================================

HOW TO RUN
----------
Double-click "PDF Redactor.exe" in this folder. A console window opens (leave it
running) and your default browser opens automatically to the app. If the browser
doesn't open by itself, copy the address printed in the console window
(http://127.0.0.1:5000 or similar) into any browser.

To stop the app, close the console window or press Ctrl+C inside it.

WHAT'S IN THIS FOLDER
----------------------
Everything here is required - please don't delete or move files out of this
folder, and keep "PDF Redactor.exe" together with the rest of it.

The app starts blank: no PDFs or name lists are bundled. Files it creates as you
use it (names_to_redact.txt, emails_to_redact.txt, phones_to_redact.txt, an
uploads\ folder for anything you upload through the browser) are written next to
the .exe, in this same folder, so your work is easy to find again.

WORKFLOW
--------
1. Extract  - point it at a PDF or folder to find candidate names/emails/phones.
2. Review   - check the results, add/remove anything by hand, save the curated list.
3. Redact   - run the actual redaction against a PDF or folder using that list.
4. Verify   - re-scan the redacted output to confirm nothing leaked through.
5. Interactive - for one-off manual redaction: open a PDF, select text on the
   page directly, and redact just that selection (or every occurrence).

KNOWN LIMITATIONS
------------------
- OCR fallback for scanned/image-only PDFs needs the Tesseract and Poppler
  system tools installed separately and on your PATH - same requirement as the
  source install, this portable build doesn't change that.
- This is a local, single-user tool with no login and no built-in HTTPS. It's
  meant to be run on your own machine, not exposed to a network.
