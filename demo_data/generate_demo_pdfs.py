#!/usr/bin/env python3
"""Generates the synthetic demo PDFs shipped in demo_data/raw/, plus the matching
names/emails/phones lists in demo_data/, so the README's documented workflow has
something real to run against out of the box.

Every name, email, phone number, institution, and case reference below is entirely
fabricated for this demo - none of it refers to any real person, organisation, or
document. Re-run this script any time to regenerate the PDFs from scratch:

    python demo_data/generate_demo_pdfs.py
"""
import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "raw")

CASE_REF = "DEMO-2024-0001"
APPLICANT = "Alex Whitmore"
APPLICANT_EMAIL = "alex.whitmore@example.com"
APPLICANT_PHONE = "+353 1 555 0142"
APPLICANT_LINKEDIN = "linkedin.com/in/alex-whitmore-demo"
INSTITUTION = "Meridian Institute of Technology"

EVALUATORS = [
    ("Priya Nandakumar", "Tech Transfer Advisor", "Meridian Institute of Technology",
     "priya.nandakumar@example.com"),
    ("Tomasz Kaczmarek", "Industry Reviewer", "Northbridge Ventures",
     "tomasz.kaczmarek@example.com"),
]
EVALUATOR_PHONE = "+353 1 555 0199"

PANEL_MEMBERS = ["Samuel Whitcombe", "Elena Vasquez", "Marcus Delacroix"]

CITATION_AUTHORS = [
    ("Harborne", "C", 1), ("Delacote", "F", 2), ("Winslow", "J", 3),
]


def _draw_boilerplate(c, page_width, page_height, page_num, total_pages):
    """Repeated header/footer on every page - lets the demo exercise the tool's
    boilerplate-stripping feature (--boilerplate_graphics / --boilerplate_text)."""
    c.setFillColorRGB(0.85, 0.85, 0.9)
    c.rect(0, page_height - 20 * mm, page_width, 20 * mm, fill=1, stroke=0)
    c.setFillColorRGB(0.2, 0.2, 0.3)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(15 * mm, page_height - 13 * mm, "DEMO FUNDING PROGRAMME")
    c.setFont("Helvetica", 8)
    c.drawString(15 * mm, page_height - 18 * mm, "CONFIDENTIAL - SAMPLE DOCUMENT")

    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.setFont("Helvetica", 7)
    c.drawString(15 * mm, 10 * mm,
                 f"Demo Funding Programme - Internal Use Only - Page {page_num} of {total_pages}")
    c.setFillColorRGB(0, 0, 0)


def _new_page(c, page_num, total_pages):
    width, height = A4
    _draw_boilerplate(c, width, height, page_num, total_pages)
    return width, height


def _wrap_and_draw(c, text, x, y, width_chars=95, leading=13, font="Helvetica", size=10):
    import textwrap
    c.setFont(font, size)
    for line in textwrap.wrap(text, width_chars):
        c.drawString(x, y, line)
        y -= leading
    return y


def build_cfp_compiled():
    path = os.path.join(OUT_DIR, "CFP_Compiled_WPV_FINAL.pdf")
    c = canvas.Canvas(path, pagesize=A4)
    total_pages = 3

    # Page 1: label-form cover sheet
    width, height = _new_page(c, 1, total_pages)
    y = height - 40 * mm
    c.setFont("Helvetica-Bold", 13)
    c.drawString(20 * mm, y, "Compiled Funding Programme Application")
    y -= 12 * mm
    c.setFont("Helvetica", 10)
    c.drawString(20 * mm, y, f"Reference No: {CASE_REF}    Requested Budget: e250,000.00")
    y -= 8 * mm
    c.drawString(20 * mm, y,
                 f"Lead Applicant Name: {APPLICANT}    Institution Name: {INSTITUTION}")
    y -= 8 * mm
    c.drawString(20 * mm, y, f"Email: {APPLICANT_EMAIL}    Phone: {APPLICANT_PHONE}")
    y -= 8 * mm
    c.drawString(20 * mm, y, f"Applicant profile: {APPLICANT_LINKEDIN}")
    c.showPage()

    # Page 2: project description prose + a citation list
    width, height = _new_page(c, 2, total_pages)
    y = height - 40 * mm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, y, "Project Description")
    y -= 10 * mm
    body = (
        f"This application, led by {APPLICANT} at {INSTITUTION}, proposes a twelve-month "
        "feasibility study into low-power sensor calibration for industrial monitoring "
        "equipment. The work builds on prior published results and will be delivered in "
        "three work packages covering algorithm design, hardware integration, and field "
        "trials. Software Development: prototype firmware is implemented in stage one. "
        "Project Management: reporting follows the standard quarterly cycle."
    )
    y = _wrap_and_draw(c, body, 20 * mm, y)
    y -= 6 * mm
    authors = ", ".join(f"{surname} {initial}.{n}" for surname, initial, n in CITATION_AUTHORS)
    y = _wrap_and_draw(c, f"References: {authors}.", 20 * mm, y)
    c.showPage()

    # Page 3: signature block
    width, height = _new_page(c, 3, total_pages)
    y = height - 40 * mm
    c.setFont("Helvetica", 10)
    c.drawString(20 * mm, y, f"Signed: {APPLICANT}")
    y -= 8 * mm
    c.drawString(20 * mm, y, "Date: 2024-01-15")
    c.showPage()

    c.save()
    return path


def build_consolidated_evaluations():
    path = os.path.join(OUT_DIR, "Consolidated_Evaluations_For_Applicant_WPV.pdf")
    c = canvas.Canvas(path, pagesize=A4)
    total_pages = 2

    # Page 1: evaluator table
    width, height = _new_page(c, 1, total_pages)
    y = height - 40 * mm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, y, f"Consolidated Evaluations - {CASE_REF}")
    y -= 14 * mm

    col_x = [20 * mm, 90 * mm, 150 * mm]
    headers = ["Name", "Role", "Institution"]
    c.setFont("Helvetica-Bold", 10)
    for x, h in zip(col_x, headers):
        c.drawString(x, y, h)
    c.line(20 * mm, y - 3 * mm, 190 * mm, y - 3 * mm)
    y -= 10 * mm

    c.setFont("Helvetica", 10)
    for name, role, inst, _email in EVALUATORS:
        for x, val in zip(col_x, [name, role, inst]):
            c.drawString(x, y, val)
        y -= 8 * mm
    c.showPage()

    # Page 2: written evaluation comments
    width, height = _new_page(c, 2, total_pages)
    y = height - 40 * mm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, y, "Evaluator Comments")
    y -= 10 * mm
    for name, role, _inst, email in EVALUATORS:
        comment = (
            f"{name} ({role}) noted that the proposal is well scoped and the budget is "
            "realistic for the stated deliverables. Contact for follow-up questions: "
            f"{email}, {EVALUATOR_PHONE}."
        )
        y = _wrap_and_draw(c, comment, 20 * mm, y)
        y -= 6 * mm
    c.showPage()

    c.save()
    return path


def build_panel_recommendations():
    path = os.path.join(OUT_DIR, "Panel_Recommendations_WPV.pdf")
    c = canvas.Canvas(path, pagesize=A4)
    total_pages = 1

    width, height = _new_page(c, 1, total_pages)
    y = height - 40 * mm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, y, f"Panel Recommendation - {CASE_REF}")
    y -= 12 * mm
    body = (
        f"The panel reviewed the application submitted by {APPLICANT} on behalf of "
        f"{INSTITUTION} and, having considered the consolidated evaluations, recommends "
        "this application for funding. The panel commends the clarity of the work plan "
        "and the realistic budget."
    )
    y = _wrap_and_draw(c, body, 20 * mm, y)
    y -= 8 * mm
    c.setFont("Helvetica-Bold", 10)
    c.drawString(20 * mm, y, "Recommendation: FUNDED")
    y -= 10 * mm
    c.setFont("Helvetica", 10)
    y = _wrap_and_draw(c, "Panel: " + ", ".join(PANEL_MEMBERS) + ".", 20 * mm, y)
    c.showPage()

    c.save()
    return path


def write_redact_lists():
    names = sorted({
        APPLICANT,
        *[e[0] for e in EVALUATORS],
        *PANEL_MEMBERS,
        *[f"{surname}" for surname, _i, _n in CITATION_AUTHORS],
    })
    emails = sorted({APPLICANT_EMAIL, *[e[3] for e in EVALUATORS]})
    phones = sorted({APPLICANT_PHONE, EVALUATOR_PHONE})

    with open(os.path.join(HERE, "names_to_redact.txt"), "w") as f:
        f.write("\n".join(names) + "\n")
    with open(os.path.join(HERE, "emails_to_redact.txt"), "w") as f:
        f.write("\n".join(emails) + "\n")
    with open(os.path.join(HERE, "phones_to_redact.txt"), "w") as f:
        f.write("\n".join(phones) + "\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = [build_cfp_compiled(), build_consolidated_evaluations(), build_panel_recommendations()]
    write_redact_lists()
    for p in paths:
        print("wrote", p)
    print("wrote demo_data/names_to_redact.txt, emails_to_redact.txt, phones_to_redact.txt")


if __name__ == "__main__":
    main()
