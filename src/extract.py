"""
extract.py — turn a Victorian legislation PDF into section-aware chunks.

Why this exists
----------------
Victorian "Authorised Version" legislation PDFs are formatted with a
running header (Act name + "No. X of YYYY" + Part), a footer
("Authorised by the Chief Parliamentary Counsel" + page number), and a
narrow left-hand margin column full of amendment history notes
("S. 35(1) amended by No. 38/2022 s. 91(1)."). None of that is part of
the actual law — but it's interleaved with the real text in a plain
text dump, so it has to be stripped before we chunk by section.

Approach
--------
1. Extract the whole PDF as plain text with `pdftotext`.
2. Skip the Table of Provisions at the front. We detect the real start
   of the Act's body by looking for the FIRST section heading whose
   number is "1" that is immediately followed by prose rather than
   another heading — in the ToC, a section's number and its title live
   in different text columns, so they are never glued together on one
   line the way they are in the real body text. This turns out to be a
   reliable, generic signal across all the Acts tested.
2. Walk the remaining lines. Track the current Part / Division from
   heading lines ("Part I—Offences", "Division 1—...") and start a new
   section chunk whenever we see a line matching
   `<section number> <Title Case heading text>`.
3. Drop lines that are running headers/footers or that match common
   margin-note vocabulary ("inserted by", "amended by", "S. 12(3)",
   "No. 47/2016", "s. 3(1),", standalone "*", etc).

Known limitation
----------------
This is a heuristic parser, not a real PDF layout parser. Some margin
note *fragments* (e.g. a single lowercase word like "vagina" that is
the tail end of a note like "S. 35(1) def. of vagina") aren't caught
by the vocabulary filter and can leak a stray word into a section's
text. Always spot check a few sections after running this (see the
`inspect` command in cli.py) before trusting the output.
"""

import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# A section heading: a number (optionally with trailing letters, e.g. "34AG",
# "319", "2B") followed by a Title-Case heading on the SAME line.
HEADING_RE = re.compile(r"^(\d+[A-Z]{0,3})\s+([A-Z][A-Za-z].{1,110})$")

# "Part I—Offences", "Part 1—Preliminary"
PART_RE = re.compile(r"^Part\s+([0-9A-Z]+)\s*[—-]\s*(.+)$")

# "Division 1—Offences against the person", "Division (4) Offences..."
DIVISION_RE = re.compile(r"^Division\s+([0-9A-Za-z()]+)\s*[—-]\s*(.+)$")

# A bare date like "2 September 2026" — matches HEADING_RE by accident
# because month names are Title Case. Filter these out explicitly.
_MONTHS = (
    "January|February|March|April|May|June|July|August|September|"
    "October|November|December"
)
DATE_FALSE_POSITIVE_RE = re.compile(rf"^\d+\s+(?:{_MONTHS})\s+\d{{4}}$")

# Lines that are pure margin-note / amendment-history noise.
MARGIN_NOISE_RES = [
    re.compile(r"^S\.\s*\d"),                      # "S. 35(1) def. of"
    re.compile(r"^Ss?\.\s*\d"),                     # "Ss. 12–14"
    re.compile(r"^Pt\s+\d"),                        # "Pt 1 Div. 1 Subdiv..."
    re.compile(r"^Div\.\s"),                        # "Div. 2 heading..."
    re.compile(r"^No\.?s?\s*\d+/\d{2,4}"),           # "No. 5/2018" / "Nos 68/2009"
    re.compile(r"^s\.\s*\d"),                       # "s. 3(1),"
    re.compile(r"^ss\.\s*\d"),                      # "ss 2, 7(a)"
    re.compile(r"^(inserted|amended|substituted|repealed|renumbered|"
               r"reinserted|new)\b", re.IGNORECASE),
    re.compile(r"^\*$"),                            # repealed-provision marker
    re.compile(r"^item\s+\d"),                      # "item 34)"
]

FOOTER_RE = re.compile(r"^Authorised by the Chief Parliamentary Counsel$")
PAGE_NUM_RE = re.compile(r"^[ivxlc]+$|^\d{1,4}$", re.IGNORECASE)
FORM_FEED = "\x0c"


@dataclass
class Section:
    act: str
    part: str
    division: str
    section: str
    heading: str
    text: str

    def to_dict(self):
        return asdict(self)


def run_pdftotext(pdf_path: Path) -> str:
    """Extract plain text from a PDF.

    Prefers poppler's `pdftotext` (best quality, matches the layout this
    parser was tuned against). Falls back to `pypdf` if poppler isn't
    installed, so this still works out of the box on a machine that
    hasn't installed poppler — see README for installing poppler for
    best results.
    """
    try:
        result = subprocess.run(
            ["pdftotext", str(pdf_path), "-"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return _pypdf_fallback(pdf_path)


def _pypdf_fallback(pdf_path: Path) -> str:
    from pypdf import PdfReader
    print(f"  (poppler's pdftotext not found — falling back to pypdf for "
          f"{pdf_path.name}; quality may be slightly lower, see README)")
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def detect_act_title(text: str) -> str:
    """Pull the Act's official name + year off the cover page.

    Cover pages look like:
        Authorised Version No. 322

        Crimes Act 1958
        No. 6231 of 1958
        Authorised Version incorporating amendments as at
    """
    m = re.search(
        r"Authorised Version No\.\s*\d+\s*\n\s*\n(.+?)\nNo\.\s*\d+ of \d{4}",
        text, re.DOTALL,
    )
    if m:
        title = " ".join(line.strip() for line in m.group(1).splitlines() if line.strip())
        return title
    return "Unknown Act"


def find_body_start(lines: list[str]) -> int:
    """Return the index of the first line of the Act's real body text.

    We look for a line matching HEADING_RE with section number "1" where
    the very next non-blank line is NOT itself a heading (i.e. it's real
    prose, not another entry in the Table of Provisions).
    """
    for i, line in enumerate(lines):
        m = HEADING_RE.match(line.strip())
        if not m or m.group(1) != "1":
            continue
        if DATE_FALSE_POSITIVE_RE.match(line.strip()):
            continue
        # look ahead to the next non-blank line
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines):
            continue
        next_line = lines[j].strip()
        if HEADING_RE.match(next_line) and not DATE_FALSE_POSITIVE_RE.match(next_line):
            continue  # still densely packed — looks like ToC, keep searching
        return i
    return 0  # fallback: couldn't detect, parse everything


def is_noise_line(line: str, act_header_lines: set[str]) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped == FORM_FEED or FORM_FEED in stripped and len(stripped) <= 2:
        return True
    if stripped in act_header_lines:
        return True
    if FOOTER_RE.match(stripped):
        return True
    if PAGE_NUM_RE.match(stripped) and len(stripped) <= 4:
        return True
    for pat in MARGIN_NOISE_RES:
        if pat.match(stripped):
            return True
    return False


def parse_sections(text: str, act_title: str) -> list[Section]:
    lines = text.split("\n")
    body_start = find_body_start(lines)
    lines = lines[body_start:]

    # Build the set of running-header lines to strip: the Act title line
    # itself and the "No. X of YYYY" line repeat on every page.
    no_of_match = re.search(r"No\.\s*\d+ of \d{4}", text)
    header_lines = {act_title}
    if no_of_match:
        header_lines.add(no_of_match.group(0))

    sections: list[Section] = []
    current_part = ""
    current_division = ""
    current_section = None
    current_heading = ""
    buffer: list[str] = []

    def flush():
        if current_section is not None:
            body = " ".join(buffer).strip()
            body = re.sub(r"\s+", " ", body)
            sections.append(Section(
                act=act_title,
                part=current_part,
                division=current_division,
                section=current_section,
                heading=current_heading,
                text=body,
            ))

    for raw_line in lines:
        stripped = raw_line.strip()

        if is_noise_line(raw_line, header_lines):
            continue

        part_m = PART_RE.match(stripped)
        if part_m:
            current_part = f"Part {part_m.group(1)}—{part_m.group(2)}"
            continue

        div_m = DIVISION_RE.match(stripped)
        if div_m:
            current_division = f"Division {div_m.group(1)}—{div_m.group(2)}"
            continue

        head_m = HEADING_RE.match(stripped)
        if head_m and not DATE_FALSE_POSITIVE_RE.match(stripped):
            # starting a new section — flush the previous one first
            flush()
            current_section = head_m.group(1)
            current_heading = head_m.group(2).strip()
            buffer = []
            continue

        buffer.append(stripped)

    flush()
    return sections


def extract_pdf(pdf_path: Path) -> list[Section]:
    text = run_pdftotext(pdf_path)
    act_title = detect_act_title(text)
    return parse_sections(text, act_title)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {RAW_DIR}. Drop your Act PDFs there first.")
        sys.exit(1)

    all_sections = []
    for pdf_path in pdf_files:
        print(f"Extracting {pdf_path.name} ...")
        sections = extract_pdf(pdf_path)
        print(f"  -> {len(sections)} sections found, act title detected as: "
              f"{sections[0].act if sections else '(none)'}")
        out_name = pdf_path.stem.lower().replace(" ", "_").replace(",", "") + ".json"
        out_path = PROCESSED_DIR / out_name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in sections], f, indent=2, ensure_ascii=False)
        print(f"  -> wrote {out_path}")
        all_sections.extend(sections)

    combined_path = PROCESSED_DIR / "all_sections.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in all_sections], f, indent=2, ensure_ascii=False)
    print(f"\nTotal sections across all Acts: {len(all_sections)}")
    print(f"Combined file: {combined_path}")


if __name__ == "__main__":
    main()
