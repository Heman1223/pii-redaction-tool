"""End-to-end tests: replacement consistency, tables, and DOCX integrity.

These cover the requirements that cannot be checked by looking at one string:
the same entity must map to the same fake value everywhere, two people sharing
a name must not be merged, table cells must be processed, and the written file
must still be a valid, openable DOCX.
"""

from pathlib import Path

import pytest
from docx import Document

from src.document_utils import iter_text_units, load_document
from src.evaluator import CategoryMetrics, evaluate
from src.redactor import redact_document
from src.replacements import ReplacementMap


# ---------------------------------------------------------------------------
# Consistent replacement
# ---------------------------------------------------------------------------


def test_same_entity_gets_same_fake_value():
    """The central requirement: repeated PII must redact identically."""
    mapping = ReplacementMap(seed=1)
    first = mapping.get("PERSON", "Rashi Patil")
    second = mapping.get("PERSON", "Rashi Patil")
    assert first == second


def test_case_and_spacing_variants_share_one_value():
    """Cover pages shout names in capitals; the body uses title case."""
    mapping = ReplacementMap(seed=1)
    assert mapping.get("PERSON", "RAJESH  KUSHAL HEGDE") == mapping.get(
        "PERSON", "Rajesh Kushal Hegde"
    )


def test_different_entities_get_different_values():
    mapping = ReplacementMap(seed=1)
    assert mapping.get("PERSON", "Rashi Patil") != mapping.get("PERSON", "Rohan Dey")


def test_same_name_different_role_gets_different_identity():
    """The same-name edge case from the brief.

    Two different people both called "Rahul Sharma" are told apart by the role
    stated beside each one, so they receive separate fake identities.
    """
    mapping = ReplacementMap(seed=1)
    director = mapping.get("PERSON", "Rahul Sharma", context="director")
    cfo = mapping.get("PERSON", "Rahul Sharma", context="chief financial officer")
    assert director != cfo


def test_same_name_same_role_is_merged_known_limitation():
    """Documented limitation: identical name AND role cannot be separated.

    With no distinguishing context there is no signal to split on, so the two
    people are merged. This test pins the behaviour so it stays a known,
    deliberate trade rather than an accident.
    """
    mapping = ReplacementMap(seed=1)
    a = mapping.get("PERSON", "Rahul Sharma", context="director")
    b = mapping.get("PERSON", "Rahul Sharma", context="director")
    assert a == b


def test_replacements_are_type_compatible():
    mapping = ReplacementMap(seed=7)
    assert "@" in mapping.get("EMAIL", "someone@example.org")
    assert mapping.get("PHONE", "+91 9876543210").startswith("+91")
    assert mapping.get("IP_ADDRESS", "10.0.0.1").startswith("192.0.2.")
    assert mapping.get("SSN", "123-45-6789").count("-") == 2


def test_replacements_are_deterministic_across_runs():
    """A fixed seed keeps the evaluation report reproducible."""
    a = ReplacementMap(seed=42).get("PERSON", "Rashi Patil")
    b = ReplacementMap(seed=42).get("PERSON", "Rashi Patil")
    assert a == b


# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------


@pytest.fixture
def docx_with_table(tmp_path) -> Path:
    """A document whose PII lives only inside a table."""
    doc = Document()
    doc.add_paragraph("Directory of contacts follows.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Name"
    table.cell(0, 1).text = "Email"
    table.cell(1, 0).text = "Priya Nair"
    table.cell(1, 1).text = "priya.nair@example.org"

    path = tmp_path / "table.docx"
    doc.save(path)
    return path


def test_table_cells_are_processed(docx_with_table, tmp_path):
    """PII inside tables must be redacted, not just document.paragraphs."""
    out = tmp_path / "table_redacted.docx"
    result = redact_document(str(docx_with_table), str(out))

    assert "EMAIL" in result.counts_by_type

    text = "\n".join(u.text for u in iter_text_units(load_document(str(out))))
    assert "priya.nair@example.org" not in text


def test_output_is_a_valid_openable_docx(docx_with_table, tmp_path):
    out = tmp_path / "valid.docx"
    redact_document(str(docx_with_table), str(out))
    assert out.exists() and out.stat().st_size > 0
    Document(str(out))  # raises if the file is corrupt


def test_original_document_is_not_modified(docx_with_table, tmp_path):
    before = docx_with_table.read_bytes()
    redact_document(str(docx_with_table), str(tmp_path / "copy.docx"))
    assert docx_with_table.read_bytes() == before


def test_consistency_across_the_whole_document(tmp_path):
    """One person mentioned three times must redact to one fake name."""
    doc = Document()
    doc.add_paragraph("Rashi Patil joined the team.")
    doc.add_paragraph("Later, Rashi Patil was promoted.")
    doc.add_paragraph("Finally Rashi Patil resigned.")
    src = tmp_path / "repeat.docx"
    doc.save(src)

    out = tmp_path / "repeat_redacted.docx"
    result = redact_document(str(src), str(out))

    text = "\n".join(u.text for u in iter_text_units(load_document(str(out))))
    assert "Rashi Patil" not in text

    # Exactly one person was seen, so exactly one fake identity was created...
    person_fakes = {
        fake for key, fake in result.replacement_map.items()
        if key.startswith("PERSON:")
    }
    assert len(person_fakes) == 1

    # ...and it must appear in all three sentences.
    assert text.count(person_fakes.pop()) == 3


def test_images_are_neutralised(tmp_path):
    """Images cannot be read without OCR, so they are blanked wholesale.

    The reference prospectus hides a PAN card - name, father's name, date of
    birth and PAN number - inside an image where no text rule can reach it.
    """
    from docx.shared import Inches
    import base64, io

    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
        "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )
    doc = Document()
    doc.add_paragraph("See the attached identity document.")
    doc.add_picture(io.BytesIO(png), width=Inches(1))
    src = tmp_path / "with_image.docx"
    doc.save(src)

    result = redact_document(str(src), str(tmp_path / "img_redacted.docx"))
    assert result.images_found >= 1
    assert result.images_blanked == result.images_found


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def test_metrics_arithmetic():
    m = CategoryMetrics(tp=8, fp=2, fn=2)
    assert m.precision == pytest.approx(0.8)
    assert m.recall == pytest.approx(0.8)
    assert m.f1 == pytest.approx(0.8)
    assert m.accuracy == pytest.approx(8 / 12)   # TP / (TP + FP + FN)


def test_metrics_are_zero_when_nothing_predicted():
    m = CategoryMetrics(tp=0, fp=0, fn=5)
    assert m.precision == 0.0 and m.recall == 0.0 and m.f1 == 0.0


def test_phone_formatting_differences_still_match():
    report = evaluate(
        [("+91 20 4505 3237", "PHONE")], [("+91 20 45053237", "PHONE")]
    )
    assert report.overall.tp == 1 and report.overall.fp == 0


def test_dash_style_differences_still_match():
    """En-dash vs hyphen must not be scored as an error."""
    report = evaluate(
        [("Pune – 411 044, Maharashtra, India", "ADDRESS")],
        [("Pune - 411 044, Maharashtra, India", "ADDRESS")],
    )
    assert report.overall.tp == 1 and report.overall.fp == 0


def test_address_fragments_count_as_one_detection():
    """A multi-paragraph address split into fragments is one true positive."""
    report = evaluate(
        [("12 Example Road", "ADDRESS"), ("Pune - 411 001", "ADDRESS")],
        [("12 Example Road, Pune - 411 001", "ADDRESS")],
    )
    assert report.overall.tp == 1
    assert report.overall.fp == 0


def test_wrong_type_is_not_a_match():
    report = evaluate([("Acme Ltd", "PERSON")], [("Acme Ltd", "COMPANY")])
    assert report.overall.tp == 0
    assert report.overall.fp == 1 and report.overall.fn == 1
