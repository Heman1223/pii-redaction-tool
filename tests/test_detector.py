"""Detection tests, including the false-positive cases that matter most.

Recall tests ("does it find an email?") are easy and rarely fail. The tests
that earn their keep here are the negative ones: invoice numbers shaped like
SSNs, ticket IDs that pass a Luhn check, version strings shaped like IPs, and
the 270 ordinary dates in the reference document that must not become dates of
birth.
"""

import pytest

from src.detector import (
    detect_contact_persons,
    detect_entities,
    detect_regex_entities,
    luhn_valid,
    resolve_overlaps,
    Entity,
)


def types_of(entities):
    return {e.type for e in entities}


def texts_of(entities, etype):
    return {e.text for e in entities if e.type == etype}


# ---------------------------------------------------------------------------
# Structured types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("Email: rashhi.patil@gmail.com", "rashhi.patil@gmail.com"),
    ("Write to cs.connect@kshinternational.com today", "cs.connect@kshinternational.com"),
    ("kshinternational.ipo@in.mpms.mufg.com is the registrar", "kshinternational.ipo@in.mpms.mufg.com"),
])
def test_email_detection(text, expected):
    assert expected in texts_of(detect_regex_entities(text), "EMAIL")


@pytest.mark.parametrize("text", [
    "Telephone: +91 20 45053237",
    "Tel: +91 22 4009 4400",
    "Call + 91 8879770456 now",          # space after the plus
    "Mobile: +91-9876543210",
    "Landline + 91 (20) 6729 5100",      # parenthesised area code
])
def test_phone_formats(text):
    assert texts_of(detect_regex_entities(text), "PHONE")


def test_phone_requires_country_code():
    """Bare 10-digit runs are not phones - a deliberate precision choice."""
    entities = detect_regex_entities("Total shares allotted: 26704570 units")
    assert not texts_of(entities, "PHONE")


def test_ssn_detected():
    assert "123-45-6789" in texts_of(detect_regex_entities("SSN: 123-45-6789"), "SSN")


def test_ssn_not_matched_for_invoice_number():
    """An invoice number has the same shape as an SSN; context disambiguates."""
    entities = detect_regex_entities("Invoice number 456-78-9012 was raised")
    assert not texts_of(entities, "SSN")


def test_credit_card_detected():
    entities = detect_regex_entities("Card on file: 4111 1111 1111 1111")
    assert texts_of(entities, "CREDIT_CARD")


def test_credit_card_rejects_non_luhn():
    """Long digit runs that fail the checksum are not card numbers."""
    assert not luhn_valid("1234567890123")
    entities = detect_regex_entities("Reference 1234567890123 applies")
    assert not texts_of(entities, "CREDIT_CARD")


def test_credit_card_not_matched_for_ticket_id():
    entities = detect_regex_entities("Reference ticket ID 998877665544332211 is open")
    assert not texts_of(entities, "CREDIT_CARD")


def test_ip_detected():
    assert "192.168.14.22" in texts_of(
        detect_regex_entities("Last login from IP 192.168.14.22"), "IP_ADDRESS"
    )


def test_ip_not_matched_for_version_string():
    entities = detect_regex_entities("Software build version 10.2.14.3 deployed")
    assert not texts_of(entities, "IP_ADDRESS")


def test_ip_rejects_out_of_range_octets():
    assert not texts_of(detect_regex_entities("Value 999.1.1.1 here"), "IP_ADDRESS")


# ---------------------------------------------------------------------------
# Date of birth: the highest-value false-positive guard in the tool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "Date of Birth: December 10, 1985",
    "DOB: 15/03/1992",
    "Born on 2 February 1975",
])
def test_dob_detected_with_context(text):
    assert texts_of(detect_regex_entities(text), "DOB")


@pytest.mark.parametrize("text", [
    "Dated December 10, 2025",
    "incorporated on July 30, 1979 under the Companies Act, 1956",
    "The policy was approved by our Board on May 17, 2025",
    "Bonus issue undertaken on February 21, 2025",
])
def test_ordinary_dates_are_not_dob(text):
    """The reference document holds 270 dates and no text-layer DOB.

    An ungated date rule would emit ~270 false positives here, which is why
    DOB detection requires a birth-context keyword.
    """
    assert not texts_of(detect_regex_entities(text), "DOB")


# ---------------------------------------------------------------------------
# Addresses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "Registered Office: 11/3, 11/4 and 11/5, Village Birdewadi, Chakan Taluka - Khed, Pune - 410 501, Maharashtra, India",
    "Address: 1600 Pennsylvania Avenue NW, Washington, DC 20500, United States",
    "Office: 221B Baker Street, Marylebone, London, NW1 6XE, United Kingdom",
])
def test_addresses_across_countries(text):
    assert texts_of(detect_entities(text, use_ner=False), "ADDRESS")


@pytest.mark.parametrize("text", [
    "Peer review number: 014680",
    "Registration number: 141032",
    "Firm registration number: 105215W/ W100057",
    "Corporate identity number: U28129PN1979PLC141032",
])
def test_identifiers_are_not_addresses(text):
    """Six-digit identifiers must not be mistaken for postcodes."""
    assert not texts_of(detect_entities(text, use_ner=False), "ADDRESS")


# ---------------------------------------------------------------------------
# Contact-person rule
# ---------------------------------------------------------------------------


def test_contact_person_list_is_split():
    """NER returns this as one mangled span; the rule recovers every name."""
    text = "Contact Person: Eric Bacha/ Sachin Gawade/ Pravin Teli"
    names = texts_of(detect_contact_persons(text), "PERSON")
    assert {"Eric Bacha", "Sachin Gawade", "Pravin Teli"} <= names


def test_contact_person_stops_at_next_label():
    text = "Contact Person: Shanti Gopalkrishnan Website: www.example.com"
    names = texts_of(detect_contact_persons(text), "PERSON")
    assert "Shanti Gopalkrishnan" in names
    assert not any("www" in n for n in names)


# ---------------------------------------------------------------------------
# Overlap resolution
# ---------------------------------------------------------------------------


def test_person_inside_email_is_not_separately_detected():
    """The core overlap case from the brief.

    "rashhi.patil@gmail.com" must be one EMAIL, never an email plus a PERSON
    found inside it - otherwise replacement corrupts the substituted value.
    """
    text = "Contact Rashi Patil at rashhi.patil@gmail.com for details"
    entities = detect_entities(text)

    emails = [e for e in entities if e.type == "EMAIL"]
    assert len(emails) == 1

    email = emails[0]
    for other in entities:
        if other is not email:
            assert not other.overlaps(email)


def test_higher_priority_type_wins_overlap():
    email = Entity("a@b.com", "EMAIL", 0, 7)
    person = Entity("a", "PERSON", 0, 1)
    kept = resolve_overlaps([person, email])
    assert [e.type for e in kept] == ["EMAIL"]


def test_non_overlapping_entities_all_kept():
    a = Entity("a@b.com", "EMAIL", 0, 7)
    b = Entity("192.0.2.1", "IP_ADDRESS", 20, 29)
    assert len(resolve_overlaps([a, b])) == 2
