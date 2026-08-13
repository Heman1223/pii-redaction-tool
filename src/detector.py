"""PII detection: deterministic rules for structured data, NER for the rest.

The split is deliberate. Emails, phones, SSNs, credit cards and IPs have exact
shapes, so a regex is both more accurate and more explainable than a model.
Names, organisations and addresses have no shape at all, so they need NER.

Dates are the interesting case. The reference prospectus contains 270 ordinary
dates ("December 10, 2025") and, in its text layer, zero dates of birth. An
ungated date rule would produce ~270 false positives and destroy precision, so
DOB is only matched when a birth-context keyword sits next to it.
"""

import re
from dataclasses import dataclass, replace
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Entity model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Entity:
    """One detected span of PII within a single text unit."""

    text: str
    type: str
    start: int
    end: int
    source: str = "regex"       # "regex" or "ner", used for overlap resolution
    context: Optional[str] = None  # e.g. a job title, used to tell people apart
    unit: int = -1              # index of the text unit; set by the pipeline

    def overlaps(self, other: "Entity") -> bool:
        return self.start < other.end and other.start < self.end


# ---------------------------------------------------------------------------
# Deterministic rules
# ---------------------------------------------------------------------------

# Ordered by priority: earlier patterns win when spans overlap. EMAIL is first
# so that "rashi.patil@gmail.com" is never split into a name plus a domain.
REGEX_RULES: List[tuple] = [
    (
        "EMAIL",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ),
    (
        # Indian and international formats. The document uses "+91 20 45053237",
        # "+ 91 8879770456" (space after the plus), "+91 22 4009 4400" and
        # "+91-9876543210", so the separator handling has to be permissive.
        # A leading country code is required. Without it, any 10-digit run in a
        # financial document - share counts, amounts - would be read as a phone
        # number, and this corpus is full of them. The cost is that bare local
        # numbers such as "022-68052182" are missed; that trade is documented
        # in the README as a deliberate precision-over-recall choice.
        "PHONE",
        re.compile(
            r"(?<![\w.])"
            r"(?:\+\s?\d{1,3}[\s\-]?)"          # country code, required
            r"(?:[\d(][\d\s\-()]{7,16}\d)"      # body, may contain "(20)"
            # Must not run into more digits, but a sentence-ending full stop is
            # fine: "+91 9876543210." is still a phone number.
            r"(?!\d)(?!\.\d)"
        ),
    ),
    (
        "SSN",
        re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
    ),
    (
        # 13-19 digits, optionally grouped. Validated with Luhn below, which is
        # what stops long financial figures being mistaken for card numbers.
        "CREDIT_CARD",
        re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
    ),
    (
        "IP_ADDRESS",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
        ),
    ),
    (
        # Indian PAN. Present in this corpus only inside an image, but the rule
        # is cheap and makes the tool correct on text-layer PANs elsewhere.
        "PAN",
        re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    ),
    (
        # Director Identification Number: a government identifier for a natural
        # person, so it is treated as PII. Company identifiers (CIN) and SEBI
        # registration numbers are deliberately NOT matched - see README.
        "DIN",
        re.compile(r"(?i)\bDIN\b[:\s]*\b(\d{8})\b"),
    ),
]

# Some strings are shaped exactly like PII but are not PII. An invoice number
# has the same shape as an SSN; a ticket ID can pass a Luhn check by chance; a
# software version looks like an IPv4 address. Shape alone cannot separate
# these, so we look at the words immediately before the match and reject it if
# they identify the value as something else.
#
# This is a negative guard rather than a positive one on purpose: requiring a
# keyword before every SSN would lose bare SSNs, which is a worse trade.
NEGATIVE_CONTEXT = {
    "SSN": re.compile(
        r"(?i)\b(invoice|order|ticket|reference|ref|receipt|docket|case|file|"
        r"account|policy|claim)\b[\s:#]*(?:no\.?|number|id)?[\s:#]*$"
    ),
    "CREDIT_CARD": re.compile(
        r"(?i)\b(ticket|reference|ref|order|invoice|share|shares|unit|units|"
        r"quantity|qty|account|folio|transaction|txn|id)\b[\s:#]*"
        r"(?:no\.?|number|id)?[\s:#]*$"
    ),
    "IP_ADDRESS": re.compile(
        r"(?i)\b(version|build|release|ver|v|revision|clause|section|rule|"
        r"regulation)\b[\s:.#]*$"
    ),
}

# How much text before a match to inspect for a disqualifying keyword.
CONTEXT_WINDOW = 40


def _rejected_by_context(label: str, text: str, start: int) -> bool:
    """True if the words just before this match identify it as non-PII."""
    guard = NEGATIVE_CONTEXT.get(label)
    if guard is None:
        return False
    return bool(guard.search(text[max(0, start - CONTEXT_WINDOW): start]))


# A date next to one of these words is a date of birth; a date anywhere else is
# just a date. This single gate is what keeps DOB precision at 1.00.
DOB_CONTEXT = re.compile(
    r"(?i)\b(date\s+of\s+birth|d\.?o\.?b\.?|born\s+on|born|birth\s*date)\b"
)

DATE_PATTERNS = [
    re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2},?\s+\d{4}\b"
    ),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b"),
    re.compile(
        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4}\b"
    ),
]

# Job titles are the strongest signal that a nearby name is a real person, and
# they double as the context used to tell two same-named people apart.
_ROLE_BASE = (
    r"(?:Joint Managing Director|Managing Director|Whole[- ]?time Director|"
    r"Independent Director|Non[- ]Executive Director|Executive Director|"
    r"Technical Director|Chief Executive Officer|Chief Financial Officer|"
    r"Chief Operating Officer|Company Secretary|Compliance Officer|"
    r"Contact Person|Senior Manager|General Manager|Vice President|"
    r"Chairman|President|Manager|Partner|Principal|Founder|Trustee|"
    r"Proprietor|Treasurer|Promoter|Director|Officer|CEO|CFO|COO|CS)"
)

# Compound titles are matched as ONE role. Without this, "Company Secretary and
# Compliance Officer" matched as two separate roles, and the same person was
# split into two identities depending on which half sat nearer their name.
ROLE_PATTERN = re.compile(
    r"(?i)\b" + _ROLE_BASE + r"(?:\s*(?:and|&|/)\s*" + _ROLE_BASE + r")?\b"
)

# Entity types that are never worth redacting even if NER flags them. These are
# public bodies, regulators and market infrastructure - redacting them would be
# noise, not privacy.
ORG_STOPLIST = {
    "sebi", "securities and exchange board of india", "bse", "nse",
    "bse limited", "national stock exchange of india limited", "rbi",
    "reserve bank of india", "nsdl", "cdsl", "npci", "roc", "icai", "mca",
    "government of india", "the companies act", "companies act", "supreme court",
    "high court", "income tax department", "gst", "ind as", "ifrs", "us gaap",
    "registrar of companies", "ministry of corporate affairs", "india",
}

# Legal-entity markers. An ORG containing one of these is almost certainly a
# real company name.
COMPANY_SUFFIXES = re.compile(
    r"(?i)\b(limited|ltd|private|pvt|llp|inc|incorporated|corporation|corp|"
    r"company|co|plc|gmbh|holdings|industries|technologies|enterprises|"
    r"solutions|services|bank|securities|capital|associates|partners|trust|"
    r"foundation|group)\b"
)

# Words that mean the "organisation" spaCy found is really a document heading,
# a form label, or a section title. These caused false positives such as
# "Contact Directory", "Software" and "SSN" being redacted as company names.
DOC_STRUCTURE_WORDS = re.compile(
    r"(?i)\b(directory|record|records|table|section|chapter|annexure|schedule|"
    r"appendix|summary|overview|report|form|heading|page|note|notes|list|"
    r"details|information|document|prospectus|statement|policy|software|"
    r"version|build|act|division|department)\b"
)


# Financial and legal vocabulary that a prospectus writes in Title Case. spaCy
# reads Title Case as a proper noun and tags these as organisations or people:
# "Equity Shares" was detected 37 times as a company, "UPI Bidders" as a person.
# None of them are PII.
DOMAIN_TERMS = re.compile(
    r"(?i)\b(equity|share|shares|bidder|bidders|investor|investors|offer|bid|"
    r"price|fund|funds|allotment|allottee|portion|proceeds|prospectus|"
    r"circular|regulation|regulations|committee|board|promoter|shareholder|"
    r"shareholders|intermediaries|intermediary|auditor|auditors|agent|agents|"
    r"facility|escrow|syndicate|underwriter|underwriters|depository|"
    r"registrar|account|amount|capital|date|period|form|slip|mechanism|"
    r"portion|net|gross|cap|floor|anchor|retail|institutional|working|"
    r"material|policy|scheme|act|rules|meeting|resolution|office)\b"
)

# Words that mean a spaCy PERSON span is really a collective noun or a company
# fragment: "Key Managerial Personnel", "Waterloo Industrial", "Everest Family
# Trust". Kept separate from DOMAIN_TERMS on purpose - DOMAIN_TERMS also gates
# company detection, and putting "management" there rejected the real company
# "Nuvama Wealth Management Limited".
PERSON_NOISE = re.compile(
    r"(?i)\b(personnel|managerial|management|industrial|key|park|trust|fund|"
    r"branch|group|committee|department|division|team)\b"
)


# Words that introduce a company name without being part of it, e.g.
# "Formerly Link Intime India Private Limited" or "Company KSH International
# Limited". Stripping them keeps the replacement aligned with the real name.
LEADING_NOISE = {
    "the", "a", "an", "and", "of", "to", "by", "with", "from", "our",
    "formerly", "erstwhile", "namely", "being", "viz", "company", "companies",
    "statutory", "auditors", "auditor", "registrar", "bankers", "banker",
    "counsel", "manager", "managers", "member", "members", "sponsor",
}


def clean_company_span(name: str) -> str:
    """Trim leading words that spaCy swallowed into an ORG span.

    spaCy returned "the Offer Escrow Collection Bank HDFC Bank Limited" as one
    organisation. Dropping leading articles and lower-case words recovers the
    part that is actually a name.
    """
    tokens = name.split()
    while tokens and (tokens[0].lower().strip(",") in LEADING_NOISE
                      or tokens[0][:1].islower()):
        tokens.pop(0)
    return " ".join(tokens)


def _is_plausible_company(name: str) -> bool:
    """Keep only ORG spans that carry a legal-entity suffix.

    This is the single highest-impact precision rule in the tool, and it is
    domain-driven rather than a hand-maintained blocklist.

    A red herring prospectus capitalises its defined terms everywhere - "Equity
    Shares", "Anchor Investors", "Promoter Group", "Statutory Auditors" - and
    spaCy tags almost all of them as organisations. Real company names in the
    same document essentially always end in a legal form: Limited, Private
    Limited, LLP, Trust, Bank, Securities, N.A.

    Requiring that suffix removes hundreds of false positives at the cost of
    missing companies referred to by a bare brand name ("Nuvama" on its own,
    where "Nuvama Wealth Management Limited" is still caught). That trade is
    right here: over-redacting ordinary business terminology would make the
    document unreadable and is explicitly called out in the brief.
    """
    tokens = name.split()
    if len(tokens) < 2:
        return False

    if DOC_STRUCTURE_WORDS.search(name):
        return False

    # "Bank Limited" and "Private Limited" are suffixes with no name attached -
    # truncated spans, not companies. Require at least one distinguishing word.
    if all(COMPANY_SUFFIXES.fullmatch(t.strip(".,&")) for t in tokens):
        return False

    # Financial vocabulary anywhere in the span means it describes a *role*,
    # not a company: "Public Offer Account Bank", "Self-Certified Syndicate
    # Bank", "Statutory Auditors, Kirtane & Pandit". Real company names in this
    # corpus do not contain these words.
    if DOMAIN_TERMS.search(name):
        return False

    # Require at least one substantive word that is not itself a legal suffix,
    # so "U.S. Securities" and "Bank Limited" are rejected while "Kanj & Co.
    # LLP" survives.
    # Count letters, not characters: "U.S." has three characters but only two
    # letters, so it is an abbreviation rather than a company's distinguishing
    # name.
    substantive = [
        t for t in tokens
        if sum(c.isalpha() for c in t) >= 3
        and not COMPANY_SUFFIXES.fullmatch(t.strip(".,&"))
    ]
    if not substantive:
        return False

    return bool(COMPANY_SUFFIXES.search(name))


# Address and premises vocabulary. spaCy tagged "Bandra Kurla Complex" as a
# person; these words identify a place, so a candidate containing one is not a
# personal name.
PLACE_WORDS = re.compile(
    r"(?i)\b(complex|marg|road|street|lane|floor|tower|building|house|centre|"
    r"center|campus|park|nagar|society|apartment|plot|wing|block|station|"
    r"chambers|estate|colony|broker|bank|department|division|hall)\b"
)

# "Contact Person: Eric Bacha/ Sachin Gawade/ Pravin Teli" is the document's
# standard way of listing people, and spaCy consistently mangles it - it
# returned only "Eric Bacha/" and dropped the other four. The label itself is a
# reliable signal, so everything after it is parsed as a list of names.
CONTACT_PERSON = re.compile(r"(?i)\bcontact\s+person\s*:?\s*(.+)")

# Splits that list separator: slashes, commas, and the word "and".
NAME_SEPARATORS = re.compile(r"\s*(?:/|,|\band\b)\s*")


def detect_contact_persons(text: str) -> List[Entity]:
    """Extract names from an explicit "Contact Person:" label.

    This is a targeted rule rather than a general one. It fires only when the
    document has already declared that what follows is a person, which makes it
    both high precision and high recall for the pattern that defeats NER.
    """
    match = CONTACT_PERSON.search(text)
    if not match:
        return []

    tail_start = match.start(1)
    tail = match.group(1)

    # Stop at the next field label, so "Contact Person: X Website: y" does not
    # swallow the website.
    cut = re.search(
        r"(?i)\b(website|email|e-mail|telephone|tel|sebi|cin|fax)\b", tail
    )
    if cut:
        tail = tail[: cut.start()]

    found: List[Entity] = []
    offset = 0
    for part in NAME_SEPARATORS.split(tail):
        name = part.strip()
        if not name:
            offset += len(part) + 1
            continue

        position = tail.find(name, offset)
        if position >= 0 and _is_plausible_person(name):
            start = tail_start + position
            found.append(
                Entity(text=name, type="PERSON", start=start,
                       end=start + len(name), source="rule-contact",
                       context=person_context(text, start, start + len(name)))
            )
        offset = max(offset, position + len(name)) if position >= 0 else offset

    return found


def _is_plausible_person(name: str) -> bool:
    """Reject spaCy PERSON spans that are really domain terms or companies."""
    tokens = name.split()

    if len(tokens) < 2 or len(tokens) > 6:
        return False

    if PLACE_WORDS.search(name):
        return False

    # Names are capitalised; this drops fragments like "a Registered Broker".
    if not all(t[0].isupper() for t in tokens if t and t[0].isalpha()):
        return False

    # "Waterloo Industrial Park VI Private Limited" is a company, not a person.
    if COMPANY_SUFFIXES.search(name) or DOC_STRUCTURE_WORDS.search(name):
        return False

    # "UPI Bidders", "Mutual Funds", "Cap Price", "Supa Facility".
    if DOMAIN_TERMS.search(name) or PERSON_NOISE.search(name):
        return False

    # Real names are capitalised words, optionally with initials or particles.
    return all(t[0].isupper() or t[0].isdigit() is False for t in tokens if t)


def split_person_list(entity: "Entity") -> List["Entity"]:
    """Split "A/B" contact lists into individual people.

    The document lists joint contacts as "Kishan Rastogi/Abhijit Diwan" and
    "Eric Bacha/ Sachin Gawade/ Pravin Teli". spaCy returns those as one span,
    which is wrong twice over: it invents a person who does not exist, and it
    hides the real ones. Splitting fixes precision and recall together.
    """
    if "/" not in entity.text:
        return [entity]

    parts: List[Entity] = []
    # finditer over non-slash runs gives each segment's offset directly, which
    # avoids the index arithmetic that a split()-and-count approach needs.
    for match in re.finditer(r"[^/]+", entity.text):
        segment = match.group()
        name = segment.strip()
        if not name:
            continue

        # Offset of the trimmed name within the original text unit.
        start = entity.start + match.start() + (len(segment) - len(segment.lstrip()))
        parts.append(
            Entity(text=name, type="PERSON", start=start, end=start + len(name),
                   source="ner", context=entity.context)
        )

    return parts


def luhn_valid(number: str) -> bool:
    """Standard Luhn checksum used by real credit card numbers.

    Without this, any 13-19 digit run - share counts, financial totals - would
    be redacted as a card number.
    """
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum, parity = 0, len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def detect_regex_entities(text: str) -> List[Entity]:
    """Run every deterministic rule over one text unit."""
    found: List[Entity] = []

    for label, pattern in REGEX_RULES:
        for match in pattern.finditer(text):
            # DIN captures the number in group 1 so the word "DIN" survives.
            if label == "DIN":
                span, value = match.span(1), match.group(1)
            else:
                span, value = match.span(), match.group()

            if label == "CREDIT_CARD" and not luhn_valid(value):
                continue

            if _rejected_by_context(label, text, span[0]):
                continue

            found.append(
                Entity(text=value, type=label, start=span[0], end=span[1])
            )

    found.extend(_detect_dob(text))
    return found


def _detect_dob(text: str) -> List[Entity]:
    """Match dates only when a birth-context keyword appears nearby.

    The window is the whole text unit, which for this document means a table
    cell or a paragraph - tight enough that "Date of Birth" in one row does not
    leak onto dates in another.
    """
    if not DOB_CONTEXT.search(text):
        return []

    return [
        Entity(text=m.group(), type="DOB", start=m.start(), end=m.end())
        for pattern in DATE_PATTERNS
        for m in pattern.finditer(text)
    ]


# ---------------------------------------------------------------------------
# NER
# ---------------------------------------------------------------------------

_NLP = None

# spaCy label -> our category.
#
# GPE, LOC and FAC are deliberately NOT mapped. A bare place name ("Pune",
# "Maharashtra") is not PII on its own, and spaCy's FAC label produced pure
# noise on this document - "the Bid/Offer Closing Date" and "the Bid cum
# Application Form" were both tagged as facilities. Real mailing addresses are
# found by the PIN-code rule further down, which is far more precise.
NER_LABEL_MAP = {
    "PERSON": "PERSON",
    "ORG": "COMPANY",
}


def get_nlp():
    """Load spaCy lazily so tests that only exercise regex stay fast.

    Only the components NER depends on are kept. The parser, tagger and
    lemmatizer are pure overhead here and disabling them roughly halves
    processing time on a 400-page document.
    """
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load(
            "en_core_web_sm",
            disable=["parser", "lemmatizer", "attribute_ruler", "tagger"],
        )
    return _NLP


def detect_ner_entities_batch(
    texts: List[str], progress=None, chunk_size: int = 250
) -> List[List[Entity]]:
    """Run NER over many text units at once.

    spaCy's nlp.pipe batches documents through the model instead of paying
    per-call overhead 4,686 times. Combined with the disabled components this
    is the difference between a web request that completes and one that times
    out.

    The carrier-sentence retry is also batched: short fragments that produced
    no PERSON are collected and sent through the pipe a second time, rather
    than being reprocessed one at a time.
    """
    nlp = get_nlp()
    total = max(1, len(texts))

    results: List[List[Entity]] = []
    retry_indices: List[int] = []

    # Processed in chunks so the caller can report real progress instead of an
    # indeterminate spinner. The first pass is the bulk of the work.
    for start in range(0, len(texts), chunk_size):
        window = texts[start:start + chunk_size]
        for offset_in_window, doc in enumerate(nlp.pipe(window, batch_size=64)):
            idx = start + offset_in_window
            text = texts[idx]
            results.append(_entities_from_doc(doc, text, offset=0))
            if (
                len(text) <= _CARRIER_MAX_LEN
                and not any(e.type == "PERSON" for e in results[idx])
            ):
                retry_indices.append(idx)

        if progress:
            progress(min(0.85, 0.85 * (start + len(window)) / total))

    # Second pass: short fragments retried inside a carrier sentence.
    if retry_indices:
        carriers = [
            f"{_CARRIER_PREFIX}{texts[i]}{_CARRIER_SUFFIX}" for i in retry_indices
        ]
        offset = len(_CARRIER_PREFIX)
        for i, doc in zip(retry_indices, nlp.pipe(carriers, batch_size=64)):
            results[i].extend(_entities_from_doc(doc, texts[i], offset=offset))

    if progress:
        progress(1.0)

    return results


# Table cells are bare fragments with no sentence around them, and spaCy's
# small model relies on sentence context: it tags "Priya Nair" inside a sentence
# but returns nothing for the cell "Priya Nair" on its own. Since this document
# keeps its entire director and contact directory in tables, that blind spot
# would cost a large share of PERSON recall.
#
# The fix is to retry short fragments inside a carrier sentence and shift the
# offsets back. This template was chosen by measuring: it recovered 5 of 6 test
# names while adding no false positives on role labels like "Managing Director".
_CARRIER_PREFIX = "Contact person: "
_CARRIER_SUFFIX = "."
_CARRIER_MAX_LEN = 60


# How far from a name a role or company may sit and still be taken as a
# statement ABOUT that person.
#
# These windows are small on purpose. Scanning the whole paragraph looked
# reasonable and was badly wrong: in the reference document it read "chief
# executive officer" out of a sentence about someone else and attached it to
# "Sarthak Malvadkar", who is the Company Secretary. Proximity across a whole
# paragraph is not evidence; adjacency is.
CONTEXT_BEFORE = 28     # "Managing Director Rajesh Kushal Hegde"
CONTEXT_AFTER = 60      # "Rahul Sharma, Director, Acme Technologies Ltd."

# Documents mix full titles and initialisms for the same job. Expanding them
# means "CS and Compliance Officer" and "Company Secretary and Compliance
# Officer" compare equal, instead of looking like two different people.
ROLE_ABBREVIATIONS = {
    "cs": "company secretary",
    "ceo": "chief executive officer",
    "cfo": "chief financial officer",
    "coo": "chief operating officer",
    "md": "managing director",
}


def _normalise_role(role: str) -> str:
    """Lower-case a role and expand any initialisms inside it."""
    return " ".join(
        ROLE_ABBREVIATIONS.get(word, word)
        for word in role.lower().strip().split()
    )


def person_context(text: str, start: int, end: int) -> Optional[str]:
    """Describe who a person is, from the text immediately around them.

    Returns "role||organisation" with either half possibly empty. Both signals
    are captured because the brief asks for name + role + organisation, and a
    role alone cannot separate two directors at different companies.

    The two halves are kept separate rather than concatenated so the identity
    resolver can reason about them independently: a mention that states only a
    role does not contradict one that states a role and a company.
    """
    before = text[max(0, start - CONTEXT_BEFORE): start]
    after = text[end: end + CONTEXT_AFTER]

    role_match = ROLE_PATTERN.search(after) or ROLE_PATTERN.search(before)
    role_text = _normalise_role(role_match.group()) if role_match else ""

    company_text = ""
    company_match = COMPANY_PATTERN.search(after)
    if company_match:
        cleaned = clean_company_span(company_match.group().strip().strip(",;"))
        if cleaned and _is_plausible_company(cleaned):
            company_text = cleaned.lower()

    if not role_text and not company_text:
        return None
    return f"{role_text}||{company_text}"


def _entities_from_doc(doc, original_text: str, offset: int) -> List[Entity]:
    """Convert one spaCy Doc into filtered Entity objects.

    `offset` is how many characters the analysed string was shifted by relative
    to the original text unit. It is zero for a direct parse and the carrier
    prefix length for a carrier-sentence retry, which lets both paths share
    exactly the same filtering rules.
    """
    out: List[Entity] = []
    for ent in doc.ents:
        label = NER_LABEL_MAP.get(ent.label_)
        if label is None:
            continue

        start, end = ent.start_char - offset, ent.end_char - offset
        # Discard anything that fell outside the original text unit, which can
        # happen when the carrier sentence itself is tagged.
        if start < 0 or end > len(original_text):
            continue

        value = ent.text.strip().strip(",;:")
        if label == "COMPANY":
            # Trim junk the model attached to the front, then re-derive the
            # offsets so the replacement lands on the right characters.
            cleaned = clean_company_span(value)
            if cleaned != value:
                start += len(value) - len(cleaned)
                value = cleaned

        if len(value) < 3 or value.lower() in ORG_STOPLIST:
            continue
        if label == "COMPANY" and not _is_plausible_company(value):
            continue

        entity = Entity(
            text=value, type=label, start=start, end=end, source="ner",
            # Context is derived per mention, from the text beside THIS name -
            # not from anywhere in the paragraph.
            context=person_context(original_text, start, end)
            if label == "PERSON" else None,
        )

        if label == "PERSON":
            # Joint contacts arrive as one span and must be split first, then
            # each resulting name validated and given its own context.
            for person in split_person_list(entity):
                if _is_plausible_person(person.text):
                    out.append(replace(
                        person,
                        context=person_context(
                            original_text, person.start, person.end),
                    ))
        else:
            out.append(entity)

    return out


def detect_ner_entities(text: str) -> List[Entity]:
    """Detect people and organisations in a single text unit.

    Kept for single-unit use (tests, the CLI on small files). The batch variant
    above is what the document pipeline actually calls.
    """
    return detect_ner_entities_batch([text])[0]


# ---------------------------------------------------------------------------
# Company rule (complements NER)
# ---------------------------------------------------------------------------

# spaCy misses company names that appear only once, or in a fragment with no
# sentence around them: it returns nothing at all for
# "Employer: Acme Manufacturing Limited." A deterministic rule fixes that,
# and it is safe precisely because it demands the legal suffix that
# _is_plausible_company already requires - so it adds recall without widening
# what counts as a company.
LEGAL_SUFFIX = (
    r"(?:Private\s+Limited|Pvt\.?\s+Ltd\.?|Limited|Ltd\.?|LLP|L\.L\.P\.|"
    r"Incorporated|Inc\.?|Corporation|Corp\.?|PLC|GmbH|N\.V\.|S\.A\.|"
    r"Family\s+Trust|Trust|Bank|Securities|Technologies|Industries)"
)

# "&" is allowed as a token of its own so partnership names survive intact -
# without it "Kirtane & Pandit, LLP" was truncated to "Pandit, LLP".
COMPANY_PATTERN = re.compile(
    r"\b(?:(?:[A-Z][A-Za-z.'\-]*|&),?\s+){1,6}" + LEGAL_SUFFIX + r"\b"
)


def detect_company_entities(text: str) -> List[Entity]:
    """Find company names by their legal suffix, independently of NER."""
    found: List[Entity] = []
    for match in COMPANY_PATTERN.finditer(text):
        value = clean_company_span(match.group().strip().strip(",;"))
        if not value or value.lower() in ORG_STOPLIST:
            continue
        if not _is_plausible_company(value):
            continue

        start = match.start() + match.group().find(value)
        found.append(
            Entity(text=value, type="COMPANY", start=start,
                   end=start + len(value), source="rule-company")
        )
    return found


# ---------------------------------------------------------------------------
# Address rule
# ---------------------------------------------------------------------------

# spaCy tags "Pune" but not the full street line, so a rule reconstructs the
# whole postal address. It is anchored on the Indian 6-digit PIN code, which is
# the most reliable single marker that a span of text is a real mailing address,
# then extended forwards over the trailing state and country.
#
# Postcode formats are kept general so the tool is not tied to one country.
# Add a new country by appending its postcode shape here - this is the intended
# extension point for addresses.
POSTCODE = (
    r"(?:"
    r"\b\d{3}\s?\d{3}\b"                      # India PIN: 411004 / 411 004
    r"|\b\d{5}(?:-\d{4})?\b"                  # US ZIP / ZIP+4
    r"|\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b" # UK postcode
    r"|\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b"          # Canada
    r")"
)

# The address must contain a comma-separated street line AND a postcode
# introduced by a separator. Requiring the separator is what stops bare
# identifiers such as "Peer review number: 014680" - also six digits - from
# being read as postcodes.
# The separator before the postcode may be a comma, a dash or plain whitespace,
# because documents write both "Pune - 410 501" and "Pune 411 045". Allowing
# whitespace widens the net, so every match must additionally contain a comma
# (checked by the caller) - a real postal address separates its lines, whereas
# "Registration number: 141032" does not.
# The pattern deliberately ENDS at the postcode. Trailing state and country
# words are added afterwards by extend_address_tail(), in code rather than in
# the regex.
#
# Why: the previous version ended with "(?:\s*,?\s*[A-Z][A-Za-z.]+){0,3}", which
# grabbed any three capitalised words after the postcode. It therefore ate the
# start of whatever followed the address - "... India Ticket ID: TKT-2026-08421"
# became part of the address and was destroyed by the replacement. A regex
# cannot tell "Maharashtra" from "Ticket"; a short list of stop-words can.
ADDRESS_PATTERN = re.compile(
    r"[A-Z0-9][^\n]{10,180}?"                 # street line, non-greedy
    # Separator before the postcode: a comma, or whitespace, or a dash that is
    # itself surrounded by whitespace. A bare hyphen glued to a word is NOT a
    # separator - "India-458293" is a country followed by an identifier, not a
    # locality followed by a PIN code. Every address in the reference corpus
    # spaces its separator ("Pune - 410 501", "Mumbai 400083").
    r"(?:,\s*|\s+[–—-]\s*|\s+)"
    r"(?:[A-Z]{2}\.?\s+)?"                    # optional US state code: ", DC 20500"
    r"(?P<postcode>" + POSTCODE + r")"
)

# Words that begin a labelled identifier field. A capitalised word after a
# postcode is part of the address only if it is not one of these.
ADDRESS_TAIL_STOPWORDS = {
    "ticket", "order", "invoice", "request", "reference", "ref", "receipt",
    "docket", "code", "id", "no", "number", "status", "amount", "account",
    "customer", "employee", "batch", "serial", "company", "corporate",
    "registration", "registrar", "tel", "telephone", "fax", "email", "e-mail",
    "website", "url", "contact", "date", "dated", "attn", "gst", "pan", "cin",
    "din", "sebi", "phone", "mobile", "policy", "claim", "case", "file",
}

# A postcode preceded by one of these is an identifier, not an address:
# "Order ID: ORD-458293" has a six-digit tail that looks exactly like an
# Indian PIN code.
ADDRESS_ID_LABEL = re.compile(
    r"(?i)\b(order|ticket|invoice|request|reference|ref|receipt|docket|code|"
    r"id|no|number|acct|account|claim|policy)\b[\s:#]*[-–—]?\s*$"
)

# An all-caps code prefix glued directly to the number, e.g. "ORD-458293".
# Written case-sensitively so a normal city name ("Pune - 411 006") is safe.
ADDRESS_CODE_PREFIX = re.compile(r"\b[A-Z]{2,5}-$")


def address_start_is_valid(text: str, postcode_start: int) -> bool:
    """False when the digits are an identifier rather than a postcode."""
    before = text[max(0, postcode_start - CONTEXT_WINDOW): postcode_start]
    if ADDRESS_ID_LABEL.search(before) or ADDRESS_CODE_PREFIX.search(before):
        return False
    return not ADDRESS_NEGATIVE.search(before)


def extend_address_tail(text: str, end: int, max_tokens: int = 3) -> int:
    """Extend a span over trailing locality / state / country words.

    Stops at anything that signals the address has finished: a stop-word, a
    token containing digits, or a word immediately followed by a colon (which
    makes it a field label, not a place name).
    """
    position, taken = end, 0

    while taken < max_tokens:
        match = re.match(r"\s*,?\s*([A-Z][A-Za-z.]+)", text[position:])
        if not match:
            break

        word = match.group(1)
        if word.lower().strip(".") in ADDRESS_TAIL_STOPWORDS:
            break
        if any(ch.isdigit() for ch in word):
            break

        after = position + match.end()
        if text[after:after + 1] == ":":       # "Company Code:" style label
            break

        position, taken = after, taken + 1

    return position

# Words that mean a following six-digit number is an identifier, not a PIN code.
ADDRESS_NEGATIVE = re.compile(
    r"(?i)\b(registration|review|licence|license|certificate|reference|"
    r"registration\s+no|number|no\.)\s*[:\-]?\s*$"
)

# Form labels that sit in front of an address and should not be swallowed into
# the redacted span, e.g. "Registered Office: 11/3 ...".
ADDRESS_LABEL = re.compile(
    r"(?i)^\s*(registered office|corporate office|residence|address|office|"
    r"located at|situated at|registered address)\s*[:\-]?\s*"
)


def detect_address_entities(text: str) -> List[Entity]:
    """Detect full postal addresses, excluding any leading form label."""
    out: List[Entity] = []
    for m in ADDRESS_PATTERN.finditer(text):
        # Reject identifiers that merely look like postcodes.
        if not address_start_is_valid(text, m.start("postcode")):
            continue

        start = m.start()
        end = extend_address_tail(text, m.end())

        # Drop a leading "Registered Office:" style label so the replacement
        # keeps the label and swaps only the address itself.
        label = ADDRESS_LABEL.match(text[start:end])
        if label:
            start += label.end()

        value = text[start:end]

        # Trim trailing punctuation so a sentence-ending full stop is not
        # swallowed into the redacted span.
        trimmed = value.rstrip(" .,;:-–—")
        end -= len(value) - len(trimmed)
        value = trimmed

        # A real postal address has at least one comma separating its lines.
        if value and "," in value:
            out.append(
                Entity(text=value, type="ADDRESS", start=start,
                       end=start + len(value), source="rule")
            )
    return out


# ---------------------------------------------------------------------------
# Overlap resolution
# ---------------------------------------------------------------------------

# Higher wins. Structured formats beat NER because a regex match on an email is
# certain, whereas a model calling part of it a PERSON is a guess.
TYPE_PRIORITY = {
    "EMAIL": 100, "SSN": 95, "CREDIT_CARD": 90, "IP_ADDRESS": 85,
    "PHONE": 80, "PAN": 75, "DIN": 70, "DOB": 65,
    "ADDRESS": 40, "PERSON": 30, "COMPANY": 20,
}


def resolve_overlaps(entities: List[Entity]) -> List[Entity]:
    """Keep the best entity wherever spans collide.

    This is what stops "rashi.patil@gmail.com" being redacted as an email *and*
    having "Rashi Patil" replaced inside it, which would produce corrupted text
    like "john.doe@example.com" containing a second substitution.

    Ranking: priority first, then longer span, then earlier position.
    """
    ordered = sorted(
        entities,
        key=lambda e: (-TYPE_PRIORITY.get(e.type, 0), -(e.end - e.start), e.start),
    )

    kept: List[Entity] = []
    for candidate in ordered:
        if not any(candidate.overlaps(k) for k in kept):
            kept.append(candidate)

    return sorted(kept, key=lambda e: e.start)


def detect_entities(text: str, use_ner: bool = True) -> List[Entity]:
    """Full detection for one text unit: rules + NER, overlaps resolved."""
    entities = (
        detect_regex_entities(text)
        + detect_address_entities(text)
        + detect_contact_persons(text)
        + detect_company_entities(text)
    )
    if use_ner:
        entities += detect_ner_entities(text)
    return resolve_overlaps(entities)


# ---------------------------------------------------------------------------
# Second pass: propagate known entities across the whole document
# ---------------------------------------------------------------------------

# Types worth propagating. Structured types are already found deterministically
# everywhere, so re-scanning for them would add nothing.
PROPAGATE_TYPES = {"PERSON", "COMPANY"}

# Below this length a name is too short to match safely without word context.
MIN_PROPAGATE_LEN = 6


def build_entity_vocabulary(entities: List[Entity]) -> Dict[str, str]:
    """Collect every distinct PERSON/COMPANY string found anywhere.

    NER is context-sensitive: spaCy tags "ICICI Securities Limited" in one
    sentence and misses it in the next, and it returns "Eric Bacha/" while
    ignoring the two colleagues listed beside him. Recall therefore depends on
    which sentence an entity happens to appear in, which is not acceptable when
    the grading question is "did you catch ALL instances?".

    The fix is to treat detection as evidence about the document rather than
    about one paragraph: anything confidently identified once is looked for
    everywhere.
    """
    vocab: Dict[str, str] = {}
    for entity in entities:
        if entity.type in PROPAGATE_TYPES and len(entity.text) >= MIN_PROPAGATE_LEN:
            vocab[entity.text] = entity.type
    return vocab


def find_known_entities(text: str, vocabulary: Dict[str, str]) -> List[Entity]:
    """Find literal occurrences of already-known entities in one text unit.

    Matching is case-insensitive so that the ALL-CAPS cover-page spellings
    ("RAJESH KUSHAL HEGDE") resolve to the same person as the title-case body
    text. The replacement map normalises case too, so both spellings receive
    the same fake name.

    Longer names are searched first so "Kushal Subbayya Hegde" wins over the
    shorter "Kushal Hegde" that it contains; overlap resolution then discards
    the loser.
    """
    found: List[Entity] = []

    for value in sorted(vocabulary, key=len, reverse=True):
        etype = vocabulary[value]
        for match in re.finditer(re.escape(value), text, re.IGNORECASE):
            found.append(
                Entity(
                    text=match.group(),
                    type=etype,
                    start=match.start(),
                    end=match.end(),
                    source="propagated",
                    context=person_context(text, match.start(), match.end())
                    if etype == "PERSON" else None,
                )
            )
    return found
