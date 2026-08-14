"""Consistent, type-compatible fake replacements.

Two requirements drive this file.

1. Consistency. The same real entity must always map to the same fake value, so
   a reader of the redacted document sees a coherent story: if "Rajesh Kushal
   Hegde" becomes "John Doe" on page 4, he is still "John Doe" on page 250.
   That means generating a value once and caching it, never generating per
   occurrence.

2. Type compatibility. An email must be replaced by something shaped like an
   email, a phone by a phone. The redacted document should remain realistic
   enough to be useful for testing or demos.
"""

import re
from difflib import SequenceMatcher
from typing import Dict, Optional, Tuple

from faker import Faker

# A fixed seed makes every run reproducible, which matters because the
# evaluation report must be regenerable and tests must be deterministic.
DEFAULT_SEED = 42


def normalise_entity_text(text: str) -> str:
    """Canonical form of an entity string, used wherever identity is compared.

    Shared with the identity resolver in redactor.py so the two cannot drift
    apart - if they normalised differently, a person could be resolved as one
    identity and then given two replacements.
    """
    return re.sub(r"\s+", " ", text).strip().lower()


class ReplacementMap:
    """Maps real PII to fake PII, one stable fake value per logical entity."""

    def __init__(self, seed: int = DEFAULT_SEED):
        self.faker = Faker("en_IN")
        Faker.seed(seed)
        self._map: Dict[Tuple[str, str, str], str] = {}
        # Lets an email for a known person reuse that person's fake name,
        # so "rajesh.hegde@ksh.com" becomes "john.doe@example.com".
        self._person_names: Dict[str, str] = {}
        self._counters: Dict[str, int] = {}
        # First spelling seen for each key. The key itself is canonical - for
        # phones that is a bare digit string - so this keeps the mapping report
        # readable ("+91 98765 43210" rather than "919876543210").
        self._first_seen: Dict[Tuple[str, str, str], str] = {}

    # -- key construction ---------------------------------------------------

    @staticmethod
    def _normalise(text: str) -> str:
        """Collapse whitespace and case so trivial variants share one mapping.

        "KUSHAL SUBBAYYA HEGDE" and "Kushal Subbayya Hegde" are the same person
        and must receive the same fake name.
        """
        return normalise_entity_text(text)

    @staticmethod
    def _canonical(entity_type: str, text: str) -> str:
        """Reduce a mention to the form that decides entity identity.

        Two mentions are "the same entity" when their canonical forms match, so
        this is what controls whether they share a fake value.

        Phone numbers need more than whitespace and case folding, because their
        separators are purely cosmetic. A document may print one number as
        "+91 98765 43210" in a contact block and "+91-9876543210" in a footer;
        those are one number and must receive one fake identity. Comparing on
        digits alone makes the formatting irrelevant.

        The evaluator already compared phones digit-wise, so before this the two
        layers disagreed: scoring treated the spellings as one number while the
        replacement map issued two different fakes.
        """
        if entity_type == "PHONE":
            return re.sub(r"[^\d]", "", text)
        return ReplacementMap._normalise(text)

    def _key(self, entity_type: str, text: str, context: Optional[str]) -> Tuple[str, str, str]:
        """Build the identity key for an entity.

        For people the context (their job title) is part of the key. This is the
        same-name edge case: two different people called "Rahul Sharma" who hold
        different roles get different fake identities, because the key differs.

        This is a heuristic, not real entity resolution. Two people with the
        same name AND the same role are still merged - documented as a known
        limitation in the README rather than papered over.
        """
        base = self._canonical(entity_type, text)
        if entity_type == "PERSON" and context:
            return (entity_type, base, context)
        return (entity_type, base, "")

    # -- generation ---------------------------------------------------------

    def _next(self, entity_type: str) -> int:
        self._counters[entity_type] = self._counters.get(entity_type, 0) + 1
        return self._counters[entity_type]

    def _generate(self, entity_type: str, original: str) -> str:
        """Produce one fake value of the right shape for this type."""
        n = self._next(entity_type)

        if entity_type == "PERSON":
            return self.faker.name()

        if entity_type == "EMAIL":
            # Reuse the matching fake person if we have already seen them, so
            # names and emails stay consistent with each other.
            local = original.split("@")[0]
            for real_name, fake_name in self._person_names.items():
                if _name_matches_email_local(real_name, local):
                    return _email_from_name(fake_name)
            return f"user{n}@example.com"

        if entity_type == "PHONE":
            # 90-series keeps it a plausible Indian mobile number.
            return f"+91 90{n:08d}"[:16]

        if entity_type == "COMPANY":
            return f"{self.faker.last_name()} {_COMPANY_SUFFIXES[n % len(_COMPANY_SUFFIXES)]}"

        if entity_type == "ADDRESS":
            # Addresses often arrive as fragments of a multi-paragraph block
            # ("Pune - 410 501", "Maharashtra, India"). Substituting a full
            # postal address for each fragment would triple the block's length
            # and wreck the layout, so the fake value is matched to the shape
            # of what it replaces.
            if len(original) <= 30:
                return _ADDRESS_LINES[n % len(_ADDRESS_LINES)]
            return (
                f"{100 + n} Example Street, Sector {n % 20 + 1}, "
                f"Exampleton - 411 0{n % 90:02d}, Example State, India"
            )

        if entity_type == "SSN":
            return f"{100 + n % 800:03d}-{10 + n % 89:02d}-{1000 + n % 8999:04d}"

        if entity_type == "CREDIT_CARD":
            # 4111-1111-1111-1111 is the standard Visa test number and is Luhn
            # valid, so redacted output still passes card validation.
            return "4111-1111-1111-1111"

        if entity_type == "IP_ADDRESS":
            # 192.0.2.0/24 is TEST-NET-1, reserved by RFC 5737 for documentation
            # and guaranteed never to route to a real host.
            return f"192.0.2.{n % 254 + 1}"

        if entity_type == "DOB":
            return f"{n % 28 + 1:02d} January 1990"

        if entity_type == "PAN":
            return "ABCDE1234F"

        if entity_type == "DIN":
            return f"{9 * 10**7 + n:08d}"

        return f"[REDACTED-{entity_type}-{n}]"

    # -- public API ---------------------------------------------------------

    def get(self, entity_type: str, text: str, context: Optional[str] = None) -> str:
        """Return the fake value for this entity, creating it on first sight."""
        key = self._key(entity_type, text, context)

        if key not in self._map:
            fake = self._generate(entity_type, text)
            self._map[key] = fake
            self._first_seen[key] = text.strip()
            if entity_type == "PERSON":
                self._person_names[self._normalise(text)] = fake

        return self._map[key]

    def as_dict(self) -> Dict[str, str]:
        """Flatten to {original: fake} for reporting and debugging.

        Reported against the first spelling encountered, not the canonical key,
        so a phone entry reads "+91 98765 43210" rather than "919876543210".
        """
        return {
            f"{etype}:{self._first_seen.get(key, text)}"
            + (f" [{ctx}]" if ctx else ""): fake
            for key, fake in self._map.items()
            for etype, text, ctx in [key]
        }

    def __len__(self) -> int:
        return len(self._map)


# Short stand-ins for single lines of a multi-paragraph address block.
_ADDRESS_LINES = [
    "Exampleton - 411 001",
    "Example State, India",
    "12 Example Road, Sector 4",
    "Example City - 400 002",
    "Example District, India",
]

_COMPANY_SUFFIXES = [
    "Technologies Ltd.", "Industries Limited", "Solutions Private Limited",
    "Enterprises Ltd.", "Manufacturing Limited", "Holdings Private Limited",
]


def _name_matches_email_local(real_name: str, local_part: str) -> bool:
    """True if an email's local part looks derived from a person's name.

    Exact substring matching is too brittle for real data. The assignment's own
    example pairs "Rashi Patil" with "rashhi.patil@gmail.com" - note the typo -
    so "rashi" is not a substring of "rashhi" and a strict check would miss it.

    Fuzzy matching handles that: a name token counts as present if it closely
    resembles any token in the local part. Two matching tokens are still
    required, so a shared common first name alone will not link two people.
    """
    local_tokens = [t for t in re.split(r"[^a-z]+", local_part.lower()) if t]
    name_tokens = [t for t in real_name.lower().split() if len(t) > 2]

    hits = sum(
        1
        for nt in name_tokens
        if any(_similar(nt, lt) for lt in local_tokens)
    )
    return hits >= 2


def _similar(a: str, b: str, threshold: float = 0.8) -> bool:
    """Fuzzy string match tolerant of one or two typos."""
    if a == b:
        return True
    return SequenceMatcher(None, a, b).ratio() >= threshold


def _email_from_name(fake_name: str) -> str:
    parts = re.sub(r"[^A-Za-z ]", "", fake_name).split()
    slug = ".".join(p.lower() for p in parts[:2]) or "user"
    return f"{slug}@example.com"
