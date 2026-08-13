"""Pipeline orchestration: read -> detect -> replace -> write.

This module is intentionally thin. Each step lives in its own file, and this
file just wires them together in order, which keeps the top-level flow readable
as a single function.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

from .detector import (
    ADDRESS_PATTERN,
    Entity,
    address_start_is_valid,
    extend_address_tail,
    build_entity_vocabulary,
    detect_address_entities,
    detect_company_entities,
    detect_contact_persons,
    detect_ner_entities_batch,
    detect_regex_entities,
    find_known_entities,
    resolve_overlaps,
)
from .document_utils import (
    blank_images,
    count_images,
    iter_text_units,
    load_document,
    replace_paragraph_text,
    save_document,
)
from .replacements import ReplacementMap, normalise_entity_text


# Addresses in real documents are often typed as several short paragraphs:
#
#     11/3, 11/4 and 11/5, Village Birdewadi Chakan, Taluka-Khed
#     Pune - 410 501
#     Maharashtra, India
#
# Detection that works one paragraph at a time sees only "Pune - 410 501" and
# matches nothing, so the registered office, corporate office and every banker
# address leak through. These constants bound the window used to stitch such
# runs back together before matching.
MAX_ADDRESS_LINE = 120   # a paragraph longer than this is prose, not an address line
MAX_ADDRESS_BLOCK = 5    # how many consecutive short paragraphs to join

# Paragraphs that begin with one of these labels carry an identifier or a
# contact field, never the first line of a postal address.
NON_ADDRESS_LINE = re.compile(
    r"(?i)^\s*(firm|peer|sebi|cin|din|registration|telephone|tel|fax|email|"
    r"e-mail|website|url|contact|investor|changes|company registration|"
    r"corporate identity|registration number|address of)\b"
)

# A paragraph of the form "Something:" or "Company Code: KSH-2026-4812" is a
# labelled field. Such a paragraph must never be joined to an address block.
#
# This is the structural half of the address-boundary fix. Blocks are built by
# gluing consecutive short paragraphs together with ", ", which previously made
# "Company Code: KSH-2026-4812" look like the next line of the address above it
# - and the whole labelled field was then replaced. Ending the block at any
# labelled field stops the two from ever being considered together.
LABELLED_FIELD = re.compile(r"^\s*[A-Za-z][A-Za-z0-9 /&.\-]{1,28}:")


def find_multiline_addresses(units, existing) -> Dict[int, List[Entity]]:
    """Detect addresses spread over consecutive short paragraphs.

    Runs of short paragraphs are joined with ", " and matched as one string.
    Any match spanning more than one paragraph is then cut back into per-unit
    fragments, so each paragraph keeps its own replacement and the document's
    line structure survives.
    """
    found: Dict[int, List[Entity]] = defaultdict(list)
    separator = ", "
    index, total = 0, len(units)

    def joinable(i: int) -> bool:
        """Whether paragraph i may belong to a stitched address block."""
        text = units[i].text
        return (
            len(text) <= MAX_ADDRESS_LINE
            and not existing[i]
            and not LABELLED_FIELD.match(text)
        )

    while index < total:
        # Skip prose, labelled fields, and paragraphs already carrying an address.
        if not joinable(index):
            index += 1
            continue

        block = []
        cursor = index
        while cursor < total and joinable(cursor) and len(block) < MAX_ADDRESS_BLOCK:
            block.append(cursor)
            cursor += 1

        if len(block) >= 2:
            # Map each paragraph to its span inside the joined string.
            spans, position = [], 0
            for unit_idx in block:
                stripped = units[unit_idx].text.strip()
                spans.append((unit_idx, position, position + len(stripped)))
                position += len(stripped) + len(separator)

            joined = separator.join(units[i].text.strip() for i in block)

            # Match only from the start of a paragraph that could plausibly
            # begin an address - one containing a house or plot number. Letting
            # the regex choose its own start swallowed the preceding heading
            # ("GENERAL INFORMATION") into the address span.
            matches = []
            for _, span_start, _ in spans:
                head = joined[span_start:span_start + 40]
                if not any(ch.isdigit() for ch in head):
                    continue
                # A paragraph that opens with a field label introduces an
                # identifier, not a street. Without this the stitcher pulled in
                # "Peer review number: 014680" and "Firm registration number".
                if NON_ADDRESS_LINE.match(head):
                    continue
                match = ADDRESS_PATTERN.match(joined, span_start)
                if match and address_start_is_valid(joined, match.start("postcode")):
                    matches.append(match)
                    break

            for match in matches:
                # Same tail rules as the single-paragraph path, so a stitched
                # address stops at the same place a plain one would.
                m_start = match.start()
                m_end = extend_address_tail(joined, match.end())
                if "," not in joined[m_start:m_end]:
                    continue
                touched = [
                    (i, s, e) for i, s, e in spans if s < m_end and m_start < e
                ]
                # A single-paragraph match is already handled by the normal
                # address rule; only genuinely split addresses matter here.
                if len(touched) < 2:
                    continue

                for unit_idx, span_start, span_end in touched:
                    raw = units[unit_idx].text
                    lead = len(raw) - len(raw.lstrip())
                    lo = max(m_start, span_start) - span_start
                    hi = min(m_end, span_end) - span_start
                    fragment = raw.strip()[lo:hi]

                    if len(fragment.strip()) >= 3:
                        start = lead + lo
                        found[unit_idx].append(
                            Entity(text=fragment, type="ADDRESS", start=start,
                                   end=start + len(fragment), source="rule-multiline")
                        )

        index = cursor if len(block) >= 2 else index + 1

    return found


def _split_context(context: Optional[str]) -> tuple:
    """Split a "role||organisation" context into its two halves."""
    role, _, org = (context or "").partition("||")
    return role.strip(), org.strip()


def _contexts_conflict(a: tuple, b: tuple) -> bool:
    """True only when two contexts positively disagree.

    A missing half never conflicts. "company secretary" and "company
    secretary||ksh international limited" describe the same person: one mention
    simply said less.

    Role is the strong signal; organisation is deliberately weak. The
    organisation is only whichever company happens to be named in the same
    paragraph, which is frequently *not* that person's employer - in the
    reference document it named a different family trust in almost every
    paragraph, and treating that as evidence split one promoter into four
    people. Organisation is therefore allowed to discriminate only when both
    mentions state the SAME role, which is the case the brief actually cares
    about: two directors at two different companies.
    """
    (role_a, org_a), (role_b, org_b) = a, b

    if role_a and role_b:
        # One title containing the other is an abbreviation, not a
        # contradiction. The reference document writes the same person's title
        # in full ("Company Secretary and Compliance Officer") in one place and
        # abbreviated ("CS and Compliance Officer") in another; treating those
        # as two people split him in half. "Director" vs "Managing Director" is
        # the same situation. Genuinely different titles - "Director" against
        # "Senior Manager" - share no such containment.
        if role_a not in role_b and role_b not in role_a:
            return True
        return bool(org_a and org_b and org_a != org_b)

    return False


def resolve_person_identities(entities: List[Entity]) -> List[Entity]:
    """Decide which mentions of a name refer to the same person.

    This fixes a real defect found on the reference document: because context
    was taken from whichever paragraph a mention appeared in, "Sarthak
    Malvadkar" received THREE different fake identities and "Rohit Kushal
    Hegde" four. Consistency is the headline requirement, so per-paragraph
    context cannot be the identity key on its own.

    The rule is deliberately conservative - assume one person unless the
    document says otherwise:

    * All mentions of a name are grouped into clusters of mutually compatible
      contexts.
    * One cluster (the overwhelmingly common case) means one person, and the
      context is dropped entirely so every mention shares a replacement.
    * Two or more clusters means genuinely conflicting evidence, so the
      mentions are kept apart.
    * A mention with no context at all attaches to the first cluster, which is
      the documented limitation: it cannot be placed on evidence.
    """
    observed: Dict[str, List[tuple]] = defaultdict(list)
    for entity in entities:
        if entity.type == "PERSON" and entity.context:
            name = normalise_entity_text(entity.text)
            parsed = _split_context(entity.context)
            if parsed not in observed[name]:
                observed[name].append(parsed)

    # Sorted so clustering does not depend on the order paragraphs happened to
    # be processed in - the same document must always give the same result.
    clusters: Dict[str, List[tuple]] = {}
    for name, contexts in observed.items():
        groups: List[tuple] = []
        for context in sorted(contexts):
            for index, existing in enumerate(groups):
                if not _contexts_conflict(context, existing):
                    # Merge, so a cluster accumulates the fullest description.
                    groups[index] = (existing[0] or context[0],
                                     existing[1] or context[1])
                    break
            else:
                groups.append(context)
        clusters[name] = groups

    resolved: List[Entity] = []
    for entity in entities:
        if entity.type != "PERSON":
            resolved.append(entity)
            continue

        name = normalise_entity_text(entity.text)
        groups = clusters.get(name, [])

        if len(groups) <= 1:
            # One person: drop context so every mention shares one identity.
            resolved.append(replace(entity, context=None))
            continue

        parsed = _split_context(entity.context)
        for index, group in enumerate(groups):
            if not _contexts_conflict(parsed, group):
                resolved.append(replace(entity, context=f"#{index}"))
                break
        else:
            resolved.append(replace(entity, context="#0"))

    return resolved


@dataclass
class RedactionResult:
    """Everything the caller needs to report on a run."""

    entities: List[Entity] = field(default_factory=list)
    counts_by_type: Dict[str, int] = field(default_factory=dict)
    replacement_map: Dict[str, str] = field(default_factory=dict)
    text_units: int = 0
    images_found: int = 0
    images_blanked: int = 0

    @property
    def total_entities(self) -> int:
        return len(self.entities)


def apply_replacements(text: str, entities: List[Entity], mapping: ReplacementMap) -> str:
    """Rewrite one text unit, substituting every detected entity.

    Replacement runs right-to-left so that earlier offsets stay valid: if we
    replaced left-to-right, a fake value of a different length would shift every
    subsequent start/end index and corrupt the output.
    """
    result = text
    for entity in sorted(entities, key=lambda e: e.start, reverse=True):
        fake = mapping.get(entity.type, entity.text, entity.context)
        result = result[: entity.start] + fake + result[entity.end :]
    return result


class Cancelled(Exception):
    """Raised by a progress callback to abort a run early."""


def redact_document(
    input_path: str,
    output_path: Optional[str] = None,
    use_ner: bool = True,
    seed: int = 42,
    redact_images: bool = True,
    progress=None,
) -> RedactionResult:
    """Redact a DOCX and optionally write the result.

    This is the public API of the package:
        redact_document("input.docx", "redacted.docx")

    `progress` is an optional callable taking (stage: str, fraction: float).
    It lets the web app show a real progress bar, and it may raise Cancelled to
    stop the run - which is how the UI's Cancel button works without needing a
    task queue.
    """
    def report(stage: str, fraction: float) -> None:
        if progress:
            progress(stage, fraction)

    # Parsing a large DOCX is a single opaque call inside python-docx that can
    # take 20s+ on a 400-page file, with no way to report sub-progress. The UI
    # shows an indeterminate bar while the percentage is still this low.
    report("Opening document (large files take a moment)", 0.02)
    doc = load_document(input_path)
    mapping = ReplacementMap(seed=seed)
    result = RedactionResult()

    result.images_found = count_images(doc)

    units = list(iter_text_units(doc))
    result.text_units = len(units)

    # --- Pass 1: detect ---------------------------------------------------
    # Rules run per unit; NER runs over the whole batch at once because
    # spaCy's per-call overhead dominates otherwise.
    # Written as a loop rather than a comprehension so progress is reported
    # periodically. That keeps the bar moving during the rules pass and, more
    # importantly, gives the Cancel button somewhere to take effect - the
    # callback raises, which unwinds the pipeline.
    per_unit = []
    for index, unit in enumerate(units):
        per_unit.append(
            detect_regex_entities(unit.text)
            + detect_address_entities(unit.text)
            + detect_contact_persons(unit.text)
            + detect_company_entities(unit.text)
        )
        if index % 400 == 0:
            report("Scanning for structured PII", 0.04 + 0.06 * index / max(1, len(units)))

    if use_ner:
        # NER dominates the runtime, so its internal progress drives most of
        # the bar: it is mapped onto the 10-75% range of the overall job.
        ner_results = detect_ner_entities_batch(
            [u.text for u in units],
            progress=lambda f: report("Detecting names and organisations",
                                      0.10 + 0.65 * f),
        )
        per_unit = [
            resolve_overlaps(rules + ner)
            for rules, ner in zip(per_unit, ner_results)
        ]
    else:
        per_unit = [resolve_overlaps(rules) for rules in per_unit]

    report("Linking addresses across paragraphs", 0.78)

    # --- Pass 1b: stitch addresses split across paragraphs ----------------
    multiline = find_multiline_addresses(
        units, [bool(e) for e in per_unit]
    )
    for unit_idx, extra in multiline.items():
        per_unit[unit_idx] = resolve_overlaps(per_unit[unit_idx] + extra)

    report("Propagating entities document-wide", 0.84)

    # --- Pass 2: propagate ------------------------------------------------
    # Names and companies confirmed anywhere are then searched for everywhere,
    # because NER only recognises them in sentences it finds easy. Without this
    # the same person is redacted in one paragraph and left in clear text in
    # the next.
    if use_ner:
        vocabulary = build_entity_vocabulary(
            [e for entities in per_unit for e in entities]
        )
        per_unit = [
            resolve_overlaps(entities + find_known_entities(unit.text, vocabulary))
            for unit, entities in zip(units, per_unit)
        ]

    # --- Entity resolution -------------------------------------------------
    # Decide which same-named mentions are the same person, using evidence
    # from the whole document rather than one paragraph at a time.
    flat = [e for entities in per_unit for e in entities]
    resolved = iter(resolve_person_identities(flat))
    per_unit = [[next(resolved) for _ in entities] for entities in per_unit]

    report("Replacing detected PII", 0.90)

    # --- Apply ------------------------------------------------------------
    for index, (unit, entities) in enumerate(zip(units, per_unit)):
        if not entities:
            continue
        # Record which paragraph each entity came from, so evaluation can be
        # scoped to a section that has been exhaustively annotated.
        result.entities.extend(replace(e, unit=index) for e in entities)
        replace_paragraph_text(
            unit.paragraph, apply_replacements(unit.text, entities, mapping)
        )

    # Images cannot be inspected without OCR, so every one of them is treated as
    # a potential leak and neutralised. The reference document hides a PAN card
    # (name, father's name, date of birth, PAN number) in an image, which no
    # text rule in this tool can see.
    report("Neutralising embedded images", 0.95)
    if redact_images and result.images_found:
        result.images_blanked = blank_images(doc)

    result.counts_by_type = dict(Counter(e.type for e in result.entities))
    result.replacement_map = mapping.as_dict()

    report("Writing redacted document", 0.98)
    if output_path:
        save_document(doc, output_path)

    report("Complete", 1.0)
    return result
