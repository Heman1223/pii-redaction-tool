# PII Redaction Tool

Upload a `.docx`, get back a copy with every piece of personal information
replaced by a realistic, consistent fake — plus an evaluation report showing how
well the detection actually performed.

---

**Approach.** A hybrid detector. Deterministic **regex** handles anything with a
fixed shape — emails, phones, SSNs, credit cards (Luhn-validated), IPs, PAN,
DIN. **spaCy NER** (`en_core_web_sm`) handles names and organisations, which
have no fixed shape. Rules then clean up after the model: overlapping spans are
resolved by priority so an email is never also redacted as a person, and any
entity confidently found once is propagated across the whole document so the
same person cannot be redacted on one page and left in clear text on the next.
Each entity maps to one stable fake value (`Rashi Patil → John Doe`,
`rashhi.patil@gmail.com → john.doe@example.com`), so the output stays readable
and internally consistent.

**Measured on the real reference document: precision 0.97, recall 0.93, F1 0.95**
(entity-level, 61 hand-annotated entities). Full methodology in
[`EVALUATION.md`](EVALUATION.md); generated numbers in
[`EVALUATION_REPORT.md`](EVALUATION_REPORT.md).

**Tradeoffs and errors actually observed.** Three cost recall and were chosen
deliberately: phone numbers require a country code (a document full of 10-digit
share counts otherwise floods with false positives), company names require a
legal suffix like `Limited`/`LLP` (without it spaCy tagged `Equity Shares` as a
company 37 times), and dates only become dates of birth next to a birth keyword
(the document holds 270 ordinary dates and zero text-layer DOBs). Known misses:
`022-68052182` (no country code), `Trilegal` (single-word company), and two
names spaCy simply did not tag. Known false positive: `practicing company`, a
truncated organisation span. The one gap no metric covers is **image PII** — the
document hides a photographed PAN card with a name, date of birth and PAN number
inside `word/media/image4.png`, which no text rule can read; all images are
blanked wholesale rather than understood, and OCR is the top future improvement.

---

## 1. Problem statement

Documents that must be shared for testing, demos or analysis routinely contain
personal data: names, emails, phone numbers, home addresses, national ID
numbers. Removing that data by hand does not scale, and blanking it out with
`XXXX` destroys the document's usefulness.

This tool replaces PII with **type-compatible fakes** instead of deleting it, so
the redacted document still reads like a real document. It then **measures** its
own accuracy against hand-annotated ground truth, because a redaction tool whose
recall is unknown is a liability rather than a safeguard.

## 2. Project overview

| | |
|---|---|
| Input | `.docx`, plain text (`.txt`/`.md`/`.csv`), or text pasted into the UI |
| Output | redacted `.docx` + Markdown evaluation report |
| Interfaces | web app (upload → redact → download) and CLI |
| Detection | hybrid: deterministic rules + spaCy NER + document-wide propagation |
| Measured on real data | **Precision 0.97 · Recall 0.93 · F1 0.95** |

## 3. Architecture

```
                        ┌──────────────────────────┐
   input.docx  ────────▶│  document_utils.py       │  paragraphs, table cells,
                        │  extract as TextUnits    │  headers, footers, images
                        └────────────┬─────────────┘
                                     ▼
                        ┌──────────────────────────┐
                        │  detector.py             │  Pass 1: rules + NER
                        │  detect → resolve overlaps│  Pass 1b: stitch addresses
                        │                          │  Pass 2: propagate entities
                        └────────────┬─────────────┘
                                     ▼
                        ┌──────────────────────────┐
                        │  replacements.py         │  one stable fake per
                        │  original → fake mapping │  logical entity
                        └────────────┬─────────────┘
                                     ▼
                        ┌──────────────────────────┐
                        │  redactor.py             │  write back in place,
                        │  apply + save            │  preserve formatting
                        └────────────┬─────────────┘
                                     ▼
                   redacted.docx  +  evaluator.py → metrics
```

Everything the document contains — body paragraph, table cell, header — is
yielded as one uniform `TextUnit`. Detection and replacement never learn where
the text came from, which is what makes it structurally impossible for a rule to
work "in paragraphs but not in tables".

### Files

| File | Responsibility |
|---|---|
| `src/document_utils.py` | DOCX in/out, uniform text traversal, image handling |
| `src/detector.py` | All detection rules, NER, overlap resolution |
| `src/replacements.py` | Consistent, type-compatible fake generation |
| `src/redactor.py` | Pipeline orchestration, multi-paragraph addresses |
| `src/evaluator.py` | Entity-level precision / recall / F1 / accuracy |
| `src/app.py` | FastAPI web application |
| `src/main.py` | CLI |

## 4. Detection approach

Hybrid, because neither technique alone is sufficient.

**Deterministic rules** for anything with a fixed shape — email, phone, SSN,
credit card, IP, PAN, DIN. A regex is more accurate *and* more explainable than
a model for these, and it never varies with surrounding wording.

**spaCy NER** (`en_core_web_sm`) for names and organisations, which have no
fixed shape.

**Three refinements matter more than the base approach:**

1. **Context-gated dates.** The reference document contains 270 ordinary dates
   and no date of birth in its text. An ungated date rule produces ~270 false
   positives. A date is only a DOB when a birth keyword sits beside it.

2. **Carrier sentences for table cells.** spaCy tags `Priya Nair` inside a
   sentence but returns nothing for the bare cell `Priya Nair`. Short fragments
   are retried inside a carrier sentence and the offsets shifted back. Measured:
   recovered 5 of 6 test names, added zero false positives.

3. **Document-wide propagation.** NER recognises `ICICI Securities Limited` in
   one sentence and misses it in the next. Any entity confidently found *once*
   is then searched for *everywhere*, case-insensitively. This is what turns
   inconsistent per-sentence recall into consistent document-level recall.

### Why not Presidio?

Presidio wraps spaCy plus a recognizer registry and pulls a large dependency
tree. Its built-in recognizers are US-centric — no Indian phone or PAN support —
so the custom recognizers would have to be written anyway, just inside someone
else's abstraction. Calling spaCy directly keeps detection in ~100 readable
lines with no framework indirection.

## 5. Supported PII types

| Type | Method | Notes |
|---|---|---|
| Full names | NER + carrier sentences + `Contact Person:` rule | |
| Email | regex | |
| Phone | regex | country code required — see limitations |
| Company names | NER + **legal-suffix requirement** | |
| Addresses | postcode rule (IN / US / UK / CA) + multi-paragraph stitching | |
| SSN | regex + negative context | |
| Credit card | regex + **Luhn checksum** | |
| Date of birth | regex + **required birth context** | |
| IP address | regex + octet range + negative context | |
| PAN (India) | regex | |
| DIN (India) | regex | government ID of a *person* |

### Adding a new type

Append a `(label, compiled_regex)` pair to `REGEX_RULES` in `detector.py`, give
it a priority in `TYPE_PRIORITY`, and add a branch to `_generate()` in
`replacements.py`. Three small edits in three obvious places — no new classes.

### What is **not** treated as PII — and why

The brief asks for explicit choices here, so:

| Not redacted | Reason |
|---|---|
| **CIN** (`U28129PN1979PLC141032`) | identifies a *company*, not a person |
| **SEBI registration numbers** | identifies a regulated *entity* |
| Firm registration / peer review numbers | identify a *firm* or a *filing* |
| Regulators & exchanges (SEBI, BSE, NSE, RBI) | public bodies; redacting them destroys readability with no privacy gain |
| Ordinary dates, financial figures, share counts | business data, not personal data |

**DIN is redacted** despite looking like a corporate code, because it is issued
to a natural person and is a durable personal identifier.

## 6. Replacement strategy

Two rules govern every substitution.

**Consistency.** A fake value is generated once per logical entity and cached,
so `Rajesh Kushal Hegde` is the same fake name on page 4 and page 250. Case and
spacing are normalised first, so the cover page's `RAJESH KUSHAL HEGDE` maps to
the same person as the body text.

**Type compatibility.** `PERSON → John Doe`, `EMAIL → john.doe@example.com`,
`IP → 192.0.2.1` (RFC 5737 documentation range), `CARD → 4111-1111-1111-1111`
(Luhn-valid test number). The output stays realistic and safe.

Emails are linked back to their owner where possible, using fuzzy matching so
the brief's own example still works despite its typo:

```
Rashi Patil            → Aryan Maharaj
rashhi.patil@gmail.com → aryan.maharaj@example.com    ("rashhi" ≠ "rashi")
```

### Identity resolution and the same-name edge case

Consistency is decided for the document as a whole, not paragraph by paragraph.
All mentions of a name are grouped, and they are split into separate people
only when the evidence positively conflicts:

| Evidence | Result |
|---|---|
| Same name, same role/organisation | one person |
| Same name, **different role _and_ organisation stated beside each** | two people |
| Same name, no distinguishing context | one person (documented limitation) |

Role is read only from text **adjacent** to the name, and organisation is a
weak signal that discriminates only when roles already agree. Both rules exist
because an earlier version read context from anywhere in the paragraph and
split single real people into several fake identities — see §9.

```
Rahul Sharma, Director, Acme Technologies Ltd.          → one person
Rahul Sharma, Senior Manager, Beta Financial Services   → a different person
```

## 7. Evaluation methodology

> Full strategy, annotation policy, matching rules, per-error diagnosis and
> threats to validity: **[`EVALUATION.md`](EVALUATION.md)**.

**Entity-level, not token-level.** A PII document is >99% non-PII text, so a
model that predicts *nothing* scores ~99% token accuracy while leaking every
identifier. Token accuracy is actively misleading and is never reported here.

**"Accuracy" is defined explicitly**, since the term has no standard meaning
without true negatives:

```
accuracy = TP / (TP + FP + FN)        (Jaccard index)
```

the share of all entities involved in the comparison that were handled
correctly. It cannot be inflated by correctly-ignored text.

**Two corpora, because one cannot cover the requirement.** The real prospectus
contains no SSNs, credit cards or IP addresses, and its only date of birth is
locked inside an image. Those categories are therefore scored on a synthetic
fixture built for the purpose; the messy real-world behaviour is scored on the
prospectus. Both are reported separately and neither is blended into the other.

**Scoped annotation.** Hand-annotating 4,686 paragraphs is infeasible. The
`GENERAL INFORMATION` section (text units 468–606) was annotated *exhaustively*,
and predictions are restricted to that same region — otherwise correct
detections outside it would count as false positives against annotations that
never covered them.

**Matching is type-aware.** Phones compare on digits only; addresses compare by
containment and multiple fragments of one address count as a single detection.
Both prevent the metric from measuring typography and line breaks instead of
detection quality.

## 8. Evaluation results

Generated by `python scripts/run_evaluation.py`. Full output in
[`EVALUATION_REPORT.md`](EVALUATION_REPORT.md).

### Real document — Red Herring Prospectus (4,686 units, 840 replacements)

| PII Type | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| EMAIL | 1.00 | 1.00 | 1.00 | 16 |
| PERSON | 1.00 | 0.93 | 0.96 | 14 |
| PHONE | 1.00 | 0.90 | 0.95 | 10 |
| ADDRESS | 0.92 | 1.00 | 0.96 | 11 |
| COMPANY | 0.89 | 0.80 | 0.84 | 10 |
| **OVERALL** | **0.97** | **0.93** | **0.95** | **61** |

Two false positives and four false negatives remain across the whole annotated
section. Every one is named and diagnosed in
[`EVALUATION.md` §6](EVALUATION.md#6-error-analysis--all-six-errors) — including
one false positive that is a correct detection scored wrongly because the
document prints the same address twice with different punctuation. The
annotation was not adjusted to make it disappear.

### Synthetic fixture — all nine required types

Precision 1.00 · Recall 1.00 · F1 1.00 (19 entities).

This corpus is a controlled fixture written alongside the tool, so a perfect
score demonstrates that the plumbing works — **not** that detection is perfect.
The prospectus numbers above are the meaningful ones.

### Honest reading of these numbers

`COMPANY` remains the weakest category and is the one to improve next. It is
hard here because the tool *deliberately* constrains it: a candidate must carry
a legal suffix and must not contain financial vocabulary. That is what stops
hundreds of Title-Cased defined terms (`Equity Shares`, `Anchor Investors`,
`Public Offer Account Bank`) being redacted. The cost is that a company named
only by a bare brand word — `Trilegal` — is missed.

## 9. Known limitations and trade-offs

1. **Image PII is not read — images are neutralised, not understood.** The reference document hides a photographed PAN
   card — name, father's name, DOB, PAN number — inside `word/media/image4.png`.
   No text-based tool can see it. Every image is therefore **blanked wholesale**
   as an unreviewable disclosure risk. Adding OCR (`pytesseract`) would let
   these be detected properly; it was skipped because it needs a system binary
   outside pip.

2. **Identical name and role cannot be separated.** Two people sharing a name
   *and* a job title are merged into one fake identity. The system also
   deliberately merges when context is missing: splitting one real person into
   several fake identities makes the redacted document incoherent, which is a
   worse failure than merging two strangers. Real coreference resolution is out
   of scope for an MVP.

   Correspondingly, a person who holds several roles ("Promoter" in one place,
   "Managing Director" in another) is correctly kept as **one** identity,
   because a partial or alternative title is treated as an abbreviation rather
   than a contradiction.

3. **Phone numbers require a country code.** `022-68052182` is missed. In a
   financial document full of 10-digit share counts, a rule without that anchor
   produced unacceptable false positives — a deliberate precision-over-recall
   choice.

4. **Single-word company names are missed** (`Trilegal`), and names containing
   financial vocabulary would be too (a firm called `Capital Partners Ltd`).
   Both are consequences of the constraints that protect precision.

5. **Formatting within a replaced paragraph is flattened.** Text is written into
   the first run, so a paragraph with bold text mid-sentence loses that emphasis.
   Necessary because DOCX splits entities across runs arbitrarily.

6. **Ground truth covers one section, not the whole document.** Metrics describe
   the annotated region and should not be read as whole-document guarantees.

7. **Address boundaries depend on separator style.** A postcode is only
   recognised after a comma or a spaced separator, so `India-458293` is read as
   a country followed by an identifier — correct here, but a document writing
   `Pune-411006` with no spaces would have that address missed.

8. **English-language model only.** Non-English names will detect poorly.

## 10. Installation

Requires Python 3.9+.

```bash
git clone https://github.com/Heman1223/pii-redaction-tool
cd pii-redaction-tool

python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 11. How to run locally

### Web app

```bash
python -m uvicorn src.app:app --reload --port 8000
```

Open <http://127.0.0.1:8000>. Drag in a `.docx` or plain-text file — or switch
to the **Paste text** tab and paste content directly — then watch the progress
bar and download the redacted `.docx` and the report.

Whatever goes in, a **`.docx` comes out**: plain text is converted to a document
first, so there is only one redaction implementation to maintain and test.

Large documents take 1–2 minutes, so redaction runs on a background thread and
the page polls for status rather than blocking the request.

### CLI

```bash
python -m src.main "input/Red Herring Prospectus.docx" \
    --output "output/redacted.docx" \
    --mapping output/mapping.json
```

### Tests

```bash
python -m pytest tests/ -q          # 101 tests
```

### Regenerate the evaluation report

```bash
python scripts/make_synthetic_docx.py     # rebuild the synthetic fixture
python scripts/run_evaluation.py          # rewrite EVALUATION_REPORT.md
```

## 12. Example usage

As a library:

```python
from src import redact_document

result = redact_document("input.docx", "redacted.docx")

print(result.total_entities)        # 840
print(result.counts_by_type)        # {'PERSON': 366, 'COMPANY': 402, ...}
print(result.replacement_map)       # {'PERSON:rajesh kushal hegde': 'Pahal Balay', ...}
```

Before and after:

```
Contact Person: Sarthak Malvadkar, Company Secretary and Compliance Officer;
Telephone: + 91 20 4505 3237; E-mail: cs.connect@kshinternational.com
```

```
Contact Person: Aryan Maharaj, Company Secretary and Compliance Officer;
Telephone: +91 9000000002; E-mail: aryan.maharaj@example.com
```

## 13. Future improvements

1. **OCR for images**, closing the largest recall gap in the tool.
2. **Better company recall** — a gazetteer of known entity names, or a larger
   spaCy model (`en_core_web_trf`), to relax the legal-suffix requirement
   without losing precision.
3. **Confidence scores per detection**, so a reviewer can triage borderline
   cases instead of trusting a binary decision.
4. **Reviewer UI** — show detections in context and allow accept/reject before
   the redacted file is produced. For a genuinely sensitive document, a human
   checkpoint matters more than another point of F1.
5. **More formats** — PDF, XLSX, plain text.
6. **Expanded ground truth** covering more sections, for tighter metrics.
7. **Packaging** as `pip install pii-redactor` with a console entry point.
