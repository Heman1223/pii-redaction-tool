# PII Redaction Tool — Evaluation Strategy and Report

How this PII redaction tool is measured, why it is measured that way, and what
the numbers do and do not prove.

### Report details

| | |
|---|---|
| **System under test** | PII Redaction Tool (hybrid regex + spaCy NER) |
| **Primary corpus** | Red Herring Prospectus (`.docx`, 4,686 paragraphs and table cells) |
| **Secondary corpora** | Synthetic fixture (19 entities); negative control (11 distractor lines) |
| **Detection model** | spaCy `en_core_web_sm` v3.8 + deterministic rules |
| **Evaluation unit** | One unique entity |
| **Metrics reported** | Precision, Recall, F1, Accuracy (defined in §3) |
| **Regression suite** | 104 automated tests, all passing |
| **Reproduced by** | `python scripts/run_evaluation.py` |

Every figure in this document is produced by running the pipeline. Nothing is
hand-entered, and fake values are generated from a fixed seed, so a re-run
reproduces these numbers exactly.

---

## 1. Headline results

**Real document — Red Herring Prospectus** (4,686 paragraphs and table cells,
840 PII occurrences replaced, 258 unique entities, 8/8 images neutralised):

| | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| **Overall** | **0.97** | **0.93** | **0.95** | **0.90** |

TP 57 · FP 2 · FN 4 · 61 annotated entities.

Two supporting runs:

- **Synthetic fixture** — all nine required PII types: 1.00 across the board (19 entities).
- **Negative control** — 11 lines of PII-shaped non-PII: **0 detections, 0 characters altered.**

Section 6 names and explains every one of the 6 errors individually.

---

## 2. The central decision: entity-level, not token-level

**Token-level accuracy is not reported anywhere in this project, deliberately.**

A PII document is over 99% ordinary text. A model that predicts *nothing at
all* — leaking every name, email and account number — scores roughly **99% token
accuracy**. The metric does not merely fail to detect that failure; it rewards
it.

In the prospectus the arithmetic is stark: ~324,000 characters of text against
61 annotated entities in the evaluated section. Token accuracy would be
dominated by the correctly-ignored 99.9%.

So the unit of evaluation is **one entity**, and the question asked of each is
binary: *was this piece of personal data found, or not?*

### Unique entities, not occurrences

Entities are de-duplicated before scoring. `Rajesh Kushal Hegde` appears dozens
of times; counting each mention separately would let one frequently-repeated
name dominate the score and mask a miss on someone mentioned once.

The meaningful question is *"did the tool find this person?"* — asked once per
person.

---

## 3. What "accuracy" means here

The assignment asks for accuracy, but the term has no standard meaning for
entity extraction: there is no meaningful count of **true negatives**. Every
span of text that is not PII is a true negative, so "how many did we correctly
ignore?" has no principled denominator.

This project therefore defines and reports:

```
accuracy = TP / (TP + FP + FN)
```

the **Jaccard index** — the proportion of all entities involved in the
comparison that were handled correctly. It penalises both misses and false
alarms, and unlike `(TP + TN) / total` it cannot be inflated by the vast
quantity of correctly-ignored text.

Precision, recall and F1 carry their standard definitions:

```
precision = TP / (TP + FP)          recall = TP / (TP + FN)
F1        = 2 · precision · recall / (precision + recall)
```

Accuracy is always the most pessimistic of the four here. That is intended.

### What TP, FP and FN mean for redaction

| Term | Meaning in this system | Consequence if it happens |
|---|---|---|
| **True positive (TP)** | Annotated PII that the tool detected and replaced | Correct — the data is protected |
| **False positive (FP)** | Text the tool replaced that is **not** PII | Over-redaction: a non-sensitive word is destroyed, the document degrades |
| **False negative (FN)** | Annotated PII the tool **failed** to detect | **Data leak** — real personal data survives into the "redacted" output |
| True negative (TN) | Non-PII text correctly left alone | Not counted; see above for why |

**These two errors are not equally bad.** A false positive damages readability;
a false negative defeats the entire purpose of the tool. Recall is therefore the
number to scrutinise first, and it is the weaker of this system's two headline
figures (0.93 against precision 0.97).

---

## 4. Ground truth

### 4.1 Two corpora, because one cannot cover the requirement

The assignment requires nine PII types. The reference document **does not
contain four of them**: it has no SSNs, no credit card numbers, no IP
addresses, and its only date of birth is locked inside a photographed PAN card
image that no text-based rule can read.

Evaluating on the prospectus alone would leave those four categories blank —
indistinguishable, in a report, from a detector that is simply broken.

| Corpus | Purpose | Annotated entities |
|---|---|---|
| **Red Herring Prospectus** | Real, messy, adversarial. Measures behaviour on genuine data with heavy false-positive pressure. | 61 |
| **Synthetic fixture** | Covers all nine required types, including the four absent from the real document. | 19 |
| **Negative control** | PII-shaped non-PII. Measures precision pressure directly. | 0 (all must be ignored) |

The two scored corpora are reported **separately and never blended**. Averaging
them would let the easy fixture flatter the hard document.

### 4.2 Honest framing of the synthetic 1.00

The synthetic corpus scores 1.00 on every category. **This is not evidence that
detection is perfect.** I wrote both the fixture and the detector, so a perfect
score demonstrates that the plumbing works end to end — extraction, detection,
overlap resolution, replacement, DOCX writing — and that all nine required types
are wired up.

**The prospectus numbers are the meaningful ones.** Any assessment should weight
Section 5.1 far above Section 5.2.

### 4.3 Scoped annotation, and why the scope is essential

Hand-annotating all 4,686 paragraphs is not feasible. Instead the
**`GENERAL INFORMATION` section (text units 468–606)** was annotated
**exhaustively** — every entity of every supported type inside it, including
ones the tool is known to miss.

That section was chosen because it is where the document concentrates personal
data: registered and corporate offices, the company secretary's direct contact
details, both book running lead managers, the registrar, legal counsel, bankers
to the offer, and the statutory auditors.

Predictions are then **restricted to the same unit range** before scoring.

This restriction is not a convenience — it is what makes precision meaningful.
Scoring whole-document predictions against a partial annotation would count
every correct detection made *outside* the annotated region as a false
positive. The tool would be penalised for working. With ~840 occurrences
detected document-wide against 61 annotations, unrestricted scoring would
report a precision near 0.07 and mean nothing.

Annotation composition:

| Type | PERSON | EMAIL | PHONE | COMPANY | ADDRESS | Total |
|---|---|---|---|---|---|---|
| Annotated | 14 | 16 | 10 | 10 | 11 | **61** |

### 4.4 Annotation policy — what counts as PII

The brief asks for explicit choices. These were fixed **before** measuring, and
the ground truth was never edited afterwards to improve a score.

**Annotated as PII:** person names, emails, phone numbers, company names,
postal addresses, SSNs, credit cards, dates of birth, IP addresses, PAN, and
**DIN** — a Director Identification Number is issued to a natural person and is
a durable personal identifier.

**Deliberately not annotated:** CIN (`U28129PN1979PLC141032`), SEBI
registration numbers, firm registration and peer review numbers — these
identify *companies and filings*, not people. Also excluded: regulators and
exchanges (SEBI, BSE, NSE, RBI), ordinary dates, financial figures and share
counts.

---

## 5. Results

### 5.1 Real document — Red Herring Prospectus

Scope: units 468–606. Document-wide the run replaced 840 occurrences across 258
unique entities and neutralised 8 of 8 images.

| PII Type | Precision | Recall | F1 | Accuracy | TP | FP | FN | Support |
|---|---|---|---|---|---|---|---|---|
| EMAIL | 1.00 | 1.00 | 1.00 | 1.00 | 16 | 0 | 0 | 16 |
| PERSON | 1.00 | 0.93 | 0.96 | 0.93 | 13 | 0 | 1 | 14 |
| PHONE | 1.00 | 0.90 | 0.95 | 0.90 | 9 | 0 | 1 | 10 |
| ADDRESS | 0.92 | 1.00 | 0.96 | 0.92 | 11 | 1 | 0 | 11 |
| COMPANY | 0.89 | 0.80 | 0.84 | 0.73 | 8 | 1 | 2 | 10 |
| **OVERALL** | **0.97** | **0.93** | **0.95** | **0.90** | **57** | **2** | **4** | **61** |

Document-wide detection volume (all 4,686 units, unscored):

| COMPANY | PERSON | EMAIL | ADDRESS | PHONE |
|---|---|---|---|---|
| 331 | 328 | 70 | 64 | 47 |

### 5.2 Synthetic fixture — all nine required types

37 units, 23 occurrences replaced, 19 unique entities.

| PII Type | Precision | Recall | F1 | Accuracy | TP | FP | FN | Support |
|---|---|---|---|---|---|---|---|---|
| PERSON | 1.00 | 1.00 | 1.00 | 1.00 | 5 | 0 | 0 | 5 |
| EMAIL | 1.00 | 1.00 | 1.00 | 1.00 | 4 | 0 | 0 | 4 |
| PHONE | 1.00 | 1.00 | 1.00 | 1.00 | 3 | 0 | 0 | 3 |
| COMPANY | 1.00 | 1.00 | 1.00 | 1.00 | 2 | 0 | 0 | 2 |
| ADDRESS | 1.00 | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 | 1 |
| SSN | 1.00 | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 | 1 |
| CREDIT_CARD | 1.00 | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 | 1 |
| DOB | 1.00 | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 | 1 |
| IP_ADDRESS | 1.00 | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 | 1 |
| **OVERALL** | **1.00** | **1.00** | **1.00** | **1.00** | **19** | **0** | **0** | **19** |

This is the only corpus where all nine required types appear, which is why it
exists. See Section 4.2 for why a perfect score here is a plumbing check rather
than a quality claim.

### 5.3 Negative control — precision under pressure

Eleven lines that *look* like PII but are not, run through the full pipeline:

```
Invoice number 456-78-9012                 (SSN-shaped)
Reference ticket ID 998877665544332211     (card-shaped, passes Luhn by chance)
Software build version 10.2.14.3           (IPv4-shaped)
Board meeting on December 10, 2025         (date, not a DOB)
Corporate Identity Number: U28129PN1979PLC141032
SEBI Registration Number: INM000013004
Peer review number: 014680                 (6 digits, postcode-shaped)
Order ID: ORD-458293, Company Code KSH-2026-4812
Request ID: REQ-987654321, status code 200
Total shares allotted: 26704570            (10 digits, phone-shaped)
Amount: 45,000 paid on 13 August 2026
```

**Result: 0 entities detected, 0 lines altered.**

This is the single most informative precision measurement in the project. The
reference document contains 270 ordinary dates and zero text-layer dates of
birth; an ungated date rule alone would have produced ~270 false positives.

---

### 5.4 Sample output from an actual run


Verbatim before/after from the synthetic corpus, showing all nine PII types
replaced with type-compatible values. The synthetic corpus is used here rather
than the prospectus so that no real personal data is reproduced in this report.

**Redacted:**

| Before | After |
|---|---|
| `Full Name: Rashi Patil` | `Full Name: Aryan Maharaj` |
| `Email: rashhi.patil@gmail.com` | `Email: aryan.maharaj@example.com` |
| `Contact: +91 9876543210` | `Contact: +91 9000000001` |
| `Date of Birth: 15 March 1992` | `Date of Birth: 02 January 1990` |
| `SSN: 123-45-6789` | `SSN: 101-11-1001` |
| `Card on file: 4111 1111 1111 1111` | `Card on file: 4111-1111-1111-1111` |
| `Last login from IP 192.168.14.22` | `Last login from IP 192.0.2.2` |
| `Employer: Acme Manufacturing Limited` | `Employer: Konda Industries Limited` |
| `Residence: 42 Baker Street, Kothrud, Pune - 411 038, Maharashtra, India` | `Residence: 101 Example Street, Sector 2, Exampleton - 411 001, Example State, India` |
| `Reporting Manager: Rohan Dey` | `Reporting Manager: Liam Chaudry` |
| `Manager email: rohan.dey@gmail.com` | `Manager email: liam.chaudry@example.com` |

Two properties visible in this sample:

- **Type compatibility** — an email is replaced by something shaped like an
  email, a card by a Luhn-valid test card, an IP by an address from the RFC 5737
  documentation range.
- **Linked identity** — `Rashi Patil` becomes `Aryan Maharaj`, and her email
  becomes `aryan.maharaj@example.com`. The person and their email stay
  consistent with each other, and both remain stable everywhere they recur.

**Left untouched in the same document** — every one of these contains digits and
resembles PII, and none was altered:

```
Invoice number 456-78-9012 was raised for this onboarding.
Reference ticket ID 998877665544332211 remains open.
Software build version 10.2.14.3 was deployed to production.
The policy was approved by the board on December 10, 2025.
Corporate Identity Number: U28129PN1979PLC141032
SEBI Registration Number: INM000013004
Total shares allotted: 26704570 at face value 5 each.
```

---

## 6. Error analysis — all six errors

No error is summarised away. Each is named, diagnosed, and classified as either
a real defect or a measurement artefact.

### False positives (2)

**1. `Bandra Kurla Complex, Bandra East Mumbai 400 051` (ADDRESS)** —
*measurement artefact, not a detection error.* The document prints Nuvama's
address twice, in the Book Running Lead Managers block and again under
Syndicate Members, with different line breaks and punctuation. The annotation
records one form; the containment match narrowly fails against the other. The
detection is correct; the scoring counts it as a false alarm.

I have **not** adjusted the annotation or the matcher to remove it. Loosening a
matcher until errors disappear is how evaluations become fiction. Real
precision is therefore marginally better than 0.97.

**2. `practicing company` (COMPANY)** — *a real defect.* From "an independent
practicing company secretary, Kanj & Co. LLP". A truncated organisation span
survived the filters. Harmless in effect — it redacts two words of a job
description — but it is a genuine false positive.

### False negatives (4)

**3. `022-68052182` (PHONE)** — *a deliberate trade.* The phone rule requires a
country code. In a financial document dense with 10-digit share counts, a rule
without that anchor generated unacceptable false positives. Precision was
chosen over recall here, and this miss is the cost.

**4. `Trilegal` (COMPANY)** — *a deliberate trade.* Organisation spans must
carry a legal-entity suffix (`Limited`, `LLP`, `Private Limited`). This single
rule removed hundreds of false positives, because a prospectus Title-Cases its
vocabulary and spaCy tagged `Equity Shares` as a company 37 times. Single-word
brand names are the price.

**5. `ICICI Securities Limited` (COMPANY)** — *a real recall gap.* NER is
context-sensitive: spaCy tags this entity in some sentences and misses it in
others. The document-wide propagation pass recovers most such cases, but not
all.

**6. `Lalit Muljibhai Sarvaiya` (PERSON)** — *a real recall gap.* A three-token
name inside a long legal sentence that spaCy did not tag, and which no
contextual rule anchored.

### What the error profile says

Errors 3 and 4 are **policy**, not bugs — reversing either would raise recall
and lower precision by more. Errors 5 and 6 are the genuine ceiling of a small
NER model. Error 2 is the only outright defect, and it is cosmetic.

For a redaction tool, the asymmetry matters: **a false negative leaks personal
data; a false positive over-redacts a word.** Recall at 0.93 with two of the
four misses being deliberate policy is the number to scrutinise, and it is the
weaker of the two headline figures.

---

## 7. Matching rules

Naive exact string matching measures typography rather than detection. Three
type-aware rules were introduced, each fixing a specific measurement bug:

| Rule | Problem it fixes |
|---|---|
| **Dash normalisation** | The document writes `Pune – 411 044` (en-dash), an annotator types `-`. Without this, one correct detection scores as **both** a false positive and a false negative. |
| **Phones compare on digits** | `+91 22 40094400` and `+ 91 22 4009 4400` are one number. Predictions are also de-duplicated on digits, so a second spelling cannot become a phantom false positive. |
| **Addresses match by containment; fragments count once** | A postal address is typed across several paragraphs, so the tool reports fragments while the annotation is one string. Demanding exact equality would score a correct multi-line detection as 1 miss + 3 false alarms. |

Matching is otherwise exact after case and whitespace normalisation, and is
**greedy one-to-one**: a single prediction can satisfy at most one annotation,
so an over-long span cannot "cover" several annotations and inflate recall.

**These fixes changed measurement, not detection.** When they were introduced,
overall F1 moved from 0.62 to 0.85 with no change to any detection rule. That
distinction is stated explicitly because a reader is entitled to know which
gains came from a better detector and which from a corrected ruler.

---

## 8. Threats to validity

Stated plainly, because an evaluation that hides its own weaknesses is not
evidence.

1. **Scope.** Metrics describe units 468–606, the most PII-dense section. Other
   sections are more prose-heavy, where NER behaves differently. These numbers
   should not be read as a whole-document guarantee.

2. **Sample size.** 61 entities. A single additional miss moves recall by ~1.6
   points. Per-category figures rest on 10–16 examples each and are indicative,
   not tight.

3. **Single annotator.** I wrote both the detector and the ground truth. There
   is no second annotator and no inter-annotator agreement score, so annotation
   bias cannot be ruled out. This is the weakest link in the methodology.

4. **One document, one domain.** An Indian IPO prospectus. Performance on
   medical records, CVs or chat logs is unmeasured.

5. **Image PII is out of scope of the metrics entirely.** The PAN card carrying
   a name, father's name, date of birth and PAN number is inside an image.
   It cannot be detected without OCR and does not appear in the ground truth,
   so **no metric in this document reflects it.** All 8 images are blanked
   wholesale — mitigation by removal, not by detection.

6. **DOB, SSN, credit card and IP are scored only on synthetic data**, since
   the real document contains none. Their 1.00 scores carry the caveat in
   Section 4.2.

---

## 9. Reproducing these numbers

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

python scripts/make_synthetic_docx.py    # rebuild the synthetic fixture
python scripts/run_evaluation.py         # rewrite EVALUATION_REPORT.md
python -m pytest tests/ -q               # 104 tests
```

The real-document run needs `input/Red Herring Prospectus.docx`, which is
excluded from version control because it contains real personal data —
including that PAN card. The synthetic corpus and the full test suite run from
a clean clone.

### Regression tests as continuous evaluation

Metrics are a snapshot; the 104 tests are the standing guarantee. The valuable
ones are **negative**:

```python
test_ordinary_dates_are_not_dob()               # the 270-dates problem
test_ssn_not_matched_for_invoice_number()
test_credit_card_not_matched_for_ticket_id()
test_ip_not_matched_for_version_string()
test_identifiers_are_not_addresses()
test_person_inside_email_is_not_separately_detected()
test_same_name_same_role_is_merged_known_limitation()   # pins a known limit
```

The last one is deliberate: it makes a documented weakness visible and
regression-proof rather than leaving it to be rediscovered.

---

## 10. Summary

| Question | Answer |
|---|---|
| Headline (real document) | Precision **0.97**, Recall **0.93**, F1 **0.95**, Accuracy **0.90** |
| What is "accuracy"? | `TP / (TP + FP + FN)`, entity-level. Token accuracy is never reported. |
| How was ground truth built? | One section annotated exhaustively by hand; predictions scored in the same scope. |
| Are the numbers reproducible? | Yes — one command, fixed seed, nothing hand-entered. |
| Biggest weakness in the tool | Company recall 0.80, a deliberate precision trade. |
| Biggest weakness in the evaluation | Single annotator, 61 entities, one document. |
| What is **not** measured | Image PII; four categories on real data; every other document domain. |

This is a working MVP with measured behaviour and known, documented limits —
not a solved problem.
