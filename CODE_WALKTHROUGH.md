# Code Walkthrough — what was built, in what order, and why

This document is for **understanding and defending the code**, not for using it.
The README explains *what* the tool does; this explains *why each decision was
made*, in the order the decisions were actually taken, including the things that
went wrong along the way.

Read it top to bottom before a technical discussion. Every section ends with
**"If asked…"** — the short version of the answer.

---

## Step 0 — Profiling the document before writing any code

**What I did:** unzipped the `.docx` and inspected the raw XML before writing a
single detection rule.

**What that revealed:**

| Finding | Consequence |
|---|---|
| 4,415 paragraphs, 76 tables, 3,225 table cells | Table traversal is mandatory |
| 48,817 runs for 48,200 text nodes | Text is fragmented; an email can be split across runs |
| 52 emails, 26 phones | Real targets |
| **Zero** SSNs / IPs / DOB labels in the text | Four required categories are absent |
| 270 ordinary dates, 0 DOBs | The dominant false-positive risk |
| CINs, SEBI codes | Identifier-shaped strings that must **not** be redacted |

**Why it mattered:** the single most important design decision in the tool —
context-gating DOB — came directly from counting 270 dates. Had I written rules
first, I'd have shipped an ungated date rule and lost most of my precision.

> **If asked "how did you decide what to build?"** — I profiled the input first.
> The 270-dates-to-0-DOBs ratio dictated the detection strategy.

---

## Step 1 — The discovery that changes the whole problem

Text extraction reported **zero dates of birth**. That was true of the text
layer and completely wrong about the document.

`word/media/image4.png` is a **photographed PAN card** containing a full name, a
father's name, a date of birth and a PAN number. It is invisible to
`python-docx`, to spaCy, and to every regex in this project.

**Decision:** since the tool cannot read images without OCR, it must not pretend
they are safe. Every image is counted, reported, and **blanked** — treated as an
unreviewable disclosure risk.

**Trade-off considered:** adding `pytesseract` would detect the PAN properly,
but Tesseract is a system binary outside `pip`, which breaks a one-command
install. Documented as the number-one future improvement.

> **If asked "what was the hardest thing to find?"** — the PAN card in an image.
> A pure text pipeline scores 0 on DOB and never knows why.

---

## Step 2 — `document_utils.py`: one uniform text stream

**The design choice:** everything — body paragraph, table cell, header, footer —
is yielded as one `TextUnit`. Downstream code cannot tell them apart.

```python
@dataclass
class TextUnit:
    text: str
    location: str
    paragraph: Paragraph     # live object: writing to it edits the document
```

**Why:** "we handled paragraphs but forgot tables" is *the* classic bug in this
task. Making the traversal uniform means a rule physically cannot work in one
place and not the other. Table iteration also recurses, because tables nest.

### The run-splitting problem

DOCX splits text into *runs* at every formatting or spellcheck boundary. A
single email is often 3–4 runs. Replacing run by run would corrupt any entity
crossing a boundary.

**Solution:** write the full replacement into the first run, blank the rest.

**Cost, stated honestly:** a paragraph with bold text mid-sentence loses that
emphasis. Correctness of redaction beats preservation of mid-paragraph styling.

> **If asked about DOCX structure** — runs are the trap; paragraph-level
> replacement is the fix, and it costs intra-paragraph formatting.

---

## Step 3 — `detector.py`: rules where shape exists, NER where it doesn't

| Technique | Used for | Reason |
|---|---|---|
| Regex | email, phone, SSN, card, IP, PAN, DIN | Fixed shapes; more accurate *and* explainable than a model |
| spaCy NER | names, organisations | No fixed shape exists |
| Rules on top of NER | contact lists, company filtering | NER output needs cleaning |

### Why not Presidio

It wraps spaCy plus a registry, pulls a large dependency tree, and its default
recognizers are US-centric — no Indian phone or PAN. The custom recognizers
would have to be written anyway, just inside someone else's framework.

> **If asked "why not Presidio?"** — I'd write the same rules either way, but
> lose readability and gain dependencies.

### Luhn validation for credit cards

Any 13–19 digit run matches a card pattern — including share counts, which this
document has in quantity. The Luhn checksum is what separates a card number from
a big number.

### Context gates — the highest-value idea in the file

Some strings are *shaped* exactly like PII but are not PII:

| String | Looks like | Actually is |
|---|---|---|
| `456-78-9012` | SSN | invoice number |
| `998877665544332211` | credit card (passes Luhn!) | ticket ID |
| `10.2.14.3` | IP address | software version |
| `December 10, 2025` | DOB | ordinary date |

Shape alone cannot separate these. The tool inspects the ~40 characters
*before* a match and rejects it if they identify it as something else.

**Negative gating vs positive gating** — a deliberate asymmetry:

* **SSN / card / IP** use *negative* gates ("reject if preceded by 'invoice'").
  Requiring a keyword before every SSN would lose bare SSNs.
* **DOB** uses a *positive* gate ("require a birth keyword"). Here the base rate
  is overwhelming — 270 dates, 0 DOBs — so the default must be "not PII".

> **If asked about precision** — describe the four look-alikes above, then
> explain why DOB is gated positively and the others negatively.

---

## Step 4 — Three NER problems and their fixes

spaCy's small model is fast and free, and it fails in three specific ways here.

### 4a. Bare table cells

```
nlp("Priya Nair")                        → []            ← fails
nlp("Priya Nair is the Company Secretary.") → [PERSON]   ← works
```

The model needs sentence context; table cells have none. This document keeps its
entire contact directory in tables.

**Fix:** retry short fragments inside a carrier sentence, then shift the offsets
back.

**The template was chosen by measurement, not guesswork** — four candidates were
tested; `"Contact person: {}."` recovered 5/6 names with **0** false positives,
while `"{} is a person."` also mislabelled role titles.

### 4b. Prospectus defined terms tagged as companies

A prospectus Title-Cases its vocabulary: `Equity Shares`, `Anchor Investors`,
`Promoter Group`. spaCy reads Title Case as a proper noun and tagged
**`Equity Shares` as a company 37 times**.

**The domain insight:** real company names in this document essentially always
carry a legal suffix (`Limited`, `LLP`, `Private Limited`, `Trust`, `Bank`).
Defined terms never do.

**Fix:** require a legal-entity suffix for any ORG.

**Effect:** unique detected entities fell from **806 → 283** — almost all of the
removed items were noise.

**Cost, stated plainly:** single-word company names like `Trilegal` are missed.
Over-redacting ordinary business terminology would make the document unreadable,
which the brief explicitly warns against.

**A later refinement recovered the lost recall.** spaCy alone missed companies
mentioned only once, so a *deterministic* company rule was added: capitalised
words ending in a legal suffix, run through the same validation. Adding it
lifted recall but sank precision (0.71 → 0.46), because it matched role
descriptions like `Public Offer Account Bank`. Rejecting any span containing
financial vocabulary fixed that, giving **0.89 precision / 0.80 recall**.

> **If asked about the weakest metric** — COMPANY. Explain the constraint chain:
> legal suffix required, financial vocabulary forbidden, one substantive word
> minimum. Each one was added in response to a measured false positive.

### 4c. Joint contact lists

The document writes `Contact Person: Eric Bacha/ Sachin Gawade/ Pravin Teli`.
spaCy returned **only `Eric Bacha/`** — inventing a malformed name and hiding
two real people.

**Fix:** a targeted rule. When the document itself says `Contact Person:`, parse
what follows as a list of names, stopping at the next field label.

**Why this is safe:** it fires only where the document has already declared that
the following text is a person. High precision *and* high recall.

**Impact:** PERSON recall went **0.43 → 0.93**. The single biggest measured
improvement in the project.

---

## Step 5 — Overlap resolution

The brief calls this out directly: `rashhi.patil@gmail.com` must be **one**
email, never an email *plus* a `PERSON` found inside it — otherwise replacement
corrupts the substituted value.

```python
TYPE_PRIORITY = {"EMAIL": 100, "SSN": 95, ..., "PERSON": 30, "COMPANY": 20}
```

Ranking: **priority → longer span → earlier position**. Structured types outrank
NER because a regex match on an email is certain, whereas a model calling part
of it a name is a guess.

> **If asked about overlaps** — one sentence: a regex hit is evidence, an NER hit
> is a guess, so evidence wins.

---

## Step 6 — Two-pass detection

**The problem:** NER is context-sensitive, so recall depends on which *sentence*
an entity happens to appear in. `ICICI Securities Limited` is found in one
paragraph and missed in the next. That is unacceptable when the grading question
is *"did you catch **all** instances?"*

**The fix — treat detection as evidence about the document, not the paragraph:**

* **Pass 1** — detect per unit (rules + batched NER).
* **Pass 1b** — stitch addresses split across paragraphs (Step 7).
* **Pass 2** — build a vocabulary of every confidently-found person/company, then
  search for those strings **everywhere**, case-insensitively.

Case-insensitive matching also unifies the cover page's `RAJESH KUSHAL HEGDE`
with the body's `Rajesh Kushal Hegde`; the replacement map normalises case, so
both get the same fake name.

> **If asked "how do you guarantee consistency?"** — this is the answer.
> Detection is document-level, not paragraph-level.

---

## Step 7 — Multi-paragraph addresses

A real address is typed as several short paragraphs:

```
[470] 11/3, 11/4 and 11/5, Village Birdewadi Chakan, Taluka-Khed
[471] Pune – 410 501
[472] Maharashtra, India
```

Per-paragraph matching sees only `Pune – 410 501` and matches **nothing**, so the
registered office, corporate office and every banker address were leaking.

**Fix:** join runs of consecutive short paragraphs, match against the joined
string, then cut the match back into per-paragraph fragments.

**Two bugs found while building it** — both worth knowing:

1. The regex chose its own start point and swallowed the preceding heading, so
   `GENERAL INFORMATION` was redacted as part of an address.
   **Fix:** anchor matches to paragraph starts, and require the starting
   paragraph to contain a digit.
2. Label lines like `Peer review number: 014680` were pulled in as address
   lines. **Fix:** a stop-list of field labels that can never begin a street.

**Replacement detail:** fake values are length-aware. Substituting a full postal
address for the fragment `Pune – 410 501` would triple the block's length and
wreck the layout, so short fragments get short fake lines.

**Result:** ADDRESS recall 1.00.

---

## Step 8 — `replacements.py`: consistency and the same-name case

**Consistency** is a caching problem: generate once per logical entity, then
reuse. Text is normalised (whitespace + case) before use as a key.

**The same-name edge case** is handled by putting *context in the key*:

```python
("PERSON", "rahul sharma", "director")                  → Pahal Balay
("PERSON", "rahul sharma", "chief financial officer")   → Tejas Kaul
```

**Where this stops working, stated honestly:** identical name *and* identical
role are merged. There is no signal left to split on. A test pins this behaviour
so it stays a *known* limitation rather than an accident.

**Email↔name linking** uses fuzzy matching, because the brief's own example
contains a typo — `Rashi Patil` vs `rashhi.patil@` — and exact matching fails on
it. Two tokens must match, so a shared common first name alone won't link two
people.

**Type-compatible fakes** use deliberately safe constants: `192.0.2.x` (RFC 5737
documentation range, never routable) and `4111-1111-1111-1111` (standard Luhn-
valid test card).

**A fixed seed** keeps runs reproducible, which the evaluation report depends on.

---

## Step 9 — `evaluator.py`: the part most candidates get wrong

### Why token-level accuracy is never reported

A PII document is **>99% non-PII text**. A model predicting *nothing at all*
scores ~99% token accuracy while leaking every identifier. The metric is not
merely weak; it actively rewards failure.

### "Accuracy" is defined, because the word is ambiguous

The assignment asks for accuracy, but without true negatives it has no standard
meaning. This tool reports:

```
accuracy = TP / (TP + FP + FN)      (Jaccard index)
```

— the share of all entities in the comparison handled correctly. Unlike
`(TP+TN)/total`, it cannot be inflated by correctly-ignored text.

### Type-aware matching

Naïve exact matching measures typography, not detection. Three cases forced
changes:

| Problem | Effect before fix | Fix |
|---|---|---|
| En-dash vs hyphen (`Pune – 411 044`) | Same address counted as FP **and** FN | Normalise dashes |
| `+91 22 40094400` vs `+ 91 22 4009 4400` | Second spelling became an FP | Compare digits only; dedupe by digits |
| Address detected as 3 fragments | 1 TP + 2 FP | Fragments of one address count once |

Fixing these three measurement bugs moved overall F1 from **0.62 → 0.85** —
without changing a single detection rule. Worth stating clearly: that gain was
in the *measurement*, not the detector.

### Scoped annotation

Hand-annotating 4,686 paragraphs is infeasible. The `GENERAL INFORMATION`
section (units 468–606) was annotated **exhaustively**, and predictions are
restricted to the same region.

**Why the restriction is essential:** without it, every correct detection
elsewhere in the document counts as a false positive against annotations that
never covered it. The metric would punish the tool for working.

> **If asked "how do I know these numbers are real?"** — `scripts/run_evaluation.py`
> regenerates the entire report from an actual run. Nothing is typed by hand.

---

## Step 10 — Two corpora

The real prospectus contains **no** SSNs, credit cards or IP addresses, and its
only DOB is inside an image. Four required categories cannot be scored on it.

* **Synthetic fixture** — all nine required types plus deliberate distractors
  (invoice numbers, ticket IDs, version strings, ordinary dates). Scores 1.00.
* **Real prospectus** — messy, adversarial, false-positive-heavy. Scores 0.91 F1.

**Say this before anyone else does:** the synthetic 1.00 is a *fixture I wrote
myself*. It proves the plumbing works; it is not evidence of perfect detection.
The prospectus numbers are the meaningful ones.

---

## Step 11 — `app.py`: why background jobs

Redacting the full document takes ~60s — far too long to hold an HTTP request
open.

**Design:** upload → job on a background thread → browser polls a small JSON
endpoint → results page. This also produces a *real* progress indicator rather
than a spinner that lies.

**No database:** a job is a temporary artefact of one upload. Persistence would
be infrastructure with no purpose.

### The honesty decision in the UI

An arbitrary uploaded document **has no ground truth**, so precision and recall
are mathematically undefined for it.

The UI therefore always shows detection counts, but shows metrics **only** when
annotations exist. Showing invented numbers for an arbitrary upload would be
fabricating evaluation results — exactly what the brief forbids.

> **If asked "why no metrics on my upload?"** — because there's nothing to
> compare against. Making numbers up would be worse than showing none.

---

## Step 12 — Performance

First working version: **147s**. Now: **~60s**.

1. **Disabled unused spaCy components** (`parser`, `tagger`, `lemmatizer`) —
   NER doesn't need them.
2. **Batched with `nlp.pipe`** — pays model overhead once instead of 4,686
   times. The carrier-sentence retry is batched too.

No algorithmic cleverness; just not doing unnecessary work.

---

## Step 13 — Tests (54)

The tests that matter are the **negative** ones. "Does it find an email?" almost
never fails. These do the real work:

```python
def test_ordinary_dates_are_not_dob()          # 270-dates problem
def test_ssn_not_matched_for_invoice_number()  # shape collision
def test_credit_card_not_matched_for_ticket_id()
def test_ip_not_matched_for_version_string()
def test_identifiers_are_not_addresses()       # 6-digit ≠ postcode
def test_person_inside_email_is_not_separately_detected()
def test_same_name_different_role_gets_different_identity()
def test_same_name_same_role_is_merged_known_limitation()   # pins the limitation
def test_table_cells_are_processed()
def test_output_is_a_valid_openable_docx()
```

Note the pinned limitation test: it makes a known weakness *deliberate and
visible* instead of an accident waiting to regress.

---

## Numbers worth remembering

| Metric | Value |
|---|---|
| Real document — Precision / Recall / F1 | **0.95 / 0.93 / 0.94** |
| Synthetic fixture | 1.00 (fixture — plumbing check only) |
| Text units processed | 4,686 |
| PII occurrences replaced | 885 |
| Images neutralised | 8 / 8 |
| Tests | 54 passing |
| Runtime | ~60s |
| PERSON recall, before → after contact rule | 0.43 → 0.93 |
| Unique entities, before → after suffix rule | 806 → 283 |
| Overall F1, before → after measurement fixes | 0.62 → 0.85 |
| COMPANY P/R, before → after the suffix rule | 0.71/0.50 → 0.89/0.80 |

---

## The three questions to be ready for

**"What's the biggest weakness?"**
Company detection. It's the most constrained category by design — a candidate
needs a legal suffix and must not contain financial vocabulary — because that's
what stops hundreds of Title-Cased defined terms being redacted. The cost is
that a bare brand name like `Trilegal` is missed. I'd fix it with a gazetteer or
a larger spaCy model, not by dropping the constraints.

**"How do I know the tool actually works?"**
It reports its own error rate against exhaustive hand annotation, and the report
regenerates from a real run. Detection failures are listed by name in
`EVALUATION_REPORT.md` rather than summarised away.

**"What would you do next?"**
OCR for images. It's the largest known gap — the PAN card is real PII that this
tool currently blanks rather than understands.

---

## Step 14 — The correction pass (adversarial testing)

Running the tool on hostile *pasted text* — an address wedged between labelled
identifiers — found two defects that every earlier test had missed. Both are
worth knowing because they show where the earlier tests were too gentle.

### 14a. Address spans ran past the address

Input:

```
106 Example Street, Sector 7, Exampleton - 411 006, Example State, India
Company Code: KSH-2026-4812
Order ID: ORD-458293
Ticket ID: TKT-2026-08421
```

Output: `Company Code`, `Order ID` and `Ticket ID` were **all destroyed**.

Three independent causes, each needing its own fix:

| Cause | Fix |
|---|---|
| The regex tail took any 3 capitalised words after the postcode, so it ate `Ticket`, `ID`, `Company`, `Code` | Tail moved out of the regex into `extend_address_tail()`, which stops at a stop-word, a token containing digits, or a word followed by `:` |
| `POSTCODE` matches any six digits, so `ORD-458293` looked like an Indian PIN | `address_start_is_valid()` rejects a postcode preceded by an ID label or an all-caps code prefix |
| Paragraphs were glued with `", "`, so a label line looked like the address's next line | `LABELLED_FIELD` ends a block at any `Something:` paragraph |

A fourth surfaced from the test suite: `India-458293`. The separator now has to
be a comma or *spaced* dash, because every real address in the corpus spaces it.

> **If asked about span precision** — a regex can find where an address
> *starts*; deciding where it *ends* needs to know what the following words
> mean, which is a job for a stop-list, not a pattern.

### 14b. The same person got several fake identities — the serious one

`Sarthak Malvadkar` received **three** different fake names; `Rohit Kushal
Hegde` four. Eleven people were affected. Consistency is the headline
requirement, and it was broken on the real document while every test passed.

Cause: context was read from **anywhere in the paragraph**, so it was proximity
noise rather than a statement about that person. A paragraph mentioning the CEO
made the Company Secretary look like a CEO.

Four changes, each measured:

| Change | People with >1 identity |
|---|---|
| starting point | 11 |
| Global identity resolution instead of per-paragraph keys | 9 |
| Organisation demoted to a weak signal (roles must agree first) | 6 |
| Context read only from text **adjacent** to the name | 1 |
| Compound and abbreviated titles normalised (`CS` → `Company Secretary`) | **0** |

The resolver defaults to *merge*: mentions are one person unless the evidence
positively conflicts. Splitting one real person into several fakes makes the
document incoherent; merging two strangers is the milder failure.

> **If asked "what did adversarial testing find?"** — that the consistency
> guarantee was quietly broken on real data. Tests written from the happy path
> did not catch it; running the tool on hostile input did.

### 14c. A regression I introduced and caught

Filtering "Key Managerial Personnel" out of PERSON, I added `management` to
`DOMAIN_TERMS` — which also gates company detection, so it silently rejected
the real company `Nuvama Wealth Management Limited` and dropped COMPANY recall
from 0.80 to 0.70. Fixed by splitting the noun list into `PERSON_NOISE`
(person filter only) and `DOMAIN_TERMS` (both).

> Worth saying out loud: a shared blocklist between two filters is a trap.

### Before / after

| Metric | Before | After |
|---|---|---|
| Precision | 0.95 | **0.97** |
| Recall | 0.93 | 0.93 |
| F1 | 0.94 | **0.95** |
| False positives | 3 | **2** |
| People with >1 fake identity | **11** | **0** |
| Tests | 54 | **101** |

Precision and consistency improved; recall held. No ground truth was edited and
no metric was hand-adjusted.
