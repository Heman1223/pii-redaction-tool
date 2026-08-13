from src.document_utils import document_from_text, load_document, iter_text_units
from src import redact_document
import tempfile, os

TXT = """Ananya Mehta joined the company last year.
Ananya Mehta, Director, Acme Technologies Ltd.
Ananya Mehta signed the register again.
Phone: +91 98765 43210
Alt contact: +91-9876543210
Tejas Kaul, Director at Choudhury Industries Limited
Tejas Kaul, Director at Choudhury Industries Limited
Rushil Saini, Senior Manager at Chahal Solutions Private Limited
Email: rushil.saini@example.com
Mobile: +91 90000 00004
Server: 192.0.2.4
Example State, India
12 Example Road, Sector 4: ORD-458293
Ticket ID: TKT-2026-08421
Status code: 200"""

d = tempfile.mkdtemp()
src = os.path.join(d, "adv.docx"); document_from_text(TXT, src)
out = os.path.join(d, "adv_r.docx")
r = redact_document(src, out)

print("=== DETECTED ===")
for e in r.entities:
    print(f"   {e.type:<11} {e.text!r}")
print()
print("=== MAPPING ===")
for k, v in sorted(r.replacement_map.items()):
    print(f"   {k:<62} -> {v}")
print()
print("=== OUTPUT ===")
for u in iter_text_units(load_document(out)):
    print("  ", u.text)
