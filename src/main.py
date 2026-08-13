"""Command-line entry point.

    python -m src.main input.docx --output redacted.docx

Kept deliberately thin: it parses arguments, calls the library, and prints a
summary. All logic lives in the modules it calls, so the CLI and the web app
share exactly one implementation.
"""

import argparse
import json
import sys
from pathlib import Path

from .evaluator import evaluate, format_report, load_ground_truth
from .redactor import redact_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pii-redactor",
        description="Replace personal information in a DOCX with realistic fakes.",
    )
    parser.add_argument("input", help="path to the .docx file to redact")
    parser.add_argument(
        "-o", "--output",
        help="output path (default: '<input> - REDACTED.docx')",
    )
    parser.add_argument(
        "--ground-truth",
        help="optional annotations JSON; enables precision/recall scoring",
    )
    parser.add_argument(
        "--mapping",
        help="optional path to write the original -> fake mapping as JSON",
    )
    parser.add_argument(
        "--no-ner", action="store_true",
        help="disable spaCy and use deterministic rules only (much faster)",
    )
    parser.add_argument(
        "--keep-images", action="store_true",
        help="do not blank embedded images (images may contain unreadable PII)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="seed for fake value generation (default: 42, keeps runs reproducible)",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    source = Path(args.input)
    if not source.exists():
        print(f"error: no such file: {source}", file=sys.stderr)
        return 1

    destination = Path(args.output) if args.output else source.with_name(
        f"{source.stem} - REDACTED.docx"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    result = redact_document(
        str(source),
        str(destination),
        use_ner=not args.no_ner,
        seed=args.seed,
        redact_images=not args.keep_images,
    )

    print(f"Redacted: {destination}")
    print(f"  text units scanned      : {result.text_units}")
    print(f"  PII occurrences replaced: {result.total_entities}")
    print(f"  unique entities         : {len(result.replacement_map)}")
    print(f"  images blanked          : {result.images_blanked}/{result.images_found}")

    if result.counts_by_type:
        print("\n  By category:")
        for etype, count in sorted(result.counts_by_type.items(), key=lambda kv: -kv[1]):
            print(f"    {etype:<12} {count}")

    if args.mapping:
        Path(args.mapping).write_text(
            json.dumps(result.replacement_map, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n  mapping written to {args.mapping}")

    if args.ground_truth:
        report = evaluate(
            [(e.text, e.type) for e in result.entities],
            load_ground_truth(args.ground_truth),
            corpus=source.name,
        )
        print("\n" + format_report(report))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
