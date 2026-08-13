"""FastAPI web application: upload a document or paste text, get a redacted DOCX.

Redaction of a large document takes about a minute, which is far too long to
hold an HTTP request open. Each submission therefore becomes a job processed on
a background thread, and the browser polls a small JSON endpoint for progress.
That is also what gives the UI a real percentage instead of a spinner that lies,
and what makes the Cancel button possible without a task queue.

Jobs are held in a dictionary in memory. There is no database because nothing
here needs to outlive the process: a job is a temporary artefact of one upload.
"""

import shutil
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .document_utils import document_from_text
from .evaluator import EvaluationReport, evaluate, format_report, load_ground_truth
from .redactor import Cancelled, redact_document

ROOT = Path(__file__).resolve().parents[1]
JOBS_DIR = ROOT / "output" / "jobs"
TEMPLATES = Jinja2Templates(directory=str(ROOT / "templates"))

# Plain-text inputs are converted to DOCX before processing, so there is only
# ever one redaction implementation to maintain.
TEXT_SUFFIXES = {".txt", ".md", ".text", ".log", ".csv"}
DOCX_SUFFIXES = {".docx"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

app = FastAPI(title="SecureRedact")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

JOBS: Dict[str, dict] = {}
_LOCK = threading.Lock()


def _update(job_id: str, **fields) -> None:
    with _LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(fields)


def _make_progress(job_id: str):
    """Build the progress callback handed to the pipeline.

    It doubles as the cancellation check: raising inside the callback unwinds
    the pipeline immediately, which is why Cancel needs no extra machinery.
    """
    def progress(stage: str, fraction: float) -> None:
        if JOBS.get(job_id, {}).get("cancel_requested"):
            raise Cancelled()
        _update(job_id, stage=stage, percent=round(fraction * 100))

    return progress


def _process(job_id: str, docx_path: Path, gt_path: Optional[Path]) -> None:
    """Run the redaction pipeline for one job. Executed on a worker thread."""
    try:
        _update(job_id, status="running", stage="Starting", percent=0)

        out_path = docx_path.parent / f"{docx_path.stem} - REDACTED.docx"
        result = redact_document(
            str(docx_path), str(out_path), progress=_make_progress(job_id)
        )

        _update(job_id, stage="Building report", percent=99)
        report_md = _build_report(job_id, result, gt_path)
        report_path = docx_path.parent / "evaluation_report.md"
        report_path.write_text(report_md, encoding="utf-8")

        _update(
            job_id,
            status="done",
            stage="Complete",
            percent=100,
            result=result,
            output_path=str(out_path),
            report_path=str(report_path),
            finished_at=datetime.now().isoformat(timespec="seconds"),
        )
    except Cancelled:
        _update(job_id, status="cancelled", stage="Cancelled by user")
    except Exception as exc:                      # noqa: BLE001 - surfaced to UI
        _update(
            job_id,
            status="error",
            stage="Failed",
            error=str(exc),
            traceback=traceback.format_exc(),
        )


def _build_report(job_id: str, result, gt_path: Optional[Path]) -> str:
    """Assemble the Markdown evaluation report for one run.

    Metrics are produced only when ground truth was supplied. For an arbitrary
    document there is nothing to compare against, so precision and recall are
    undefined - quoting numbers anyway would mean inventing them.
    """
    job = JOBS[job_id]
    lines = [
        "# PII Redaction - Evaluation Report",
        "",
        f"- **Document:** {job['filename']}",
        f"- **Generated:** {datetime.now().isoformat(timespec='seconds')}",
        f"- **Text units scanned:** {result.text_units}",
        f"- **PII occurrences replaced:** {result.total_entities}",
        f"- **Unique entities:** {len(result.replacement_map)}",
        f"- **Images found / neutralised:** {result.images_found} / {result.images_blanked}",
        "",
        "## Detected entities by category",
        "",
        "| PII Type | Occurrences |",
        "|---|---|",
    ]
    for etype, count in sorted(result.counts_by_type.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {etype} | {count} |")

    lines += ["", "## Detection quality", ""]

    if gt_path is None:
        lines += [
            "No ground-truth annotations were supplied for this document, so "
            "precision, recall and F1 cannot be computed.",
            "",
            "These metrics require knowing the correct answer in advance. "
            "Quoting numbers without annotations would mean inventing them. To "
            "score a document, upload a ground-truth JSON alongside it: a list "
            'of `{"text": "...", "type": "..."}` objects.',
        ]
    else:
        report: EvaluationReport = evaluate(
            [(e.text, e.type) for e in result.entities],
            load_ground_truth(str(gt_path)),
            corpus=job["filename"],
        )
        with _LOCK:
            JOBS[job_id]["evaluation"] = report

        lines += [
            "Entity-level evaluation. *Accuracy* here is TP / (TP + FP + FN) - "
            "see the README for why token-level accuracy is not reported.",
            "",
            format_report(report),
        ]
        if report.false_positives:
            lines += ["", "### False positives", ""]
            lines += [f"- `{t}` ({ty})" for t, ty in report.false_positives[:40]]
        if report.false_negatives:
            lines += ["", "### False negatives", ""]
            lines += [f"- `{t}` ({ty})" for t, ty in report.false_negatives[:40]]

    return "\n".join(lines)


def _start_job(job_dir: Path, docx_path: Path, display_name: str,
               gt_path: Optional[Path]) -> str:
    job_id = job_dir.name
    with _LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "filename": display_name,
            "status": "queued",
            "stage": "Queued",
            "percent": 0,
            "has_ground_truth": gt_path is not None,
            "cancel_requested": False,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }

    threading.Thread(
        target=_process, args=(job_id, docx_path, gt_path), daemon=True
    ).start()
    return job_id


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return TEMPLATES.TemplateResponse("index.html", {"request": request})


@app.post("/redact")
async def start_redaction(
    document: Optional[UploadFile] = File(None),
    pasted_text: Optional[str] = Form(None),
    ground_truth: Optional[UploadFile] = File(None),
):
    """Accept a file upload or pasted text and start a background job."""
    job_dir = JOBS_DIR / uuid.uuid4().hex[:12]
    job_dir.mkdir(parents=True, exist_ok=True)

    # --- Pasted text ------------------------------------------------------
    if pasted_text and pasted_text.strip():
        docx_path = job_dir / "pasted_text.docx"
        document_from_text(pasted_text, str(docx_path))
        display_name = "pasted_text.docx"

    # --- Uploaded file ----------------------------------------------------
    elif document is not None and document.filename:
        suffix = Path(document.filename).suffix.lower()
        if suffix not in DOCX_SUFFIXES | TEXT_SUFFIXES:
            raise HTTPException(
                400,
                "Unsupported file type. Upload a .docx or a plain-text file "
                "(.txt, .md, .csv, .log).",
            )

        raw_path = job_dir / document.filename
        with raw_path.open("wb") as fh:
            shutil.copyfileobj(document.file, fh)

        if raw_path.stat().st_size > MAX_UPLOAD_BYTES:
            raise HTTPException(400, "File exceeds the 50 MB limit.")

        if suffix in TEXT_SUFFIXES:
            # Convert to DOCX so a single pipeline handles every input, and so
            # the output is always a .docx as required.
            text = raw_path.read_text(encoding="utf-8", errors="replace")
            docx_path = job_dir / f"{raw_path.stem}.docx"
            document_from_text(text, str(docx_path))
        else:
            docx_path = raw_path

        display_name = document.filename
    else:
        raise HTTPException(400, "Provide a file or paste some text.")

    gt_path = None
    if ground_truth is not None and ground_truth.filename:
        gt_path = job_dir / "ground_truth.json"
        with gt_path.open("wb") as fh:
            shutil.copyfileobj(ground_truth.file, fh)

    job_id = _start_job(job_dir, docx_path, display_name, gt_path)
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(request: Request, job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    return TEMPLATES.TemplateResponse(
        "result.html", {"request": request, "job": job, "job_id": job_id}
    )


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """Ask a running job to stop; the progress callback does the rest."""
    if job_id not in JOBS:
        raise HTTPException(404, "Unknown job.")
    _update(job_id, cancel_requested=True)
    return JSONResponse({"ok": True})


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    """Small JSON payload the results page polls while a job runs."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")

    payload = {
        "status": job["status"],
        "stage": job.get("stage", ""),
        "percent": job.get("percent", 0),
        "filename": job["filename"],
        "error": job.get("error"),
    }

    result = job.get("result")
    if result is not None:
        payload["summary"] = {
            "text_units": result.text_units,
            "total_entities": result.total_entities,
            "unique_entities": len(result.replacement_map),
            "images_found": result.images_found,
            "images_blanked": result.images_blanked,
            "counts_by_type": result.counts_by_type,
        }

    report = job.get("evaluation")
    if report is not None:
        payload["metrics"] = {
            "overall": {
                "precision": round(report.overall.precision, 4),
                "recall": round(report.overall.recall, 4),
                "f1": round(report.overall.f1, 4),
                "accuracy": round(report.overall.accuracy, 4),
                "tp": report.overall.tp,
                "fp": report.overall.fp,
                "fn": report.overall.fn,
            },
            "by_type": {
                etype: {
                    "precision": round(m.precision, 4),
                    "recall": round(m.recall, 4),
                    "f1": round(m.f1, 4),
                    "support": m.support,
                }
                for etype, m in report.by_type.items()
            },
        }
    else:
        payload["metrics"] = None
        payload["metrics_note"] = (
            "Scoring needs a ground-truth file. Without one there is nothing "
            "to compare against, so precision and recall are undefined."
        )

    return JSONResponse(payload)


@app.get("/jobs/{job_id}/download/{kind}")
def download(job_id: str, kind: str):
    job = JOBS.get(job_id)
    if job is None or job.get("status") != "done":
        raise HTTPException(404, "Result not ready.")

    if kind == "docx":
        path = Path(job["output_path"])
        media = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif kind == "report":
        path = Path(job["report_path"])
        media = "text/markdown"
    else:
        raise HTTPException(404, "Unknown download.")

    return FileResponse(path, media_type=media, filename=path.name)
