"""Any-policy ingestion: a single front door that turns whatever a user has —
a PDF on disk, a web page, a URL, or pasted text — into clean policy text the
coverage extractor can consume.

This is what backs the "drop in *any* policy" promise. The rest of the atlas
pipeline (extract → ground → x-ray → adjudicate) only ever sees the normalized
`IngestedPolicy.text`, so adding a new input format means adding a branch here
and nothing downstream changes.

Files reuse the retrieval loaders (`load_any`, already handling md/pdf/txt/
html). URLs are fetched with httpx and dispatched by content-type. Raw text is
passed through. Title and source are best-effort so the extractor and the
generated artifacts can attribute every clause.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.logging import get_logger
from app.retrieval.loaders import html_to_text, load_any

log = get_logger(__name__)

# A polite, real-browser-ish UA — some publishers 403 the default httpx UA.
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_MAX_BYTES = 20 * 1024 * 1024  # 20 MB guard for a single policy fetch


@dataclass
class IngestedPolicy:
    """Normalized policy text plus where it came from."""

    text: str
    title: str
    source: str
    fmt: str  # markdown | pdf | text | html
    meta: dict


def _looks_like_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _title_from_text(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            # use the first non-empty line if it's heading-ish/short
            return (line[2:].strip() if line.startswith("# ") else line)[:120]
    return fallback


def _pdf_bytes_to_text(data: bytes) -> str:
    try:
        import fitz  # type: ignore[import-untyped]  # PyMuPDF
    except ImportError as e:  # pragma: no cover - dep is declared
        raise RuntimeError(
            "PDF ingestion requires PyMuPDF (pymupdf), which is a declared dependency."
        ) from e
    doc = fitz.open(stream=data, filetype="pdf")
    return "\n\n".join(page.get_text("text") for page in doc).strip()


def ingest_from_url(url: str, *, timeout: float = 30.0) -> IngestedPolicy:
    """Fetch a policy from the web. Dispatches PDF vs HTML vs text by the
    response content-type (falling back to the URL suffix)."""
    import httpx

    with httpx.Client(follow_redirects=True, timeout=timeout, headers={"User-Agent": _UA}) as client:
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.content[:_MAX_BYTES]
        ctype = resp.headers.get("content-type", "").lower()

    suffix = Path(url.split("?", 1)[0]).suffix.lower()
    is_pdf = "application/pdf" in ctype or suffix == ".pdf" or data[:5] == b"%PDF-"
    is_html = "text/html" in ctype or "application/xhtml" in ctype or suffix in (".html", ".htm")

    if is_pdf:
        text, fmt = _pdf_bytes_to_text(data), "pdf"
    elif is_html:
        text, fmt = html_to_text(data.decode("utf-8", errors="replace")), "html"
    else:
        text, fmt = data.decode("utf-8", errors="replace"), "text"

    text = text.strip()
    if not text:
        raise ValueError(f"Fetched {url} but extracted no text (content-type: {ctype!r}).")
    title = _title_from_text(text, url.rsplit("/", 1)[-1] or url)
    log.info("atlas.ingest.url", url=url, fmt=fmt, chars=len(text))
    return IngestedPolicy(text=text, title=title, source=url, fmt=fmt, meta={"content_type": ctype})


def ingest_from_file(path: Path) -> IngestedPolicy:
    loaded = load_any(path)
    text = loaded.text.strip()
    if not text:
        raise ValueError(f"{path} produced no text.")
    log.info("atlas.ingest.file", path=str(path), fmt=loaded.metadata.get("format"), chars=len(text))
    return IngestedPolicy(
        text=text,
        title=loaded.title,
        source=str(path),
        fmt=str(loaded.metadata.get("format", path.suffix.lstrip("."))),
        meta=loaded.metadata,
    )


def ingest_from_text(text: str, *, title: str = "", source: str = "pasted-text") -> IngestedPolicy:
    text = text.strip()
    if not text:
        raise ValueError("Empty policy text.")
    return IngestedPolicy(
        text=text,
        title=title or _title_from_text(text, "Untitled policy"),
        source=source,
        fmt="text",
        meta={},
    )


def ingest_policy(source: str) -> IngestedPolicy:
    """The one entry point. `source` may be an http(s) URL, a path to a local
    file (pdf/txt/md/html), or — if it's neither — raw policy text.

    The raw-text fallback only triggers when `source` is not a URL and not an
    existing path, so a typo'd path surfaces as a clear file error rather than
    being silently treated as a 12-character "policy".
    """
    if _looks_like_url(source):
        return ingest_from_url(source)
    # Decide path-vs-text BEFORE touching the filesystem: a multiline or very
    # long string is policy text, not a path — and Path.exists() raises OSError
    # ("File name too long" / embedded NUL) on such input rather than returning
    # False.
    looks_like_text = "\n" in source or len(source) > 200 or "\x00" in source
    if not looks_like_text:
        try:
            p = Path(source).expanduser()
            if p.exists():
                return ingest_from_file(p)
        except OSError:
            pass
    if looks_like_text:
        return ingest_from_text(source)
    raise FileNotFoundError(
        f"Not a URL, not an existing file, and too short to be policy text: {source!r}"
    )
