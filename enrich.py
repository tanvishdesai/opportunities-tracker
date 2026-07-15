"""
Enrichment layer: turns raw scraped postings into a filtered, readable
digest using the Gemini API.

For each new posting this tries to:
  1. Locate the underlying official notification (usually a PDF) --
     either directly (for "pages" sources) or by fetching the
     Faculty Tick article and pulling its "Reference: ..." / "View
     Official Notification" link out.
  2. Send the title (+ PDF, if fetched) to Gemini, asking it to judge
     relevance against your research profile and extract the facts
     that actually matter (deadline, stipend, eligibility, how to
     apply) as strict JSON.
  3. Render a clean HTML email and a condensed WhatsApp text from the
     structured results.

If GEMINI_API_KEY isn't set, or a given item can't be enriched for any
reason, callers fall back to the plain unfiltered listing -- nothing
is ever silently dropped just because the LLM step had a problem.
"""

import base64
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

USER_AGENT = (
    "Mozilla/5.0 (compatible; PersonalResearchTracker/1.0; "
    "personal-use, low-frequency polling)"
)

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

ANCHOR_RE = re.compile(r'<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', re.I | re.S)
TAG_STRIP_RE = re.compile(r"<[^>]+>")
REFERENCE_TEXT_RE = re.compile(r"Reference:\s*(https?://\S+)", re.I)
NOTIFICATION_ANCHOR_HINTS = ("official notification", "notification", "advertisement", "view official")
GDRIVE_FILE_RE = re.compile(r"drive\.google\.com/file/d/([^/]+)/")

DEFAULT_PROFILE = (
    "Final-year Computer Science undergraduate researcher. Core interests: "
    "multimodal deepfake detection, adversarial and certified robustness, "
    "explainable AI (XAI), trustworthy/safe machine learning, computer "
    "vision, NLP, and machine learning broadly (including general Data "
    "Science / AI roles). NOT interested in postings in unrelated "
    "disciplines (pure civil, mechanical, chemical, biology, medical, "
    "humanities, law, physics, pure mathematics) UNLESS the posting "
    "explicitly involves AI/ML/computer-vision/data-science methods -- "
    "e.g. 'AI for structural health monitoring' in a Civil Engineering "
    "department DOES count."
)


def clean_text(raw_html_fragment):
    text = TAG_STRIP_RE.sub("", raw_html_fragment)
    text = re.sub(r"\s+", " ", text).strip()
    for a, b in (("&amp;", "&"), ("&nbsp;", " "), ("&#8211;", "-"), ("&#8217;", "'")):
        text = text.replace(a, b)
    return text


def fetch_bytes(url, timeout=25, max_bytes=20_000_000):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content_type = resp.headers.get("Content-Type", "")
        data = resp.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"response exceeded {max_bytes} bytes, refusing")
        return data, content_type


def normalize_pdf_url(url):
    """Google Drive 'view' links don't return raw PDF bytes -- rewrite
    them to the direct-download form."""
    m = GDRIVE_FILE_RE.search(url)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    return url


def find_notification_url(article_html):
    """Best-effort: pull the official notification/PDF link out of a
    Faculty Tick article page. Returns None if nothing recognizable is
    found -- callers fall back to title-only classification.

    Checked in order of reliability: a direct .pdf href (works
    regardless of what the link text says -- many articles just show
    the raw URL as the link text, which hint-word matching would miss
    entirely), a Google Drive share link, the plain-text "Reference:
    <url>" pattern some articles use, and finally hint words in the
    anchor text as a last resort."""
    anchors = ANCHOR_RE.findall(article_html)

    for href, _ in anchors:
        if href.lower().split("?")[0].endswith(".pdf"):
            return href

    for href, _ in anchors:
        if GDRIVE_FILE_RE.search(href):
            return normalize_pdf_url(href)

    m = REFERENCE_TEXT_RE.search(article_html)
    if m:
        return normalize_pdf_url(m.group(1).rstrip(".,)"))

    for href, inner in anchors:
        text = clean_text(inner).lower()
        if any(hint in text for hint in NOTIFICATION_ANCHOR_HINTS):
            return normalize_pdf_url(href)
    return None


def try_fetch_pdf(url):
    """Returns base64-encoded PDF bytes, or None if the URL doesn't
    actually yield a PDF (wrong content-type, too big, network error,
    login wall, etc). Never raises -- this is best-effort."""
    if not url:
        return None
    try:
        data, content_type = fetch_bytes(url, max_bytes=20_000_000)
    except Exception as e:
        print(f"[enrich] couldn't fetch candidate doc {url}: {e}", file=sys.stderr)
        return None
    looks_like_pdf = data[:5] == b"%PDF-" or "pdf" in content_type.lower() or url.lower().endswith(".pdf")
    if not looks_like_pdf:
        return None
    return base64.b64encode(data).decode("ascii")


def resolve_document(item):
    """For a scraped item, work out the best URL to try fetching a PDF
    from, then attempt the fetch. Returns base64 PDF data or None.

    Content-sniffs FIRST rather than guessing from the file extension --
    plenty of government sites serve PDFs from extension-less routes
    (e.g. /notices/download?id=123), and guessing wrong used to mean we
    tried to parse raw PDF bytes as HTML and found nothing."""
    url = item["url"]
    try:
        data, content_type = fetch_bytes(url, max_bytes=20_000_000)
    except Exception as e:
        print(f"[enrich] couldn't fetch {url}: {e}", file=sys.stderr)
        return None

    if data[:5] == b"%PDF-" or "pdf" in content_type.lower():
        return base64.b64encode(data).decode("ascii")

    # Not a PDF itself -- treat it as an article/listing page and look
    # for an embedded notification link.
    page_html = data.decode("utf-8", errors="replace")
    doc_url = find_notification_url(page_html)
    if not doc_url:
        return None
    return try_fetch_pdf(doc_url)


def call_gemini(model, api_key, parts, timeout=60, max_retries=4):
    """Calls generateContent, retrying with backoff on 429 (rate limit)
    and transient 5xx errors. Honors the Retry-After header when the
    API sends one; otherwise backs off 5s, 10s, 20s, 40s."""
    url = GEMINI_ENDPOINT.format(model=model)
    body = json.dumps(
        {
            "contents": [{"parts": parts}],
            "generationConfig": {"responseMimeType": "application/json"},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )

    backoff = 5
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            retryable = e.code == 429 or 500 <= e.code < 600
            if retryable and attempt < max_retries:
                retry_after = e.headers.get("Retry-After") if e.headers else None
                wait = float(retry_after) if retry_after else backoff
                print(f"[enrich] Gemini {e.code}, retrying in {wait:.0f}s (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
                time.sleep(wait)
                backoff *= 2
                continue
            raise


PROMPT_TEMPLATE = """You are helping a researcher evaluate whether an academic job/research posting from an Indian IIT/NIT/IIIT is relevant to them, and extract the key facts.

Their research profile:
{profile}

Posting title: {title}
Source: {source}

{doc_note}

Return ONLY a JSON object, no markdown code fences, with exactly these fields:
{{
  "relevant": true or false,
  "confidence": "high" or "medium" or "low",
  "institute": string,
  "department": string or null,
  "role": string,
  "topic_one_line": string or null,
  "eligibility_summary": string or null (max 2 short sentences),
  "stipend": string or null,
  "deadline": string or null,
  "apply_method": string or null (a link or email address),
  "contact": string or null
}}

Judge "relevant" on genuine fit to the profile, not keyword overlap. A Civil Engineering posting that merely mentions "data collection" is NOT relevant. A Mechanical Engineering posting on "AI-driven predictive maintenance" or a Civil Engineering posting on "ML for structural health monitoring" IS relevant."""


def classify_and_extract(item, profile, model, api_key, pdf_b64):
    if pdf_b64:
        doc_note = "A copy of the official notification PDF is attached above -- use it as the primary source."
        parts = [
            {"inline_data": {"mime_type": "application/pdf", "data": pdf_b64}},
            {"text": PROMPT_TEMPLATE.format(profile=profile, title=item["title"], source=item["source"], doc_note=doc_note)},
        ]
    else:
        doc_note = "No PDF could be retrieved for this posting -- judge from the title alone, and set confidence to \"low\"."
        parts = [{"text": PROMPT_TEMPLATE.format(profile=profile, title=item["title"], source=item["source"], doc_note=doc_note)}]

    raw_text = call_gemini(model, api_key, parts)
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```[a-zA-Z]*\n?", "", raw_text)
        raw_text = re.sub(r"\n?```$", "", raw_text)
    return json.loads(raw_text)


def enrich_new_items(new_items, profile, model, api_key, max_items=40, request_delay=7):
    """Returns (relevant, not_relevant, unprocessed) -- three lists of
    dicts. `unprocessed` holds items where enrichment itself failed
    (network error, bad JSON, etc) so nothing gets silently lost; these
    should still be shown to the user, just without the nice details.

    request_delay paces the Gemini calls (seconds between each item) to
    stay under free-tier RPM limits (~10 RPM for gemini-2.5-flash) --
    call_gemini's own retry/backoff handles it if you still get 429'd."""
    relevant, not_relevant, unprocessed = [], [], []
    items_to_process = new_items[:max_items]

    for i, item in enumerate(items_to_process):
        print(f"[enrich] processing {i + 1}/{len(items_to_process)}: {item['title'][:70]}", file=sys.stderr)
        try:
            pdf_b64 = resolve_document(item)
            result = classify_and_extract(item, profile, model, api_key, pdf_b64)
            result["_source_item"] = item
            result["_had_pdf"] = pdf_b64 is not None
            if result.get("relevant"):
                relevant.append(result)
            else:
                not_relevant.append(result)
        except Exception as e:
            print(f"[enrich] failed to enrich '{item['title']}': {e}", file=sys.stderr)
            unprocessed.append(item)

        if i < len(items_to_process) - 1:
            time.sleep(request_delay)

    skipped = len(new_items) - len(items_to_process)
    if skipped > 0:
        print(f"[enrich] {skipped} items skipped this run (max_items_to_enrich cap)", file=sys.stderr)

    return relevant, not_relevant, unprocessed


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------

def _field(d, key, fallback="Not specified"):
    val = d.get(key)
    return html.escape(str(val)) if val else fallback


def render_email_html(relevant, unprocessed, run_date):
    cards = []
    for r in relevant:
        item = r["_source_item"]
        confidence_note = "" if r.get("confidence") == "high" else (
            f'<div style="color:#a15c00;font-size:12px;margin-top:6px;">'
            f'⚠ extracted with {html.escape(r.get("confidence","low"))} confidence -- verify against the source</div>'
        )
        cards.append(f"""
<div style="border:1px solid #ddd;border-radius:10px;padding:16px 20px;margin-bottom:16px;font-family:Arial,sans-serif;">
  <div style="font-size:16px;font-weight:bold;color:#1a1a1a;">{_field(r,'role')}</div>
  <div style="font-size:14px;color:#444;margin-top:2px;">{_field(r,'institute')}{' — ' + html.escape(r['department']) if r.get('department') else ''}</div>
  {'<div style="margin-top:8px;font-size:14px;color:#222;">' + html.escape(r['topic_one_line']) + '</div>' if r.get('topic_one_line') else ''}
  <table style="margin-top:10px;font-size:13px;color:#333;">
    <tr><td style="padding:2px 10px 2px 0;color:#777;">Deadline</td><td><b>{_field(r,'deadline')}</b></td></tr>
    <tr><td style="padding:2px 10px 2px 0;color:#777;">Stipend</td><td>{_field(r,'stipend')}</td></tr>
    <tr><td style="padding:2px 10px 2px 0;color:#777;">Eligibility</td><td>{_field(r,'eligibility_summary')}</td></tr>
    <tr><td style="padding:2px 10px 2px 0;color:#777;">Apply via</td><td>{_field(r,'apply_method')}</td></tr>
  </table>
  <div style="margin-top:10px;"><a href="{html.escape(item['url'])}" style="font-size:13px;color:#1a73e8;">Original posting →</a></div>
  {confidence_note}
</div>""")

    unprocessed_html = ""
    if unprocessed:
        rows = "".join(
            f'<li style="margin-bottom:4px;"><a href="{html.escape(it["url"])}">{html.escape(it["title"])}</a> '
            f'<span style="color:#888;">({html.escape(it["source"])})</span></li>'
            for it in unprocessed
        )
        unprocessed_html = f"""
<div style="margin-top:24px;font-family:Arial,sans-serif;font-size:13px;color:#555;">
  <b>Couldn't auto-filter these (check manually):</b>
  <ul>{rows}</ul>
</div>"""

    body = "".join(cards) if cards else '<p style="font-family:Arial,sans-serif;">No relevant postings this run.</p>'
    return f"""<html><body style="margin:0;padding:20px;background:#fafafa;">
<h2 style="font-family:Arial,sans-serif;color:#1a1a1a;">Research opportunities — {run_date}</h2>
{body}
{unprocessed_html}
</body></html>"""


def render_whatsapp_text(relevant, unprocessed):
    lines = [f"Research opportunities — {len(relevant)} relevant"]
    for r in relevant:
        item = r["_source_item"]
        lines.append(f"\n*{r.get('role','?')}* — {r.get('institute','?')}")
        if r.get("deadline"):
            lines.append(f"Deadline: {r['deadline']}")
        lines.append(item["url"])
    if unprocessed:
        lines.append(f"\n({len(unprocessed)} more couldn't be auto-filtered -- check email/repo)")
    return "\n".join(lines)