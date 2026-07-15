#!/usr/bin/env python3
"""
IIT/NIT Research Opportunity Tracker
-------------------------------------
Pulls Research Associate / JRF / SRF / Project Associate / Postdoc style
postings from:
  1. Faculty Tick (facultytick.com) — an existing aggregator that already
     scrapes these notices across IITs/NITs/IIITs, via its RSS feeds.
  2. A handful of official institute pages (config.json -> "pages"),
     scraped with a best-effort regex link extractor.

Only genuinely NEW postings (not seen in a previous run) are written to
digest.md. A running log of everything ever seen is kept in
all_postings.csv. State (which URLs have been seen, when the tracker
last ran) is kept in seen_urls.json / last_run.json so it survives
between cron runs.

Zero third-party dependencies — only the Python standard library.
Tested logic (RSS parsing, HTML link extraction) against real fetched
markup; the live network calls themselves could not be exercised in the
environment this was written in, so do one manual test run before you
trust the cron job blindly. See README.md.
"""

import csv
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

HERE = Path(__file__).parent
CONFIG_FILE = HERE / "config.json"
STATE_FILE = HERE / "seen_urls.json"
LAST_RUN_FILE = HERE / "last_run.json"
CSV_FILE = HERE / "all_postings.csv"
DIGEST_FILE = HERE / "digest.md"

USER_AGENT = (
    "Mozilla/5.0 (compatible; PersonalResearchTracker/1.0; "
    "personal-use, low-frequency polling)"
)

# WordPress post permalinks look like /2026/07/15/some-post-title/
WP_POST_URL_RE = re.compile(r'https?://[^\s"\'<>]+/\d{4}/\d{2}/\d{2}/[a-z0-9\-]+/?')
ANCHOR_RE = re.compile(r'<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', re.I | re.S)
TAG_STRIP_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def load_json(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[WARN] {path.name} is corrupt, starting fresh", file=sys.stderr)
    return default


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def fetch(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def clean_text(raw_html_fragment):
    text = TAG_STRIP_RE.sub("", raw_html_fragment)
    text = WS_RE.sub(" ", text).strip()
    # Unescape the handful of entities that show up constantly in these titles
    for a, b in (("&amp;", "&"), ("&nbsp;", " "), ("&#8211;", "-"), ("&#8217;", "'")):
        text = text.replace(a, b)
    return text


def parse_rss(raw_bytes):
    """Returns a list of {title, url, published} or None if not valid RSS."""
    try:
        root = ET.fromstring(raw_bytes)
    except ET.ParseError:
        return None
    channel = root.find("channel")
    if channel is None:
        return None
    items = []
    for item in channel.findall("item"):
        title = clean_text(item.findtext("title") or "")
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        if title and link:
            items.append({"title": title, "url": link, "published": pub})
    return items


def extract_wp_style_links(html_text):
    """Fallback for facultytick-style pages if the feed is unavailable:
    pull anything that looks like a WordPress post permalink plus its
    anchor text."""
    items = []
    seen_urls = set()
    for href, inner in ANCHOR_RE.findall(html_text):
        if not WP_POST_URL_RE.match(href):
            continue
        title = clean_text(inner)
        if not title or href in seen_urls:
            continue
        seen_urls.add(href)
        items.append({"title": title, "url": href, "published": ""})
    return items


def extract_generic_links(html_text, keywords):
    """Best-effort fallback for official institute listing pages: grab
    every <a> whose visible text matches one of our keywords. Works well
    for simple bullet-list notice pages; will find nothing useful on
    JS-rendered portals (SAP apps, login-gated systems, React SPAs) --
    those need a headless browser, which is out of scope here."""
    items = []
    for href, inner in ANCHOR_RE.findall(html_text):
        title = clean_text(inner)
        if not title:
            continue
        low = title.lower()
        if any(k in low for k in keywords):
            items.append({"title": title, "url": href, "published": ""})
    return items


def matches_keywords(title, include_kw, field_kw):
    t = title.lower()
    if not any(k in t for k in include_kw):
        return False
    if field_kw and not any(k.lower() in t for k in field_kw):
        return False
    return True


def collect_feed(name, url):
    try:
        raw = fetch(url)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"[WARN] Could not fetch feed '{name}' ({url}): {e}", file=sys.stderr)
        return []

    items = parse_rss(raw)
    if items is None:
        print(f"[INFO] '{name}' didn't parse as RSS, trying HTML link fallback", file=sys.stderr)
        try:
            html_text = raw.decode("utf-8", errors="replace")
        except Exception:
            return []
        items = extract_wp_style_links(html_text)

    for it in items:
        it["source"] = name
    print(f"[INFO] {name}: {len(items)} items", file=sys.stderr)
    return items


def collect_page(name, url, include_kw):
    try:
        raw = fetch(url)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"[WARN] Could not fetch page '{name}' ({url}): {e}", file=sys.stderr)
        return []
    html_text = raw.decode("utf-8", errors="replace")
    items = extract_generic_links(html_text, include_kw)
    for it in items:
        it["source"] = name
        # relative hrefs are common on these pages; best-effort absolutize
        if it["url"].startswith("/"):
            m = re.match(r"https?://[^/]+", url)
            if m:
                it["url"] = m.group(0) + it["url"]
    print(f"[INFO] {name}: {len(items)} candidate links", file=sys.stderr)
    return items


def should_run_now(interval_days):
    last = load_json(LAST_RUN_FILE, None)
    if last is None:
        return True
    last_dt = datetime.fromisoformat(last["last_run"])
    elapsed = datetime.now(timezone.utc) - last_dt
    return elapsed.days >= interval_days


def main():
    config = load_json(CONFIG_FILE, {})
    interval_days = config.get("run_interval_days", 4)

    if not should_run_now(interval_days):
        print(f"[INFO] Ran less than {interval_days} days ago, skipping this trigger.")
        return

    include_kw = [k.lower() for k in config.get("keywords_include", [])]
    field_kw = config.get("keywords_field_filter", [])

    seen = set(load_json(STATE_FILE, []))
    raw_items = []

    for feed in config.get("feeds", []):
        raw_items.extend(collect_feed(feed["name"], feed["url"]))
    for page in config.get("pages", []):
        raw_items.extend(collect_page(page["name"], page["url"], include_kw))

    filtered = [it for it in raw_items if matches_keywords(it["title"], include_kw, field_kw)]
    new_items = [it for it in filtered if it["url"] not in seen]

    fetched_at = datetime.now(timezone.utc).isoformat()
    write_header = not CSV_FILE.exists()
    with CSV_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fetched_at", "source", "title", "url", "published"])
        if write_header:
            writer.writeheader()
        for it in new_items:
            writer.writerow({"fetched_at": fetched_at, **it})

    seen.update(it["url"] for it in filtered)
    save_json(STATE_FILE, list(seen)[-5000:])  # cap unbounded growth
    save_json(LAST_RUN_FILE, {"last_run": fetched_at})

    lines = [f"# New research opportunities — {datetime.now().strftime('%Y-%m-%d')}", ""]
    if new_items:
        for it in new_items:
            pub = f"  _{it['published']}_" if it["published"] else ""
            lines.append(f"- **[{it['source']}]** [{it['title']}]({it['url']}){pub}")
    else:
        lines.append("No new postings matching your keywords since the last run.")
    DIGEST_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nDone. {len(new_items)} new / {len(filtered)} total matched this run.")

    # Only ping you when there's actually something new -- no "nothing
    # happened" noise every 4 days. Remove the `if new_items:` guard if
    # you'd rather get a heartbeat every run to confirm it's alive.
    if new_items:
        notify_telegram(lines)
        notify_email()
        notify_whatsapp()


def notify_telegram(digest_lines):
    """Optional: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID as env vars
    (e.g. GitHub Actions secrets) to also get pinged in Telegram."""
    import os

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    text = "\n".join(digest_lines)[:4000]  # Telegram message length limit
    import urllib.parse

    api_url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode(
        {"chat_id": chat_id, "text": text, "disable_web_page_preview": "true"}
    ).encode()
    try:
        req = urllib.request.Request(api_url, data=data, method="POST")
        urllib.request.urlopen(req, timeout=15)
        print("[INFO] Telegram notification sent.")
    except Exception as e:
        print(f"[WARN] Telegram notification failed: {e}", file=sys.stderr)


def notify_email():
    """Optional: set GMAIL_ADDRESS and GMAIL_APP_PASSWORD as env vars /
    GitHub secrets to get an email digest via Gmail's SMTP server. Sends
    to itself by default; set GMAIL_TO to send somewhere else."""
    import os
    import smtplib
    from email.mime.text import MIMEText

    addr = os.environ.get("GMAIL_ADDRESS")
    app_password = os.environ.get("GMAIL_APP_PASSWORD")
    to_addr = os.environ.get("GMAIL_TO", addr)
    if not addr or not app_password:
        return

    body = DIGEST_FILE.read_text(encoding="utf-8")
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = f"Research opportunities digest — {datetime.now().strftime('%Y-%m-%d')}"
    msg["From"] = addr
    msg["To"] = to_addr

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as server:
            server.login(addr, app_password)
            server.sendmail(addr, [to_addr], msg.as_string())
        print("[INFO] Email digest sent.")
    except Exception as e:
        print(f"[WARN] Email send failed: {e}", file=sys.stderr)


def notify_whatsapp():
    """Optional: set CALLMEBOT_PHONE and CALLMEBOT_APIKEY as env vars /
    GitHub secrets to get a WhatsApp digest via the CallMeBot API
    (callmebot.com) -- an unofficial, personal-use-only third-party
    service (not run by Meta/WhatsApp). Fine for a personal notification;
    has no SLA, so don't make it your only channel."""
    import os
    import urllib.parse

    phone = os.environ.get("CALLMEBOT_PHONE")
    apikey = os.environ.get("CALLMEBOT_APIKEY")
    if not phone or not apikey:
        return

    text = DIGEST_FILE.read_text(encoding="utf-8")
    if len(text) > 1500:
        text = text[:1500] + "\n...(truncated -- see the repo for the full digest)"

    url = "https://api.callmebot.com/whatsapp.php?" + urllib.parse.urlencode(
        {"phone": phone, "text": text, "apikey": apikey}
    )
    try:
        urllib.request.urlopen(url, timeout=20)
        print("[INFO] WhatsApp digest sent.")
    except Exception as e:
        print(f"[WARN] WhatsApp send failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()