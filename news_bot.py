"""
Breaking News Monitor -> Telegram Notifier (GitHub Actions edition)
=====================================================================
Runs ONCE per invocation: checks RSS feeds across finance/stocks, crypto,
AI, and international news, asks an AI model (Llama 3.3 70B via Groq's
free tier) whether each new article is genuinely significant, and if so
drafts a short breaking-news alert -- picking a label (BREAKING / JUST IN
/ DEVELOPING / UPDATE / NEW) that fits the story -- then pushes a Telegram
notification (label + headline + image + source link) so you can review
and post it to X yourself.

This is meant to be triggered on a schedule by the GitHub Actions workflow
in .github/workflows/news-bot.yml (every 15 minutes by default). It is NOT
a persistent process -- each run checks once and exits. It does NOT post
to X, and does NOT require a paid API key or a credit card anywhere.

--------------------------------------------------------------------
SETUP (one-time, all free, no server, no credit card)
--------------------------------------------------------------------
1. Create a new GitHub repo (PUBLIC -- see note below) and push these
   three files, keeping the folder structure:
     news_bot.py
     requirements.txt
     .github/workflows/news-bot.yml

   Why public: public repos get unlimited free Actions minutes. Private
   repos only get 2,000/month, and a 15-minute schedule can use more than
   that -- which would require adding a card for overage. Your API keys
   stay safe either way (see step 5), so the code itself being public
   costs you nothing.

2. Create a Telegram bot:
     - In Telegram, message @BotFather -> /newbot -> follow the prompts
     - Copy the token it gives you

3. Get your chat_id:
     - Send any message to your new bot
     - Visit: https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
     - Find "chat":{"id": ...} in the response -- that number is your chat_id

4. Get a free Groq API key at https://console.groq.com -- no credit card needed

5. In your GitHub repo: Settings -> Secrets and variables -> Actions ->
   New repository secret. Add three secrets (exact names matter):
     TELEGRAM_BOT_TOKEN
     TELEGRAM_CHAT_ID
     GROQ_API_KEY

6. Done. The workflow now runs automatically every 15 minutes. To trigger
   it right away instead of waiting: repo -> Actions tab -> "Breaking News
   Monitor" -> Run workflow.

Heads up: the very first run has no history yet, so it'll treat whatever
is currently in each feed as "new" and check all of it -- expect that one
run to take a few minutes longer than normal. After that, each run only
looks at what's actually new since the last one.
--------------------------------------------------------------------
"""

import feedparser
import requests
import json
import re
import time
import calendar
import hashlib
import os
from datetime import datetime

# ============================ CONFIG ================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "PASTE_YOUR_TELEGRAM_BOT_TOKEN").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "PASTE_YOUR_TELEGRAM_CHAT_ID").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "PASTE_YOUR_GROQ_API_KEY").strip()

GROQ_MODEL = "openai/gpt-oss-120b"        # free tier, no card required
GROQ_MIN_SECONDS_BETWEEN_CALLS = 2.5      # keeps us safely under the ~30/min free cap

ENTRIES_PER_FEED_CHECKED = 15   # newest N entries looked at per feed each run
                                 # (higher than a continuous-poller needs, since
                                 # this only runs once per schedule interval)
MAX_ARTICLE_AGE_HOURS = 24       # skip anything older than this, so alerts stay
                                  # same-day -- lower this (e.g. to 6) for a
                                  # tighter "right now" feel
SEEN_FILE = "seen_articles.json"

VALID_LABELS = ["BREAKING", "JUST IN", "DEVELOPING", "UPDATE", "NEW"]

# Starter feed list -- each URL verified live as of Aug 2026. Outlets do
# change their RSS endpoints occasionally, so watch the Actions run log for
# "[!] Failed to fetch" and swap in a replacement (rss.feedspot.com has
# a big searchable directory) if one ever breaks.
FEEDS = {
    "finance": [
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # CNBC Top News
        "https://finance.yahoo.com/news/rssindex",                 # Yahoo Finance
        "https://www.investing.com/rss/news_14.rss",                # Investing.com Economy
    ],
    "crypto": [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://decrypt.co/feed",
        "https://cryptoslate.com/feed",
    ],
    "ai": [
        "https://techcrunch.com/category/artificial-intelligence/feed/",
        "https://venturebeat.com/category/ai/feed/",
        "https://arstechnica.com/ai/feed",
    ],
    "international": [
        "http://feeds.bbci.co.uk/news/world/rss.xml",
        "https://www.aljazeera.com/xml/rss/all.xml",
    ],
}
# ======================================================================


def load_seen():
    if os.path.exists(SEEN_FILE):
        try:
            with open(SEEN_FILE) as f:
                return set(json.load(f))
        except (json.JSONDecodeError, IOError):
            return set()
    return set()


def save_seen(seen):
    with open(SEEN_FILE, "w") as f:
        json.dump(list(seen)[-3000:], f)  # keep the file bounded


def article_id(entry):
    key = entry.get("link") or entry.get("title", "")
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def article_age_hours(entry):
    """Hours since the article's own published/updated timestamp. Returns
    None if the feed entry doesn't carry a parseable date -- in that case
    we let it through rather than risk dropping something genuinely fresh."""
    struct = entry.get("published_parsed") or entry.get("updated_parsed")
    if not struct:
        return None
    published_ts = calendar.timegm(struct)  # feedparser normalizes this to UTC
    return (time.time() - published_ts) / 3600


CATEGORY_PLACEHOLDER = {
    "finance": "https://placehold.co/1200x630/1a3c34/ffffff?text=FINANCE",
    "crypto": "https://placehold.co/1200x630/2d2140/ffffff?text=CRYPTO",
    "ai": "https://placehold.co/1200x630/1e2a4a/ffffff?text=AI",
    "international": "https://placehold.co/1200x630/3a1f1f/ffffff?text=WORLD",
}


def fetch_og_image(article_url):
    """Best-effort fetch of an article page's og:image meta tag, for feed
    entries that don't embed one directly."""
    try:
        resp = requests.get(article_url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        match = re.search(
            r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
            resp.text, re.IGNORECASE,
        ) or re.search(
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
            resp.text, re.IGNORECASE,
        )
        return match.group(1) if match else None
    except requests.RequestException:
        return None


def extract_image(entry, category, link):
    if entry.get("media_thumbnail"):
        return entry["media_thumbnail"][0].get("url")
    if entry.get("media_content"):
        return entry["media_content"][0].get("url")
    for l in entry.get("links", []):
        if str(l.get("type", "")).startswith("image"):
            return l.get("href")
    return fetch_og_image(link) or CATEGORY_PLACEHOLDER.get(category)


_last_groq_call = 0.0  # tracks pacing between API calls, keeps us under the free rate limit


def analyze_and_draft(category, title, summary):
    """Ask the model: is this genuinely significant, and if so, draft the alert."""
    global _last_groq_call
    wait = GROQ_MIN_SECONDS_BETWEEN_CALLS - (time.time() - _last_groq_call)
    if wait > 0:
        time.sleep(wait)

    prompt = f"""You are screening for a fast-moving breaking-news account covering {category}.

Article title: {title}
Summary: {summary}

Decide if this is genuinely significant, breaking news -- major market-moving,
geopolitical, or industry-defining. Be selective: reject routine, minor, or
speculative stories. Most articles should be rejected.

Output ONLY this JSON shape, nothing else:
{{"significant": true or false, "label": "...", "headline": "..."}}

If significant is true:
- Choose "label" to match how big the story actually is -- vary it, don't
  default to the same one every time:
    BREAKING    major, sudden, high-impact (wars, crashes, disasters, a
                surprise central-bank move, a company collapsing)
    JUST IN     a fresh official statement, decision, or announcement
    DEVELOPING  a big situation still unfolding, more likely to come
    UPDATE      a real development on a story that's already ongoing
    NEW         notable but not urgent (a launch, a report, a filing)
- "headline" is a punchy, strictly factual 1-2 sentence summary under 250
  characters, with one fitting emoji. Do not editorialize or exaggerate
  beyond what the source supports.
If significant is false, set label and headline to empty strings."""

    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "content-type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "max_tokens": 300,
            "messages": [
                {"role": "system", "content": "You only output valid JSON. No prose, no markdown fences, no explanation."},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=25,
    )
    _last_groq_call = time.time()
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return {"significant": False, "label": "", "headline": ""}

    if result.get("significant"):
        label = str(result.get("label", "")).strip().upper()
        result["label"] = label if label in VALID_LABELS else "JUST IN"
    return result


def send_telegram(text, image_url=None):
    base = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    try:
        if image_url:
            r = requests.post(
                f"{base}/sendPhoto",
                data={"chat_id": TELEGRAM_CHAT_ID, "photo": image_url, "caption": text[:1024]},
                timeout=15,
            )
        else:
            r = requests.post(
                f"{base}/sendMessage",
                data={"chat_id": TELEGRAM_CHAT_ID, "text": text},
                timeout=15,
            )
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"[!] Telegram send failed: {e}")


def run():
    for name, val in [("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN),
                       ("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID),
                       ("GROQ_API_KEY", GROQ_API_KEY)]:
        if not val or val.startswith("PASTE_YOUR_"):
            raise SystemExit(f"[!] {name} is missing or empty -- check it's saved as a repo secret with that exact name.")

    if os.environ.get("TEST_TELEGRAM") == "true":
        send_telegram("\u2705 Test message from the news bot -- if you see this, Telegram delivery is working.")
        print(f"[{datetime.now()}] Sent Telegram test message, skipping feed check.")
        return

    seen = load_seen()
    total_feeds = sum(len(v) for v in FEEDS.values())
    print(f"[{datetime.now()}] Checking {total_feeds} feeds across {len(FEEDS)} categories...")

    for category, urls in FEEDS.items():
        for url in urls:
            try:
                parsed = feedparser.parse(url)
                if parsed.bozo and not parsed.entries:
                    raise ValueError(parsed.get("bozo_exception", "unknown parse error"))
            except Exception as e:
                print(f"[!] Failed to fetch {url}: {e}")
                continue

            for entry in parsed.entries[:ENTRIES_PER_FEED_CHECKED]:
                aid = article_id(entry)
                if aid in seen:
                    continue

                title = entry.get("title", "")
                summary = re.sub("<[^<]+?>", "", entry.get("summary", ""))[:500]  # strip HTML tags
                link = entry.get("link", "")
                if not title:
                    seen.add(aid)  # never usable -- fine to permanently skip
                    continue

                age_hours = article_age_hours(entry)
                if age_hours is not None and age_hours > MAX_ARTICLE_AGE_HOURS:
                    seen.add(aid)  # too old, and it won't get fresher -- permanently skip
                    continue

                try:
                    result = analyze_and_draft(category, title, summary)
                except Exception as e:
                    print(f"[!] Analysis failed for '{title[:60]}': {e}")
                    continue  # NOT marked seen -- retried automatically next run

                seen.add(aid)  # mark seen only once we actually have a verdict

                if result.get("significant") and result.get("headline"):
                    image = extract_image(entry, category, link)
                    post_text = f"{result['label']}: {result['headline']}"
                    message = f"{post_text}\n\n(reply with source: {link})"
                    send_telegram(message, image)
                    print(f"[{datetime.now()}] Notified ({category}): {result['label']}: {result['headline']}")

    save_seen(seen)
    print(f"[{datetime.now()}] Run complete. Tracking {len(seen)} seen articles.")


if __name__ == "__main__":
    run()
