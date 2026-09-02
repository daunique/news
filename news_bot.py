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
import io
from datetime import datetime, timezone
from urllib.parse import urljoin
from PIL import Image, ImageDraw, ImageFont, ImageOps

# ============================ CONFIG ================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "PASTE_YOUR_TELEGRAM_BOT_TOKEN").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "PASTE_YOUR_TELEGRAM_CHAT_ID").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "PASTE_YOUR_GROQ_API_KEY").strip()

GROQ_MODEL = "openai/gpt-oss-120b"        # free tier, no card required
# Free tier for this model: 30 req/min, 1,000 req/day, 8,000 tokens/min, 200,000 tokens/day.
# Token-per-minute is the binding constraint once the recent-headlines block is
# included in every prompt, so pacing is more conservative than the RPM alone implies.
GROQ_MIN_SECONDS_BETWEEN_CALLS = 4.5
MAX_RECENT_HEADLINES_IN_PROMPT = 15   # bounds prompt size so token usage can't creep up over the day

ENTRIES_PER_FEED_CHECKED = 15   # newest N entries looked at per feed each run
                                 # (higher than a continuous-poller needs, since
                                 # this only runs once per schedule interval)
MAX_ARTICLE_AGE_HOURS = 24       # skip anything older than this, so alerts stay
                                  # same-day -- lower this (e.g. to 6) for a
                                  # tighter "right now" feel
SEEN_FILE = "seen_articles.json"
RECENT_HEADLINES_FILE = "recent_headlines.json"
DUPLICATE_WINDOW_HOURS = 8   # how long a story counts as "already covered," across sources
DAILY_COUNT_FILE = "daily_count.json"
MIN_DAILY_POSTS = 2          # for Facebook -- ensures at least this many posts even on quiet news days
CATCHUP_GRACE_HOUR_UTC = 18  # start relaxing the significance bar from this hour (UTC) if still short

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
        "https://www.investing.com/rss/news_1.rss",                 # Investing.com Forex/currencies
        "https://oilprice.com/rss/main",                            # OilPrice.com - energy & geopolitics
    ],
    "crypto": [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://decrypt.co/feed",
        "https://cryptoslate.com/feed",
        "https://blockworks.co/feed/",                              # institutional/macro crypto angle
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


def load_recent_headlines():
    """Headlines already sent within the dedup window, regardless of which
    feed/source they came from -- used to stop two outlets covering the same
    underlying story from both triggering a notification."""
    if not os.path.exists(RECENT_HEADLINES_FILE):
        return []
    try:
        with open(RECENT_HEADLINES_FILE) as f:
            items = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []
    cutoff = time.time() - DUPLICATE_WINDOW_HOURS * 3600
    return [i for i in items if i.get("ts", 0) > cutoff]


def save_recent_headlines(items):
    with open(RECENT_HEADLINES_FILE, "w") as f:
        json.dump(items[-100:], f)  # keep it bounded


def load_daily_count():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    data = {}
    if os.path.exists(DAILY_COUNT_FILE):
        try:
            with open(DAILY_COUNT_FILE) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            data = {}
    if data.get("date") != today:
        return {"date": today, "count": 0}  # new day -- reset the counter
    return data


def save_daily_count(data):
    with open(DAILY_COUNT_FILE, "w") as f:
        json.dump(data, f)


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
        return urljoin(article_url, match.group(1)) if match else None
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


# ======================= BRANDED CARD IMAGE =======================
# Turns the raw article image into a designed card: your photo as the
# background, a dark gradient for legibility, your wordmark, a label pill,
# and a bold word-wrapped headline with the key phrase picked out in your
# accent color -- the same genre as Watcher.Guru/Bitcoin Magazine-style cards.
BRAND_NAME = "APEX WIRE"          # <-- change this if you land on a different name
CARD_W, CARD_H = 1080, 1920       # Facebook Story size; still reads fine in an X/FB feed
BRAND_NAVY = (16, 26, 46)
BRAND_AMBER = (251, 191, 36)
CARD_WHITE = (255, 255, 255)

FONT_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/anton/Anton-Regular.ttf"
FONT_PATH = "Anton-Regular.ttf"


def ensure_font():
    """Downloads the display font once and reuses it; falls back to PIL's
    built-in font if the download ever fails, so a font hiccup never kills
    a notification -- it just looks plainer."""
    if os.path.exists(FONT_PATH):
        return FONT_PATH
    try:
        r = requests.get(FONT_URL, timeout=15)
        r.raise_for_status()
        with open(FONT_PATH, "wb") as f:
            f.write(r.content)
        return FONT_PATH
    except requests.RequestException as e:
        print(f"[!] Could not download display font, using a plainer fallback: {e}")
        return None


def _wrap_lines(draw, text, font, max_width):
    words, lines, current = text.split(), [], []
    for word in words:
        trial = current + [word]
        if draw.textlength(" ".join(trial), font=font) > max_width and current:
            lines.append(current)
            current = [word]
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def _fit_headline(draw, text, font_path, max_width, max_height, start_size=64, min_size=38):
    """Shrinks the font until the wrapped headline fits the available box,
    so long headlines never run off the bottom of the card."""
    size = start_size
    while size >= min_size:
        font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
        line_h = int(size * 1.18)
        lines = _wrap_lines(draw, text, font, max_width)
        if len(lines) * line_h <= max_height:
            return font, lines, line_h
        size -= 4
    font = ImageFont.truetype(font_path, min_size) if font_path else ImageFont.load_default()
    line_h = int(min_size * 1.18)
    return font, _wrap_lines(draw, text, font, max_width), line_h


def build_card_image(image_url, label, headline, highlight, category):
    """Composites the branded card. Raises on failure -- callers should
    catch and fall back to the plain source image rather than lose the
    notification over a design step."""
    font_path = ensure_font()
    resp = requests.get(image_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        raise ValueError(f"not an image (Content-Type: {content_type or 'unknown'})")
    bg = Image.open(io.BytesIO(resp.content)).convert("RGB")
    bg = ImageOps.fit(bg, (CARD_W, CARD_H), method=Image.LANCZOS)

    gradient = Image.new("L", (1, CARD_H), 0)
    for y in range(CARD_H):
        t = max(0, (y - CARD_H * 0.32) / (CARD_H * 0.68))
        gradient.putpixel((0, y), int(255 * min(1, t) * 0.93))
    gradient = gradient.resize((CARD_W, CARD_H))
    overlay = Image.new("RGBA", (CARD_W, CARD_H), BRAND_NAVY + (0,))
    overlay.putalpha(gradient)
    card = Image.alpha_composite(bg.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(card)

    brand_font = ImageFont.truetype(font_path, 30) if font_path else ImageFont.load_default()
    label_font = ImageFont.truetype(font_path, 32) if font_path else ImageFont.load_default()

    draw.text((50, 60), BRAND_NAME, font=brand_font, fill=CARD_WHITE)
    draw.text((50, 100), category.upper(), font=label_font, fill=BRAND_AMBER)

    bottom_margin = 90
    max_text_zone_h = CARD_H - bottom_margin - 220  # leaves room for the pill above it
    font, lines, line_h = _fit_headline(draw, headline, font_path, CARD_W - 100, max_text_zone_h)
    block_h = len(lines) * line_h
    text_start_y = CARD_H - bottom_margin - block_h

    pill_h = 62
    pill_y = text_start_y - pill_h - 24
    pill_w = draw.textlength(label, font=label_font) + 50
    draw.rounded_rectangle([50, pill_y, 50 + pill_w, pill_y + pill_h], radius=12, fill=BRAND_AMBER)
    draw.text((75, pill_y + 13), label, font=label_font, fill=BRAND_NAVY)

    highlight_words = set(w.strip(".,:;!?$").lower() for w in (highlight.split() if highlight else []))
    y = text_start_y
    for line in lines:
        cx = 50
        for w in line:
            color = BRAND_AMBER if w.strip(".,:;!?$").lower() in highlight_words else CARD_WHITE
            draw.text((cx, y), w, font=font, fill=color)
            cx += draw.textlength(w + " ", font=font)
        y += line_h

    buf = io.BytesIO()
    card.convert("RGB").save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return buf
# ====================================================================


_last_groq_call = 0.0  # tracks pacing between API calls, keeps us under the free rate limit


def _groq_request(payload):
    """POST to Groq, retrying once on a rate limit using its Retry-After header.
    A long retry-after (over a minute) means we've likely hit the daily cap,
    not just a per-minute burst -- in that case give up and let this article
    get picked up on a later run instead of stalling the whole job."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "content-type": "application/json",
    }
    for attempt in range(2):
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=payload, timeout=25,
        )
        if resp.status_code != 429:
            return resp
        try:
            wait_s = float(resp.headers.get("Retry-After", ""))
        except ValueError:
            wait_s = 5.0
        if wait_s > 60 or attempt == 1:
            raise RuntimeError(f"Groq rate limited (retry-after {wait_s:.0f}s) -- deferring to next run")
        print(f"[!] Groq rate limited, waiting {wait_s:.0f}s and retrying once...")
        time.sleep(wait_s)
    return resp


def analyze_and_draft(category, title, summary, recent_headlines, relaxed=False):
    """Ask the model: is this genuinely significant, and if so, draft the alert.
    relaxed=True is used for the end-of-day catch-up pass, when the daily
    minimum hasn't been hit yet -- it lowers the bar so a solid-but-not-huge
    story can still qualify, so there's something to post that day."""
    global _last_groq_call
    wait = GROQ_MIN_SECONDS_BETWEEN_CALLS - (time.time() - _last_groq_call)
    if wait > 0:
        time.sleep(wait)

    recent_block = "\n".join(f"- {h}" for h in recent_headlines[-MAX_RECENT_HEADLINES_IN_PROMPT:]) or "(none)"

    if relaxed:
        selectivity = ("Today hasn't produced enough major stories yet and a daily minimum "
                        "needs filling. For THIS screening only, be notably more lenient: flag it "
                        "significant if it's a real, solid, on-topic story -- it doesn't need to be "
                        "dramatic, just genuinely newsworthy and accurate. Still reject pure filler, "
                        "ads, or opinion pieces with no real news value. If flagged significant here, "
                        "the label should almost always be NEW (or UPDATE for a development on an "
                        "ongoing story) -- do not use BREAKING/JUST IN/DEVELOPING unless fully earned.")
    else:
        selectivity = ("Decide if this is genuinely significant, breaking news -- major "
                        "market-moving, geopolitical, or industry-defining. Be selective: reject "
                        "routine, minor, or speculative stories. Most articles should be rejected.")

    prompt = f"""You are screening for a fast-moving breaking-news account covering {category}.

Article title: {title}
Summary: {summary}

{selectivity}

Headlines already sent in the last {DUPLICATE_WINDOW_HOURS} hours -- if this
article covers the same underlying story or event as any of these (even from
a different source, with different wording), set significant to false even
if it would otherwise qualify:
{recent_block}

Output ONLY this JSON shape, nothing else:
{{"significant": true or false, "label": "...", "headline": "...", "highlight": "..."}}

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
  characters. Only start it with an emoji when there's a clear country or
  nationality angle -- one or two flag emoji for the country/countries
  directly involved (e.g. "🇮🇷 Iran's currency crashes..." or "🇺🇸🇮🇷 Treasury
  Secretary..."). Most headlines should have NO emoji at all -- don't add a
  flag just to have one, and never use generic symbolic emoji like 🚨 or 📈.
  Do not editorialize or exaggerate beyond what the source supports.
- "highlight" is the single most impactful word or short phrase copied
  VERBATIM from within "headline" (a number, dollar amount, name, or the
  key outcome) -- used to visually pick it out on the card. Keep it short
  (1-5 words). If nothing stands out clearly, repeat the first few words.
If significant is false, set label, headline, and highlight to empty strings."""

    resp = _groq_request({
        "model": GROQ_MODEL,
        "max_tokens": 300,
        "messages": [
            {"role": "system", "content": "You only output valid JSON. No prose, no markdown fences, no explanation."},
            {"role": "user", "content": prompt},
        ],
    })
    _last_groq_call = time.time()
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return {"significant": False, "label": "", "headline": "", "highlight": ""}

    if result.get("significant"):
        label = str(result.get("label", "")).strip().upper()
        result["label"] = label if label in VALID_LABELS else "JUST IN"
    return result


def send_telegram(text, image_url=None, image_bytes=None):
    base = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    try:
        if image_bytes:
            r = requests.post(
                f"{base}/sendPhoto",
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": text[:1024]},
                files={"photo": ("card.jpg", image_bytes, "image/jpeg")},
                timeout=20,
            )
        elif image_url:
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
    recent = load_recent_headlines()
    daily = load_daily_count()
    total_feeds = sum(len(v) for v in FEEDS.values())
    print(f"[{datetime.now()}] Checking {total_feeds} feeds across {len(FEEDS)} categories... "
          f"({daily['count']}/{MIN_DAILY_POSTS} posted today)")

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

                relaxed = (daily["count"] < MIN_DAILY_POSTS
                           and datetime.now(timezone.utc).hour >= CATCHUP_GRACE_HOUR_UTC)
                try:
                    result = analyze_and_draft(category, title, summary, [r["headline"] for r in recent], relaxed)
                except Exception as e:
                    print(f"[!] Analysis failed for '{title[:60]}': {e}")
                    continue  # NOT marked seen -- retried automatically next run

                seen.add(aid)  # mark seen only once we actually have a verdict

                if result.get("significant") and result.get("headline"):
                    image = extract_image(entry, category, link)
                    post_text = f"{result['label']}: {result['headline']}"
                    message = f"{post_text}\n\n(reply with source: {link})"

                    card_bytes = None
                    if image:
                        try:
                            card_bytes = build_card_image(
                                image, result["label"], result["headline"], result.get("highlight", ""), category)
                        except Exception as e:
                            print(f"[!] Card generation failed for {image}, sending plain image instead: {e}")

                    if card_bytes:
                        send_telegram(message, image_bytes=card_bytes)
                    else:
                        send_telegram(message, image_url=image)

                    print(f"[{datetime.now()}] Notified ({category}){' [catch-up]' if relaxed else ''}: "
                          f"{result['label']}: {result['headline']}")
                    recent.append({"headline": result["headline"], "ts": time.time()})
                    daily["count"] += 1

    save_seen(seen)
    save_recent_headlines(recent)
    save_daily_count(daily)
    print(f"[{datetime.now()}] Run complete. Tracking {len(seen)} seen articles. "
          f"{daily['count']}/{MIN_DAILY_POSTS} posted today.")


if __name__ == "__main__":
    run()
