import os
import re
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
from google import genai

# ── Load environment variables from .env file ─────────────────────────────────
load_dotenv()

GEMINI_API_KEY     = os.environ["GEMINI_API_KEY"]
NEWSAPI_KEY        = os.environ["NEWSAPI_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]

# ── Config ────────────────────────────────────────────────────────────────────
CATEGORIES = ["Germany", "Europe", "US", "Asia", "World"]
NEWS_TYPES  = ["Culture", "Business", "War", "Stocks & Markets", "Technology"]

CATEGORY_QUERIES = {
    "Germany":  "Germany",
    "Europe":   "Europe",
    "US":       "United States",
    "Asia":     "Asia",
    "World":    "world",
}

NEWSTYPE_QUERIES = {
    "Culture":          "culture OR arts OR entertainment",
    "Business":         "business OR economy",
    "War":              "war OR conflict OR military",
    "Stocks & Markets": "stock market OR finance OR economy",
    "Technology":       "technology OR AI OR tech",
}

# ── Fetch headlines from NewsAPI ──────────────────────────────────────────────
def fetch_headlines(category: str, news_type: str) -> list:
    query = f"{CATEGORY_QUERIES[category]} {NEWSTYPE_QUERIES[news_type]}"
    resp = requests.get("https://newsapi.org/v2/everything", params={
        "q":        query,
        "language": "en",
        "sortBy":   "publishedAt",
        "pageSize": 5,
        "apiKey":   NEWSAPI_KEY,
    })
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    return [
        {"title": a["title"], "description": a.get("description", ""), "url": a["url"]}
        for a in articles if a.get("title") and a.get("url")
    ]


# ── Summarize with Gemini ─────────────────────────────────────────────────────
def summarize_with_gemini(category: str, news_type: str, articles: list) -> dict:
    client = genai.Client(api_key=GEMINI_API_KEY)

    articles_text = "\n".join(
        f"- {a['title']}: {a['description']} ({a['url']})"
        for a in articles
    )

    prompt = f"""You are a professional news editor. Based on these headlines about {news_type} in {category}:

{articles_text}

Return ONLY a valid JSON object (no markdown, no backticks):
{{
  "headline": "<single most important headline, concise>",
  "summary": "<2-3 sentence summary of the key story>",
  "links": [
    {{"title": "<article title>", "url": "<url>"}},
    {{"title": "<article title>", "url": "<url>"}}
  ]
}}"""

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
    )
    raw = response.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ── Fetch latest Bundestag session number ────────────────────────────────────
def get_latest_bundestag_session() -> tuple[int, int]:
    """Returns (wahlperiode, sitzung) of the most recent available session."""
    # Current Wahlperiode is 21
    wahlperiode = 21
    # Try sessions counting down from a high number to find the latest
    for sitzung in range(120, 0, -1):
        url = f"https://dserver.bundestag.de/btp/{wahlperiode}/{wahlperiode:02d}{sitzung:03d}.pdf"
        try:
            resp = requests.head(url, timeout=5)
            if resp.status_code == 200:
                return wahlperiode, sitzung
        except Exception:
            continue
    return wahlperiode, 1


# ── Fetch and summarize latest Bundestag session ──────────────────────────────
def fetch_bundestag_summary() -> dict:
    print("  -> Fetching latest Bundestag session...")
    try:
        wahlperiode, sitzung = get_latest_bundestag_session()
        pdf_url = f"https://dserver.bundestag.de/btp/{wahlperiode}/{wahlperiode:02d}{sitzung:03d}.pdf"
        print(f"     Found: {wahlperiode}. Wahlperiode, {sitzung}. Sitzung")

        resp = requests.get(pdf_url, timeout=30)
        resp.raise_for_status()

        # Extract text from PDF using pdfminer if available, else use raw bytes hint
        try:
            import io
            from pdfminer.high_level import extract_text
            text = extract_text(io.BytesIO(resp.content))
        except ImportError:
            # Fallback: decode raw PDF text (works for simple PDFs)
            text = resp.content.decode("latin-1", errors="ignore")

        # Trim to first 8000 chars to stay within token limits
        text_trimmed = text[:8000]

        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = f"""The following is the transcript of the {sitzung}. Sitzung of the {wahlperiode}. Wahlperiode of the German Bundestag.

{text_trimmed}

Write a concise English summary of this session covering:
1. The date and session number
2. The main agenda topics discussed
3. Key debates or decisions made
4. Any notable statements from ministers or MPs

Keep it to 5-8 sentences total. Be factual and neutral."""

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
        )

        return {
            "wahlperiode": wahlperiode,
            "sitzung":     sitzung,
            "url":         pdf_url,
            "summary":     response.text.strip(),
        }

    except Exception as e:
        print(f"     Warning: Bundestag summary failed ({e})")
        return None


# ── Build full briefing ───────────────────────────────────────────────────────
def build_briefing() -> dict:
    today = datetime.now().strftime("%A, %B %d %Y")
    sections = []

    EMOJI_CATEGORY = {
        "Germany": "🇩🇪", "Europe": "🇪🇺", "US": "🇺🇸",
        "Asia": "🌏", "World": "🌍"
    }
    EMOJI_TYPE = {
        "Culture": "🎭", "Business": "💼", "War": "⚔️",
        "Stocks & Markets": "📈", "Technology": "💻"
    }

    for category in CATEGORIES:
        topics = []
        for news_type in NEWS_TYPES:
            print(f"  -> {category} / {news_type}...")
            try:
                articles = fetch_headlines(category, news_type)
                if not articles:
                    print(f"     No articles found, skipping.")
                    continue
                summary = summarize_with_gemini(category, news_type, articles)
                topics.append({
                    "type":     news_type,
                    "emoji":    EMOJI_TYPE.get(news_type, "📰"),
                    "headline": summary["headline"],
                    "summary":  summary["summary"],
                    "links":    summary.get("links", [])[:2],
                })
            except Exception as e:
                print(f"     Warning: Skipped ({e})")

        sections.append({
            "category": category,
            "emoji":    EMOJI_CATEGORY.get(category, "🌐"),
            "topics":   topics,
        })

    # Fetch Bundestag summary
    bundestag = fetch_bundestag_summary()

    return {"date": today, "sections": sections, "bundestag": bundestag}


# ── Format as plain text ──────────────────────────────────────────────────────
def format_telegram(data: dict) -> str:
    lines = [
        "📰 Daily News Briefing",
        f"📅 {data['date']}",
        "─────────────────────────",
    ]

    for section in data["sections"]:
        lines.append(f"\n{section['emoji']} {section['category'].upper()}")
        for topic in section["topics"]:
            lines.append(f"\n  {topic['emoji']} {topic['type'].upper()}")
            lines.append(f"  {topic['headline']}")
            lines.append(f"  {topic['summary']}")
            for link in topic.get("links", []):
                lines.append(f"  🔗 {link['title']}")
                lines.append(f"     {link['url']}")
        lines.append("─────────────────────────")

    # Add Bundestag section
    bt = data.get("bundestag")
    if bt:
        lines.append(f"\n🏛️ BUNDESTAG — {bt['wahlperiode']}. WAHLPERIODE, {bt['sitzung']}. SITZUNG")
        lines.append(bt["summary"])
        lines.append(f"  🔗 Vollständiges Protokoll (PDF)")
        lines.append(f"     {bt['url']}")
        lines.append("─────────────────────────")

    return "\n".join(lines)


# ── Send to Telegram ──────────────────────────────────────────────────────────
def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    for chunk in chunks:
        resp = requests.post(url, json={
            "chat_id":                  TELEGRAM_CHAT_ID,
            "text":                     chunk,
            "disable_web_page_preview": True,
        })
        resp.raise_for_status()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Building news briefing...")
    data = build_briefing()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Sending to Telegram...")
    msg = format_telegram(data)
    send_telegram(msg)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Done! Briefing sent.")
