import os
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

EMOJI_CATEGORY = {
    "Germany": "🇩🇪", "Europe": "🇪🇺", "US": "🇺🇸",
    "Asia": "🌏", "World": "🌍"
}
EMOJI_TYPE = {
    "Culture": "🎭", "Business": "💼", "War": "⚔️",
    "Stocks & Markets": "📈", "Technology": "💻"
}


# ── Step 1: Fetch all headlines from NewsAPI ──────────────────────────────────
def fetch_all_headlines() -> dict:
    """Fetch headlines for all category/type combos. Returns nested dict."""
    all_headlines = {}
    for category in CATEGORIES:
        all_headlines[category] = {}
        for news_type in NEWS_TYPES:
            print(f"  -> Fetching: {category} / {news_type}...")
            try:
                query = f"{CATEGORY_QUERIES[category]} {NEWSTYPE_QUERIES[news_type]}"
                resp = requests.get("https://newsapi.org/v2/everything", params={
                    "q":        query,
                    "language": "en",
                    "sortBy":   "publishedAt",
                    "pageSize": 3,
                    "apiKey":   NEWSAPI_KEY,
                })
                resp.raise_for_status()
                articles = resp.json().get("articles", [])
                all_headlines[category][news_type] = [
                    {"title": a["title"], "description": a.get("description", ""), "url": a["url"]}
                    for a in articles if a.get("title") and a.get("url")
                ]
            except Exception as e:
                print(f"     Warning: NewsAPI failed ({e})")
                all_headlines[category][news_type] = []
    return all_headlines


# ── Step 2: Summarize ALL headlines in ONE Gemini call ───────────────────────
def summarize_all_with_gemini(all_headlines: dict) -> dict:
    """Send all headlines to Gemini in one prompt. Returns structured summaries."""
    print("  -> Summarizing all headlines with Gemini (1 call)...")
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Build a compact text block of all headlines
    headlines_block = ""
    for category in CATEGORIES:
        for news_type in NEWS_TYPES:
            articles = all_headlines[category][news_type]
            if not articles:
                continue
            headlines_block += f"\n[{category} / {news_type}]\n"
            for a in articles:
                headlines_block += f"  - {a['title']} | {a['url']}\n"

    prompt = f"""You are a professional news editor. Below are headlines grouped by region and topic.

{headlines_block}

For EACH [Region / Topic] group, write a summary. Return ONLY a valid JSON object (no markdown, no backticks) in this exact structure:
{{
  "sections": [
    {{
      "category": "<region>",
      "news_type": "<topic>",
      "headline": "<single most important headline, concise>",
      "summary": "<2-3 sentence summary>",
      "links": [
        {{"title": "<article title>", "url": "<url>"}},
        {{"title": "<article title>", "url": "<url>"}}
      ]
    }}
  ]
}}

Cover every [Region / Topic] group that has at least one headline. Be factual and concise."""

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
    )
    raw = response.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ── Step 3: Fetch latest Bundestag session ────────────────────────────────────
def get_latest_bundestag_session() -> tuple:
    wahlperiode = 21
    for sitzung in range(120, 0, -1):
        url = f"https://dserver.bundestag.de/btp/{wahlperiode}/{wahlperiode:02d}{sitzung:03d}.pdf"
        try:
            resp = requests.head(url, timeout=5)
            if resp.status_code == 200:
                return wahlperiode, sitzung
        except Exception:
            continue
    return wahlperiode, 1


def fetch_bundestag_summary() -> dict:
    print("  -> Fetching latest Bundestag session...")
    try:
        wahlperiode, sitzung = get_latest_bundestag_session()
        pdf_url = f"https://dserver.bundestag.de/btp/{wahlperiode}/{wahlperiode:02d}{sitzung:03d}.pdf"
        print(f"     Found: {wahlperiode}. Wahlperiode, {sitzung}. Sitzung")

        resp = requests.get(pdf_url, timeout=30)
        resp.raise_for_status()

        try:
            import io
            from pdfminer.high_level import extract_text
            text = extract_text(io.BytesIO(resp.content))
        except ImportError:
            text = resp.content.decode("latin-1", errors="ignore")

        text_trimmed = text[:8000]

        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = f"""The following is the transcript of the {sitzung}. Sitzung of the {wahlperiode}. Wahlperiode of the German Bundestag.

{text_trimmed}

Write a concise English summary covering:
1. The date and session number
2. Main agenda topics
3. Key debates or decisions
4. Notable statements from ministers or MPs

Keep it to 5-8 sentences. Be factual and neutral."""

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


# ── Step 4: Build full briefing ───────────────────────────────────────────────
def build_briefing() -> dict:
    today = datetime.now().strftime("%A, %B %d %Y")

    # Fetch all headlines (NewsAPI — many calls but free)
    all_headlines = fetch_all_headlines()

    # Summarize all in ONE Gemini call
    gemini_result = summarize_all_with_gemini(all_headlines)

    # Organize into sections
    section_map = {}
    for item in gemini_result.get("sections", []):
        cat = item["category"]
        if cat not in section_map:
            section_map[cat] = []
        section_map[cat].append({
            "type":     item["news_type"],
            "emoji":    EMOJI_TYPE.get(item["news_type"], "📰"),
            "headline": item["headline"],
            "summary":  item["summary"],
            "links":    item.get("links", [])[:2],
        })

    sections = [
        {
            "category": cat,
            "emoji":    EMOJI_CATEGORY.get(cat, "🌐"),
            "topics":   section_map.get(cat, []),
        }
        for cat in CATEGORIES
    ]

    # Fetch Bundestag summary (1 Gemini call)
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

    # Bundestag section
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
