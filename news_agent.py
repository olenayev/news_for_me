import os
import io
import time
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from gtts import gTTS

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


# ── Gemini call with retry ────────────────────────────────────────────────────
def gemini_generate(client, prompt: str) -> str:
    """Call Gemini with up to 3 retries on 503 errors."""
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            if attempt < 2:
                print(f"     Gemini attempt {attempt + 1} failed ({e}), retrying in 30s...")
                time.sleep(30)
            else:
                raise


# ── Step 1: Fetch all headlines from NewsAPI ──────────────────────────────────
def fetch_all_headlines() -> dict:
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
    print("  -> Summarizing all headlines with Gemini (1 call)...")
    client = genai.Client(api_key=GEMINI_API_KEY)

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

    raw = gemini_generate(client, prompt)
    raw = raw.replace("```json", "").replace("```", "").strip()
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

        summary = gemini_generate(client, prompt)

        return {
            "wahlperiode": wahlperiode,
            "sitzung":     sitzung,
            "url":         pdf_url,
            "summary":     summary,
        }

    except Exception as e:
        print(f"     Warning: Bundestag summary failed ({e})")
        return None


# ── Step 4: Fetch 2 historical facts with sources ─────────────────────────────
def fetch_historical_facts() -> list:
    print("  -> Fetching historical facts...")
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = """Generate exactly 2 interesting historical facts about fashion, religion, politics, or worldwide traditions.

For each fact you MUST provide a real, verifiable source URL (Wikipedia, BBC, Smithsonian, National Geographic, britannica.com, history.com, etc.).

Return ONLY a valid JSON array (no markdown, no backticks):
[
  {
    "fact": "<interesting historical fact, 2-3 sentences>",
    "category": "<Fashion / Religion / Politics / Traditions>",
    "source_title": "<name of the source>",
    "source_url": "<real URL>"
  },
  {
    "fact": "<interesting historical fact, 2-3 sentences>",
    "category": "<Fashion / Religion / Politics / Traditions>",
    "source_title": "<name of the source>",
    "source_url": "<real URL>"
  }
]

Make the facts genuinely surprising and educational. Vary the categories each time."""

        raw = gemini_generate(client, prompt)
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)

    except Exception as e:
        print(f"     Warning: Historical facts failed ({e})")
        return []


# ── Step 5: Build full briefing ───────────────────────────────────────────────
def build_briefing() -> dict:
    today = datetime.now().strftime("%A, %B %d %Y")

    all_headlines = fetch_all_headlines()
    gemini_result = summarize_all_with_gemini(all_headlines)

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

    bundestag = fetch_bundestag_summary()
    facts     = fetch_historical_facts()

    return {"date": today, "sections": sections, "bundestag": bundestag, "facts": facts}


# ── Format as plain text ──────────────────────────────────────────────────────
def format_telegram(data: dict) -> str:
    lines = [
        "📰 Daily News Briefing",
        f"📅 {data['date']}",
        "─────────────────────────",
    ]

    for section in data["sections"]:
        if not section.get("topics"):
            continue
        lines.append(f"\n{section['emoji']} {section['category'].upper()}")
        for topic in section["topics"]:
            if not topic.get("headline") or not topic.get("summary"):
                continue
            lines.append(f"\n  {topic['emoji']} {topic['type'].upper()}")
            lines.append(f"  {topic['headline']}")
            lines.append(f"  {topic['summary']}")
            for link in topic.get("links", []):
                if link.get("title") and link.get("url"):
                    lines.append(f"  🔗 {link['title']}")
                    lines.append(f"     {link['url']}")
        lines.append("─────────────────────────")

    # Historical facts section
    facts = data.get("facts", [])
    if facts:
        lines.append("\n💡 HISTORICAL FACTS OF THE DAY")
        for i, fact in enumerate(facts, 1):
            lines.append(f"\n  {i}. {fact['category'].upper()}")
            lines.append(f"  {fact['fact']}")
            lines.append(f"  🔗 {fact['source_title']}")
            lines.append(f"     {fact['source_url']}")
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


# ── Generate voice message from Bundestag summary ────────────────────────────
def generate_voice(text: str) -> bytes:
    print("  -> Generating voice message...")
    tts = gTTS(text=text, lang="en", slow=False)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer.read()


# ── Send voice message to Telegram ───────────────────────────────────────────
def send_telegram_voice(audio_bytes: bytes, caption: str):
    print("  -> Sending voice message to Telegram...")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVoice"
    resp = requests.post(url, data={
        "chat_id": TELEGRAM_CHAT_ID,
        "caption": caption,
    }, files={
        "voice": ("bundestag_summary.mp3", audio_bytes, "audio/mpeg"),
    })
    resp.raise_for_status()


# ── Send text message to Telegram ────────────────────────────────────────────
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

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Sending text briefing to Telegram...")
    msg = format_telegram(data)
    send_telegram(msg)

    # Generate and send voice message for Bundestag summary
    bt = data.get("bundestag")
    if bt:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating Bundestag voice summary...")
        voice_text = (
            f"Hello beauty, here is Bundestag Summary for you. "
            f"{bt['wahlperiode']}th Wahlperiode, {bt['sitzung']}th Sitzung. "
            f"{bt['summary']}"
        )
        audio = generate_voice(voice_text)
        send_telegram_voice(
            audio,
            caption=f"🏛️ Bundestag {bt['wahlperiode']}. WP / {bt['sitzung']}. Sitzung — Audio Summary"
        )

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Done! Briefing and voice note sent.")
