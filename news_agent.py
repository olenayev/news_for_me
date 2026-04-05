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

    return {"date": today, "sections": sections}


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
