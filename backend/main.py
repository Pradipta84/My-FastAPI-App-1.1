from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi import Response
from pydantic import BaseModel
import asyncio
import json
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from datetime import datetime
import httpx

app = FastAPI(title="AI Q&A Backend", version="1.0.0")

# CORS for local dev (Flutter web or emulators)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    prompt: str
    model: str | None = None  # optional override
    latitude: float | None = None
    longitude: float | None = None


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
async def fetch_weather(city: str | None = None, lat: float | None = None, lon: float | None = None) -> str:
    # Free, no-key alternative: Open-Meteo
    async with httpx.AsyncClient(timeout=10) as http:
        city_name = city
        if lat is None or lon is None:
            if not city:
                return "Location not provided. Please share location or specify a city."
            # 1) Geocode city to lat/lon
            g = await http.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "en", "format": "json"},
            )
            if g.status_code != 200 or not g.json().get("results"):
                return f"Could not find location for {city}."
            res = g.json()["results"][0]
            lat = res["latitude"]
            lon = res["longitude"]
            city_name = res.get("name", city)

        # 2) Current weather
        w = await http.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current_weather": True,
                "timezone": "auto",
            },
        )
        if w.status_code != 200:
            return f"Could not fetch weather for {city_name}: {w.text}"
        data = w.json()
        cw = data.get("current_weather", {})
        temp = cw.get("temperature")
        wind = cw.get("windspeed")
        where = city_name if city_name else f"{lat},{lon}"
        return f"Weather in {where} now: {temp}°C, wind {wind} km/h."


async def reverse_country_code(lat: float, lon: float) -> str | None:
    async with httpx.AsyncClient(timeout=10) as http:
        r = await http.get(
            "https://geocoding-api.open-meteo.com/v1/reverse",
            params={"latitude": lat, "longitude": lon, "count": 1, "language": "en", "format": "json"},
        )
        if r.status_code != 200 or not r.json().get("results"):
            return None
        res = r.json()["results"][0]
        return res.get("country_code")


def parse_month_year(text: str) -> tuple[int, int]:
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    }
    text = text.lower()
    m = next((months[k] for k in months if k in text), datetime.now().month)
    y = datetime.now().year
    # find 4-digit year in text
    for token in text.replace("/", " ").replace("-", " ").split():
        if token.isdigit() and len(token) == 4:
            y = int(token)
            break
    return m, y


async def fetch_holidays(country_code: str, year: int, month: int) -> list[dict]:
    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}"
    async with httpx.AsyncClient(timeout=10) as http:
        r = await http.get(url)
        if r.status_code != 200:
            return []
        data = r.json()
        return [h for h in data if int(h.get("date", "0000-00-00").split("-")[1]) == month]


async def openai_stream(prompt: str, model: str | None = None):
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set. Create backend/.env with OPENAI_API_KEY=sk-... and restart.")

    # Some clients (e.g., Swagger default) send literal "string"; treat as unset
    if model is None or model.strip() == "" or model.strip().lower() == "string":
        chosen_model = "gpt-4o-mini"
    else:
        chosen_model = model
    # Uses Chat Completions with streaming
    try:
        system_msg = f"Current date: {datetime.now().strftime('%Y-%m-%d')}"
        stream = await client.chat.completions.create(
            model=chosen_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        async for event in stream:
            if event.choices and event.choices[0].delta and event.choices[0].delta.content:
                yield event.choices[0].delta.content
    except Exception as e:
        # Graceful fallback if quota/billing or model access fails
        yield "[error]: " + str(e) + "\n"
        # simple non-LLM fallback so the UI still works
        text = f"[{datetime.now().strftime('%Y-%m-%d')}] I cannot call the AI model right now (quota/billing). But here's an echo: {prompt}"
        for token in text.split(" "):
            yield token + " "
            await asyncio.sleep(0.03)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def generator():
        # Light intent: "weather" queries
        text_lower = req.prompt.lower()
        # Holidays intent
        if "holiday" in text_lower:
            month, year = parse_month_year(text_lower)
            cc = None
            if req.latitude is not None and req.longitude is not None:
                cc = await reverse_country_code(req.latitude, req.longitude)
            # fallback: try to infer from text (two-letter code)
            if not cc:
                for token in text_lower.replace(',', ' ').split():
                    if len(token) == 2 and token.isalpha():
                        cc = token.upper()
                        break
            if not cc:
                yield "Please share location (lat/lon) or include a 2-letter country code (e.g., IN, US)."
                return
            holidays = await fetch_holidays(cc, year, month)
            if not holidays:
                yield f"No holidays found for {cc} {year}-{month:02d}."
                return
            header = f"Public holidays in {cc} for {year}-{month:02d}:\n"
            yield header
            for h in holidays:
                line = f"- {h.get('date')}: {h.get('localName') or h.get('name')}\n"
                for tok in line.split(" "):
                    yield tok + " "
                    await asyncio.sleep(0.005)
            return
        # Links intent: generate search URLs without external calls
        if "link" in text_lower or "links" in text_lower:
            # extract query after keywords like "for" or after colon
            q = req.prompt.strip()
            parts = q.split(":", 1)
            if len(parts) > 1 and parts[1].strip():
                q = parts[1].strip()
            else:
                kw_parts = q.split(" for ", 1)
                if len(kw_parts) > 1 and kw_parts[1].strip():
                    q = kw_parts[1].strip()
            # cleanup common command words so the search looks right
            words = [w for w in q.replace("?", " ").replace(".", " ").split() if w]
            stop = {"give", "me", "link", "links", "please", "a", "an", "the", "for", "this"}
            cleaned = " ".join([w for w in words if w.lower() not in stop]).strip()
            if not cleaned:
                cleaned = q.strip()
            import urllib.parse
            enc = urllib.parse.quote_plus(cleaned)
            lines = [
                "Here are some links you can try:\n",
                f"- Google: https://www.google.com/search?q={enc}\n",
                f"- Bing: https://www.bing.com/search?q={enc}\n",
                f"- DuckDuckGo: https://duckduckgo.com/?q={enc}\n",
                f"- Wikipedia search: https://en.wikipedia.org/w/index.php?search={enc}\n",
            ]
            for line in lines:
                for tok in line.split(" "):
                    yield tok + " "
                    await asyncio.sleep(0.005)
            return
        if "weather" in text_lower:
            # Prefer explicit coordinates from client if available
            city = None
            if req.latitude is not None and req.longitude is not None:
                info = await fetch_weather(None, req.latitude, req.longitude)
            else:
                # naive city extraction: look for 'in <city>'
                parts = text_lower.split(" in ")
                if len(parts) > 1:
                    city = parts[1].split(" ")[0:3]
                    city = " ".join(city).replace("?", "").replace(".", "").strip()
                info = await fetch_weather(city)
            for tok in info.split(" "):
                yield tok + " "
                await asyncio.sleep(0.02)
            return

        try:
            async for token in openai_stream(req.prompt, req.model):
                yield token
        except Exception as e:
            yield f"\n[error]: {str(e)}\n"
    return StreamingResponse(generator(), media_type="text/plain")


# Alias single endpoint as required: POST /chat (streaming response)
@app.post("/chat")
async def chat(req: ChatRequest):
    return await chat_stream(req)


@app.get("/")
async def root():
    return {"status": "ok"}

# (image endpoints removed)



