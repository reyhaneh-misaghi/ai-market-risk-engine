import asyncio
import json
import os
from typing import Dict, List, Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class NewsSentimentSignal(BaseModel):
    """Structured schema enforced via Pydantic for LLM financial sentiment output."""
    ticker: str = Field(..., description="Asset ticker symbol (e.g., SPY, NG=F, CL=F)")
    sentiment_score: float = Field(..., ge=-1.0, le=1.0, description="Score from -1.0 (very bearish) to +1.0 (very bullish)")
    regime: Literal["Bearish", "Neutral", "Bullish"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., description="Concise financial reasoning behind the score")


class AsyncSentimentEngine:
    """Asynchronous financial news sentiment analyzer using LLM + Pydantic validation."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model_name: Optional[str] = None

    async def _resolve_active_model(self, client) -> str:
        """Automatically fetches an active chat model from Groq API."""
        if self.model_name:
            return self.model_name
        models_resp = await client.models.list()
        available = [m.id for m in models_resp.data if "whisper" not in m.id.lower() and "guard" not in m.id.lower()]
        for preferred in ["llama-3.1-8b-instant", "openai/gpt-oss-120b", "qwen/qwen3-32b"]:
            if preferred in available:
                self.model_name = preferred
                return preferred
        self.model_name = available[0]
        return self.model_name

    async def _analyze_single_headline(self, ticker: str, headline: str) -> NewsSentimentSignal:
        """Analyzes one headline asynchronously using Groq LLM or enhanced local NLP fallback."""
        if self.api_key:
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url="https://api.groq.com/openai/v1",
                )
                active_model = await self._resolve_active_model(client)
                prompt = (
                    f"Analyze the financial market impact of this headline for asset '{ticker}'.\n"
                    f"Headline: '{headline}'\n"
                    "Return ONLY a valid JSON object with keys: "
                    '"ticker" (string), "sentiment_score" (float between -1.0 and 1.0), '
                    '"regime" ("Bearish", "Neutral", or "Bullish"), '
                    '"confidence" (float between 0.0 and 1.0), '
                    '"rationale" (1 concise analytical sentence explaining the financial mechanism).'
                )
                completion = await client.chat.completions.create(
                    model=active_model,
                    messages=[
                        {"role": "system", "content": "You are a quantitative risk analyst. Output strictly JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                raw_json = json.loads(completion.choices[0].message.content)
                return NewsSentimentSignal(**raw_json)
            except Exception as e:
                print(f"[Warning] API call failed for {ticker}, using local NLP fallback: {e}")

        await asyncio.sleep(0.1)
        return self._local_financial_scorer(ticker, headline)

    def _local_financial_scorer(self, ticker: str, headline: str) -> NewsSentimentSignal:
        """Enhanced domain-specific scorer listing exact matched drivers."""
        text = headline.lower()
        bearish_words = ["surge in supply", "drop", "recession", "cuts forecast", "disruption", "plunge", "sanctions", "risk", "fall"]
        bullish_words = ["rally", "demand growth", "beat expectations", "beats expectations", "stimulus", "recovery", "record high", "tight supply"]

        matched_bearish = [w for w in bearish_words if w in text]
        matched_bullish = [w for w in bullish_words if w in text]

        score = -0.45 * len(matched_bearish) + 0.45 * len(matched_bullish)
        score = max(-1.0, min(1.0, round(score, 2)))

        regime: Literal["Bearish", "Neutral", "Bullish"] = (
            "Bearish" if score < -0.15 else ("Bullish" if score > 0.15 else "Neutral")
        )

        drivers = matched_bearish + matched_bullish
        if drivers:
            rationale = f"{regime} drivers detected: {', '.join(drivers)}."
        else:
            rationale = f"No strong directional catalysts detected for {ticker}."

        return NewsSentimentSignal(
            ticker=ticker,
            sentiment_score=score,
            regime=regime,
            confidence=0.85 if drivers else 0.65,
            rationale=rationale,
        )

    async def analyze_batch(self, news_feed: List[Dict[str, str]]) -> Dict[str, NewsSentimentSignal]:
        """Processes all asset headlines concurrently using asyncio.gather."""
        tasks = [self._analyze_single_headline(item["ticker"], item["headline"]) for item in news_feed]
        results = await asyncio.gather(*tasks)
        return {res.ticker: res for res in results}


if __name__ == "__main__":
    SAMPLE_NEWS = [
        {"ticker": "SPY", "headline": "US tech earnings beat expectations amid resilient consumer demand growth"},
        {"ticker": "FTSEMIB.MI", "headline": "European industrial output faces recession risk as borrowing costs stay elevated"},
        {"ticker": "CL=F", "headline": "Global crude oil prices drop sharply following unexpected surge in supply"},
        {"ticker": "NG=F", "headline": "European natural gas faces severe supply disruption and geopolitical sanctions"},
        {"ticker": "GC=F", "headline": "Gold prices rally to record high as investors hedge geopolitical risk"},
    ]

    engine = AsyncSentimentEngine()
    signals = asyncio.run(engine.analyze_batch(SAMPLE_NEWS))

    print("=== Structured AI Sentiment Signals ===")
    for ticker, sig in signals.items():
        print(f"{ticker:10s} | Score: {sig.sentiment_score:+5.2f} | Regime: {sig.regime:7s} | Conf: {sig.confidence:.0%} | {sig.rationale}")