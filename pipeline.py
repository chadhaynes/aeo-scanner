"""
AEO Scanner — pipeline engine.

Pure logic, no interface. Every stage you proved in the notebook, lifted into a
callable function so the server can run a full scan for ANY brand on demand.

Stages:
  1. fetch_site_text   — pull a brand's homepage, return readable text
  2. generate_prompts  — DeepSeek turns that into customer-style prompts (brand-blind)
  3. gather            — run one prompt across 5 models with forced live search
  4. detect_mention    — boundary-aware brand detection in an answer
  5. analyse_answer    — extract competitors + sentiment (normalised to parent brands)
  6. run_scan          — orchestrator: chains 1-5 for one brand, returns the report

Every outbound API call goes through _post_with_retry, so a transient network
drop retries with backoff instead of crashing a 100+ call scan.
"""

import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# One representative model per vendor, search forced via :online (Perplexity
# searches natively). This is the measured set.
MODELS = [
    "openai/gpt-5.6-luna:online",
    "anthropic/claude-sonnet-5:online",
    "anthropic/claude-opus-5:online",
    "google/gemini-3.7-flash:online",
    "perplexity/sonar",
]

# Model used to GENERATE the prompt set (cheaper, no search needed).
PROMPT_MODEL = "deepseek/deepseek-v4-pro-0813"

# Model used to ANALYSE answers (must support structured outputs — GPT).
ANALYSIS_MODEL = "openai/gpt-5.6-luna"


# ---------------------------------------------------------------------------
# Network helper — one POST, retried on transient failure
# ---------------------------------------------------------------------------
def _post_with_retry(payload, timeout=120, retries=3):
    """POST to OpenRouter, retrying transient network failures with backoff.

    A dropped SSL connection or timeout on one call shouldn't sink a whole scan.
    Retries up to `retries` times with 1s, 2s, 4s waits, then re-raises so the
    caller's own error handling can decide what to do. Returns parsed JSON.
    """
    for attempt in range(retries):
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
                json=payload,
                timeout=timeout,
            )
            return response.json()
        except requests.exceptions.RequestException:
            if attempt == retries - 1:  # last attempt — give up, let caller handle
                raise
            time.sleep(2 ** attempt)  # 1s, 2s, 4s


# ---------------------------------------------------------------------------
# Stage 1 — fetch the brand's homepage
# ---------------------------------------------------------------------------
def fetch_site_text(url):
    """Fetch a homepage and return readable text (nav menu = product taxonomy).

    Browser User-Agent header is required or many sites return a blocked/empty
    page. We trim to ~17k chars — enough for the model to read the product
    lineup without paying for the entire DOM.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()  # fail loudly on a bad fetch, don't scan garbage
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return text[:17000]


# ---------------------------------------------------------------------------
# Stage 2 — generate customer-style prompts from the site text
# ---------------------------------------------------------------------------
def generate_prompts(brand, site_text):
    """Ask DeepSeek for a structured set of customer-style questions.

    Brand-blind by design: the questions must sound like a real customer who
    doesn't know or care about {brand} — they describe a NEED, never the brand.
    Returns a flat list of dicts: {product_family, funnel_stage, question}.
    """
    system_message = (
        "You generate realistic customer search prompts for measuring a brand's "
        "visibility across AI answer engines.\n\n"
        f"You will be given the homepage text of: {brand}\n\n"
        "From it, infer the brand's product families. Then generate customer-style "
        "questions a real person would type into an AI assistant when researching "
        "these products.\n\n"
        "CRITICAL RULES:\n"
        f"- Questions must be BRAND-BLIND. Never mention {brand} or any competitor "
        "by name. A real customer describes their NEED, not the brand.\n"
        "- Cover multiple funnel stages: problem_aware (early research), "
        "comparison (weighing options), bottom_funnel (ready to act).\n"
        "- Spread across the product families you identified.\n\n"
        "Return ONLY valid JSON in this exact shape:\n"
        '{"prompts": [{"product_family": "...", "funnel_stage": "...", '
        '"question": "..."}]}'
    )

    data = _post_with_retry({
        "model": PROMPT_MODEL,
        "temperature": 0.7,  # some variety in the questions is fine here
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": site_text},
        ],
        "response_format": {"type": "json_object"},
    })
    content = data["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    return parsed["prompts"]


# ---------------------------------------------------------------------------
# Stage 3 — gather: run one prompt across all measured models
# ---------------------------------------------------------------------------
def gather(brand, prompt_obj):
    """Run a single prompt across every model in MODELS with forced search.

    Returns a list of per-model result dicts. Records model_actual so version
    drift is visible, and captures citations for the memory-vs-live-web split.
    Never trusts that a call succeeded — errors are captured per row, not raised,
    so one bad model doesn't sink the whole scan.
    """
    question = prompt_obj["question"]
    results = []

    for model in MODELS:
        row = {
            "model_requested": model,
            "model_actual": None,
            "product_family": prompt_obj.get("product_family"),
            "funnel_stage": prompt_obj.get("funnel_stage"),
            "question": question,
            "answer": "",
            "citations": [],
            "num_citations": 0,
            "brand_mentioned": False,
            "error": None,
        }
        try:
            data = _post_with_retry({
                "model": model,
                "messages": [{"role": "user", "content": question}],
            })

            if "choices" not in data:
                row["error"] = json.dumps(data)[:300]
                results.append(row)
                time.sleep(1)
                continue

            message = data["choices"][0]["message"]
            row["model_actual"] = data.get("model")
            row["answer"] = message.get("content", "")

            # citations arrive as url_citation annotations (all vendors,
            # normalised by OpenRouter). Their presence = a live search fired.
            annotations = message.get("annotations", []) or []
            citations = [
                a["url_citation"]["url"]
                for a in annotations
                if a.get("type") == "url_citation"
            ]
            row["citations"] = citations
            row["num_citations"] = len(citations)
            row["brand_mentioned"] = detect_mention(brand, row["answer"])

        except Exception as exc:  # network exhausted retries, or parse — capture
            row["error"] = str(exc)[:300]

        results.append(row)
        time.sleep(1)  # gentle on rate limits

    return results


# ---------------------------------------------------------------------------
# Stage 4 — mention detection (boundary-aware)
# ---------------------------------------------------------------------------
def detect_mention(brand, answer):
    """True if the brand is named in the answer.

    Word-boundary regex so short brand tokens (e.g. 'NAB') don't match inside
    longer words ('cannabis'), and so an acronym and its full form can both be
    caught. This is the honest version of the naive substring check.
    """
    if not answer:
        return False
    pattern = re.compile(rf"\b{re.escape(brand)}\b", re.IGNORECASE)
    return bool(pattern.search(answer))


# ---------------------------------------------------------------------------
# Stage 5 — analyse one answer: competitors + sentiment
# ---------------------------------------------------------------------------
ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "description": "Every brand or organisation named in the answer.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the entity.",
                    },
                    "isCompetitor": {
                        "type": "boolean",
                        "description": "Whether the entity is a competitor to the brand being analysed.",
                    },
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "neutral", "negative"],
                        "description": (
                            "Sentiment of the answer's coverage of this entity. "
                            "positive = recommended or led alone; neutral = "
                            "listed/viable, not singled out; negative = "
                            "criticised, and criticism overrides prominence."
                        ),
                    },
                },
                "required": ["name", "isCompetitor", "sentiment"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["entities"],
    "additionalProperties": False,
}


def analyse_answer(brand, answer):
    """Extract every named brand from an answer, with competitor flag + sentiment.

    Normalises product/card names to their parent brand so share-of-voice isn't
    fragmented (Amex Explorer -> American Express). Sentiment rubric and
    normalisation live in the system message because the model weights those
    harder than schema descriptions.
    """
    if not answer:
        return {"entities": []}

    system_message = f"""You are analysing how an AI answer covers brands in a market.

The brand being analysed is: {brand}

Extract every brand or organisation named in the answer. For each one:
- name: the PARENT COMPANY or brand, normalised to its most common name. Do NOT return product or card names as separate entities — collapse them to the parent. For example: "Amex", "American Express Low Rate", and "Amex Explorer" all become "American Express". "ANZ Rewards Black" becomes "ANZ". "Bankwest Breeze Platinum" becomes "Bankwest". Use the shortest widely-recognised form of the brand.
- isCompetitor: true if it competes with {brand} in the same market, false otherwise (regulators, government bodies, and comparison/aggregator sites are not competitors)
- sentiment: how the answer treats that entity. positive = recommended, or named first and alone as the pick. neutral = listed as one of several viable options, not singled out. negative = criticised. If a brand is named first and alone as the recommendation, that is positive, not neutral. Only use neutral when a brand is one of several listed without clear preference.

If the same parent brand appears multiple times under different product names, return it ONCE with a single sentiment reflecting its overall treatment.

Only include entities actually named in the answer. Do not infer or add ones that aren't there."""

    data = _post_with_retry({
        "model": ANALYSIS_MODEL,
        "temperature": 0,  # deterministic as possible — this is measurement
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": answer},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "entity_analysis",
                "strict": True,
                "schema": ENTITY_SCHEMA,
            },
        },
    })
    if "choices" not in data:
        return {"entities": [], "error": json.dumps(data)[:300]}
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)


# ---------------------------------------------------------------------------
# Helper — domain from a URL (for the citation leaderboard)
# ---------------------------------------------------------------------------
def get_domain(url):
    """Reduce a full URL to its bare domain (www. stripped) for aggregation."""
    return urlparse(url).netloc.replace("www.", "")


# ---------------------------------------------------------------------------
# Stage 6 — orchestrator: run the whole pipeline for one brand
# ---------------------------------------------------------------------------
def run_scan(brand, url, progress=None):
    """Run the full AEO scan for one brand and return the complete report.

    `progress` is an optional callback(str) so the caller (server/UI) can report
    live status. Everything expensive happens here; the caller caches the result.

    A failed analysis on one row leaves that row's entities empty and moves on,
    mirroring gather's per-row error handling — one bad call never kills the scan.

    Returns a dict:
      {
        "brand": ...,
        "url": ...,
        "results": [ per-answer rows, each with entities attached ],
      }
    """
    def say(msg):
        if progress:
            progress(msg)

    say(f"Fetching {url} ...")
    site_text = fetch_site_text(url)

    say("Generating customer prompts ...")
    prompts = generate_prompts(brand, site_text)
    say(f"Generated {len(prompts)} prompts.")

    results = []
    for i, prompt_obj in enumerate(prompts, start=1):
        say(f"Gathering prompt {i}/{len(prompts)} across {len(MODELS)} models ...")
        results.extend(gather(brand, prompt_obj))

    say("Analysing answers for competitors and sentiment ...")
    for i, row in enumerate(results, start=1):
        if row.get("error"):
            row["entities"] = []
            continue
        try:
            analysis = analyse_answer(brand, row["answer"])
            row["entities"] = analysis.get("entities", [])
        except Exception as exc:  # one bad analysis call must not kill the scan
            row["entities"] = []
            row["error"] = f"analysis failed: {str(exc)[:200]}"

    say("Scan complete.")
    return {"brand": brand, "url": url, "results": results}


# ---------------------------------------------------------------------------
# Quick standalone test (run: python pipeline.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    report = run_scan("NAB", "https://www.nab.com.au", progress=print)
    print(f"\nGot {len(report['results'])} result rows.")
    mentioned = sum(1 for r in report["results"] if r["brand_mentioned"])
    print(f"Brand mentioned in {mentioned} of {len(report['results'])} answers.")
    errors = sum(1 for r in report["results"] if r["error"])
    print(f"Rows with errors: {errors}")
