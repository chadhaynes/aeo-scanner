"""
AEO Scanner — FastAPI server.

Sits between the browser and the pipeline engine. Receives scan requests,
serves cached results instantly, runs a live scan (and caches it) on a miss.
"""

import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import run_scan

app = FastAPI(title="AEO Scanner")

# allow the browser page to call this server during local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)


class ScanRequest(BaseModel):
    """Shape of the incoming request body — FastAPI validates against this."""
    brand: str
    url: str


def cache_path(brand):
    """One JSON file per brand, name sanitised for the filesystem."""
    safe = "".join(c for c in brand.lower() if c.isalnum() or c in "-_")
    return os.path.join(CACHE_DIR, f"{safe}.json")


@app.post("/scan")
def scan(req: ScanRequest):
    """Return an AEO report for a brand — from cache if we have it, else live."""
    path = cache_path(req.brand)

    # cache hit — serve instantly, no API spend
    if os.path.exists(path):
        with open(path) as f:
            report = json.load(f)
        report["cached"] = True
        return report

    # cache miss — run the full pipeline, then persist
    report = run_scan(req.brand, req.url)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    report["cached"] = False
    return report


@app.get("/health")
def health():
    """Trivial endpoint to confirm the server is up."""
    return {"status": "ok"}

# serve index.html at the root — must be the LAST line, after all routes
app.mount("/", StaticFiles(directory=".", html=True), name="static")