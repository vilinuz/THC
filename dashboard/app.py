
import asyncio
import json
import logging
import os
import sys
import yaml
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

# Add parent dir to path to import backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sentiment_analysis.social_sentiment import SocialSentimentAnalyzer

app = FastAPI(title="Crypto Sentiment Dashboard")

# CORS and Static Files
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import RedirectResponse

# Ensure static dir exists
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir), html=True), name="static")

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

# Event Queue for SSE
event_queue = asyncio.Queue()

# Global Config Path
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))

@app.get("/api/config")
async def get_config():
    """Read config.yaml"""
    try:
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/config")
async def update_config(request: Request):
    """Update config.yaml"""
    try:
        new_config = await request.json()
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(new_config, f, sort_keys=False)
        return {"status": "success", "message": "Config updated"}
    except Exception as e:
        return {"error": str(e)}

async def run_analysis_worker(keywords: list):
    """Background worker that runs analysis and pushes events"""
    analyzer = SocialSentimentAnalyzer(config_path=CONFIG_PATH)
    
    def progress_callback(event, msg):
        asyncio.create_task(event_queue.put({
            "event": event,
            "data": msg
        }))
    
    try:
        await event_queue.put({"event": "START", "data": "Analysis worker started..."})
        
        # Run blocking analysis in thread pool
        loop = asyncio.get_event_loop()
        results, anomalies = await loop.run_in_executor(
            None, 
            lambda: analyzer.run_sentiment_analysis(
                keywords=keywords, 
                callback=progress_callback
            )
        )
        
        # Serialize results for frontend
        agg_json = json.dumps(results['aggregate'])
        await event_queue.put({"event": "RESULT_AGG", "data": agg_json})
        
        # Serialize anomalies text
        anom_summary = f"Most Positive: {anomalies.get('most_positive', 'N/A')[:100]}..."
        await event_queue.put({"event": "RESULT_ANOM", "data": anom_summary})

    except Exception as e:
        await event_queue.put({"event": "ERROR", "data": str(e)})

@app.post("/api/start")
async def start_analysis(background_tasks: BackgroundTasks, request: Request):
    """Start sentiment analysis in background"""
    body = await request.json()
    keywords = body.get("keywords", ["bitcoin", "ethereum"])
    background_tasks.add_task(run_analysis_worker, keywords)
    return {"status": "started", "keywords": keywords}

@app.get("/api/events")
async def sse_endpoint(request: Request):
    """SSE endpoint for real-time updates"""
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            # Wait for event
            item = await event_queue.get()
            yield {
                "event": "message", 
                "data": json.dumps(item)
            }
    return EventSourceResponse(event_generator())
