
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
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent dir to path to import backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sentiment_analysis.social_sentiment import SocialSentimentAnalyzer
from indicators.tillson_t3 import TillsonT3
from strategy.b_sniper_strategy import BSniperStrategy

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

# ... imports
import redis.asyncio as redis

# ... (Previous Code)

# Redis Client
redis_client = None

@app.on_event("startup")
async def startup_event():
    global redis_client
    # Connect to Redis (Config default)
    # In prod, read from config.yaml
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

@app.on_event("shutdown")
async def shutdown_event():
    if redis_client:
        await redis_client.close()

async def fetch_market_data():
    """Fetch latest market data from Redis Streams"""
    # Assuming config pairs, e.g. BINANCE-BTC-USDT
    # Ticker Key in Cryptofeed: ticker-BINANCE-BTC-USDT
    # Stream key usually implies XREAD
    if not redis_client: return None
    
    # Check for stream keys (simplified for demo)
    stream_key = "ticker-BINANCE-BTC-USDT" 
    
    try:
        # Read last entry from stream
        data = await redis_client.xrevrange(stream_key, count=1)
        if data:
            # Format: [(b'timestamp', {b'bid': ..., b'ask': ...})]
            # decoded: [('timestamp', {'bid': '...', ...})]
            entry_id, payload = data[0]
            payload['source'] = 'redis_stream'
            return payload
    except Exception as e:
        # Fallback for simple keys if not using streams
        pass
    return None

@app.get("/api/events")
async def sse_endpoint(request: Request):
    """SSE endpoint for real-time updates"""
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            
            # 1. Check Event Queue
            try:
                item = event_queue.get_nowait()
                yield {
                    "event": "message", 
                    "data": json.dumps(item)
                }
            except asyncio.QueueEmpty:
                pass
                
            # 2. Check Market Data (Polling Redis)
            market_data = await fetch_market_data()
            if market_data:
                yield {
                    "event": "market_data",
                    "data": json.dumps(market_data)
                }
            
            await asyncio.sleep(1) # Frequency
            
    return EventSourceResponse(event_generator())

@app.get("/api/chart_data")
async def get_chart_data():
    """
    Generate dummy OHLCV data + T3 + Signals for visualization
    In a real app, this would fetch from DB or Exchange.
    """
    # Generate 100 bars of dummy uptrend/choppy data
    periods = 200
    base_price = 45000
    
    dates = [datetime.now() - timedelta(minutes=periods-i) for i in range(periods)]
    
    # Random walk with drift
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.002, periods)
    price_curve = base_price * np.cumprod(1 + returns)
    
    data = []
    for i in range(periods):
        close = price_curve[i]
        # Add some volatility for High/Low
        high = close * (1 + abs(np.random.normal(0, 0.001)))
        low = close * (1 - abs(np.random.normal(0, 0.001)))
        open_p = close * (1 + np.random.normal(0, 0.001)) # Slight variation
        
        # Ensure High/Low envelop Open/Close
        high = max(high, open_p, close)
        low = min(low, open_p, close)
        
        data.append({
            'time': dates[i].timestamp(),
            'open': open_p,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(100, 1000)
        })
        
    df = pd.DataFrame(data)
    
    # Calculate T3
    t3 = TillsonT3.calculate(df, length=10, volume_factor=1.7)
    df['t3'] = t3.fillna(0)
    
    # Calculate Signals
    strat = BSniperStrategy(config={'t3_length': 10, 'volume_factor': 1.7})
    signals = strat.generate_signals(df)
    df['signal'] = signals
    
    # Transform to JSON structure
    # Lightweight Charts expects unix timestamp (seconds)
    chart_data = {
        'candles': [],
        't3': [],
        'markers': []
    }
    
    for i, row in df.iterrows():
        t = row['time'] 
        chart_data['candles'].append({
            'time': t,
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close']
        })
        
        if row['t3'] != 0:
            chart_data['t3'].append({
                'time': t,
                'value': row['t3']
            })
            
        if row['signal'] == 1:
            chart_data['markers'].append({
                'time': t,
                'position': 'belowBar',
                'color': '#22c55e', # Green
                'shape': 'arrowUp',
                'text': 'B'
            })
            
    return chart_data
