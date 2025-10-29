import redis
import json
import pickle
from typing import Any, Optional, Callable
from functools import wraps
import hashlib

class RedisManager:
    """Redis cache manager with distributed capabilities"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, ttl: int = 300):
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=False,
            socket_connect_timeout=5,
            socket_keepalive=True,
            health_check_interval=30
        )
        self.default_ttl = ttl
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = self.client.get(key)
            if value:
                return pickle.loads(value)
        except Exception as e:
            print(f"Cache get error: {e}")
        return None
        
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache with TTL"""
        try:
            serialized = pickle.dumps(value)
            self.client.setex(key, ttl or self.default_ttl, serialized)
        except Exception as e:
            print(f"Cache set error: {e}")
            
    def delete(self, key: str):
        """Delete key from cache"""
        self.client.delete(key)
 
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        return self.client.exists(key) > 0

    def cache_dataframe(self, key: str, df: Any, ttl: Optional[int] = None):
        """Cache pandas DataFrame efficiently"""
        try:
            json_data = df.to_json()
            self.client.setex(f"df:{key}", ttl or self.default_ttl, json_data)
        except Exception as e:
            print(f"DataFrame cache error: {e}")

    def get_dataframe(self, key: str) -> Optional[Any]:
        """Retrieve cached DataFrame"""
        try:
            import pandas as pd
            json_data = self.client.get(f"df:{key}")
            if json_data:
                return pd.read_json(json_data)
        except Exception as e:
            print(f"DataFrame retrieval error: {e}")
        return None

    def cache_decorator(self, ttl: Optional[int] = None):
        """Decorator for caching function results"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key from function name and arguments
                key_data = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                cache_key = hashlib.md5(key_data.encode()).hexdigest()

                # Try to get from cache
                cached = self.get(cache_key)
                if cached is not None:
                    return cached
                    
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl or self.default_ttl)
                return result
            return wrapper
        return decorator
        
    def publish(self, channel: str, message: dict):
        """Publish message to Redis pub/sub channel"""
        self.client.publish(channel, json.dumps(message))
        
    def subscribe(self, channels: list) -> Any:
        """Subscribe to Redis pub/sub channels"""
        pubsub = self.client.pubsub()
        pubsub.subscribe(channels)
        return pubsub
        
    def close(self):
        """Close Redis connection"""
        self.client.close()
