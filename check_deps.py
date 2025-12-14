
import sys
try:
    import pandas
    import yaml
    import uvicorn
    import fastapi
    from sentiment_analysis.social_sentiment import SocialSentimentAnalyzer
    print("All imports successful.")
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
