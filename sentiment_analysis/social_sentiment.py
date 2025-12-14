"""
Crypto Social Sentiment Analysis
Gathers and analyzes crypto sentiment from X (Twitter), Reddit, and news sources,
generates sentiment scores and PDF summary reports.
"""
import pandas as pd
import numpy as np
import requests
import datetime
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import yaml
import google.generativeai as genai
import smtplib
from email.mime.text import MIMEText
import os

class SocialSentimentAnalyzer:
    """
    Aggregates crypto sentiment from X (Twitter), Reddit, and news.
    Typical workflow:
        - Collect posts/tweets/headlines
        - Clean and preprocess
        - Score sentiment using TextBlob, VADER, or Gemini
        - Aggregate and produce time series
        - Generate summary and anomalies report
        - Send notifications/alerts
    """

    def __init__(self, config_path='config.yaml', tickers=['BTC', 'ETH'], twitter_api=None, reddit_api=None):
        self.tickers = tickers
        self.sia = SentimentIntensityAnalyzer()
        self.config = self._load_config(config_path)
        
        # Initialize APIs from config if not provided
        self.twitter_api = twitter_api
        self.reddit_api = reddit_api
        self._initialize_apis()
        
    def _load_config(self, path):
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Could not load config from {path}: {e}")
            return {}

    def _initialize_apis(self):
        # Setup Gemini
        sa_config = self.config.get('sentiment_analysis', {})
        if sa_config.get('use_gemini'):
            api_key = sa_config.get('gemini_api_key') or os.environ.get('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel('gemini-pro')
            else:
                print("Warning: Gemini API key not found.")
                self.gemini_model = None
        else:
            self.gemini_model = None

        # Setup Tweepy if API keys present and not already passed
        if not self.twitter_api and sa_config.get('twitter_api_key'):
             try:
                import tweepy
                auth = tweepy.OAuthHandler(
                    sa_config.get('twitter_api_key'),
                    sa_config.get('twitter_api_secret')
                )
                auth.set_access_token(
                    sa_config.get('twitter_access_token'),
                    sa_config.get('twitter_access_secret')
                )
                self.twitter_api = tweepy.API(auth)
             except Exception as e:
                 print(f"Warning: Failed to initialize Twitter API: {e}")

        # Setup PRAW if keys present
        if not self.reddit_api and sa_config.get('reddit_client_id'):
            try:
                import praw
                self.reddit_api = praw.Reddit(
                    client_id=sa_config.get('reddit_client_id'),
                    client_secret=sa_config.get('reddit_client_secret'),
                    user_agent="crypto_sentiment_bot"
                )
            except Exception as e:
                print(f"Warning: Failed to initialize Reddit API: {e}")

    def fetch_twitter_data(self, query=None, accounts=None, max_tweets=100):
        """
        Fetch recent tweets using Tweepy or custom Twitter API calls.
        Can fetch via query or specific accounts.
        """
        tweets = []
        try:
            if not self.twitter_api:
                return []
            
            # Fetch from specific accounts if provided or in config
            if accounts is None:
                accounts = self.config.get('sentiment_analysis', {}).get('twitter_accounts', [])
            
            if accounts:
                for account in accounts:
                    try:
                        # Fetch user timeline
                        timeline = self.twitter_api.user_timeline(
                            screen_name=account,
                            count=min(max_tweets, 20), # Limit per user
                            tweet_mode='extended'
                        )
                        for status in timeline:
                             tweets.append(status.full_text)
                    except Exception as e:
                        print(f"Error fetching timeline for {account}: {e}")

            # If query provided, also search
            if query:
                cursor = tweepy.Cursor(
                    self.twitter_api.search_tweets,
                    q=query,
                    tweet_mode='extended',
                    lang='en'
                ).items(max_tweets)
                
                for status in cursor:
                    if hasattr(status, 'full_text'):
                        tweets.append(status.full_text)
                    else:
                        tweets.append(status.text)
                    
        except ImportError:
            print("Error: Tweepy library not installed.")
        except Exception as e:
            print(f"Error fetching Twitter data: {str(e)}")
            
        return tweets

    def fetch_reddit_data(self, subreddit="CryptoCurrency", query="bitcoin", max_posts=100):
        """
        Fetch Reddit posts/comments using praw or Pushshift API.
        Returns list of post/comment texts.
        """
        posts = []
        try:
            if not self.reddit_api:
                print("Warning: Reddit API client not provided.")
                return []
                
            # Attempt to use PRAW interface
            # self.reddit_api should be a praw.Reddit instance
            results = self.reddit_api.subreddit(subreddit).search(query, limit=max_posts)
            for submission in results:
                # Combine title and body for better sentiment context
                content = f"{submission.title} {submission.selftext}"
                posts.append(content)
                
        except Exception as e:
            print(f"Error fetching Reddit data: {str(e)}")
            
        return posts

    def fetch_news_data(self, tickers, n_articles=40):
        """
        Use requests to fetch relevant news headlines from a news API.
        Returns list of headlines or article snippets.
        """
        headlines = []
        try:
            # Placeholder URL - normally specific to a News API provider
            # e.g., https://newsapi.org/v2/everything
            base_url = "https://newsapi.org/v2/everything"
            
            # Check for API key in environment or assume it's set
            import os
            api_key = os.environ.get("NEWS_API_KEY", "")
            if not api_key:
                print("Warning: NEWS_API_KEY not found in environment.")
                # We return empty but don't crash, allowing other data sources to work
                return []

            for ticker in tickers:
                params = {
                    'q': ticker,
                    'apiKey': api_key,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': n_articles
                }
                response = requests.get(base_url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                if 'articles' in data:
                    for article in data['articles']:
                        if article.get('title'):
                            headlines.append(article['title'])
                            
        except requests.exceptions.RequestException as re:
            print(f"Network error fetching news: {str(re)}")
        except Exception as e:
            print(f"Error fetching news data: {str(e)}")
            
        return headlines

    def clean_texts(self, texts):
        """
        Remove URLs, emojis, and excessive whitespace.
        """
        import re
        clean_list = []
        for text in texts:
            txt = re.sub('https?:\\/\\/\\S+', '', text)
            txt = re.sub(r"[^A-Za-z0-9 .,!?']", " ", txt)
            clean_list.append(txt.strip())
        return clean_list

    def score_sentiment(self, texts):
        """
        Score sentiment using TextBlob/VADER or Gemini if configured.
        """
        results = []
        use_gemini = self.config.get('sentiment_analysis', {}).get('use_gemini', False)

        for txt in texts:
            # Basic analysis (always run)
            tb_polarity = TextBlob(txt).sentiment.polarity
            tb_subjectivity = TextBlob(txt).sentiment.subjectivity
            vader = self.sia.polarity_scores(txt)
            
            res = {
                "text": txt,
                "polarity": tb_polarity,
                "subjectivity": tb_subjectivity,
                "vader_compound": vader['compound'],
                "vader_pos": vader['pos'],
                "vader_neu": vader['neu'],
                "vader_neg": vader['neg'],
                "gemini_score": 0.0,
                "market_impact": "NONE"
            }

            if use_gemini and self.gemini_model:
                try:
                    gemini_res = self.score_sentiment_gemini(txt)
                    res['gemini_score'] = gemini_res.get('score', 0.0)
                    res['market_impact'] = gemini_res.get('market_impact', 'NONE')
                    # Optionally override compound score or keep separate
                except Exception as e:
                    print(f"Gemini scoring failed: {e}")

            results.append(res)
        return pd.DataFrame(results)

    def score_sentiment_gemini(self, text):
        """
        Uses Gemini API to score sentiment and assess market impact.
        Returns dict with score and impact.
        """
        prompt = f"""
        Analyze the sentiment of the following text regarding cryptocurrency markets.
        Text: '{text}'
        
        Return a valid JSON object with:
        - sentiment: (positive/negative/neutral)
        - score: (float between -1.0 and 1.0)
        - market_impact: (HIGH/MEDIUM/LOW/NONE)
        - explanation: (brief string)
        
        Do not include markdown formatting or backticks. Just the raw JSON.
        """
        response = self.gemini_model.generate_content(prompt)
        try:
            import json
            # Sanitize response just in case
            clean_resp = response.text.replace('```json', '').replace('```', '').strip()
            return json.loads(clean_resp)
        except Exception:
            return {"score": 0.0, "market_impact": "NONE"}

    def aggregate_sentiment(self, sentiment_df):
        """
        Aggregate scores into overall daily/period sentiment.
        """
        avg_compound = sentiment_df['vader_compound'].mean()
        pos_pct = (sentiment_df['vader_compound'] > 0.20).mean()
        neg_pct = (sentiment_df['vader_compound'] < -0.20).mean()
        neutral_pct = 1 - pos_pct - neg_pct
        return {
            "avg_compound": avg_compound,
            "positive_pct": pos_pct,
            "negative_pct": neg_pct,
            "neutral_pct": neutral_pct
        }

    def detect_anomalies(self, sentiment_df, headlines=None):
        """
        Find extreme positive/negative posts/headlines and surfacing news keywords.
        Triggers notifications if thresholds are breached.
        """
        anomalies = {}
        if sentiment_df.empty:
            return anomalies
            
        anomalies['most_positive'] = sentiment_df.loc[sentiment_df['vader_compound'].idxmax()]['text']
        anomalies['most_negative'] = sentiment_df.loc[sentiment_df['vader_compound'].idxmin()]['text']
        
        # News Analysis
        if headlines is not None and len(headlines) > 0:
            anomalies['news_signals'] = self.summarize_headlines(headlines)
        
        # Check thresholds for alerts
        alert_config = self.config.get('notifications', {}).get('alerts', {})
        sent_thresh = float(alert_config.get('sentiment_threshold', 0.5))
        
        alerts_triggered = []
        
        # Check extreme VADER scores
        max_vader = sentiment_df['vader_compound'].max()
        min_vader = sentiment_df['vader_compound'].min()
        
        if max_vader > sent_thresh:
            alerts_triggered.append(f"High Positive Sentiment Detected: {max_vader:.2f}")
        if min_vader < -sent_thresh:
            alerts_triggered.append(f"High Negative Sentiment Detected: {min_vader:.2f}")

        # Check Gemini Impact (if available)
        if 'market_impact' in sentiment_df.columns:
            high_impact = sentiment_df[sentiment_df['market_impact'] == 'HIGH']
            if not high_impact.empty:
                count = len(high_impact)
                alerts_triggered.append(f"{count} High Impact posts detected by Gemini.")

        # Send Notifications if alerts triggered
        if alerts_triggered:
            subject = "CRYPTO SENTIMENT ALERT"
            msg = "\n".join(alerts_triggered)
            self.send_notifications(subject, msg)

        return anomalies
        
    def send_notifications(self, subject, message):
        """
        Sends notifications via enabled channels (Email, Telegram).
        """
        notif_config = self.config.get('notifications', {})
        
        # Email
        email_conf = notif_config.get('email', {})
        if email_conf.get('enabled'):
            try:
                msg = MIMEText(message)
                msg['Subject'] = subject
                sender = email_conf.get('sender') or os.environ.get('EMAIL_SENDER')
                msg['From'] = sender
                recipients = email_conf.get('recipients', [])
                msg['To'] = ", ".join(recipients)
                
                # SMTP Setup (Basic)
                smtp_server = email_conf.get('smtp_server')
                smtp_port = email_conf.get('smtp_port', 587)
                password = email_conf.get('password') or os.environ.get('EMAIL_PASSWORD')
                
                if smtp_server and sender and password:
                    with smtplib.SMTP(smtp_server, smtp_port) as server:
                        server.starttls()
                        server.login(sender, password)
                        server.send_message(msg)
                    print("Email notification sent.")
            except Exception as e:
                print(f"Failed to send email: {e}")

        # Telegram
        tg_conf = notif_config.get('telegram', {})
        if tg_conf.get('enabled'):
            try:
                bot_token = tg_conf.get('bot_token') or os.environ.get('TELEGRAM_BOT_TOKEN')
                chat_id = tg_conf.get('chat_id') or os.environ.get('TELEGRAM_CHAT_ID')
                
                if bot_token and chat_id:
                    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                    data = {"chat_id": chat_id, "text": f"{subject}\n\n{message}"}
                    requests.post(url, data=data, timeout=10)
                    print("Telegram notification sent.")
            except Exception as e:
                 print(f"Failed to send Telegram message: {e}")

    def summarize_headlines(self, headlines):
        """
        Use LLM (e.g., OpenAI, Perplexity) or local summarizer to distill signals.
        Returns summary string with positive/negative market factors.
        """
        pos_signals, neg_signals = [], []
        for headline in headlines:
            tb = TextBlob(headline)
            if tb.sentiment.polarity > 0.25:
                pos_signals.append(headline)
            elif tb.sentiment.polarity < -0.25:
                neg_signals.append(headline)
        return {
            "positive": pos_signals,
            "negative": neg_signals,
            "summary": "Positive news signals detected: {}. Negative signals detected: {}.".format(
                len(pos_signals), len(neg_signals)
            )
        }

    def generate_pdf_report(self, sentiment_results, anomalies, output_path):
        """
        Generate PDF report using reportlab/fpdf2
        Includes: summary, anomaly highlights, plots, tables
        """
        from reportlab.platypus import SimpleDocTemplate, Table, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.pagesizes import letter
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        story.append(Paragraph("Crypto Sentiment Analysis Report", styles['Title']))
        story.append(Spacer(1, 12))
        today = datetime.date.today().isoformat()
        story.append(Paragraph(f"Generated on {today}", styles['Normal']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Summary:", styles['Heading2']))
        agg = sentiment_results['aggregate']
        data = [
            ['Score', 'Percent'],
            ['Positive', f"{agg['positive_pct']:.2%}"],
            ['Negative', f"{agg['negative_pct']:.2%}"],
            ['Neutral', f"{agg['neutral_pct']:.2%}"],
            ['Average Compound', f"{agg['avg_compound']:.4f}"]
        ]
        story.append(Table(data))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Anomaly Highlights:", styles['Heading2']))
        story.append(Paragraph("Most Positive Post/Tweet: " + anomalies['most_positive'], styles['Normal']))
        story.append(Paragraph("Most Negative Post/Tweet: " + anomalies['most_negative'], styles['Normal']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("News Signals Summary:", styles['Heading2']))
        story.append(Paragraph(anomalies['news_signals']['summary'], styles['Normal']))
        story.append(Spacer(1, 12))
        # TODO: Add chart image generation
        doc.build(story)
        return output_path

    def run_sentiment_analysis(self, keywords=['bitcoin', 'ethereum'], days=1, output_pdf='crypto_sentiment_report.pdf', callback=None):
        """
        Top-level function: Gathers posts, scores, aggregates, and generates PDF report.
        Callback signature: func(event_type: str, message: str)
        """
        def log(event, msg):
            if callback:
                try:
                    callback(event, msg)
                except Exception:
                    pass
            print(f"[{event}] {msg}")

        log("START", f"Starting analysis for {keywords}")

        # Fetch data
        twitter_texts = []
        reddit_texts = []
        news_headlines = []
        
        for keyword in keywords:
            log("FETCH", f"Fetching Twitter data for {keyword}...")
            t_data = self.fetch_twitter_data(keyword, max_tweets=120)
            twitter_texts += t_data
            log("FETCH", f"Fetched {len(t_data)} tweets.")
            
            log("FETCH", f"Fetching Reddit data for {keyword}...")
            r_data = self.fetch_reddit_data(query=keyword, max_posts=120)
            reddit_texts += r_data
            log("FETCH", f"Fetched {len(r_data)} posts.")
            
            log("FETCH", f"Fetching News for {keyword}...")
            n_data = self.fetch_news_data([keyword], n_articles=20)
            news_headlines += n_data
            log("FETCH", f"Fetched {len(n_data)} headlines.")

        texts = twitter_texts + reddit_texts
        log("PROCESS", f"Cleaning {len(texts)} total text items...")
        cleaned = self.clean_texts(texts)
        
        log("SCORE", "Scoring sentiment...")
        sentiment_df = self.score_sentiment(cleaned)
        log("SCORE", "Sentiment scoring complete.")
        
        log("AGGREGATE", "Aggregating results...")
        agg = self.aggregate_sentiment(sentiment_df)
        
        log("ANALYZE", "Detecting anomalies and news signals...")
        anomalies = self.detect_anomalies(sentiment_df, news_headlines)
        
        sentiment_results = {
            "aggregate": agg,
            "sentiment_df": sentiment_df
        }

        # Generate PDF
        log("REPORT", f"Generating PDF report: {output_pdf}")
        self.generate_pdf_report(sentiment_results, anomalies, output_pdf)
        
        log("COMPLETE", "Analysis run finished successfully.")
        return sentiment_results, anomalies

