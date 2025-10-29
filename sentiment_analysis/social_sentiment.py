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

class SocialSentimentAnalyzer:
    """
    Aggregates crypto sentiment from X (Twitter), Reddit, and news.
    Typical workflow:
        - Collect posts/tweets/headlines
        - Clean and preprocess
        - Score sentiment using TextBlob and VADER
        - Aggregate and produce time series
        - Generate summary and anomalies report
    """

    def __init__(self, tickers=['BTC', 'ETH'], twitter_api=None, reddit_api=None):
        self.tickers = tickers
        self.twitter_api = twitter_api
        self.reddit_api = reddit_api
        self.sia = SentimentIntensityAnalyzer()

    def fetch_twitter_data(self, query, max_tweets=100):
        """
        Fetch recent tweets using Tweepy or custom Twitter API calls.
        Returns list of tweet texts.
        """
        # Implement Tweepy or Twitter API v2 here (pseudo for self-contained)
        # For production, use: tweepy.Cursor(api.search_tweets, q=query, tweet_mode='extended').items(max_tweets)
        tweets = []  # Replace with actual API code
        return tweets

    def fetch_reddit_data(self, subreddit="CryptoCurrency", query="bitcoin", max_posts=100):
        """
        Fetch Reddit posts/comments using praw or Pushshift API.
        Returns list of post/comment texts.
        """
        posts = []  # Replace with actual API code
        return posts

    def fetch_news_data(self, tickers, n_articles=40):
        """
        Use Perplexity or ContextualWeb/NewsAPI to fetch relevant news headlines.
        Returns list of headlines or article snippets.
        """
        headlines = []
        # Use Perplexity API or similar
        for ticker in tickers:
            # query = f"{ticker} crypto latest news"
            # headlines += [headline objects] # Replace with actual API/Perplexity call
            pass
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
        Vectorized sentiment scoring on cleaned texts.
        Returns list of dicts with polarity, subjectivity, and VADER scores.
        """
        results = []
        for txt in texts:
            tb_polarity = TextBlob(txt).sentiment.polarity
            tb_subjectivity = TextBlob(txt).sentiment.subjectivity
            vader = self.sia.polarity_scores(txt)
            results.append({
                "text": txt,
                "polarity": tb_polarity,
                "subjectivity": tb_subjectivity,
                "vader_compound": vader['compound'],
                "vader_pos": vader['pos'],
                "vader_neu": vader['neu'],
                "vader_neg": vader['neg']
            })
        return pd.DataFrame(results)

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
        """
        anomalies = {}
        anomalies['most_positive'] = sentiment_df.loc[sentiment_df['vader_compound'].idxmax()]['text']
        anomalies['most_negative'] = sentiment_df.loc[sentiment_df['vader_compound'].idxmin()]['text']
        if headlines is not None and len(headlines) > 0:
            anomalies['news_signals'] = self.summarize_headlines(headlines)
        return anomalies

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

    def run_sentiment_analysis(self, keywords=['bitcoin', 'ethereum'], days=1, output_pdf='crypto_sentiment_report.pdf'):
        """
        Top-level function: Gathers posts, scores, aggregates, and generates PDF report.
        """
        # Fetch data
        twitter_texts = []
        reddit_texts = []
        news_headlines = []
        for keyword in keywords:
            twitter_texts += self.fetch_twitter_data(keyword, max_tweets=120)
            reddit_texts += self.fetch_reddit_data(query=keyword, max_posts=120)
            news_headlines += self.fetch_news_data([keyword], n_articles=20)
        texts = twitter_texts + reddit_texts
        cleaned = self.clean_texts(texts)
        sentiment_df = self.score_sentiment(cleaned)
        agg = self.aggregate_sentiment(sentiment_df)
        anomalies = self.detect_anomalies(sentiment_df, news_headlines)
        sentiment_results = {
            "aggregate": agg,
            "sentiment_df": sentiment_df
        }

        # Generate PDF
        self.generate_pdf_report(sentiment_results, anomalies, output_pdf)
        return sentiment_results, anomalies

