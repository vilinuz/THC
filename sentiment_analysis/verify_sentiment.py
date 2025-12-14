
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Mock dependencies
sys.modules['pandas'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['textblob'] = MagicMock()
sys.modules['nltk'] = MagicMock()
sys.modules['nltk.sentiment.vader'] = MagicMock()
sys.modules['tweepy'] = MagicMock()
sys.modules['google'] = MagicMock()
sys.modules['google.generativeai'] = MagicMock()
sys.modules['praw'] = MagicMock()

# Ensure we can import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sentiment_analysis.social_sentiment import SocialSentimentAnalyzer

class TestSocialSentiment(unittest.TestCase):

    def setUp(self):
        # Mock load_config to return a test config
        self.test_config = {
            'sentiment_analysis': {
                'use_gemini': True,
                'gemini_api_key': 'fake_key',
                'twitter_accounts': ['elonmusk']
            },
            'notifications': {
                'email': {'enabled': True, 'sender': 'me@test.com', 'smtp_server': 'smtp.test.com', 'password': 'pass'},
                'telegram': {'enabled': True, 'bot_token': 'fake_token', 'chat_id': '123'},
                'alerts': {'sentiment_threshold': 0.8}
            }
        }
    
    @patch('sentiment_analysis.social_sentiment.SocialSentimentAnalyzer._load_config')
    def test_initialization(self, mock_load):
        mock_load.return_value = self.test_config
        # Since we mocked google.generativeai in sys.modules, we can check it directly
        mock_genai = sys.modules['google.generativeai']
        mock_genai.configure.reset_mock()
        
        analyzer = SocialSentimentAnalyzer()
        # Ensure model is initialized
        self.assertIsNotNone(analyzer.gemini_model)
        self.assertTrue(analyzer.config['sentiment_analysis']['use_gemini'])

    @patch('sentiment_analysis.social_sentiment.SocialSentimentAnalyzer._load_config')
    def test_fetch_twitter_accounts(self, mock_load):
        mock_load.return_value = self.test_config
        analyzer = SocialSentimentAnalyzer()
        
        # Mock twitter api
        mock_api = MagicMock()
        analyzer.twitter_api = mock_api
        
        analyzer.fetch_twitter_data()
        # Should have called user_timeline for 'elonmusk'
        mock_api.user_timeline.assert_called_with(screen_name='elonmusk', count=20, tweet_mode='extended')

    @patch('sentiment_analysis.social_sentiment.SocialSentimentAnalyzer._load_config')
    def test_gemini_scoring(self, mock_load):
        mock_load.return_value = self.test_config
        analyzer = SocialSentimentAnalyzer()
        
        # Mock gemini model response
        mock_response = MagicMock()
        mock_response.text = '{"sentiment": "positive", "score": 0.9, "market_impact": "HIGH", "explanation": "test"}'
        analyzer.gemini_model = MagicMock()
        analyzer.gemini_model.generate_content.return_value = mock_response
        
        # Mock pandas DataFrame creation (since we mocked pandas module)
        # We need score_sentiment to return a list or dict if pandas is mocked away, 
        # but in usage it returns a DataFrame.
        # Let's just check if it calls generate_content
        
        # We need to minimally mock DataFrame behavior or just rely on the fact that
        # the method will try to run line by line.
        # The mocked pandas.DataFrame will consume the input list.
        
        analyzer.score_sentiment(['Bitcoin is going to the moon!'])
        analyzer.gemini_model.generate_content.assert_called()

    @patch('sentiment_analysis.social_sentiment.SocialSentimentAnalyzer._load_config')
    @patch('smtplib.SMTP')
    @patch('requests.post')
    def test_notifications(self, mock_post, mock_smtp, mock_load):
        mock_load.return_value = self.test_config
        analyzer = SocialSentimentAnalyzer()
        
        # Create a mock dataframe that mocks the high sentiment
        # df['vader_compound'].max() > 0.8
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.__getitem__.return_value.max.return_value = 0.9
        mock_df.__getitem__.return_value.min.return_value = -0.1
        
        # Mock loc to return text
        mock_df.loc.__getitem__.return_value.__getitem__.return_value = "Positive text"
        
        # Also mock 'market_impact' column check
        # We can simplify by just triggering the vader check
        
        analyzer.detect_anomalies(mock_df)
        
        # Check if email sent
        mock_smtp.return_value.__enter__.return_value.send_message.assert_called()
        
        # Check if telegram sent
        mock_post.assert_called()

if __name__ == "__main__":
    unittest.main()
