from textwrap import dedent
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

import os, certifi

os.environ["SSL_CERT_FILE"] = certifi.where()
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
PHI_API_KEY = os.getenv('PHI_API_KEY')


class NLPIntentClassifier:
    """
    NLP-based intent classifier using a pre-trained model
    """

    def __init__(self):
        # Initialize with a financial-specific model if available, or a general one
        try:
            # Try to load a model fine-tuned on financial text
            self.classifier = pipeline(
                "text-classification",
                model="yiyanghkust/finbert-tone",  # Financial sentiment model
                top_k=1
            )
            # Map financial sentiment to our intents (this would need adjustment)
            self.intent_map = {
                'Positive': 'market_analysis',
                'Negative': 'credit_score',
                'Neutral': 'research'
            }
        except:
            # Fallback to a general purpose model
            self.classifier = pipeline(
                "text-classification",
                model="joeddav/distilbert-base-uncased-go-emotions-student",
                top_k=1
            )
            # Map general emotions to our intents
            self.intent_map = {
                'curiosity': 'research',
                'confusion': 'research',
                'approval': 'market_analysis',
                'disapproval': 'credit_score',
                'neutral': 'general'
            }

    def classify_intent(self, query: str) -> str:
        """
        Classify user intent using NLP model
        """
        try:
            result = self.classifier(query)[0][0]
            label = result['label']
            confidence = result['score']

            # Map the model's output to our intent categories
            if label in self.intent_map:
                intent = self.intent_map[label]
                print(f"Detected Intent: {intent} (Confidence: {confidence:.2f})")
                return intent
            else:
                # Fallback to keyword matching if model output isn't in our map
                return self.keyword_fallback(query)
        except Exception as e:
            print(f"NLP classification failed: {e}, falling back to keywords")
            return self.keyword_fallback(query)

    def keyword_fallback(self, query: str) -> str:
        """
        Fallback to keyword-based classification if NLP fails
        """
        query_lower = query.lower()
        if any(word in query_lower for word in ["stock", "market", "price", "analysis", "pe ratio", "eps"]):
            return "market_analysis"
        elif any(word in query_lower for word in ["research", "trend", "future", "impact", "implication", "report"]):
            return "research"
        elif any(word in query_lower for word in ["credit", "risk", "default", "rating", "score", "worthiness"]):
            return "credit_score"
        else:
            return "general"


# Initialize the NLP classifier
intent_classifier = NLPIntentClassifier()

financial_intelligence_agent = Agent(
    model=Groq(id="llama3-70b-8192"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            historical_prices=True,
            company_info=True,
            company_news=True,
        ),
        DuckDuckGoTools(),
        Newspaper4kTools(),
        # FredTools()
    ],
    description="A unified Financial Intelligence Agent handling analysis, research, and credit scoring.",
    instructions=dedent("""\
        You are the **Financial Intelligence Agent** – adapt based on intent classification.

        ## Role Mapping:
        - **Market Analysis** → Provide stock price, key metrics, industry trends, sentiment, and forward-looking outlook.
        - **Research** → Conduct investigative research using multiple sources, summarize, analyze trends, and predict implications.
        - **Credit Score** → Assess financial ratios, risks, management, assign creditworthiness score, explain methodology.

        ## Reporting Rules:
        - Always begin with a short Executive Summary.
        - Use tables for financial/quantitative data.
        - Assign creditworthiness score and its interpretation.
        - Justification for the assigned score and score partition.

    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
)

# Example usage
user_queries = [
    "Evaluate Meta vs Google creditworthiness and provide a risk assessment",
    "What's the current stock price of Tesla and its P/E ratio?",
    "Research the impact of AI on financial markets over the next 5 years",
    "How is the semiconductor industry performing this quarter?",
    "Should I be worried about Microsoft's debt levels?"
]

for user_query in user_queries:
    print(f"\nQuery: {user_query}")
    intent = intent_classifier.classify_intent(user_query)
    financial_intelligence_agent.print_response(user_query, stream=True)