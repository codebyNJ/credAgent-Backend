# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from textwrap import dedent
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools
from transformers import pipeline
import logging
from typing import List, Optional
import uuid
from datetime import datetime
import os
import certifi

# Set up environment
os.environ["SSL_CERT_FILE"] = certifi.where()
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
PHI_API_KEY = os.getenv('PHI_API_KEY')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Financial Intelligence API", version="1.0.0")

# CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: str
    timestamp: str

class Session(BaseModel):
    id: str
    created_at: str
    messages: List[dict] = []

# In-memory storage for sessions (use a database in production)
sessions = {}

class NLPIntentClassifier:
    """
    NLP-based intent classifier using a pre-trained model
    """
    def __init__(self):
        # Initialize with a financial-specific model
        try:
            # Using a model that understands financial context
            self.classifier = pipeline(
                "text-classification", 
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                top_k=3
            )
        except Exception as e:
            print(f"Error loading NLP model: {e}")
            self.classifier = None
    
    def classify_intent(self, query: str) -> str:
        """
        Classify user intent using NLP model with fallback to keywords
        """
        # Try NLP classification first
        if self.classifier:
            try:
                results = self.classifier(query)
                
                # Extract labels and scores
                labels = [result['label'] for result in results]
                scores = [result['score'] for result in results]
                
                logger.info(f"NLP Classification Results: {list(zip(labels, scores))}")
                
                # Map model outputs to our intent categories
                query_lower = query.lower()
                
                # Check for credit-related terms (highest priority)
                if any(term in query_lower for term in ["credit", "risk", "default", "rating", "score", "worthiness", "debt", "leverage"]):
                    return "credit_score"
                
                # Check for market analysis terms
                if any(term in query_lower for term in ["stock", "market", "price", "analysis", "pe ratio", "eps", "financial", "earnings"]):
                    return "market_analysis"
                
                # Check for research terms
                if any(term in query_lower for term in ["research", "trend", "future", "impact", "implication", "report", "study"]):
                    return "research"
                
                # Default based on sentiment if no specific terms found
                positive_score = sum(score for label, score in zip(labels, scores) if 'positive' in label.lower())
                negative_score = sum(score for label, score in zip(labels, scores) if 'negative' in label.lower())
                
                if negative_score > positive_score:
                    return "credit_score"  # Negative sentiment often relates to risk/credit
                else:
                    return "market_analysis"  # Positive sentiment often relates to investments
                    
            except Exception as e:
                logger.error(f"NLP classification error: {e}, falling back to keywords")
                return self.keyword_fallback(query)
        else:
            return self.keyword_fallback(query)
    
    def keyword_fallback(self, query: str) -> str:
        """
        Fallback to keyword-based classification if NLP fails
        """
        query_lower = query.lower()
        if any(word in query_lower for word in ["stock", "market", "price", "analysis", "pe ratio", "eps", "financial", "earnings"]):
            return "market_analysis"
        elif any(word in query_lower for word in ["research", "trend", "future", "impact", "implication", "report", "study"]):
            return "research"
        elif any(word in query_lower for word in ["credit", "risk", "default", "rating", "score", "worthiness", "debt", "leverage"]):
            return "credit_score"
        else:
            return "general"

# Initialize the NLP classifier
intent_classifier = NLPIntentClassifier()

# Initialize the financial agent
financial_agent = Agent(
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

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Create new session if none provided
        if not request.session_id or request.session_id not in sessions:
            session_id = str(uuid.uuid4())
            sessions[session_id] = Session(
                id=session_id,
                created_at=datetime.now().isoformat(),
                messages=[]
            )
        else:
            session_id = request.session_id
        
        # Classify intent
        intent = intent_classifier.classify_intent(request.message)
        logger.info(f"Detected intent: {intent} for message: {request.message}")
        
        # Get response from financial agent
        response = financial_agent.run(request.message)
        
        # Store message in session
        sessions[session_id].messages.append({
            "type": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        })
        sessions[session_id].messages.append({
            "type": "assistant",
            "content": response.content,
            "timestamp": datetime.now().isoformat()
        })
        
        return ChatResponse(
            response=response.content,
            session_id=session_id,
            intent=intent,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"message": "Session deleted"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
