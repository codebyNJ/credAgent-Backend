# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from textwrap import dedent
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools
import logging
from typing import List, Optional
import uuid
from datetime import datetime
import os
import certifi
import time

# Set up environment
os.environ["SSL_CERT_FILE"] = certifi.where()
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
PHI_API_KEY = os.getenv('PHI_API_KEY')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Financial Intelligence API",
    description="A unified Financial Intelligence Agent handling analysis, research, and credit scoring.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


# In-memory storage for sessions
sessions = {}


def classify_intent(query: str) -> str:
    """
    Enhanced keyword-based intent classification (replacing the problematic NLP classifier)
    """
    query_lower = query.lower()

    # Credit score intent
    credit_terms = ["credit", "risk", "default", "rating", "score", "worthiness",
                    "debt", "leverage", "solvency", "bankruptcy", "liquidity", "loan"]
    if any(term in query_lower for term in credit_terms):
        return "credit_score"

    # Market analysis intent
    market_terms = ["stock", "market", "price", "analysis", "pe ratio", "eps",
                    "financial", "earnings", "dividend", "valuation", "investment",
                    "invest", "share", "trading", "portfolio", "buy", "sell", "hold"]
    if any(term in query_lower for term in market_terms):
        return "market_analysis"

    # Research intent
    research_terms = ["research", "trend", "future", "impact", "implication",
                      "report", "study", "outlook", "forecast", "predict", "analysis",
                      "industry", "sector", "growth", "development"]
    if any(term in query_lower for term in research_terms):
        return "research"

    return "general"


# Initialize the financial agent with retry configuration
financial_agent = Agent(
    model=Groq(
        id="llama3-70b-8192",
        max_retries=3,
        timeout=30
    ),
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

        ## Important:
        - If you encounter API rate limits or connection issues, provide a helpful response explaining the limitation.
        - For market data, use YFinanceTools as the primary source.
        - For general research, use DuckDuckGoTools and Newspaper4kTools.
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Return a simple HTML page with API information"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Financial Intelligence API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #333; }
            .endpoint { background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0; }
            code { background: #eee; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>Financial Intelligence API</h1>
        <p>A unified Financial Intelligence Agent handling analysis, research, and credit scoring.</p>

        <h2>API Endpoints</h2>

        <div class="endpoint">
            <h3>POST /chat</h3>
            <p>Send a message to the financial intelligence agent.</p>
            <p><strong>Request body:</strong> <code>{"message": "your question here"}</code></p>
        </div>

        <div class="endpoint">
            <h3>GET /sessions/{session_id}</h3>
            <p>Retrieve conversation history for a session.</p>
        </div>

        <div class="endpoint">
            <h3>DELETE /sessions/{session_id}</h3>
            <p>Delete a session and its conversation history.</p>
        </div>

        <div class="endpoint">
            <h3>GET /health</h3>
            <p>Check the health status of the API.</p>
        </div>

        <p>Visit <a href="/docs">/docs</a> for interactive API documentation.</p>
    </body>
    </html>
    """


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Process a user message and return a response from the financial agent"""
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

        # Classify intent using the improved keyword-based approach
        intent = classify_intent(request.message)
        logger.info(f"Detected intent: {intent} for message: {request.message}")

        # Get response from financial agent with timeout handling
        try:
            response = financial_agent.run(request.message)
            response_content = response.content
        except Exception as agent_error:
            logger.warning(f"Agent error: {agent_error}, providing fallback response")
            response_content = f"""
            I encountered a technical issue while processing your request: {str(agent_error)}

            This might be due to:
            - API rate limits or temporary service unavailability
            - Network connectivity issues
            - High demand on the AI service

            Please try again in a moment, or rephrase your question. For market data queries, 
            you might want to try specific questions like "What is Tesla's current stock price?"
            """

        # Store message in session
        sessions[session_id].messages.append({
            "type": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        })
        sessions[session_id].messages.append({
            "type": "assistant",
            "content": response_content,
            "timestamp": datetime.now().isoformat()
        })

        return ChatResponse(
            response=response_content,
            session_id=session_id,
            intent=intent,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Sorry, I encountered an error processing your request. Please try again later. Error: {str(e)}"
        )


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Retrieve conversation history for a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its conversation history"""
    if session_id in sessions:
        del sessions[session_id]
    return {"message": "Session deleted"}


@app.get("/health")
async def health_check():
    """Check the health status of the API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(sessions)
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)