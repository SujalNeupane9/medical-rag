from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
import uvicorn
from contextlib import asynccontextmanager

# Import your existing PDFProcessor
from main import PDFProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store the processor
pdf_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the PDF processor on startup"""
    global pdf_processor
    try:
        logger.info("Initializing PDF processor...")
        pdf_processor = PDFProcessor()
        pdf_processor.initialize()
        logger.info("PDF processor initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize PDF processor: {str(e)}")
        raise
    finally:
        logger.info("Shutting down...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="PDF Chat API",
    description="Chat with your PDF documents using AI",
    version="1.0.0",
    lifespan=lifespan
)

# Request/Response models
class ChatRequest(BaseModel):
    question: str
    
class SourceDocument(BaseModel):
    filename: str
    page: int
    content_preview: str
    chunk_index: int

class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDocument]
    total_sources: int

class HealthResponse(BaseModel):
    status: str
    message: str
    documents_loaded: bool

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global pdf_processor
    
    if pdf_processor is None:
        return HealthResponse(
            status="error",
            message="PDF processor not initialized",
            documents_loaded=False
        )
    
    documents_loaded = pdf_processor.vector_store is not None
    
    return HealthResponse(
        status="healthy",
        message="PDF processor is running",
        documents_loaded=documents_loaded
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_with_pdfs(request: ChatRequest):
    """
    Chat endpoint to ask questions about your PDF documents
    
    Args:
        request: ChatRequest containing the question to ask
        
    Returns:
        ChatResponse with the answer and source documents
        
    Raises:
        HTTPException: If the processor is not initialized or query fails
    """
    global pdf_processor
    
    if pdf_processor is None:
        raise HTTPException(
            status_code=500, 
            detail="PDF processor not initialized. Check server logs."
        )
    
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    try:
        logger.info(f"Processing chat request: {request.question[:100]}...")
        
        # Query the PDF processor
        response = pdf_processor.query(request.question)
        
        # Extract source documents
        sources = []
        source_docs = response.get('source_documents', [])
        
        for doc in source_docs:
            sources.append(SourceDocument(
                filename=doc.metadata.get('filename', 'Unknown'),
                page=doc.metadata.get('page', 0),
                content_preview=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                chunk_index=doc.metadata.get('chunk_index', 0)
            ))
        
        # Create response
        chat_response = ChatResponse(
            question=request.question,
            answer=response.get('answer', 'No answer found'),
            sources=sources,
            total_sources=len(sources)
        )
        
        logger.info(f"Chat request processed successfully. Answer length: {len(chat_response.answer)}")
        return chat_response
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing your question: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {
        "message": "PDF Chat API",
        "description": "Upload PDFs to the 'data' folder and chat with them using the /chat endpoint",
        "endpoints": {
            "health": "/health - Check API health status",
            "chat": "/chat - Ask questions about your PDFs",
            "docs": "/docs - Interactive API documentation"
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "message": "Check /docs for available endpoints"}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return {"error": "Internal server error", "message": "Check server logs for details"}

if __name__ == "__main__":
    uvicorn.run(
        "api:app",  # Assuming this file is named api.py
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )