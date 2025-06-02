from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
import uvicorn
import os
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
        logger.info("Initializing PDF processor with AWS Bedrock and S3...")
        
        # Get configuration from environment variables
        opensearch_endpoint = os.getenv("OPENSEARCH_ENDPOINT")
        opensearch_index = os.getenv("OPENSEARCH_INDEX", "pdf-documents")
        aws_region = os.getenv("AWS_REGION", "us-east-1")
        s3_prefix = os.getenv("S3_PREFIX", "")
        
        if not opensearch_endpoint:
            raise ValueError("OPENSEARCH_ENDPOINT environment variable is required")
        
        # Initialize PDF processor with AWS services
        pdf_processor = PDFProcessor(
            s3_prefix=s3_prefix,
            opensearch_endpoint=opensearch_endpoint,
            opensearch_index=opensearch_index,
            aws_region=aws_region
        )
        
        pdf_processor.initialize()
        logger.info("PDF processor initialized successfully with AWS Bedrock and S3")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize PDF processor: {str(e)}")
        raise
    finally:
        logger.info("Shutting down...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="VoxMed PDF Chat API",
    description="Medical document chat assistant using AWS Bedrock and S3",
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
    s3_location: Optional[str] = None
    document_type: Optional[str] = None

class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDocument]
    total_sources: int
    processing_info: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    documents_loaded: bool
    aws_services: Dict[str, str]
    configuration: Dict[str, Any]

class SystemInfoResponse(BaseModel):
    s3_bucket: str
    s3_prefix: str
    opensearch_endpoint: str
    opensearch_index: str
    aws_region: str
    llm_model: str
    embedding_model: str


@app.post("/chat", response_model=ChatResponse)
async def chat_with_pdfs(request: ChatRequest):
    """
    Chat endpoint to ask questions about your PDF documents stored in S3
    
    Args:
        request: ChatRequest containing the question to ask
        
    Returns:
        ChatResponse with the answer and source documents from S3
        
    Raises:
        HTTPException: If the processor is not initialized or query fails
    """
    global pdf_processor
    
    if pdf_processor is None:
        raise HTTPException(
            status_code=500, 
            detail="PDF processor not initialized. Check server logs and AWS configuration."
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
                chunk_index=doc.metadata.get('chunk_index', 0),
                s3_location=doc.metadata.get('source', 'Unknown'),
                document_type=doc.metadata.get('document_type', 'pdf')
            ))
        
        # Create response with additional processing info
        chat_response = ChatResponse(
            question=request.question,
            answer=response.get('answer', 'No answer found'),
            sources=sources,
            total_sources=len(sources),
            processing_info={
                "retrieval_method": "OpenSearch similarity search",
                "llm_provider": "AWS Bedrock",
                "embedding_provider": "AWS Bedrock"
            }
        )
        
        logger.info(f"Chat request processed successfully. Answer length: {len(chat_response.answer)}")
        return chat_response
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing your question: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint with AWS services status
    
    Returns:
        HealthResponse with system health and AWS service status
    """
    global pdf_processor
    
    try:
        is_healthy = pdf_processor is not None
        
        aws_services = {
            "bedrock": "Connected" if is_healthy else "Not Connected",
            "s3": "Connected" if is_healthy else "Not Connected", 
            "opensearch": "Connected" if is_healthy else "Not Connected"
        }
        
        configuration = {}
        if pdf_processor:
            configuration = {
                "s3_bucket": pdf_processor.s3_bucket,
                "s3_prefix": pdf_processor.s3_prefix,
                "opensearch_index": pdf_processor.opensearch_index,
                "aws_region": pdf_processor.aws_region
            }
        
        return HealthResponse(
            status="healthy" if is_healthy else "unhealthy",
            message="PDF processor is ready" if is_healthy else "PDF processor not initialized",
            documents_loaded=is_healthy,
            aws_services=aws_services,
            configuration=configuration
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            message=f"Health check failed: {str(e)}",
            documents_loaded=False,
            aws_services={"error": str(e)},
            configuration={}
        )

@app.get("/system-info", response_model=SystemInfoResponse)
async def get_system_info():
    """
    Get system configuration information
    
    Returns:
        SystemInfoResponse with current system configuration
        
    Raises:
        HTTPException: If processor is not initialized
    """
    global pdf_processor
    
    if pdf_processor is None:
        raise HTTPException(
            status_code=500,
            detail="PDF processor not initialized"
        )
    
    try:
        return SystemInfoResponse(
            s3_bucket=pdf_processor.s3_bucket,
            s3_prefix=pdf_processor.s3_prefix,
            opensearch_endpoint=pdf_processor.opensearch_endpoint,
            opensearch_index=pdf_processor.opensearch_index,
            aws_region=pdf_processor.aws_region,
            llm_model="AWS Bedrock - Claude 3 Sonnet",
            embedding_model="AWS Bedrock - Titan Embeddings"
        )
        
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving system information: {str(e)}"
        )

@app.post("/reload-documents")
async def reload_documents():
    """
    Reload documents from S3 and rebuild the vector store
    
    Returns:
        Dict with reload status and statistics
        
    Raises:
        HTTPException: If reload fails
    """
    global pdf_processor
    
    if pdf_processor is None:
        raise HTTPException(
            status_code=500,
            detail="PDF processor not initialized"
        )
    
    try:
        logger.info("Starting document reload from S3...")
        
        pdf_processor.initialize()
        
        pdf_files = pdf_processor.find_pdf_files_in_s3()
        
        logger.info("Document reload completed successfully")
        
        return {
            "status": "success",
            "message": "Documents reloaded successfully from S3",
            "pdf_files_found": len(pdf_files),
            "s3_bucket": pdf_processor.s3_bucket,
            "s3_prefix": pdf_processor.s3_prefix
        }
        
    except Exception as e:
        logger.error(f"Error reloading documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error reloading documents: {str(e)}"
        )


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Endpoint not found", 
        "message": "Check /docs for available endpoints",
        "available_endpoints": ["/", "/health", "/system-info", "/chat", "/reload-documents", "/docs"]
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return {
        "error": "Internal server error", 
        "message": "Check server logs for details",
        "suggestion": "Verify AWS credentials and service configurations"
    }

