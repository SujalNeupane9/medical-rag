from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
import uvicorn
import os
import tempfile
import shutil
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
        
        # Initialize PDF processor for local file processing
        pdf_processor = PDFProcessor()
        
        logger.info("PDF processor initialized successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize PDF processor: {str(e)}")
        raise
    finally:
        logger.info("Shutting down...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="VoxMed PDF Chat API",
    description="Medical document chat assistant with user-specific collections",
    version="2.0.0",
    lifespan=lifespan
)

# Uniform Response Format
class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Request models
class ChatRequest(BaseModel):
    question: str
    user_id: str

class AddDocumentRequest(BaseModel):
    user_id: str

# Data models for response
class SourceDocument(BaseModel):
    filename: str
    page: int
    content_preview: str
    chunk_index: int
    file_path: Optional[str] = None
    document_type: Optional[str] = None

class ChatData(BaseModel):
    question: str
    answer: str
    sources: List[SourceDocument]
    total_sources: int
    user_id: str

class AddDocumentData(BaseModel):
    user_id: str
    files_processed: List[str]
    total_documents: int
    processing_status: str

@app.post("/chat", response_model=APIResponse)
async def chat_with_documents(request: ChatRequest):
    """
    Chat endpoint to ask questions about user's uploaded PDF documents
    
    Args:
        request: ChatRequest containing the question and user_id
        
    Returns:
        APIResponse with uniform format containing chat data
    """
    global pdf_processor
    
    if pdf_processor is None:
        return APIResponse(
            success=False,
            message="PDF processor not initialized",
            error="Server initialization error"
        )
    
    if not request.question.strip():
        return APIResponse(
            success=False,
            message="Question cannot be empty",
            error="Invalid input"
        )
    
    if not request.user_id.strip():
        return APIResponse(
            success=False,
            message="User ID is required",
            error="Invalid input"
        )
    
    try:
        logger.info(f"Processing chat request for user {request.user_id}: {request.question[:100]}...")
        
        # Check if user has any documents
        if request.user_id not in pdf_processor.list_users():
            return APIResponse(
                success=False,
                message=f"No documents found for user {request.user_id}. Please upload documents first.",
                error="No documents found"
            )
        
        # Query the PDF processor for the specific user
        response = pdf_processor.query_user(request.question, request.user_id)
        
        # Extract source documents
        sources = []
        source_docs = response.get('source_documents', [])
        
        for doc in source_docs:
            sources.append(SourceDocument(
                filename=doc.metadata.get('filename', 'Unknown'),
                page=doc.metadata.get('page', 0),
                content_preview=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                chunk_index=doc.metadata.get('chunk_index', 0),
                file_path=doc.metadata.get('source', 'Unknown'),
                document_type=doc.metadata.get('document_type', 'pdf')
            ))
        
        # Create chat data
        chat_data = ChatData(
            question=request.question,
            answer=response.get('answer', 'No answer found'),
            sources=sources,
            total_sources=len(sources),
            user_id=request.user_id
        )
        
        logger.info(f"Chat request processed successfully for user {request.user_id}. Answer length: {len(chat_data.answer)}")
        
        return APIResponse(
            success=True,
            message="Question answered successfully",
            data=chat_data.dict()
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request for user {request.user_id}: {str(e)}")
        return APIResponse(
            success=False,
            message="Error processing your question",
            error=str(e)
        )

@app.post("/add-documents", response_model=APIResponse)
async def add_documents(
    user_id: str,
    files: List[UploadFile] = File(...)
):
    """
    Add documents endpoint to upload and process PDF files for a specific user
    
    Args:
        user_id: User identifier
        files: List of PDF files to upload and process
        
    Returns:
        APIResponse with uniform format containing processing results
    """
    global pdf_processor
    
    if pdf_processor is None:
        return APIResponse(
            success=False,
            message="PDF processor not initialized",
            error="Server initialization error"
        )
    
    if not user_id.strip():
        return APIResponse(
            success=False,
            message="User ID is required",
            error="Invalid input"
        )
    
    if not files:
        return APIResponse(
            success=False,
            message="No files provided",
            error="No files uploaded"
        )
    
    # Validate file types
    pdf_files = []
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            return APIResponse(
                success=False,
                message=f"File '{file.filename}' is not a PDF. Only PDF files are allowed.",
                error="Invalid file type"
            )
        pdf_files.append(file)
    
    temp_file_paths = []
    processed_files = []
    
    try:
        logger.info(f"Processing {len(pdf_files)} PDF files for user {user_id}")
        
        # Create temporary directory for this user's upload
        temp_dir = tempfile.mkdtemp(prefix=f"user_{user_id}_")
        
        # Save uploaded files to temporary location
        for file in pdf_files:
            # Create temporary file
            temp_file_path = os.path.join(temp_dir, file.filename)
            
            # Save file content
            with open(temp_file_path, "wb") as temp_file:
                content = await file.read()
                temp_file.write(content)
            
            temp_file_paths.append(temp_file_path)
            processed_files.append(file.filename)
            logger.info(f"Saved {file.filename} to temporary location")
        
        # Process PDFs for the user
        pdf_processor.process_pdfs_for_user(temp_file_paths, user_id)
        
        # Get total document count for user
        total_documents = pdf_processor.get_user_document_count(user_id)
        
        # Create response data
        add_document_data = AddDocumentData(
            user_id=user_id,
            files_processed=processed_files,
            total_documents=total_documents,
            processing_status="completed"
        )
        
        logger.info(f"Successfully processed {len(processed_files)} files for user {user_id}")
        
        return APIResponse(
            success=True,
            message=f"Successfully processed {len(processed_files)} PDF files for user {user_id}",
            data=add_document_data.dict()
        )
        
    except Exception as e:
        logger.error(f"Error processing files for user {user_id}: {str(e)}")
        return APIResponse(
            success=False,
            message="Error processing uploaded files",
            error=str(e)
        )
    
    finally:
        # Clean up temporary files
        for temp_path in temp_file_paths:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Could not remove temporary file {temp_path}: {str(e)}")
        
        # Clean up temporary directory
        try:
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Could not remove temporary directory {temp_dir}: {str(e)}")

# Health check endpoint (optional)
@app.get("/health", response_model=APIResponse)
async def health_check():
    """Health check endpoint"""
    global pdf_processor
    
    if pdf_processor is None:
        return APIResponse(
            success=False,
            message="PDF processor not initialized",
            error="Service unavailable"
        )
    
    # Get system info
    users = pdf_processor.list_users()
    total_users = len(users)
    
    return APIResponse(
        success=True,
        message="Service is healthy",
        data={
            "status": "healthy",
            "total_users": total_users,
            "users_with_documents": users,
            "service": "VoxMed PDF Chat API"
        }
    )

# Root endpoint
@app.get("/", response_model=APIResponse)
async def root():
    """Root endpoint with API information"""
    return APIResponse(
        success=True,
        message="Welcome to VoxMed PDF Chat API",
        data={
            "version": "2.0.0",
            "description": "Medical document chat assistant with user-specific collections",
            "endpoints": {
                "/chat": "POST - Chat with user's documents",
                "/add-documents": "POST - Upload and process PDF documents",
                "/health": "GET - Health check",
                "/docs": "GET - API documentation"
            }
        }
    )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return APIResponse(
        success=False,
        message="Endpoint not found",
        error="The requested endpoint does not exist. Check /docs for available endpoints."
    ).model_dump()

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return APIResponse(
        success=False,
        message="Internal server error",
        error="An unexpected error occurred. Please try again later."
    ).model_dump()

