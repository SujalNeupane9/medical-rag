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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_processor_api.log'),
        logging.StreamHandler()
    ]
)
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
        logger.info(f"Loaded existing collections for users: {pdf_processor.list_users()}")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize PDF processor: {str(e)}")
        raise
    finally:
        logger.info("Shutting down PDF processor...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="VoxMed PDF Chat API",
    description="Medical document chat assistant with user-specific collections using AWS Bedrock and Chroma vector stores",
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
    chunk_size: Optional[int] = None

class ChatData(BaseModel):
    question: str
    answer: str
    sources: List[SourceDocument]
    total_sources: int
    user_id: str
    response_metadata: Optional[Dict[str, Any]] = None

class AddDocumentData(BaseModel):
    user_id: str
    files_processed: List[str]
    total_documents: int
    processing_status: str
    existing_files: Optional[List[str]] = None
    new_documents_added: Optional[int] = None

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
        logger.error("PDF processor not initialized")
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
            logger.warning(f"No documents found for user {request.user_id}")
            return APIResponse(
                success=False,
                message=f"No documents found for user {request.user_id}. Please upload documents first.",
                error="No documents found"
            )
        
        # Check if user actually has documents (not just an empty collection)
        if not pdf_processor.user_has_documents(request.user_id):
            logger.warning(f"User {request.user_id} has empty document collection")
            return APIResponse(
                success=False,
                message=f"No documents found for user {request.user_id}. Please upload documents first.",
                error="Empty document collection"
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
                document_type=doc.metadata.get('document_type', 'pdf'),
                chunk_size=doc.metadata.get('chunk_size', len(doc.page_content))
            ))
        
        # Create chat data
        chat_data = ChatData(
            question=request.question,
            answer=response.get('answer', 'No answer found'),
            sources=sources,
            total_sources=len(sources),
            user_id=request.user_id,
            response_metadata={
                "processing_time": "N/A",  # Could add timing if needed
                "model_used": "anthropic.claude-3-haiku-20240307-v1:0",
                "embedding_model": "amazon.titan-embed-text-v2:0",
                "retrieval_method": "similarity_search",
                "max_sources": 3
            }
        )
        
        logger.info(f"Chat request processed successfully for user {request.user_id}. Answer length: {len(chat_data.answer)}, Sources: {len(sources)}")
        
        return APIResponse(
            success=True,
            message="Question answered successfully",
            data=chat_data.dict()
        )
        
    except ValueError as ve:
        logger.error(f"Validation error for user {request.user_id}: {str(ve)}")
        return APIResponse(
            success=False,
            message="Invalid request or user data",
            error=str(ve)
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
    files: List[UploadFile] = File(...),
    append_to_existing: bool = True
):
    """
    Add documents endpoint to upload and process PDF files for a specific user
    
    Args:
        user_id: User identifier
        files: List of PDF files to upload and process
        append_to_existing: Whether to append to existing collection or replace it
        
    Returns:
        APIResponse with uniform format containing processing results
    """
    global pdf_processor
    
    if pdf_processor is None:
        logger.error("PDF processor not initialized")
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
    
    # Validate file types and sizes
    pdf_files = []
    max_file_size = 50 * 1024 * 1024  # 50MB limit per file
    
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            return APIResponse(
                success=False,
                message=f"File '{file.filename}' is not a PDF. Only PDF files are allowed.",
                error="Invalid file type"
            )
        
        # Check file size (if available)
        if hasattr(file, 'size') and file.size and file.size > max_file_size:
            return APIResponse(
                success=False,
                message=f"File '{file.filename}' is too large. Maximum file size is 50MB.",
                error="File too large"
            )
        
        pdf_files.append(file)
    
    temp_file_paths = []
    processed_files = []
    temp_dir = None
    
    try:
        logger.info(f"Processing {len(pdf_files)} PDF files for user {user_id} (append: {append_to_existing})")
        
        # Get existing files for this user
        existing_files = pdf_processor.get_user_document_files(user_id) if append_to_existing else []
        
        # Create temporary directory for this user's upload
        temp_dir = tempfile.mkdtemp(prefix=f"voxmed_user_{user_id}_")
        logger.info(f"Created temporary directory: {temp_dir}")
        
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
            logger.info(f"Saved {file.filename} ({len(content)} bytes) to temporary location")
        
        # Get initial document count
        initial_count = pdf_processor.get_user_document_count(user_id)
        
        # Process PDFs for the user
        pdf_processor.process_pdfs_for_user(temp_file_paths, user_id, append_to_existing)
        
        # Get final document count
        final_count = pdf_processor.get_user_document_count(user_id)
        new_documents_added = final_count - initial_count
        
        # Create response data
        add_document_data = AddDocumentData(
            user_id=user_id,
            files_processed=processed_files,
            total_documents=final_count,
            processing_status="completed",
            existing_files=existing_files,
            new_documents_added=new_documents_added
        )
        
        logger.info(f"Successfully processed {len(processed_files)} files for user {user_id}. Total documents: {final_count}")
        
        return APIResponse(
            success=True,
            message=f"Successfully processed {len(processed_files)} PDF files for user {user_id}. Added {new_documents_added} new document chunks.",
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
        # Clean up temporary files and directory
        cleanup_errors = []
        
        for temp_path in temp_file_paths:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    logger.debug(f"Removed temporary file: {temp_path}")
            except Exception as e:
                cleanup_errors.append(f"Could not remove temporary file {temp_path}: {str(e)}")
        
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.debug(f"Removed temporary directory: {temp_dir}")
            except Exception as e:
                cleanup_errors.append(f"Could not remove temporary directory {temp_dir}: {str(e)}")
        
        if cleanup_errors:
            for error in cleanup_errors:
                logger.warning(error)

# Health check endpoint
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
    
    try:
        # Get system info
        users = pdf_processor.list_users()
        total_users = len(users)
        
        # Get document counts per user
        user_doc_counts = {}
        total_documents = 0
        
        for user_id in users:
            doc_count = pdf_processor.get_user_document_count(user_id)
            user_doc_counts[user_id] = doc_count
            total_documents += doc_count
        
        return APIResponse(
            success=True,
            message="Service is healthy",
            data={
                "status": "healthy",
                "service": "VoxMed PDF Chat API",
                "version": "2.0.0",
                "total_users": total_users,
                "total_documents": total_documents,
                "users_with_documents": users,
                "user_document_counts": user_doc_counts,
                "backend_services": {
                    "aws_bedrock": "Connected",
                    "chroma_vector_store": "Active",
                    "embedding_model": "amazon.titan-embed-text-v2:0",
                    "llm_model": "anthropic.claude-3-haiku-20240307-v1:0"
                }
            }
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return APIResponse(
            success=False,
            message="Health check failed",
            error=str(e)
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
            "description": "Medical document chat assistant with user-specific collections using AWS Bedrock and Chroma vector stores",
            "features": [
                "User-specific document collections",
                "AWS Bedrock integration",
                "Semantic text splitting",
                "Conversational memory",
                "Medical-focused responses"
            ],
            "endpoints": {
                "/chat": "POST - Chat with user's documents",
                "/add-documents": "POST - Upload and process PDF documents",
                "/health": "GET - Health check with system status",
                "/docs": "GET - Interactive API documentation",
                "/redoc": "GET - Alternative API documentation"
            },
            "supported_formats": ["PDF"],
            "max_file_size": "50MB per file",
            "backend": {
                "llm": "Claude 3 Haiku (AWS Bedrock)",
                "embeddings": "Amazon Titan Embed Text v2",
                "vector_store": "Chroma",
                "text_splitter": "Semantic (sentence-boundary)"
            }
        }
    )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return APIResponse(
        success=False,
        message="Endpoint not found",
        error="The requested endpoint does not exist. Check /docs for available endpoints."
    ).model_dump()

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(exc)}")
    return APIResponse(
        success=False,
        message="Internal server error",
        error="An unexpected error occurred. Please try again later."
    ).model_dump()

@app.exception_handler(413)
async def request_entity_too_large_handler(request, exc):
    """Handle file too large errors"""
    return APIResponse(
        success=False,
        message="File too large",
        error="The uploaded file exceeds the maximum allowed size of 50MB."
    ).model_dump()

