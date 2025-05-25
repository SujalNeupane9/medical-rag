from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
from tempfile import NamedTemporaryFile
from typing import Optional, List
import uvicorn
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Import the RAG pipeline
from main import SimpleRAGPipeline

app = FastAPI(
    title="RAG API",
    description="API for Retrieval-Augmented Generation using PDFs",
    version="1.0.0"
)

origins = [
    "http://localhost:8080",  # Example frontend URL
]

# Add CORSMiddleware to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify allowed origins
    allow_credentials=True,  # Allow credentials (cookies, authorization headers)
    allow_methods=["*"],     # Allow all HTTP methods
    allow_headers=["*"],     # Allow all headers
)

class QueryRequest(BaseModel):
    question: str

class PromptTemplate(BaseModel):
    template: str

class ResponseModel(BaseModel):
    query_response: str
    data: list = []
    type: str = "normal_message"
    data_fetch_status: str

class Selected_PDF(BaseModel):
    user_id: str
    query: str
    pdfs: list = ["originalName"]

def create_response(response: str, processed_files: List[str] = None, status: str = "success") -> dict:
    return {
        "query_response": response,
        "data": processed_files or [],  
        "type": "normal_message",
        "data_fetch_status": status
    }

@app.post("/upload")
async def upload_pdfs(
    files: List[UploadFile] = File(...),
    chunk_size: Optional[int] = 1000,
    user_id: Optional[int] = 1,  # Changed to int to match database schema
    chunk_overlap: Optional[int] = 200
):
    """
    Upload and process multiple PDF files for a specific user.
    """
    # Initialize RAG pipeline with user_id
    rag = SimpleRAGPipeline(user_id=user_id)

    # Ensure user_id is set to default if not provided
    user_id = user_id or 1

    errors = []
    processed_files = []

    for file in files:
        if not file.filename.endswith('.pdf'):
            errors.append(f"File {file.filename} is not a PDF.")
            continue

        try:
            # Create user-specific directory if it doesn't exist
            user_upload_dir = Path('uploaded_files') / str(user_id)
            user_upload_dir.mkdir(parents=True, exist_ok=True)

            # Save the file to the user-specific directory
            file_path = user_upload_dir / file.filename

            with file_path.open('wb') as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Upload PDFs to RAG pipeline 
            # Note: We're removing file_name_aliases and using the actual filename
            rag.upload_pdfs(
                pdf_paths=[str(file_path)],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            processed_files.append(file.filename)

        except Exception as e:
            errors.append(f"Error processing file {file.filename}: {str(e)}")

    # Construct response
    if processed_files:
        message = f"Successfully processed {len(processed_files)} file(s)."
        if errors:
            message += f" Errors occurred with the following files: {', '.join(errors)}."
        return JSONResponse(
            content=create_response(message, processed_files, "partial_success" if errors else "success"),
            status_code=200 if not errors else 207  # 207: Multi-Status
        )
    else:
        return JSONResponse(
            content=create_response("No files were successfully processed. " + ", ".join(errors), None, "failed"),
            status_code=400
        )

@app.post("/query")
async def query(request: QueryRequest, user_id: int = 1):  # Changed to int to match database schema
    """
    Query the RAG system with a question.
    """
    try:
        # Initialize RAG pipeline with user_id
        rag = SimpleRAGPipeline(user_id=user_id)

        # Get the answer 
        retrieved_text = rag.query(request.question)

        # Construct the response with the retrieved text
        return {
            "query_response": retrieved_text,
            "data": [],    # List of retrieved passages
            "type": "normal_message",
            "data_fetch_status": "success"
        }
    except Exception as e:
        return {
            "query_response": str(e),
            "data": [],
            "type": "normal_message",
            "data_fetch_status": "failed"
        }
    
@app.post("/selected_query")
async def selected_query(SP: Selected_PDF):
    try:
        rag = SimpleRAGPipeline(user_id=SP.user_id)

        retrieved_text = rag.query_(user_id = SP.user_id, question = SP.query, pdfs = SP.pdfs)

        # Construct the response with the retrieved text
        return {
            "query_response": retrieved_text,
            "data": [],    # List of retrieved passages
            "type": "normal_message",
            "data_fetch_status": "success"
        }

    
    except Exception as e:
        return create_response(
            str(e), 
            None, 
            status="failed"
        )

@app.get("/display-pdfs")
async def display_pdfs(user_id: int = 1):  # Changed to int to match database schema
    """
    Retrieve the list of uploaded and processed PDF files for a specific user.
    """
    try:
        # Initialize RAG pipeline with user_id
        rag = SimpleRAGPipeline(user_id=user_id)
        
        # Retrieve processed files
        processed_files = rag.display_uploaded_files()
        
        # Use create_response function to format the response
        return create_response(
            "Uploaded PDF files are", 
            [file['fileName'] for file in processed_files]  # Using 'fileName' directly
        )
    except Exception as e:
        # In case of an error, use create_response with an error message
        return create_response(
            str(e), 
            None, 
            status="failed"
        )
    
@app.post("/delete-embeddings")
async def delete_embeddings(
    file_names: List[str], 
    user_id: int = 1  # Changed to int to match database schema
):
    """
    Remove the embeddings of specified PDF files for a specific user.
    
    Args:
        file_names: List of names of PDF files whose embeddings should be deleted.
        user_id: User identifier to ensure deletion from correct collection.
    
    Returns:
        A success message or an error message.
    """
    # Initialize RAG pipeline with user_id
    rag = SimpleRAGPipeline(user_id=user_id)
    
    errors = []
    
    for file_name in file_names:
        try:
            rag.delete_embedding(file_name)
        except Exception as e:
            errors.append(f"Error deleting embedding for {file_name}: {str(e)}")

    if errors:
        return JSONResponse(
            content=create_response("Some embeddings could not be deleted: " + ", ".join(errors), None, "partial_success"),
            status_code=207  # Multi-Status for partial success with errors
        )
    
    return create_response("All specified embeddings deleted successfully.")