import os
import logging
import glob
from pathlib import Path
from typing import List, Dict, Any
import boto3

from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')
nltk.download('punkt')

## json retriever
from question_pair_db import MedicalRAGRetriever

from dotenv import load_dotenv

load_dotenv()
os.environ['aws_access_key_id'] = os.getenv("aws_access_key_id")
os.environ['aws_secret_access_key'] = os.getenv("aws_secret_access_key")
os.environ['AWS_DEFAULT_REGION'] = os.getenv("AWS_DEFAULT_REGION")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SemanticTextSplitter:
    def __init__(self, chunk_size=512, chunk_overlap=256):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize token splitter as backup
        self.token_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name="cl100k_base",
            add_start_index=True
        )
    
    def split_documents(self, documents):
        """
        Split documents while preserving semantic meaning at sentence boundaries
        
        Args:
            documents: List of Document objects with page_content and metadata
            
        Returns:
            List of Document objects with split content
        """
        splits = []
        
        for doc in documents:
            # Get sentences from the document
            sentences = sent_tokenize(doc.page_content)
            
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                # If adding this sentence would exceed chunk size
                if current_length + sentence_length > self.chunk_size:
                    if current_chunk:
                        # Create new document with current chunk
                        splits.append(
                            Document(
                                page_content=" ".join(current_chunk),
                                metadata=doc.metadata.copy()
                            )
                        )
                    
                    # Start new chunk with overlap
                    if self.chunk_overlap > 0:
                        # Calculate how many sentences to keep for overlap
                        overlap_length = 0
                        overlap_sentences = []
                        for prev_sentence in reversed(current_chunk):
                            if overlap_length + len(prev_sentence) > self.chunk_overlap:
                                break
                            overlap_sentences.insert(0, prev_sentence)
                            overlap_length += len(prev_sentence)
                        current_chunk = overlap_sentences
                        current_length = overlap_length
                    else:
                        current_chunk = []
                        current_length = 0
                
                current_chunk.append(sentence)
                current_length += sentence_length
            
            # Add the last chunk if it exists
            if current_chunk:
                splits.append(
                    Document(
                        page_content=" ".join(current_chunk),
                        metadata=doc.metadata.copy()
                    )
                )
        
        # If no valid splits were created, fall back to token splitter
        if not splits:
            return self.token_splitter.split_documents(documents)
            
        return splits

class PDFProcessor:
    def __init__(self, 
                 aws_region: str = os.getenv("AWS_DEFAULT_REGION"),
                 chroma_base_dir: str = "./chroma_db"):
        self.aws_region = aws_region
        self.chroma_base_dir = chroma_base_dir

        self.json_retriever = MedicalRAGRetriever(force_rebuild=False)
        
        # Initialize AWS clients
        self.bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=self.aws_region
        )
        
        # Initialize Bedrock embeddings and LLM
        self.embeddings = BedrockEmbeddings(
            client=self.bedrock_client,
            model_id="amazon.titan-embed-text-v2:0",  
            region_name=self.aws_region
        )
        
        self.llm = ChatBedrock(
            model="anthropic.claude-3-haiku-20240307-v1:0",  # Bedrock model ID
            region_name="us-east-1",                            # Your AWS region
            temperature=0.1,                                    # Optional model parameters
            max_tokens=4096,
            # You can pass other model_kwargs as needed
        )
        
        # Dictionary to store vector stores for each user
        self.user_vector_stores = {}
        self.user_qa_chains = {}
        
        # Default template 
        self.default_template = """
        ## Core Identity
        You are VoxMed, a helpful Virtual Medical Assistant supporting both medical consultants and patients. Your role is to provide accurate, accessible medical information while maintaining the highest safety standards.

        ## Response Style
        - Write in a warm, conversational tone as if speaking with a friend or family member
        - Use clear, everyday language while remaining professional
        - Be empathetic and understanding of patient concerns
        - Keep responses concise but thorough enough to be helpful

        ## What You CAN Do:
        1. Provide general educational information from approved medical sources
        2. Share clinic policies and administrative information  
        3. Quote directly from medical literature with proper citations
        4. Answer routine questions using the provided context
        5. Offer general wellness and preventive care guidance

        ## What You CANNOT Do:
        1. Provide diagnostic advice or interpret specific symptoms
        2. Recommend treatments or medications for individual cases
        3. Make triage decisions or assess medical urgency
        4. Generate medical content not found in your sources

        ## When Information Isn't Available:
        Instead of saying "the answer is not in the provided context," use phrases like:
        - "I don't have specific information about that in my current resources"
        - "That's a great question, but I'd need to refer you to your healthcare provider for detailed guidance on that topic"
        - "While I don't have comprehensive information about that particular aspect, I can tell you..."
        - "I wish I could give you more specifics on that, but the best person to answer would be your doctor"

        ## Safety Redirects:
        For diagnostic questions, treatment advice, or emergency situations, gently redirect with phrases like:
        - "That sounds like something worth discussing with your healthcare provider"
        - "I'd encourage you to bring that up at your next appointment"
        - "For personalized medical advice like that, your doctor would be the best resource"

        ## Context and Query Processing
        Context: {context}
        Chat History: {chat_history}
        Question: {question}

        ## Instructions:
        Answer the question using information from the provided context. Be conversational and helpful while maintaining medical safety standards. If the specific information isn't available, acknowledge the limitation naturally and suggest appropriate next steps.

        Answer:
        """
        
        # Initialize text splitter
        self.text_splitter = SemanticTextSplitter(
            chunk_size=512,
            chunk_overlap=128,
        )
        
        # Load existing user collections on initialization
        self.load_existing_collections()
        
        logger.info("PDFProcessor initialized for local file processing")
    
    def load_existing_collections(self):
        """Load existing user collections from persistent storage on startup"""
        try:
            logger.info("Loading existing user collections...")
            
            # Create base directory if it doesn't exist
            os.makedirs(self.chroma_base_dir, exist_ok=True)
            
            # Look for existing user directories
            user_dirs = [d for d in os.listdir(self.chroma_base_dir) 
                        if os.path.isdir(os.path.join(self.chroma_base_dir, d))]
            
            for user_dir in user_dirs:
                user_id = user_dir
                persist_directory = os.path.join(self.chroma_base_dir, user_dir)
                
                try:
                    # Check if this directory contains a valid Chroma collection
                    if self.is_valid_chroma_collection(persist_directory):
                        logger.info(f"Loading existing collection for user: {user_id}")
                        
                        # Load the existing vector store
                        collection_name = f"user_{user_id}_collection"
                        self.user_vector_stores[user_id] = Chroma(
                            collection_name=collection_name,
                            embedding_function=self.embeddings,
                            persist_directory=persist_directory,
                        )
                        
                        # Setup QA chain for this user
                        self.setup_user_qa_chain(user_id)
                        
                        document_count = self.get_user_document_count(user_id)
                        logger.info(f"Loaded collection for user {user_id} with {document_count} documents")
                        
                except Exception as e:
                    logger.warning(f"Failed to load collection for user {user_id}: {str(e)}")
                    continue
            
            logger.info(f"Loaded {len(self.user_vector_stores)} existing user collections")
            
        except Exception as e:
            logger.error(f"Error loading existing collections: {str(e)}")
    
    def is_valid_chroma_collection(self, persist_directory: str) -> bool:
        """Check if a directory contains a valid Chroma collection"""
        try:
            # Check for essential Chroma files
            required_files = ['chroma.sqlite3']
            for file in required_files:
                if not os.path.exists(os.path.join(persist_directory, file)):
                    return False
            
            # Try to load the collection to verify it's valid
            collection_name = f"user_{os.path.basename(persist_directory)}_collection"
            test_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_directory,
            )
            
            # Check if collection has documents
            count = test_store._collection.count()
            return count > 0
            
        except Exception as e:
            logger.debug(f"Invalid Chroma collection at {persist_directory}: {str(e)}")
            return False
    
    def user_has_documents(self, user_id: str) -> bool:
        """Check if a user already has documents in their collection"""
        try:
            if user_id in self.user_vector_stores:
                document_count = self.get_user_document_count(user_id)
                return document_count > 0
            return False
        except Exception as e:
            logger.error(f"Error checking if user {user_id} has documents: {str(e)}")
            return False
    
    def get_user_document_files(self, user_id: str) -> List[str]:
        """Get list of files that have been processed for a user"""
        try:
            if user_id not in self.user_vector_stores:
                return []
            
            # Get a sample of documents to extract filenames
            retriever = self.user_vector_stores[user_id].as_retriever(search_kwargs={"k": 100})
            sample_docs = self.user_vector_stores[user_id].similarity_search("", k=100)
            
            # Extract unique filenames from metadata
            filenames = set()
            for doc in sample_docs:
                if 'filename' in doc.metadata:
                    filenames.add(doc.metadata['filename'])
            
            return list(filenames)
            
        except Exception as e:
            logger.error(f"Error getting document files for user {user_id}: {str(e)}")
            return []

    def process_single_pdf_from_path(self, file_path: str) -> List[Document]:
        """Process a single PDF file from local path and return document objects"""
        try:
            logger.info(f"Processing PDF from path: {file_path}")
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return []
            
            # Load PDF from local path
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from {file_path}")
            
            # Split documents
            splits = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(splits)} text chunks from {file_path}")
            
            # Create document objects with metadata
            document_objects = [
                Document(
                    page_content=split.page_content,
                    metadata={
                        "source": file_path,
                        "filename": Path(file_path).name,
                        "page": split.metadata.get("page", 0),
                        "chunk_size": len(split.page_content),
                        "chunk_index": i,
                        "document_type": "pdf",
                        "file_path": file_path
                    }
                )
                for i, split in enumerate(splits)
            ]
            
            logger.info(f"Successfully processed {file_path}: {len(document_objects)} document objects created")
            return document_objects
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return []

    def process_pdfs_for_user(self, file_paths: List[str], user_id: str, append_to_existing: bool = True) -> None:
        """Process multiple PDF files for a specific user and create user-specific collection"""
        try:
            logger.info(f"Processing {len(file_paths)} PDF files for user: {user_id}")
            
            all_documents = []
            
            # Process each PDF file
            for file_path in file_paths:
                documents = self.process_single_pdf_from_path(file_path)
                if documents:
                    # Add user_id to metadata for each document
                    for doc in documents:
                        doc.metadata["user_id"] = user_id
                    all_documents.extend(documents)
                else:
                    logger.warning(f"No documents extracted from {file_path}")
            
            if not all_documents:
                logger.warning(f"No documents processed for user {user_id}")
                return
            
            logger.info(f"Total documents processed for user {user_id}: {len(all_documents)}")
            
            # Create or update user-specific vector store
            if user_id in self.user_vector_stores and append_to_existing:
                # Add to existing collection
                logger.info(f"Adding documents to existing collection for user {user_id}")
                batch_size = 100
                for i in range(0, len(all_documents), batch_size):
                    batch = all_documents[i:i + batch_size]
                    self.user_vector_stores[user_id].add_documents(documents=batch)
                    logger.info(f"Added batch {i//batch_size + 1}/{(len(all_documents) + batch_size - 1)//batch_size} for user {user_id}")
            else:
                # Create new collection
                self.create_user_vector_store(all_documents, user_id)
            
            # Setup QA chain for this user (will update existing one)
            self.setup_user_qa_chain(user_id)
            
            logger.info(f"Successfully processed and stored documents for user: {user_id}")
            
        except Exception as e:
            logger.error(f"Error processing PDFs for user {user_id}: {str(e)}")
            raise

    def create_user_vector_store(self, documents: List[Document], user_id: str) -> None:
        """Create and populate user-specific vector store"""
        try:
            logger.info(f"Creating vector store for user: {user_id}")
            
            # Create Chroma vector store with user-specific collection name
            collection_name = f"user_{user_id}_collection"
            persist_directory = os.path.join(self.chroma_base_dir, user_id)
            
            # Create directory if it doesn't exist
            os.makedirs(persist_directory, exist_ok=True)
            
            self.user_vector_stores[user_id] = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_directory,
            )
            
            if documents:
                logger.info(f"Adding {len(documents)} documents to vector store for user {user_id}...")
                # Split documents into batches to avoid overwhelming the vector store
                batch_size = 100
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    self.user_vector_stores[user_id].add_documents(documents=batch)
                    logger.info(f"Added batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size} for user {user_id}")
                
                logger.info(f"Documents added to vector store successfully for user {user_id}")
            else:
                logger.warning(f"No documents to add to vector store for user {user_id}")
                
        except Exception as e:
            logger.error(f"Error creating vector store for user {user_id}: {str(e)}")
            raise

    def setup_user_qa_chain(self, user_id: str) -> None:
        """Setup the QA chain with memory for a specific user"""
        try:
            logger.info(f"Setting up QA chain for user: {user_id}")
            
            if user_id not in self.user_vector_stores:
                raise ValueError(f"Vector store not found for user {user_id}. Process documents first.")
            
            # Create memory for this user
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            # Create custom prompt template
            prompt_template = PromptTemplate(
                input_variables=["context", "chat_history", "question"],
                template=self.default_template
            )
            
            # Create QA chain with user-specific vector store
            self.user_qa_chains[user_id] = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.user_vector_stores[user_id].as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                ),
                memory=memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": prompt_template},
                verbose=True
            )
            
            logger.info(f"QA chain setup complete for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error setting up QA chain for user {user_id}: {str(e)}")
            raise

    def query_user(self, question: str, user_id: str) -> Dict[str, Any]:
        """Query the QA chain for a specific user"""
        try:
            logger.info(f"Processing query for user {user_id}: {question}")
            
            if user_id not in self.user_qa_chains:
                raise ValueError(f"QA chain not found for user {user_id}. Process documents first.")
            
            response = self.json_retriever.retrieve_answer(question)
            
            if response is None:
                response = self.user_qa_chains[user_id].invoke({"question": question})
                logger.info(f"Query processed successfully for user {user_id}. Answer length: {len(response.get('answer', ''))}")
                response['clinically_approved'] = False
                return response
            else:
                logger.info(f"Query processed successfully for user {user_id}. Answer obtained from json file.")
                response['clinically_approved'] = True
                return response
                
        except Exception as e:
            logger.error(f"Error processing query for user {user_id}: {str(e)}")
            raise

    def get_user_document_count(self, user_id: str) -> int:
        """Get the number of documents stored for a user"""
        try:
            if user_id in self.user_vector_stores:
                # Get collection info
                collection = self.user_vector_stores[user_id]._collection
                return collection.count()
            else:
                return 0
        except Exception as e:
            logger.error(f"Error getting document count for user {user_id}: {str(e)}")
            return 0

    def list_users(self) -> List[str]:
        """List all users with processed documents"""
        return list(self.user_vector_stores.keys())

    def delete_user_data(self, user_id: str) -> bool:
        """Delete all data for a specific user"""
        try:
            logger.info(f"Deleting data for user: {user_id}")
            
            # Remove from memory
            if user_id in self.user_vector_stores:
                del self.user_vector_stores[user_id]
            
            if user_id in self.user_qa_chains:
                del self.user_qa_chains[user_id]
            
            # Remove persistent directory
            persist_directory = os.path.join(self.chroma_base_dir, user_id)
            if os.path.exists(persist_directory):
                import shutil
                shutil.rmtree(persist_directory)
            
            logger.info(f"Successfully deleted data for user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting data for user {user_id}: {str(e)}")
            return False