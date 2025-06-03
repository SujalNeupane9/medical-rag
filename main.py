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
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import S3FileLoader, S3DirectoryLoader
from langchain_aws import ChatBedrock
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.vectorstores import Chroma
# from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')
nltk.download('punkt')

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
    # Class attribute for S3 bucket name
    S3_BUCKET = "demo-ai-agent"  
    
    def __init__(self, 
                 s3_prefix: str = "uploads/",
                 # opensearch_endpoint: str = None,
                 # opensearch_index: str = "pdf-documents",
                 aws_region: str = os.getenv("AWS_DEFAULT_REGION")):
        self.s3_bucket = self.S3_BUCKET
        self.s3_prefix = "uploads/"
        self.aws_region = aws_region
        # self.opensearch_endpoint = opensearch_endpoint
        # self.opensearch_index = opensearch_index
        
        # Initialize AWS clients
        self.bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=self.aws_region
        )
        
        # Initialize Bedrock embeddings and LLM
        self.embeddings = BedrockEmbeddings(
            client=self.bedrock_client,
            model_id="amazon.titan-embed-text-v1",  
            region_name=self.aws_region
        )
        
        self.llm = ChatBedrock(
            model="anthropic.claude-3-sonnet-20240229-v1:0",  # Bedrock model ID
            region_name="us-east-1",                            # Your AWS region
            temperature=0.3,                                    # Optional model parameters
            max_tokens=4096,
            # You can pass other model_kwargs as needed
        )
        
        self.vector_store = None
        self.qa_chain = None
        
        # Default template 
        self.default_template = """
            ## Core Identity
            You are VoxMed, a Virtual Medical Assistant designed to support medical consultants by handling routine patient inquiries while maintaining strict safety standards.
            Response Guidelines
            W# hat You CAN Do:

            1. Provide general educational information from approved medical sources
            2. Share clinic policies and administrative information
            3. Quote directly from medical literature with proper citations
            4. Answer routine questions using the provided context

            What You CANNOT Do:

            1. Provide diagnostic advice or symptom interpretation
            2. Recommend treatments or medications
            3. Make triage decisions or assess urgency
            4. Generate medical content not found in your sources

            Response Format:
            Provide clear, direct answers using information from the context provided. Always include source citations when referencing medical information. If the answer isn't available in the provided context, politely state that the information isn't available in the current documents.
            Safety Redirects:
            For diagnostic questions, treatment advice, or emergency situations, redirect patients to contact their healthcare provider immediately.
            
            
            ## Context and Query Processing
            Context: {context}

            Chat History: {chat_history}

            Question: {question}

            ## Instructions: 
            Answer the question using only information from the provided context. Be direct and helpful while maintaining medical safety standards. If information isn't available in the context, clearly state this limitation.

            Answer:
        """
        
        # Initialize text splitter
        self.text_splitter = SemanticTextSplitter(
            chunk_size=256,
            chunk_overlap=128,
        )
        
        logger.info(f"PDFProcessor initialized with S3 bucket: {self.s3_bucket}, prefix: {s3_prefix}")

    def find_pdf_files_in_s3(self) -> List[str]:
        """Find all PDF files in the S3 bucket"""
        try:
            s3_client = boto3.client('s3', region_name=self.aws_region)
            response = s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=self.s3_prefix
            )
            
            pdf_files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'].lower().endswith('.pdf'):
                        pdf_files.append(obj['Key'])
            
            logger.info(f"Found {len(pdf_files)} PDF files in S3: {pdf_files}")
            return pdf_files
            
        except Exception as e:
            logger.error(f"Error finding PDF files in S3: {str(e)}")
            return []

    def process_single_pdf_from_s3(self, s3_key: str) -> List[Document]:
        """Process a single PDF file from S3 and return document objects"""
        try:
            logger.info(f"Processing PDF from S3: {s3_key}")
            
            # Load PDF from S3
            loader = S3FileLoader(self.s3_bucket, s3_key)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from {s3_key}")
            
            # Split documents
            splits = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(splits)} text chunks from {s3_key}")
            
            # Create document objects with metadata
            document_objects = [
                Document(
                    page_content=split.page_content,
                    metadata={
                        "source": f"s3://{self.s3_bucket}/{s3_key}",
                        "filename": Path(s3_key).name,
                        "page": split.metadata.get("page", 0),
                        "chunk_size": len(split.page_content),
                        "chunk_index": i,
                        "document_type": "pdf",
                        "s3_bucket": self.s3_bucket,
                        "s3_key": s3_key
                    }
                )
                for i, split in enumerate(splits)
            ]
            
            logger.info(f"Successfully processed {s3_key}: {len(document_objects)} document objects created")
            return document_objects
            
        except Exception as e:
            logger.error(f"Error processing PDF {s3_key}: {str(e)}")
            return []

    def process_all_pdfs_from_s3(self) -> List[Document]:
        """Process all PDFs in the S3 bucket"""
        pdf_files = self.find_pdf_files_in_s3()
        
        if not pdf_files:
            logger.warning(f"No PDF files found in S3 bucket {self.s3_bucket} with prefix {self.s3_prefix}")
            return []
        
        all_documents = []
        for pdf_file in pdf_files:
            documents = self.process_single_pdf_from_s3(pdf_file)
            all_documents.extend(documents)
        
        logger.info(f"Total documents processed: {len(all_documents)}")
        return all_documents

    def create_opensearch_vector_store(self, documents: List[Document]) -> None:
        """Create and populate OpenSearch vector store"""
        try:
            logger.info("Creating OpenSearch vector store...")
            
            # if not self.opensearch_endpoint:
            #     raise ValueError("OpenSearch endpoint not provided")

            # Create FAISS vector store with empty docstore and index
            self.vector_store =  Chroma(
                collection_name="test_collection",
                embedding_function=self.embeddings,
                persist_directory="./chroma_db",
            )
            
            if documents:
                logger.info(f"Adding {len(documents)} documents to OpenSearch vector store...")
                # Split documents into batches to avoid overwhelming OpenSearch
                batch_size = 100
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    self.vector_store.add_documents(documents=batch)
                    logger.info(f"Added batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
                
                logger.info("Documents added to OpenSearch vector store successfully")
            else:
                logger.warning("No documents to add to vector store")
                
        except Exception as e:
            logger.error(f"Error creating OpenSearch vector store: {str(e)}")
            raise

    def setup_qa_chain(self) -> None:
        """Setup the QA chain with memory using Bedrock LLM"""
        try:
            logger.info("Setting up QA chain with Bedrock LLM...")
            
            # if not self.vector_store:
            #     raise ValueError("Vector store not initialized. Call create_opensearch_vector_store first.")
            
            # Create memory
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
            
            # Create QA chain with Bedrock LLM
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                ),
                memory=memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": prompt_template},
                verbose=True
            )
            
            logger.info("QA chain setup complete with Bedrock LLM")
            
        except Exception as e:
            logger.error(f"Error setting up QA chain: {str(e)}")
            raise

    def query(self, question: str) -> Dict[str, Any]:
        """Query the QA chain"""
        try:
            logger.info(f"Processing query: {question}")
            
            if not self.qa_chain:
                raise ValueError("QA chain not initialized. Call setup_qa_chain first.")
            
            response = self.qa_chain.invoke({"question":question})
            
            logger.info(f"Query processed successfully. Answer length: {len(response.get('answer', ''))}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    def initialize(self) -> None:
        """Initialize the complete pipeline"""
        try:
            logger.info("Starting PDF processing pipeline with AWS Bedrock and S3...")
            
            # Process all PDFs from S3
            documents = self.process_all_pdfs_from_s3()
            
            # Create OpenSearch vector store
            self.create_opensearch_vector_store(documents)
            
            # Setup QA chain with Bedrock
            self.setup_qa_chain()
            
            logger.info("PDF processing pipeline initialized successfully with AWS services")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline: {str(e)}")
            raise

