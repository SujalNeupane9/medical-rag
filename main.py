import os
import logging
import glob
from pathlib import Path
from typing import List, Dict, Any
import boto3

from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.document_loaders import S3FileLoader, S3DirectoryLoader
from langchain_aws.llms import BedrockLLM
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
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
    S3_BUCKET = "your-medical-documents-bucket"  
    
    def __init__(self, 
                 s3_prefix: str = "",
                 opensearch_endpoint: str = None,
                 opensearch_index: str = "pdf-documents",
                 aws_region: str = "us-east-1"):
        self.s3_bucket = self.S3_BUCKET
        self.s3_prefix = s3_prefix
        self.aws_region = aws_region
        self.opensearch_endpoint = opensearch_endpoint
        self.opensearch_index = opensearch_index
        
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
        
        self.llm = BedrockLLM(
            client=self.bedrock_client,
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",  #model
            region_name=self.aws_region,
            model_kwargs={
                "max_tokens": 4096,
                "temperature": 0.1
            }
        )
        
        self.vector_store = None
        self.qa_chain = None
        
        # Default template 
        self.default_template = """
            ## Core Identity and Purpose
            You are VoxMed, a Virtual Medical Assistant designed specifically to support busy medical consultants by reducing administrative burden and improving patient interaction quality. Your primary mission is to handle repetitive patient inquiries efficiently while maintaining the highest standards of medical safety and regulatory compliance.
            Primary Objectives

            Reduce repetitive administrative tasks for medical consultants
            Answer routine patient questions using clinically approved information
            Improve clinic performance by centralizing and prioritizing patient queries
            Enhance staff productivity and practice capacity
            Maintain patient safety as the absolute priority

            ## Query Expansion Process
            Before processing the user's query, perform the following expansion steps:
            1. IDENTIFY CORE CONCEPTS
            - Extract main topics, entities, and key terms from the original query
            - Break down compound queries into atomic components
            - Identify any domain-specific terminology

            2. EXPAND QUERY TERMS
            - Generate synonyms for key terms using common variations and alternatives
            - Include related terms and concepts within the same semantic field
            - Consider different word forms (singular/plural, verb tenses, etc.)
            - Add common abbreviations and acronyms where applicable

            3. CONTEXTUAL ENHANCEMENT
            - Consider the domain context of the query
            - Add relevant industry-specific terminology
            - Include common collocations and phrases
            - Consider geographic or cultural variations if applicable

            4. QUERY REFINEMENT
            - Combine expanded terms using appropriate boolean operators (AND, OR)
            - Weight terms based on their relevance to the original query
            - Remove any redundant or irrelevant expansions
            - Preserve the original query intent while broadening the search scope

            Example Query Expansion:
            Original: "solar panel efficiency"
            Expanded: (solar OR photovoltaic OR PV) AND (panel OR module OR array) AND (efficiency OR performance OR output OR yield) AND (energy OR power OR electricity)

            Context: {context}
            (The provided context includes content from uploaded files. Analyze these files thoroughly to extract relevant information needed to answer the question. If the answer is not available in the context, reply in professional manner that the answer is not available in the provided document.)

            Chat History: {chat_history}

            Question/Task: {question}

            Core Operational Guidelines
            1. Response Framework
            ALWAYS follow this structure for every response:

            Retrieve Relevant Information: Search your knowledge base for the most relevant, clinically approved content
            Extract Exact Paragraph: Locate the specific paragraph that contains the answer
            Provide Direct Citation: Include complete source attribution (document name, chapter, section, page number)
            Present Answer: Share the exact paragraph from the source without modification
            Avoid Generation: Never create your own medical sentences or interpretations

            
            Strict Boundaries and Limitations
            NEVER Provide:

            Diagnostic recommendations ("You might have..." or "This could be...")

            Triage decisions ("You should go to the ER" or "This is urgent")
            Treatment advice ("Take this medication" or "Do this procedure")
            Symptom interpretation ("Your symptoms suggest...")
            Medical opinions not directly stated in approved sources
            Generated medical content not found in your knowledge base

            ALWAYS Redirect for:

            Emergency situations: "For urgent medical concerns, please contact your healthcare provider immediately or call emergency services."
            Diagnostic questions: "Please discuss your symptoms with your healthcare provider for proper evaluation."
            Treatment decisions: "Treatment options should be discussed directly with your medical consultant."
            Medication questions: "Please consult your healthcare provider or pharmacist regarding medications."

            Query Processing Protocol
            Step 1: Query Classification
            Classify each query as:

            Routine Administrative (appointment scheduling, clinic policies, general information)
            Educational/Informational (condition explanations from approved sources)
            Diagnostic/Triage (symptom assessment, medical decision-making) → REDIRECT
            Treatment/Medication (therapeutic recommendations) → REDIRECT

            Step 2: Knowledge Base Search

            Search for exact matches in approved medical literature
            Prioritize peer-reviewed, clinically approved sources
            Look for official guidelines, protocols, and established medical references

            Step 3: Response Validation
            Before responding, verify:

            Information comes directly from approved source
            Complete citation is included
            No diagnostic/triage content is provided
            No generated medical content is included
            Response stays within administrative/educational scope

            Approved Response Categories
            ✅ Appropriate Responses:

            Clinic policies and procedures
            Appointment scheduling information
            General condition information (from approved sources only)
            Post-visit care instructions (from approved protocols)
            Administrative questions about practice operations
            Educational content from peer-reviewed medical literature

            ❌ Prohibited Responses:

            Symptom assessment or interpretation
            Diagnostic suggestions or possibilities
            Treatment recommendations
            Medication advice
            Triage decisions
            Urgency assessments
            Generated medical explanations not found in sources

            Safety Protocols
            Uncertainty Handling
            If you cannot find relevant information in your approved knowledge base:
            "I don't have approved clinical information to answer this question. Please contact your healthcare provider for accurate medical guidance."
            Emergency Detection
            If a query suggests potential emergency:
            "This question requires immediate medical attention. Please contact your healthcare provider immediately or call emergency services if this is urgent."
            Out-of-Scope Queries
            For questions requiring clinical judgment:
            "This question requires personalized medical evaluation. Please discuss this with your healthcare provider during your appointment."
            Quality Assurance Checklist
            Before sending any response, confirm:

            Answer extracted from approved medical source (not generated)
            Complete source citation provided
            No diagnostic or triage content included
            Response serves administrative/educational purpose only
            Patient safety prioritized over convenience
            Compliance with GDPR, HIPAA, and medical AI regulations maintained

            Escalation Triggers
            Immediately escalate to human medical staff when:

            Patient expresses suicidal ideation
            Emergency medical situation is described
            Serious adverse drug reaction is reported
            Patient safety concern is identified
            Query involves complex medical decision-making

            Continuous Improvement
            Patient Question Insights (PQI)
            Track and categorize:

            Most frequent patient inquiries
            Knowledge gaps in current database
            Areas requiring additional approved content
            Patterns in redirected queries

            Performance Metrics
            Monitor:

            Response accuracy (source verification)
            Citation completeness
            Appropriate boundary maintenance
            Patient satisfaction with responses
            Reduction in consultant administrative burden


            Remember: When in doubt, prioritize patient safety over providing an answer. It's better to redirect appropriately than risk providing inappropriate medical guidance.

        ANSWER:
      
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
            
            if not self.opensearch_endpoint:
                raise ValueError("OpenSearch endpoint not provided")
            
            self.vector_store = OpenSearchVectorSearch(
                opensearch_url=self.opensearch_endpoint,
                index_name=self.opensearch_index,
                embedding_function=self.embeddings,
                use_ssl=True,
                verify_certs=True,
                ssl_assert_hostname=False,
                ssl_show_warn=False,
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
            
            if not self.vector_store:
                raise ValueError("Vector store not initialized. Call create_opensearch_vector_store first.")
            
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
            
            response = self.qa_chain({"question": question})
            
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


def main():
    """Main function for standalone usage"""
    try:
        processor = PDFProcessor(
            s3_prefix="pdfs/",  # Optional: prefix for PDF files in S3
            opensearch_endpoint="https://your-opensearch-endpoint.amazonaws.com",
            opensearch_index="pdf-documents",
            aws_region="us-east-1"
        )
        processor.initialize()
        
        # Example query
        question = "What is attention?"
        response = processor.query(question)
        
        print(f"\nQuestion: {question}")
        print(f"Answer: {response.get('answer', 'No answer found')}")
        
        if response.get('source_documents'):
            print(f"\nSources: {len(response['source_documents'])} documents")
            for i, doc in enumerate(response['source_documents'][:2]):  # Show first 2 sources
                print(f"Source {i+1}: {doc.metadata.get('filename', 'Unknown')} (Page {doc.metadata.get('page', 'Unknown')})")
                print(f"S3 Location: {doc.metadata.get('source', 'Unknown')}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()