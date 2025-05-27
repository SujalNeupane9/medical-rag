import os
import logging
import glob
from pathlib import Path
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.embeddings import OpenAIEmbeddings
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
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

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
    def __init__(self, data_folder: str = "data", persist_directory: str = "./chroma_db"):
        self.data_folder = data_folder
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.qa_chain = None
        
        # Default template (you can modify this)
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
        
        logger.info(f"PDFProcessor initialized with data folder: {data_folder}")

    def find_pdf_files(self) -> List[str]:
        """Find all PDF files in the data folder"""
        pdf_pattern = os.path.join(self.data_folder, "*.pdf")
        pdf_files = glob.glob(pdf_pattern)
        logger.info(f"Found {len(pdf_files)} PDF files: {pdf_files}")
        return pdf_files

    def process_single_pdf(self, pdf_path: str) -> List[Document]:
        """Process a single PDF file and return document objects"""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Load PDF
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from {pdf_path}")
            
            # Split documents
            splits = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(splits)} text chunks from {pdf_path}")
            
            # Create document objects with metadata
            document_objects = [
                Document(
                    page_content=split.page_content,
                    metadata={
                        "source": pdf_path,
                        "filename": Path(pdf_path).name,
                        "page": split.metadata.get("page", 0),
                        "chunk_size": len(split.page_content),
                        "chunk_index": i,
                        "document_type": "pdf",
                    }
                )
                for i, split in enumerate(splits)
            ]
            
            logger.info(f"Successfully processed {pdf_path}: {len(document_objects)} document objects created")
            return document_objects
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return []

    def process_all_pdfs(self) -> List[Document]:
        """Process all PDFs in the data folder"""
        pdf_files = self.find_pdf_files()
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.data_folder}")
            return []
        
        all_documents = []
        for pdf_file in pdf_files:
            documents = self.process_single_pdf(pdf_file)
            all_documents.extend(documents)
        
        logger.info(f"Total documents processed: {len(all_documents)}")
        return all_documents

    def create_vector_store(self, documents: List[Document]) -> None:
        """Create and populate vector store"""
        try:
            logger.info("Creating vector store...")
            
            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            
            if documents:
                logger.info(f"Adding {len(documents)} documents to vector store...")
                self.vector_store.add_documents(documents=documents)
                logger.info("Documents added to vector store successfully")
            else:
                logger.warning("No documents to add to vector store")
                
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise

    def setup_qa_chain(self) -> None:
        """Setup the QA chain with memory"""
        try:
            logger.info("Setting up QA chain...")
            
            if not self.vector_store:
                raise ValueError("Vector store not initialized. Call create_vector_store first.")
            
            # Create memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            # Create QA chain
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(),
                retriever=self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                ),
                memory=memory,
                return_source_documents=True,
                verbose=True
            )
            
            logger.info("QA chain setup complete")
            
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
            logger.info("Starting PDF processing pipeline...")
            
            # Create data folder if it doesn't exist
            os.makedirs(self.data_folder, exist_ok=True)
            
            # Process all PDFs
            documents = self.process_all_pdfs()
            
            # Create vector store
            self.create_vector_store(documents)
            
            # Setup QA chain
            self.setup_qa_chain()
            
            logger.info("PDF processing pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline: {str(e)}")
            raise


def main():
    """Main function for standalone usage"""
    try:
        # Initialize processor
        processor = PDFProcessor()
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
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()