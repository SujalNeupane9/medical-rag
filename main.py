from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_core.documents import Document
from typing import List, Optional, Union
import os
import shutil
from dotenv import load_dotenv
from pathlib import Path
import logging

# Import DatabaseManager from the database integration module
from database_integration import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
                                                                                                                                                                                                                                                           
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class SimpleRAGPipeline:
    def __init__(self, openai_api_key: Optional[str] = None, user_id: Optional[int] = None):
        """
        Initialize the RAG pipeline.
        
        Args:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
            openai_api_key: OpenAI API key. If not provided, will look for OPENAI_API_KEY in environment.
            user_id: Database user ID for logging purposes
        """
        # Initialize components
        logging.info("Initializing the SimpleRAGPipeline.")
        self.embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
        self.vector_store = None
        self.qa_chain = None
        self.splits = list()
        
        # Initialize database manager
        self.db_manager = DatabaseManager()
        self.user_id = user_id or 1  # Default to user ID 1 if not provided
        
        # Default chunk settings
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        
        # Ensure uploaded_files directory exists
        self.uploaded_files_dir = Path.cwd() / 'uploaded_files'
        self.uploaded_files_dir.mkdir(exist_ok=True)
        
        # Default prompt template
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
        self.prompt = PromptTemplate(
            template=self.default_template,
            input_variables=["context", "question","chat_history"]
        )
        logging.info("Pipeline initialized successfully.")

    def upload_pdfs(self, 
                pdf_paths: List[str], 
                file_name_aliases: Optional[List[str]] = None,
                chunk_size: Optional[int] = None,
                chunk_overlap: Optional[int] = None) -> List[int]:
        """
        Upload and process multiple PDF files for a specific user.
        
        Args:
            pdf_paths: List of paths to PDF files.
            file_name_aliases: Optional list of user-friendly names for files. 
                            If not provided, original file names will be used.
            chunk_size: Optional custom chunk size for text splitting.
            chunk_overlap: Optional custom chunk overlap for text splitting.
        
        Returns:
            List of database file IDs for the uploaded files
        """
        # Create user-specific directory if it doesn't exist
        user_dir = self.uploaded_files_dir / str(self.user_id)
        user_dir.mkdir(exist_ok=True)
        
        # Create a unique collection name based on user_id
        collection_name = f"user_{self.user_id}_collection"
        
        # Update text splitter if custom parameters provided
        if chunk_size or chunk_overlap:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size or self.chunk_size,
                chunk_overlap=chunk_overlap or self.chunk_overlap,
            )
        
        # Validate and process each PDF file in the list
        processed_files = []
        db_file_ids = []
        
        # If no aliases provided, create default aliases
        if file_name_aliases is None:
            file_name_aliases = [f"Research paper {i+1}" for i in range(len(pdf_paths))]
        
        for pdf_path, file_name_alias in zip(pdf_paths, file_name_aliases):
            pdf_path = Path(pdf_path)
            
            # Check if file exists
            if not pdf_path.exists():
                logging.error(f"PDF file not found at {pdf_path}. Skipping this file.")
                continue
            
            # Determine destination path, ensuring it's unique
            dest_path = user_dir / pdf_path.name
            
            # If file is already in the destination, use the existing path
            if pdf_path.resolve() == dest_path.resolve():
                logging.info(f"File {pdf_path.name} is already in the correct directory.")
            else:
                # Copy PDF to user's directory, overwriting if exists
                shutil.copy2(pdf_path, dest_path)
                logging.info(f"Copied {pdf_path.name} to {dest_path}")
            
            # Add file to database with actual filename and alias
            try:
                # Insert file record and get its database ID
                # Now passing the actual filename instead of file path
                db_file_id = self.db_manager.add_file_to_database(
                    file_path=dest_path, 
                    file_name_alias=pdf_path.name, 
                    user_id=self.user_id
                )
                db_file_ids.append(db_file_id)
            except Exception as e:
                logging.error(f"Error adding file {pdf_path.name} to database: {e}")
                continue
            
            logging.info(f"Loading PDF from {dest_path}...")
            loader = PyPDFLoader(str(dest_path))
            documents = loader.load()

            logging.info("Splitting documents into chunks...")
            splits = self.text_splitter.split_documents(documents)

            # Prepare Document instances with metadata for each split
            # Use pdf_path.name as the source to match actual filename
            document_objects = [
                Document(page_content=split.page_content, 
                        metadata={
                            "source": pdf_path.name, 
                            "user_id": self.user_id,
                            # "description": file_name_alias  # Keep alias in metadata for reference
                        })
                for split in splits
            ]

            # Initialize vector store with user-specific collection name
            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory="./chroma_db",
            )

            # Add documents with metadata
            try:
                self.vector_store.add_documents(documents=document_objects)
                logging.info(f"Documents from {dest_path} added successfully for user {self.user_id}.")
                processed_files.append(pdf_path.name)
            except Exception as e:
                logging.error(f"Error adding documents from {dest_path}: {e}")

        # Initialize QA chain after processing all PDFs
        if self.vector_store:
            self._initialize_qa_chain()
            logging.info(f"All PDFs processed for user {self.user_id}. Ready for querying.")
        else:
            logging.error("No PDFs were successfully processed.")
        
        return db_file_ids

    def query(self, question: str) -> str:
        """
        Query the RAG system.
        
        Args:
            question: Question to ask about the uploaded document(s)
        
        Returns:
            Answer from the system
        """
        # Log the query in the database
        self.db_manager.log_query(query=question, user_id=self.user_id)
        
        # Reinitialize vector store with user-specific collection
        collection_name = f"user_{self.user_id}_collection"
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory="./chroma_db",
        )
        
        # Reinitialize QA chain
        self._initialize_qa_chain()
        
        if not self.qa_chain:
            logging.warning(f"Attempted to query before uploading a PDF for user {self.user_id}.")
            return "Please upload a PDF document first using upload_pdfs()"
            
        return self.qa_chain.run(question)
    
    def query_(self, user_id, question: str, pdfs: list) -> str:
        """
        Query the RAG system.
        
        Args:
            question: Question to ask about the uploaded document(s)
        
        Returns:
            Answer from the system
        """
        # Log the query in the database
        self.db_manager.log_query(query=question, user_id=self.user_id)
        
        # Reinitialize vector store with user-specific collection
        collection_name = f"user_{self.user_id}_collection"
        self.vectordb = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory="./chroma_db",
        )
        
        # Reinitialize QA chain
        self._initialize_qa_chain()

        pdf_file = f'uploaded_files/{user_id}/'
        
        if not self.qa_chain:
            logging.warning(f"Attempted to query before uploading a PDF for user {self.user_id}.")
            return "Please upload a PDF document first using upload_pdfs()"
        
        filter_criteria = None
        if pdfs:
            filter_criteria = {"source": {"$in": f'{pdf_file}{pdfs}'}}

        relevant_docs = self.vectordb.get(where=filter_criteria)
        if not relevant_docs['Document']:
            return "No relevant documents found for the selected PDFs."
        
        results = self.vectordb.get(
            where={"source": pdf_file.filename}
            # include=["ids", "metadatas"]
        )

        return results
            
        return self.qa_chain.run(input_document = relevant_docs["documents"], question=question)

    def display_uploaded_files(self) -> List[dict]:
        """
        Display the list of uploaded PDF files for the current user.
        
        Returns:
            A list of dictionaries containing information about uploaded files
        """
        return self.db_manager.get_user_files(user_id=self.user_id)

    def delete_file(self, file_id: int) -> None:
        """
        Delete a file from both the database and vector store.
        
        Args:
            file_id: ID of the file to delete
        """
        # Get the filename from the database first
        user_files = self.db_manager.get_user_files(user_id=self.user_id)
        file_to_delete = next((f for f in user_files if f['id'] == file_id), None)
        
        if not file_to_delete:
            logging.error(f"File with ID {file_id} not found for user {self.user_id}")
            return
        
        # Delete embedding from vector store
        self.delete_embedding(file_to_delete['fileName'])
        
        # Delete file from database
        try:
            self.db_manager.delete_file_from_database(file_id=file_id, user_id=self.user_id)
            logging.info(f"File {file_to_delete['fileName']} deleted successfully.")
        except Exception as e:
            logging.error(f"Error deleting file from database: {e}")

    # Rest of the previous methods remain the same as in the original implementation
    def set_custom_prompt(self, template: str) -> None:
        """
        Set a custom prompt template.
        
        Args:
            template: Custom template string. Must include {context} and {question} variables.
        """
        if "{context}" not in template or "{question}" not in template:
            logging.error("Invalid prompt template. Missing {context} or {question}.")
            raise ValueError("Template must include {context} and {question} variables")
            
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Reinitialize QA chain with new prompt if vector store exists
        if self.vector_store:
            self._initialize_qa_chain()

    def _initialize_qa_chain(self) -> None:
        """Initialize the QA chain with current prompt and vector store."""
        logging.info("Initializing the QA chain.")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm= ChatOpenAI(),
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 3}
            ),
            chain_type_kwargs={"prompt": self.prompt}
        )

    def delete_embedding(self, filename: str) -> None:
        
        """
        Delete the embedding of a specific file for the current user and remove the file physically.

        Args:
            filename: Name of the file to delete (can be alias or actual filename).
        """
        # Create user directory path
        user_dir = self.uploaded_files_dir / str(self.user_id)

        # Retrieve the actual file information from the database
        user_files = self.db_manager.get_user_files(user_id=self.user_id)
        
        # Find the matching file, checking both fileName and fileNameAlias
        matching_file = next(
            (f for f in user_files 
            if f.get('fileName') == filename), 
            None
        )

        if not matching_file:
            logging.error(f"No file found with name: {filename}")
            raise ValueError(f"File {filename} not found for user {self.user_id}")

        # Determine the actual filename and file ID
        actual_filename = matching_file.get('fileName')
        file_id = matching_file.get('id')

        if not actual_filename or not file_id:
            logging.error(f"No valid filename or file ID found for: {filename}")
            raise ValueError(f"Invalid file information for {filename}")

        file_path = user_dir / actual_filename

        # Initialize vector store for the user
        collection_name = f"user_{self.user_id}_collection"
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory="./chroma_db",
        )

        try:
            # Delete documents from the vector store
            self.vector_store.delete(where={"source": actual_filename})
            logging.info(f"Embedding for {actual_filename} deleted successfully.")
        except Exception as e:
            logging.error(f"Error deleting embedding for {actual_filename}: {e}")
            raise

        try:
            # Delete the physical file
            if file_path.exists():
                file_path.unlink()
                logging.info(f"File {actual_filename} deleted successfully.")
            else:
                logging.warning(f"File {actual_filename} does not exist on disk.")
        except Exception as e:
            logging.error(f"Error deleting file {actual_filename}: {e}")
            raise

        # Delete the file record from the database using file_id
        try:
            self.db_manager.delete_file_from_database(
                file_id=file_id, 
                user_id=self.user_id
            )
            logging.info(f"File record for {actual_filename} deleted from database.")
        except Exception as e:
            logging.error(f"Error deleting file record from database: {e}")
            raise