import json
import chromadb
from langchain.embeddings import OpenAIEmbeddings
from langchain_aws.embeddings import BedrockEmbeddings
import boto3
from langchain_chroma import Chroma
from langchain.schema import Document
from typing import List, Dict, Any
from dotenv import load_dotenv
import os

load_dotenv()
os.environ['aws_access_key_id'] = os.getenv("aws_access_key_id")
os.environ['aws_secret_access_key'] = os.getenv("aws_secret_access_key")
os.environ['AWS_DEFAULT_REGION'] = os.getenv("AWS_DEFAULT_REGION")

class MedicalRAGRetriever:
    def __init__(self, force_rebuild: bool = False):
        """
        Initialize the Medical RAG Retriever
        
        Args:
            force_rebuild: If True, rebuild the vector store even if it exists
        """
        self.json_file_path = 'merged1.json'
        self.persist_directory = "./vox_med_chroma_db"
        self.collection_name = "medical_qa"
        
        self.bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=os.getenv("AWS_DEFAULT_REGION")
        )
        
        # Initialize Bedrock embeddings
        self.embeddings = BedrockEmbeddings(
            client=self.bedrock_client,
            model_id="amazon.titan-embed-text-v2:0",  
            region_name=os.getenv("AWS_DEFAULT_REGION")
        )
        
        self.vectorstore = None
        self.initialize_vectorstore(force_rebuild)
    
    def _vectorstore_exists(self) -> bool:
        """Check if the vector store already exists"""
        if not os.path.exists(self.persist_directory):
            return False
        
        try:
            # Try to load existing collection
            client = chromadb.PersistentClient(path=self.persist_directory)
            collections = client.list_collections()
            return any(col.name == self.collection_name for col in collections)
        except Exception:
            return False
    
    def load_json_data(self) -> List[Dict[str, Any]]:
        """Load Q&A data from JSON file"""
        with open(self.json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Handle nested list structure if present
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            # Flatten if data is [[{...}, {...}], [...]]
            flattened_data = []
            for sublist in data:
                flattened_data.extend(sublist)
            return flattened_data
        return data
    
    def prepare_documents(self, qa_data: List[Dict[str, Any]]) -> List[Document]:
        """Convert Q&A data to LangChain Document format"""
        documents = []
        
        for i, item in enumerate(qa_data):
            # Create document with question as content
            doc = Document(
                page_content=item["question"],
                metadata={
                    "answer": item["answer"],
                    "is_verified": item.get("is_verified", 0),
                    "doc_id": i,
                    "question": item["question"]  # Keep original question
                }
            )
            documents.append(doc)
        
        return documents
    
    def build_vectorstore(self):
        """Build and persist the vector store"""
        print("Building vector store from scratch...")
        print("Loading Q&A data...")
        qa_data = self.load_json_data()
        print(f"Loaded {len(qa_data)} Q&A pairs")
        
        print("Preparing documents...")
        documents = self.prepare_documents(qa_data)
        
        print("Creating embeddings and vector index...")
        # Create ChromaDB vector store
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
        print("Vector index created and persisted successfully!")
    
    def load_existing_vectorstore(self):
        """Load existing vector store from disk"""
        print("Loading existing vector store...")
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        print("Existing vector store loaded successfully!")
    
    def initialize_vectorstore(self, force_rebuild: bool = False):
        """Initialize vector store - build new or load existing"""
        if force_rebuild or not self._vectorstore_exists():
            self.build_vectorstore()
        else:
            self.load_existing_vectorstore()
    
    def retrieve_answer(self, user_question: str, similarity_threshold: float = 0.8):
        """
        Retrieve the most relevant answer for a user question
        
        Args:
            user_question: The user's question
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            Dict with question, answer, is_verified if threshold met, else None
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        # Search for similar questions (k=1 for top 1 result)
        results = self.vectorstore.similarity_search_with_score(
            query=user_question,
            k=1
        )
        
        if not results:
            return None
        
        # Get the top result
        doc, similarity_score = results[0]
        
        # Convert distance to similarity (ChromaDB returns distance, lower is better)
        similarity = 1 - similarity_score
        
        # Check if similarity meets threshold
        if similarity < similarity_threshold:
            return None
        
        # Return only the original JSON keys
        return {
            "question": doc.metadata["question"],
            "answer": doc.metadata["answer"],
            "is_verified": doc.metadata["is_verified"]
        }
    
    def search_multiple(self, user_question: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Get top k similar questions (for debugging/analysis)
        
        Args:
            user_question: The user's question
            k: Number of results to return
        """
        results = self.vectorstore.similarity_search_with_score(
            query=user_question,
            k=k
        )
        
        formatted_results = []
        for doc, score in results:
            similarity = 1 - score
            formatted_results.append({
                "question": doc.metadata["question"],
                "answer": doc.metadata["answer"],
                "similarity_score": similarity,
                "is_verified": doc.metadata["is_verified"]
            })
        
        return formatted_results
    
    def rebuild_index(self):
        """Manually rebuild the vector store index"""
        print("Manually rebuilding vector store...")
        self.build_vectorstore()
    
    def get_collection_info(self):
        """Get information about the loaded collection"""
        if not self.vectorstore:
            return "Vector store not initialized"
        
        try:
            # Get collection count
            collection = self.vectorstore._collection
            count = collection.count()
            return f"Collection '{self.collection_name}' contains {count} documents"
        except Exception as e:
            return f"Error getting collection info: {str(e)}"