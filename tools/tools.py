import asyncio
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from datetime import datetime

class VectorDatabaseTool:
    """Tool for managing and querying ChromaDB vector database with BGE-M3 embeddings"""
    
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "knowledge_base"):
        """
        Initialize the vector database tool
        
        Args:
            db_path: Path to store ChromaDB
            collection_name: Name of the collection to use
        """
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize BGE-M3 embedding model
        print("🔄 Loading BGE-M3 embedding model...")
        self.embedding_model = SentenceTransformer('BAAI/bge-m3')
        print("✅ BGE-M3 model loaded successfully!")
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        # Sample data for demonstration
        self._load_sample_data()
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            collection = self.client.get_collection(name=self.collection_name)
            print(f"📚 Using existing collection: {self.collection_name}")
        except:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Knowledge base for multi-agent system"}
            )
            print(f"📚 Created new collection: {self.collection_name}")
        
        return collection
    
    def _load_sample_data(self):
        """Load sample data into the database for demonstration"""
        # Check if collection is empty
        if self.collection.count() == 0:
            print("📝 Loading sample data...")
            
            sample_documents = [
                {
                    "id": "doc_1",
                    "text": "Artificial Intelligence (AI) is transforming healthcare by enabling early disease detection, personalized treatment plans, and automated medical imaging analysis.",
                    "metadata": {"category": "healthcare", "topic": "AI applications", "source": "medical_research"}
                },
                {
                    "id": "doc_2", 
                    "text": "Machine learning algorithms are revolutionizing financial services through fraud detection, algorithmic trading, and credit risk assessment.",
                    "metadata": {"category": "finance", "topic": "ML applications", "source": "fintech_research"}
                },
                {
                    "id": "doc_3",
                    "text": "Natural Language Processing (NLP) is enhancing education by providing personalized learning experiences, automated grading, and intelligent tutoring systems.",
                    "metadata": {"category": "education", "topic": "NLP applications", "source": "edtech_research"}
                },
                {
                    "id": "doc_4",
                    "text": "Computer vision technology is being used in autonomous vehicles for object detection, lane recognition, and traffic sign identification.",
                    "metadata": {"category": "automotive", "topic": "computer vision", "source": "autonomous_research"}
                },
                {
                    "id": "doc_5",
                    "text": "Deep learning models like transformers have revolutionized natural language processing tasks including translation, summarization, and question answering.",
                    "metadata": {"category": "AI", "topic": "deep learning", "source": "nlp_research"}
                },
                {
                    "id": "doc_6",
                    "text": "Blockchain technology provides decentralized, secure, and transparent record-keeping for financial transactions and digital assets.",
                    "metadata": {"category": "blockchain", "topic": "distributed systems", "source": "blockchain_research"}
                },
                {
                    "id": "doc_7",
                    "text": "Internet of Things (IoT) connects physical devices to the internet, enabling smart homes, industrial automation, and environmental monitoring.",
                    "metadata": {"category": "IoT", "topic": "connected devices", "source": "iot_research"}
                },
                {
                    "id": "doc_8",
                    "text": "Cloud computing provides scalable computing resources on-demand, enabling businesses to reduce infrastructure costs and improve flexibility.",
                    "metadata": {"category": "cloud", "topic": "distributed computing", "source": "cloud_research"}
                },
                {
                    "id": "doc_9",
                    "text": "Cybersecurity is crucial for protecting digital assets, with techniques including encryption, authentication, and intrusion detection systems.",
                    "metadata": {"category": "security", "topic": "digital protection", "source": "security_research"}
                },
                {
                    "id": "doc_10",
                    "text": "Data science combines statistics, programming, and domain expertise to extract insights from large datasets and inform decision-making.",
                    "metadata": {"category": "data science", "topic": "analytics", "source": "data_research"}
                }
            ]
            
            # Add documents to collection
            self.add_documents(sample_documents)
            print(f"✅ Loaded {len(sample_documents)} sample documents")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the vector database
        
        Args:
            documents: List of documents with 'id', 'text', and 'metadata' keys
        """
        try:
            # Extract data
            ids = [doc['id'] for doc in documents]
            texts = [doc['text'] for doc in documents]
            metadatas = [doc.get('metadata', {}) for doc in documents]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts).tolist()
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"✅ Added {len(documents)} documents to database")
            
        except Exception as e:
            print(f"❌ Error adding documents: {str(e)}")
            raise
    
    def query_database(self, query: str, n_results: int = 5, filter_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Query the vector database
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            Dictionary containing query results
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            # Perform query
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                where=filter_metadata
            )
            
            # Format results
            formatted_results = {
                "query": query,
                "results": [],
                "total_found": len(results['documents'][0]) if results['documents'] else 0
            }
            
            if results['documents'] and results['documents'][0] and results['metadatas'] and results['metadatas'][0] and results['distances'] and results['distances'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    formatted_results["results"].append({
                        "rank": i + 1,
                        "document": doc,
                        "metadata": metadata,
                        "similarity_score": 1 - distance,  # Convert distance to similarity
                        "id": results['ids'][0][i] if results['ids'] else f"result_{i}"
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error querying database: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "results": [],
                "total_found": 0
            }
    
    def semantic_search(self, query: str, n_results: int, categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform semantic search with optional category filtering
        
        Args:
            query: Search query
            n_results: Number of results to return
            categories: Optional list of categories to filter by
            
        Returns:
            Search results
        """
        filter_metadata = None
        if categories:
            filter_metadata = {"category": {"$in": categories}}
        
        return self.query_database(query, n_results, filter_metadata)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "total_documents": count,
                "database_path": self.db_path,
                "embedding_model": "BGE-M3"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def delete_documents(self, document_ids: List[str]):
        """Delete documents by IDs"""
        try:
            self.collection.delete(ids=document_ids)
            print(f"✅ Deleted {len(document_ids)} documents")
        except Exception as e:
            print(f"❌ Error deleting documents: {str(e)}")
    
    def update_document(self, doc_id: str, new_text: str, new_metadata: Optional[Dict] = None):
        """Update a document's text and metadata"""
        try:
            # Generate new embedding
            new_embedding = self.embedding_model.encode([new_text]).tolist()
            
            # Update document
            self.collection.update(
                ids=[doc_id],
                embeddings=new_embedding,
                documents=[new_text],
                metadatas=[new_metadata or {}]
            )
            
            print(f"✅ Updated document: {doc_id}")
            
        except Exception as e:
            print(f"❌ Error updating document: {str(e)}")

# Example usage and testing
async def test_vector_database():
    """Test the vector database functionality"""
    print("🧪 Testing Vector Database Tool...")
    
    # Initialize tool
    db_tool = VectorDatabaseTool()
    
    # Test queries
    test_queries = [
        "AI in healthcare",
        "machine learning finance",
        "blockchain technology",
        "cybersecurity protection"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        results = db_tool.semantic_search(query, 3)
        
        if results.get("error"):
            print(f"❌ Error: {results['error']}")
        else:
            print(f"📊 Found {results['total_found']} results:")
            for result in results['results']:
                print(f"  {result['rank']}. {result['document'][:100]}...")
                print(f"     Similarity: {result['similarity_score']:.3f}")
                print(f"     Category: {result['metadata'].get('category', 'N/A')}")
    
    # Get stats
    stats = db_tool.get_collection_stats()
    print(f"\n📈 Database Stats: {stats}")

if __name__ == "__main__":
    # Run test
    asyncio.run(test_vector_database())
